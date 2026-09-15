"""Single-clip inference: video file -> behavior probabilities + evidence.

Used by the FastAPI backend and the Streamlit app. No LLM is involved; the
"evidence" is derived directly from model outputs (PDF sections 8 & 11).
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, resolve_device
from src.datasets.avasd import frames_to_tensor, mel_to_tensor
from src.evaluation.metrics import expected_calibration_error  # noqa: F401 (re-export convenience)
from src.fusion.model import AutiLensNet
from src.preprocessing.features import log_mel, sample_frames


#: The two head types this project serves. ``multilabel`` is the 9-behavior
#: sigmoid head trained here; ``multiclass`` is the softmax head used by
#: externally trained checkpoints, where exactly one class wins per clip.
MULTILABEL = "multilabel"
MULTICLASS = "multiclass"


@dataclass
class Prediction:
    behaviors: list[dict] = field(default_factory=list)   # {label, probability, detected, threshold}
    evidence: list[dict] = field(default_factory=list)    # {label, window_s, note}
    modality: str = "av"
    #: ``multilabel`` (independent per-behavior sigmoids, per-class thresholds)
    #: or ``multiclass`` (softmax over mutually exclusive classes). Consumers
    #: branch on this instead of assuming the multi-label shape.
    task: str = MULTILABEL
    #: Multi-class only: {label, probability} for the winning class. ``None``
    #: for multi-label, where any number of behaviors may fire at once.
    top_class: dict | None = None
    #: Two-stage only: Stage 1 behavior families, each
    #: {label, probability, detected, threshold}. Families are MULTI-LABEL (a clip
    #: can show more than one), so this is deliberately not ``top_class``, which
    #: carries mutually-exclusive multi-class semantics. ``behaviors`` continues to
    #: mean the 9-behavior Stage 2 output.
    families: list[dict] | None = None
    disclaimer: str = (
        "This result is a research screening aid, not a diagnosis of autism. "
        "It reports recognition of dataset-defined behaviors only and must not "
        "replace assessment by qualified professionals."
    )

    def to_dict(self) -> dict:
        # ``behaviors``/``evidence``/``modality``/``disclaimer`` keep their exact
        # previous meaning and position so existing API clients are unaffected.
        out = {
            "behaviors": self.behaviors,
            "evidence": self.evidence,
            "modality": self.modality,
            "task": self.task,
            "disclaimer": self.disclaimer,
        }
        if self.top_class is not None:
            out["top_class"] = self.top_class
        if self.families is not None:
            out["families"] = self.families
        return out


class AutiLensPredictor:
    def __init__(self, checkpoint: str, config_path: str | None = None):
        self.cfg = load_config(config_path)
        self.device = resolve_device(self.cfg["train"]["device"])
        state = torch.load(checkpoint, map_location=self.device)
        if "cfg" in state:
            self.cfg["model"].update(state["cfg"].get("model", {}))
        self.labels = list(self.cfg["labels"])
        self.model = AutiLensNet(self.cfg).to(self.device).eval()
        self.model.load_state_dict(state["model"])
        self.thr = np.asarray(state.get("thresholds", [self.cfg["eval"]["default_threshold"]] * len(self.labels)))
        self.modality = self.cfg["model"]["modality"]
        self.nf = int(self.cfg["video"]["num_frames"])
        self.sz = int(self.cfg["video"]["frame_size"])

    # ----------------------------------------------------------------- #
    @staticmethod
    def _extract_wav(video_path: Path) -> Path:
        wav = Path(tempfile.mkstemp(suffix=".wav")[1])
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", "-vn", str(wav)],
            check=False, capture_output=True,
        )
        return wav

    @torch.no_grad()
    def predict(self, video_path: str | Path) -> Prediction:
        video_path = Path(video_path)
        batch: dict[str, torch.Tensor] = {}
        duration = _duration_s(video_path)

        frames_u8 = None
        if self.modality in ("vision", "av"):
            frames_u8 = sample_frames(video_path, self.nf, self.sz)
            batch["frames"] = frames_to_tensor(frames_u8, train=False).unsqueeze(0).to(self.device)
        if self.modality in ("audio", "av"):
            wav = self._extract_wav(video_path)
            src = wav if wav.stat().st_size > 1024 else video_path
            batch["mel"] = mel_to_tensor(log_mel(src, self.cfg["audio"])).unsqueeze(0).unsqueeze(0).to(self.device)
            wav.unlink(missing_ok=True)

        want_frames = self.modality in ("vision", "av")
        out = self.model(batch, return_frame_scores=want_frames)
        logits, frame_scores = out if isinstance(out, tuple) else (out, None)
        probs = torch.sigmoid(logits)[0]

        if self.cfg["eval"].get("tta", False) and "frames" in batch:
            flip = dict(batch)
            flip["frames"] = torch.flip(batch["frames"], dims=[-1])
            out2 = self.model(flip)
            logits2 = out2[0] if isinstance(out2, tuple) else out2
            probs = 0.5 * (probs + torch.sigmoid(logits2)[0])
        probs = probs.cpu().numpy()

        pred = Prediction(modality=self.modality)
        order = np.argsort(-probs)
        for i in order:
            pred.behaviors.append({
                "label": self.labels[i],
                "probability": round(float(probs[i]), 4),
                "detected": bool(probs[i] >= self.thr[i]),
                "threshold": round(float(self.thr[i]), 3),
            })

        # Evidence: for each detected behavior, the temporal window with the
        # strongest per-frame activation.
        if frame_scores is not None and duration > 0:
            fs = torch.sigmoid(frame_scores)[0].cpu().numpy()  # (T, n_cls)
            T = fs.shape[0]
            for i in order:
                if probs[i] < self.thr[i]:
                    continue
                t_star = int(np.argmax(fs[:, i]))
                centre = duration * (t_star + 0.5) / T
                half = max(duration / T, 0.5)
                pred.evidence.append({
                    "label": self.labels[i],
                    "window_s": [round(max(0.0, centre - half), 2), round(min(duration, centre + half), 2)],
                    "peak_frame_score": round(float(fs[t_star, i]), 3),
                    "note": "highest model activation for this behavior in the clip (approximate)",
                })
        return pred


def _duration_s(path: Path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0
