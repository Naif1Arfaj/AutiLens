"""Inference for the cached-feature pipeline (frozen backbone + k fold heads).

Serves the k-fold ENSEMBLE, not a single fold. The previous API resolved to one
fold checkpoint, so the demo ran a weaker model than the reported numbers.

Per-window scoring also gives genuine evidence timestamps: each window maps to a
real (start_s, end_s) in the clip, so the Evidence panel cites where in the video
a behaviour scored highest instead of an approximate saliency guess.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, resolve_device
from src.evaluation.calibration import apply_calibrators, load_calibrators
from src.fusion.head import FeatureHead
from src.inference import MULTICLASS, MULTILABEL, Prediction
from src.preprocessing.extract_features import (
    _Norm, _mel_to_image, build_audio_encoder, build_vision_backbone,
)
from src.preprocessing.features import log_mel
from src.preprocessing.windows import sample_windows, window_time_spans

_KINETICS = ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])

#: Legacy hint only. Selection is now by measured accuracy -- see
#: ``src/model_registry.py``. Kept so older scripts importing it still work.
PRODUCTION_CKPT = "autilens_v2.pt"

_MULTICLASS_ALIASES = {MULTICLASS, "multi-class", "single-label", "categorical"}
_MULTILABEL_ALIASES = {MULTILABEL, "multi-label", "multi_label"}


def detect_task(state: dict, cfg) -> tuple[str, list[str]]:
    """Work out whether a checkpoint holds a sigmoid or a softmax head.

    The heads trained in this repo are multi-label over the 9 dataset behaviors.
    Externally trained multi-class checkpoints carry their own class list and no
    per-class threshold vector, because a softmax head has nothing to threshold.

    Resolution order, most explicit first:
      1. ``state["task"]`` / ``state["cfg"]["model"]["task"]``
      2. a ``classes``/``class_names`` list with no ``thresholds`` -> multiclass
      3. default: multilabel (every checkpoint currently in ``models/``)
    """
    declared = (state.get("task")
                or state.get("cfg", {}).get("model", {}).get("task")
                or "")
    # ``target_names`` is written by src/cv_fast.py and is how a Stage-1 family
    # checkpoint declares its 4 outputs. Without it the predictor would load 9
    # label names against 4 logits and mislabel every prediction.
    classes = (state.get("classes") or state.get("class_names")
               or state.get("target_names"))

    if str(declared).lower() in _MULTICLASS_ALIASES:
        task = MULTICLASS
    elif str(declared).lower() in _MULTILABEL_ALIASES:
        task = MULTILABEL
    elif classes is not None and state.get("thresholds") is None:
        task = MULTICLASS
    else:
        task = MULTILABEL

    labels = list(classes) if classes else list(cfg["labels"])
    return task, labels


def _load_any_calibrators(ckpt: Path):
    """Calibrator files, most-correct first.

    `_calibrators_final.pkl` is refitted on the pooled outer-OOF predictions of all
    development clips after a candidate is locked, and is what production should
    use. The per-fold files exist because nested CV fits one set per outer fold;
    fold 0 is a reasonable stand-in before the final refit exists.
    """
    for suffix in ("_calibrators_final.pkl", "_calibrators.pkl", "_calibrators_fold0.pkl"):
        cals = load_calibrators(ckpt.with_name(ckpt.stem + suffix))
        if cals is not None:
            return cals
    return None


class FastEnsemblePredictor:
    def __init__(self, checkpoint: str | Path, config_path: str | None = None):
        ckpt = Path(checkpoint)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = load_config(config_path)
        for section in ("features", "model", "windows", "audio", "head", "eval"):
            if section in state.get("cfg", {}):
                self.cfg[section].update(state["cfg"][section])

        self.device = resolve_device(self.cfg["train"]["device"])
        self.task, self.labels = detect_task(state, self.cfg)
        # FeatureHead sizes its classifier from cfg["labels"], so a multi-class
        # checkpoint with its own class list must overwrite it before the heads
        # are built -- otherwise load_state_dict fails on the output dimension.
        self.cfg["labels"] = list(self.labels)
        self.modality = self.cfg["model"]["modality"]
        # Softmax heads have nothing to threshold: the winning class is argmax.
        self.thr = (np.zeros(len(self.labels)) if self.task == MULTICLASS
                    else np.asarray(state.get(
                        "thresholds",
                        [self.cfg["eval"]["default_threshold"]] * len(self.labels))))
        # Calibrators are per-class isotonic fits over sigmoid outputs; they do
        # not transfer to a softmax head, whose probabilities must sum to 1.
        self.cals = None if self.task == MULTICLASS else _load_any_calibrators(ckpt)

        w = self.cfg["windows"]
        self.n_windows = int(w["n_windows"])
        self.num_frames = int(w["num_frames"])
        self.frame_size = int(w["frame_size"])
        self.target_fps = float(w["target_fps"])
        self.flips = int(self.cfg["features"]["flips"])

        # Frozen encoders (shared by every fold head).
        self.vis_model = self.aud_model = None
        vdim = adim = 0
        if self.modality in ("vision", "av"):
            self.vis_model, vdim, self.in_size = build_vision_backbone(
                self.cfg["features"]["vision_backbone"], self.device)
            self.vnorm = _Norm(*_KINETICS, self.device)
        if self.modality in ("audio", "av"):
            self.aud_model, adim = build_audio_encoder(
                self.cfg["features"]["audio_encoder"], self.device)

        self.heads = []
        for sd in state["folds"]:
            h = FeatureHead(self.cfg, vdim, adim).to(self.device).eval()
            h.load_state_dict(sd)
            self.heads.append(h)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_wav(video_path: Path) -> Path:
        wav = Path(tempfile.mkstemp(suffix=".wav")[1])
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000",
             "-vn", str(wav)],
            check=False, capture_output=True,
        )
        return wav

    @torch.no_grad()
    def _vision_features(self, video_path: Path) -> torch.Tensor:
        arr = sample_windows(video_path, self.num_frames, self.frame_size,
                             self.n_windows, self.target_fps)
        x = torch.from_numpy(arr).to(self.device).float().div_(255.0)
        w, t = x.shape[0], x.shape[1]
        x = x.permute(0, 1, 4, 2, 3).reshape(w * t, 3, *x.shape[2:4])
        if x.shape[-1] != self.in_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)
        x = self.vnorm(x).view(w, t, 3, self.in_size, self.in_size)

        feats = []
        for f in range(self.flips):
            xf = torch.flip(x, dims=[-1]) if f == 1 else x
            feats.append(self.vis_model(xf.permute(0, 2, 1, 3, 4)))   # (W,D)
        return torch.stack(feats, dim=1)                              # (W,F,D)

    @torch.no_grad()
    def _audio_features(self, video_path: Path) -> torch.Tensor:
        wav = self._extract_wav(video_path)
        src = wav if wav.exists() and wav.stat().st_size > 1024 else video_path
        mel = log_mel(src, self.cfg["audio"])
        wav.unlink(missing_ok=True)
        return self.aud_model(_mel_to_image(mel, self.device))        # (1,D)

    @torch.no_grad()
    def extract_features(self, video_path: str | Path) -> dict:
        """Run the frozen backbones ONCE and return their outputs.

        Split out from ``predict`` so a two-stage system can compute features a
        single time and feed both heads. Re-running Swin3D-T per head would roughly
        double latency for no benefit — nothing in it is stage-specific.
        """
        video_path = Path(video_path)
        return {
            "vision": self._vision_features(video_path) if self.vis_model is not None else None,
            "audio": self._audio_features(video_path) if self.aud_model is not None else None,
            "path": video_path,
        }

    def feature_signature(self) -> tuple:
        """What a cached feature dict is valid for. Two predictors may share
        features only if these match exactly."""
        return (self.cfg["features"]["vision_backbone"],
                self.cfg["features"]["audio_encoder"],
                self.modality, self.n_windows, self.num_frames,
                self.frame_size, self.target_fps, self.flips)

    @torch.no_grad()
    def predict(self, video_path: str | Path) -> Prediction:
        return self.score_features(self.extract_features(video_path))

    @torch.no_grad()
    def score_features(self, feats: dict) -> Prediction:
        """Apply this predictor's heads to already-computed backbone features."""
        video_path = feats["path"]
        vfeat, afeat = feats["vision"], feats["audio"]

        n_w = vfeat.shape[0] if vfeat is not None else 1
        # (W, C) per-window probabilities, averaged over folds and flips.
        per_window = np.zeros((n_w, len(self.labels)), dtype=np.float64)
        for w in range(n_w):
            batch = {}
            if vfeat is not None:
                batch["vision"] = vfeat[w].unsqueeze(0)          # (1,F,D)
            if afeat is not None:
                batch["audio"] = afeat.unsqueeze(0)              # (1,1,D)
            logits = [h(batch) for h in self.heads]
            act = torch.softmax if self.task == MULTICLASS else torch.sigmoid
            kw = {"dim": -1} if self.task == MULTICLASS else {}
            probs = [act(lg, **kw)[0].cpu().numpy() for lg in logits]
            per_window[w] = np.mean(probs, axis=0)

        overall = per_window.mean(axis=0)
        if self.cals:
            overall = apply_calibrators(self.cals, overall[None])[0]
            per_window = apply_calibrators(self.cals, per_window)

        spans = window_time_spans(video_path, self.num_frames, self.n_windows,
                                  self.target_fps)
        return build_prediction(self.labels, self.task, self.modality,
                                overall, per_window, spans, self.thr)


def build_prediction(labels, task, modality, overall, per_window, spans,
                     thresholds=None) -> Prediction:
    """Turn averaged probabilities into a Prediction, per head type.

    Split out of ``predict`` so both shapes can be exercised with plain arrays,
    without instantiating a frozen video backbone -- which matters most for the
    multi-class path, whose checkpoints do not exist in this repo yet.

    ``overall``      (C,)  probabilities averaged over folds and windows
    ``per_window``   (W,C) the same before the window average, for evidence
    ``spans``        [(start_s, end_s)] one per window
    ``thresholds``   (C,)  multi-label only; ignored for multi-class
    """
    overall = np.asarray(overall, dtype=np.float64)
    per_window = np.atleast_2d(np.asarray(per_window, dtype=np.float64))
    pred = Prediction(modality=modality, task=task)
    order = np.argsort(-overall)
    winner = int(order[0])

    if task == MULTICLASS:
        # Renormalise after averaging folds and windows: the mean of several
        # simplex points stays on the simplex, but rounding drifts.
        total = float(overall.sum()) or 1.0
        for i in order:
            pred.behaviors.append({
                "label": labels[i],
                "probability": round(float(overall[i]) / total, 4),
                "detected": bool(i == winner),
                "threshold": None,
            })
        pred.top_class = {"label": labels[winner],
                          "probability": round(float(overall[winner]) / total, 4)}
        fires = [winner]
    else:
        thr = (np.asarray(thresholds) if thresholds is not None
               else np.full(len(labels), 0.5))
        for i in order:
            pred.behaviors.append({
                "label": labels[i],
                "probability": round(float(overall[i]), 4),
                "detected": bool(overall[i] >= thr[i]),
                "threshold": round(float(thr[i]), 3),
            })
        fires = [int(i) for i in order if overall[i] >= thr[i]]

    for i in fires:
        w_star = int(np.argmax(per_window[:, i]))
        s, e = spans[w_star] if w_star < len(spans) else (0.0, 0.0)
        pred.evidence.append({
            "label": labels[i],
            "window_s": [s, e],
            "peak_frame_score": round(float(per_window[w_star, i]), 3),
            "note": f"window {w_star + 1}/{per_window.shape[0]} scored highest "
                    f"for this {'class' if task == MULTICLASS else 'behavior'}",
        })
    return pred


class TwoStagePredictor:
    """Stage 1 (behavior families) + Stage 2 (9 behaviors), one backbone pass.

    Stage 2 is the existing 9-behavior checkpoint, used unchanged. Its output is
    masked to the families Stage 1 actually detected, so the specific behavior named
    is always consistent with the family reported above it.

    Both stages read the SAME frozen features: the backbones run once per clip and
    both head-sets are applied to the result.
    """

    def __init__(self, stage1_ckpt, stage2_ckpt, config_path: str | None = None):
        self.stage1 = FastEnsemblePredictor(stage1_ckpt, config_path)
        self.stage2 = FastEnsemblePredictor(stage2_ckpt, config_path)
        if self.stage1.feature_signature() != self.stage2.feature_signature():
            raise ValueError(
                "Stage 1 and Stage 2 were trained on different feature setups, so a "
                "shared backbone pass would be invalid:\n"
                f"  stage1 {self.stage1.feature_signature()}\n"
                f"  stage2 {self.stage2.feature_signature()}")

        st = torch.load(Path(stage1_ckpt), map_location="cpu", weights_only=False)
        self.member_groups = st.get("member_groups")
        if self.member_groups is None:
            raise ValueError(f"{stage1_ckpt} has no `member_groups`; not a Stage-1 "
                             "family checkpoint (retrain with --target families)")
        self.modality = self.stage1.modality

    @torch.no_grad()
    def predict(self, video_path: str | Path) -> Prediction:
        # ONE backbone pass, shared by both stages.
        feats = self.stage1.extract_features(video_path)
        fam = self.stage1.score_features(feats)
        beh = self.stage2.score_features(feats)

        detected = {f["label"] for f in fam.behaviors if f["detected"]}
        member_of = {self.stage1.labels[i]: {self.stage2.labels[j] for j in g}
                     for i, g in enumerate(self.member_groups)}
        allowed_behaviors = set().union(*[member_of[f] for f in detected]) if detected else set()

        # Stage 2 may only name behaviors inside a detected family.
        masked = [dict(b, detected=bool(b["detected"] and b["label"] in allowed_behaviors))
                  for b in beh.behaviors]

        return Prediction(
            behaviors=masked,
            evidence=fam.evidence + [e for e in beh.evidence
                                     if e["label"] in allowed_behaviors],
            modality=self.modality,
            task=fam.task,
            families=fam.behaviors,
        )


def resolve_ensemble(models_dir: Path) -> Path | None:
    """Pick the checkpoint to serve: the one with the best measured accuracy.

    Selection used to be by filename. Plain alphabetical order picked
    ``audio_fast.pt`` -- the weakest model in the directory -- and the later
    ``PRODUCTION_CKPT`` fix hardcoded a name that training has since overtaken.
    ``src/model_registry`` ranks on the test-set scores in ``reports/``, so the
    best checkpoint wins on merit and new ones are picked up automatically.
    """
    from src.model_registry import select_best

    # Single-model serving is the 9-behavior scope (Stage 2 / legacy). Ranking is
    # on OUT-OF-FOLD score, restricted to that scope so a 4-family model cannot win
    # on a number from a different exam. Two-stage serving uses TwoStagePredictor.
    card = select_best(models_dir, n_classes=9)
    return card.path if card else None
