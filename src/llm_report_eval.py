"""Manual LLM-output evaluation harness (bootcamp guide §9: "manually evaluate
a diverse sample of outputs"; §12: accuracy/alignment, clarity, format
compliance, no invented info).

Picks a few structurally different test clips (a clear multi-behavior clip, a
borderline single-behavior clip, and a Background/no-detection clip), runs
BOTH prompt versions from src/llm_report.py on each, and writes everything
side by side to reports/llm_eval_samples.md for a human to grade.

Needs a real ANTHROPIC_API_KEY -- this script does not fabricate model output.
Without a key it writes a report saying so instead of failing silently.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_device
from src.datasets.avasd import AVASDClips
from src.fusion.model import AutiLensNet
from src.inference import Prediction
from src.llm_report import PROMPT_VERSIONS, generate_llm_report
from src.train import _collate, _infer


def _pick_examples(tag="vision_r2p1d_cv", n=3):
    cfg = load_config()
    device = resolve_device(cfg["train"]["device"])
    labels = list(cfg["labels"])
    cks = sorted(glob.glob(str(cfg.resolve_path("models_dir") / f"{tag}_fold*.pt")))
    st0 = torch.load(cks[0], map_location=device)
    cfg["model"].update(st0["cfg"]["model"])
    thr = np.load(cfg.resolve_path("models_dir") / f"{tag}_thresholds.npy")

    ds = AVASDClips(cfg, "test", train=False)
    dl = DataLoader(ds, batch_size=8, collate_fn=_collate)
    models = []
    for c in cks:
        m = AutiLensNet(cfg).to(device).eval()
        m.load_state_dict(torch.load(c, map_location=device)["model"])
        models.append(m)
    probs = np.mean([_infer(m, dl, device, tta=True)[0] for m in models], axis=0)
    ids = list(ds.meta["video_id"])
    y = ds.meta[labels].values

    n_true = y.sum(1)
    multi = int(np.argmax(n_true))                       # richest clip
    single = next((i for i in range(len(ids)) if n_true[i] == 1), 0)
    bg = next((i for i in range(len(ids)) if y[i, -1] == 1), len(ids) - 1)  # Background

    picks = []
    for idx, tagname in [(multi, "multi-behavior"), (single, "single-behavior"), (bg, "background/none")]:
        order = np.argsort(-probs[idx])
        pred = Prediction(modality=cfg["model"]["modality"])
        for j in order:
            pred.behaviors.append({
                "label": labels[j], "probability": round(float(probs[idx, j]), 4),
                "detected": bool(probs[idx, j] >= thr[j]), "threshold": round(float(thr[j]), 3),
            })
        picks.append((ids[idx], tagname, pred))
    return picks


def main():
    cfg = load_config()
    out_path = cfg.resolve_path("reports_dir") / "llm_eval_samples.md"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        out_path.write_text(
            "# LLM evaluation samples\n\n"
            "**Not run**: `ANTHROPIC_API_KEY` was not set, so no real LLM calls were made. "
            "Set the key and re-run `python -m src.llm_report_eval` to populate this file with "
            "actual model outputs for manual grading. No fabricated outputs are included here.\n"
        )
        print(out_path.read_text())
        return

    lines = ["# LLM evaluation samples (bootcamp guide §9/§12)", "",
             "Same 3 clips, both prompt versions, for manual side-by-side grading.", ""]
    for vid, kind, pred in _pick_examples():
        lines.append(f"## {vid}  ({kind})")
        lines.append(f"CV input: {json.dumps(pred.to_dict())}")
        for v in PROMPT_VERSIONS:
            out = generate_llm_report(pred, version=v)
            lines.append(f"\n### prompt {v} (source={out['source']})")
            lines.append("```")
            lines.append(json.dumps(out.get("structured", out.get("text")), indent=2))
            lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
