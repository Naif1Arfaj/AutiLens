"""Qualitative error analysis (bootcamp guide §7/§8): dump correct vs. failed
predictions on the held-out test split, with a representative frame thumbnail
for each, so the short report can show real examples instead of only metrics.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_device
from src.datasets.avasd import AVASDClips
from src.fusion.model import AutiLensNet
from src.preprocessing.features import sample_frames
from src.train import _collate, _infer


def _load_ensemble(tag: str, cfg, device):
    cks = sorted(glob.glob(str(cfg.resolve_path("models_dir") / f"{tag}_fold*.pt")))
    if not cks:
        ck = cfg.resolve_path("models_dir") / f"{tag}.pt"
        cks = [str(ck)]
    st0 = torch.load(cks[0], map_location=device)
    if "cfg" in st0:
        cfg["model"].update(st0["cfg"]["model"])
    models = []
    for c in cks:
        m = AutiLensNet(cfg).to(device).eval()
        m.load_state_dict(torch.load(c, map_location=device)["model"])
        models.append(m)
    thr_path = cfg.resolve_path("models_dir") / f"{tag}_thresholds.npy"
    thr = np.load(thr_path) if thr_path.exists() else np.asarray(st0.get("thresholds", [0.5] * len(cfg["labels"])))
    return models, thr


def _thumbnail(video_dir: Path, vid: str, out: Path, size: int = 224) -> None:
    frames = sample_frames(video_dir / f"{vid}.mp4", num_frames=1, size=size)
    cv2.imwrite(str(out), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))


def run(tag: str, config_path: str | None = None, n_examples: int = 4) -> dict:
    cfg = load_config(config_path)
    device = resolve_device(cfg["train"]["device"])
    labels = list(cfg["labels"])
    models, thr = _load_ensemble(tag, cfg, device)

    ds = AVASDClips(cfg, "test", train=False)
    dl = DataLoader(ds, batch_size=8, collate_fn=_collate)
    probs = np.mean([_infer(m, dl, device, tta=cfg["eval"].get("tta", False))[0] for m in models], axis=0)
    _, y = _infer(models[0], dl, device)
    ids = list(ds.meta["video_id"])
    pred = (probs >= thr).astype(int)

    out_dir = cfg.resolve_path("reports_dir") / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = cfg.resolve_path("clips_video")

    rows = []
    for i, vid in enumerate(ids):
        true_labs = [labels[j] for j in range(len(labels)) if y[i, j] == 1]
        pred_labs = [labels[j] for j in range(len(labels)) if pred[i, j] == 1]
        exact = set(true_labs) == set(pred_labs)
        fp = sorted(set(pred_labs) - set(true_labs))
        fn = sorted(set(true_labs) - set(pred_labs))
        # per-clip F1 over the 10 labels: the realistic "how good was this
        # prediction" score for a multi-label task (exact-match is almost
        # always 0 with 10 independent binary decisions and this little data).
        tp = len(set(true_labs) & set(pred_labs))
        f1 = (2 * tp / (2 * tp + len(fp) + len(fn))) if (true_labs or pred_labs) else 1.0
        rows.append({
            "video_id": vid, "exact": exact, "clip_f1": f1,
            "true": true_labs, "predicted": pred_labs,
            "false_positives": fp, "false_negatives": fn,
            "max_true_prob": float(probs[i, [labels.index(l) for l in true_labs]].max()) if true_labs else None,
            "n_errors": len(fp) + len(fn),
        })

    with_true = [r for r in rows if r["true"]]  # ignore Background-only clips for "best"
    correct_rows = sorted(with_true, key=lambda r: (-r["clip_f1"], -r["max_true_prob"]))[:n_examples]
    failed_rows = sorted(rows, key=lambda r: (r["clip_f1"], -r["n_errors"]))[:n_examples]

    for r in correct_rows + failed_rows:
        try:
            _thumbnail(video_dir, r["video_id"], out_dir / f"{r['video_id']}.jpg")
            r["thumbnail"] = f"examples/{r['video_id']}.jpg"
        except Exception as e:  # noqa: BLE001
            r["thumbnail"] = None

    summary = {
        "tag": tag, "n_test": len(ids),
        "n_exact_correct": sum(r["exact"] for r in rows),
        "mean_clip_f1": float(np.mean([r["clip_f1"] for r in rows])),
        "correct_examples": correct_rows,
        "failed_examples": failed_rows,
    }
    (cfg.resolve_path("reports_dir") / f"{tag}_examples.json").write_text(json.dumps(summary, indent=2))

    lines = [f"# Qualitative examples — {tag}", "",
             f"{summary['n_exact_correct']}/{summary['n_test']} test clips have every one of the "
             f"10 labels exactly right (rare by construction — 10 independent binary decisions per "
             f"clip). Mean per-clip F1 = {summary['mean_clip_f1']:.2f}. Below: the clips the model "
             f"got most right (by per-clip F1) and most wrong.", "", "## Correct / strong examples"]
    for r in correct_rows:
        lines.append(f"- **{r['video_id']}** (clip F1 {r['clip_f1']:.2f}) — true: "
                     f"{', '.join(r['true'])} | predicted: {', '.join(r['predicted']) or '(none)'}")
    lines.append("\n## Failed examples")
    for r in failed_rows:
        parts = []
        if r["false_negatives"]:
            parts.append(f"missed: {', '.join(r['false_negatives'])}")
        if r["false_positives"]:
            parts.append(f"false alarm: {', '.join(r['false_positives'])}")
        lines.append(f"- **{r['video_id']}** — true: {', '.join(r['true']) or '(none)'} | " + "; ".join(parts))
    (cfg.resolve_path("reports_dir") / f"{tag}_examples.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    run(a.tag, a.config, a.n)
