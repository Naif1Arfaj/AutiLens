"""Evaluate a trained checkpoint on the test split + write plots (PDF section 10)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_device
from src.datasets.avasd import AVASDClips
from src.evaluation.metrics import compute_metrics, confusion_by_class
from src.fusion.model import AutiLensNet
from src.train import _collate, _infer


def load_model(ckpt_path: str, cfg, device):
    state = torch.load(ckpt_path, map_location=device)
    if "cfg" in state:  # keep the modality/temporal the checkpoint was trained with
        for k in ("model",):
            cfg[k].update(state["cfg"].get(k, {}))
    model = AutiLensNet(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    thr = np.asarray(state.get("thresholds", [0.5] * len(cfg["labels"])))
    return model, thr


def _plot_confusion(mat, labels, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l[:18] for l in labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([l[:18] for l in labels], fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true label")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def evaluate(ckpt: str, config_path=None, split="test") -> dict:
    cfg = load_config(config_path)
    device = resolve_device(cfg["train"]["device"])
    model, thr = load_model(ckpt, cfg, device)

    ds = AVASDClips(cfg, split, train=False)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], collate_fn=_collate)
    probs, ys = _infer(model, dl, device, tta=cfg["eval"].get("tta", False))

    metrics = compute_metrics(ys, probs, cfg["labels"], thr)
    mat = confusion_by_class(ys, probs, cfg["labels"], thr)

    reports_dir = cfg.resolve_path("reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(ckpt).stem
    _plot_confusion(mat, cfg["labels"], reports_dir / f"{tag}_{split}_confusion.png")
    out = {
        "checkpoint": ckpt, "split": split, "modality": cfg["model"]["modality"],
        "metrics": metrics, "confusion_rows_true_cols_pred": mat.tolist(),
    }
    (reports_dir / f"{tag}_{split}_metrics.json").write_text(json.dumps(out, indent=2, default=float))

    print(f"\n=== {tag} on {split} ({metrics['n_samples']} clips) ===")
    print(f"macro-F1 {metrics['macro_f1']:.3f} | macro-AUPRC {metrics['macro_auprc']:.3f} "
          f"| macro-AUROC {metrics['macro_auroc']:.3f} | ECE {metrics['ece']:.3f}")
    print(f"{'behavior':42s} {'sup':>4s} {'F1':>6s} {'P':>6s} {'R':>6s} {'AUPRC':>6s}")
    for lab, d in metrics["per_class"].items():
        print(f"{lab:42s} {d['support']:4d} {d['f1']:6.2f} {d['precision']:6.2f} "
              f"{d['recall']:6.2f} {d['auprc']:6.2f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    evaluate(a.checkpoint, a.config, a.split)
