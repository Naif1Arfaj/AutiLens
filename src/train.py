"""Train an AutiLens behavior-recognition model (PDF section 9).

Usage:
  python -m src.train --config configs/default.yaml
  python -m src.train --modality vision --temporal lstm --tag vision_lstm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_device, set_seed
from src.datasets.avasd import AVASDClips
from src.evaluation.metrics import compute_metrics, tune_thresholds
from src.fusion.model import AutiLensNet


class FocalLoss(torch.nn.Module):
    """Multi-label focal loss with logits (PDF section 9: rare behavior classes)."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits, target):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        return (bce * (1 - p_t).pow(self.gamma)).mean()


def build_criterion(cfg, pos_weight):
    if cfg["train"].get("loss", "bce") == "focal":
        return FocalLoss(cfg["train"].get("focal_gamma", 2.0), pos_weight)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _collate(batch):
    out = {"video_id": [b["video_id"] for b in batch],
           "labels": torch.stack([b["labels"] for b in batch])}
    if "frames" in batch[0]:
        out["frames"] = torch.stack([b["frames"] for b in batch])
    if "mel" in batch[0]:
        w = max(b["mel"].shape[-1] for b in batch)
        out["mel"] = torch.stack([
            torch.nn.functional.pad(b["mel"], (0, w - b["mel"].shape[-1])) for b in batch
        ])
    return out


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def _infer(model, loader, device, tta: bool = False):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        dev_batch = _to_device(batch, device)
        out = model(dev_batch)
        logits = out[0] if isinstance(out, tuple) else out
        p = torch.sigmoid(logits)
        if tta and "frames" in dev_batch:
            flip = dict(dev_batch)
            flip["frames"] = torch.flip(dev_batch["frames"], dims=[-1])
            out2 = model(flip)
            logits2 = out2[0] if isinstance(out2, tuple) else out2
            p = 0.5 * (p + torch.sigmoid(logits2))
        probs.append(p.cpu().numpy())
        ys.append(batch["labels"].numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train(cfg, tag: str) -> dict:
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    print(f"device={device}  modality={cfg['model']['modality']}  temporal={cfg['model']['temporal']}")

    tr_ds = AVASDClips(cfg, "train")
    va_ds = AVASDClips(cfg, "val")
    dl_kw = dict(batch_size=cfg["train"]["batch_size"], collate_fn=_collate,
                 num_workers=cfg["train"]["num_workers"])
    tr = DataLoader(tr_ds, shuffle=True, drop_last=False, **dl_kw)
    va = DataLoader(va_ds, shuffle=False, **dl_kw)

    model = AutiLensNet(cfg).to(device)
    pos_w = tr_ds.pos_weight().to(device)
    crit = build_criterion(cfg, pos_w).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train"]["epochs"])

    models_dir = cfg.resolve_path("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"{tag}.pt"

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for batch in tr:
            batch = _to_device(batch, device)
            opt.zero_grad()
            loss = crit(model(batch), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item() * len(batch["labels"])
        sched.step()

        vp, vy = _infer(model, va, device)
        m = compute_metrics(vy, vp, cfg["labels"])
        row = {"epoch": epoch, "train_loss": tot / len(tr_ds),
               "val_macro_f1": m["macro_f1"], "val_macro_auprc": m["macro_auprc"],
               "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{epoch:03d}] loss={row['train_loss']:.4f} "
              f"val_macroF1={m['macro_f1']:.3f} val_macroAUPRC={m['macro_auprc']:.3f} ({row['sec']}s)")

        if m["macro_f1"] > best:
            best, best_epoch = m["macro_f1"], epoch
            torch.save({"model": model.state_dict(), "cfg": dict_cfg(cfg), "epoch": epoch}, ckpt_path)
        if epoch - best_epoch >= cfg["train"]["early_stop_patience"]:
            print(f"early stop at epoch {epoch} (best {best:.3f} @ {best_epoch})")
            break

    # Re-load best, tune thresholds on val, persist them alongside the checkpoint.
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    vp, vy = _infer(model, va, device, tta=cfg["eval"].get("tta", False))
    thr = tune_thresholds(vy, vp) if cfg["eval"]["tune_thresholds"] else np.full(len(cfg["labels"]), 0.5)
    state["thresholds"] = thr.tolist()
    torch.save(state, ckpt_path)

    result = {
        "tag": tag,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "val_metrics_tuned": compute_metrics(vy, vp, cfg["labels"], thr),
        "history": history,
    }
    (models_dir / f"{tag}_train.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"\nbest val macro-F1 (0.5 thr) = {best:.3f} @ epoch {best_epoch}  ->  {ckpt_path}")
    return result


def dict_cfg(cfg):
    return json.loads(json.dumps(cfg))


def apply_overrides(cfg, args):
    if args.modality:
        cfg["model"]["modality"] = args.modality
    if args.temporal:
        cfg["model"]["temporal"] = args.temporal
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.backbone:
        cfg["model"]["vision_backbone"] = args.backbone
    if args.loss:
        cfg["train"]["loss"] = args.loss
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--modality", choices=["vision", "audio", "av"])
    ap.add_argument("--temporal", choices=["mean", "lstm", "transformer"])
    ap.add_argument("--backbone")
    ap.add_argument("--loss", choices=["bce", "focal"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), a)
    tag = a.tag or f"{cfg['model']['modality']}_{cfg['model']['temporal']}"
    train(cfg, tag)
