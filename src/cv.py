"""Subject-disjoint k-fold cross-validation + test ensemble (PDF sections 9-10).

Why: the shipped validation split has only 27 usable clips, so single-run
metrics are very high variance. This merges train+val, splits the *subjects*
into k folds, trains one model per fold, and reports:

  * out-of-fold (OOF) metrics  -> honest validation estimate, mean +/- std
  * test metrics for the k-model ensemble (probabilities averaged)

The official test split is never touched during training.

Usage:
  python -m src.cv --modality av --backbone r2plus1d_18 --tag av_r2p1d_cv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_config, resolve_device, set_seed
from src.datasets.avasd import AVASDClips
from src.evaluation.metrics import compute_metrics, tune_thresholds
from src.fusion.model import AutiLensNet
from src.train import _collate, _infer, apply_overrides, build_criterion


def _fold_assignment(subjects: list[str], k: int, seed: int) -> np.ndarray:
    uniq = sorted(set(subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {s: i % k for i, s in enumerate(uniq)}
    return np.array([fold_of[s] for s in subjects])


def _train_one(cfg, tr_ds, va_ds, device) -> AutiLensNet:
    set_seed(cfg["seed"])
    dl_kw = dict(batch_size=cfg["train"]["batch_size"], collate_fn=_collate, num_workers=0)
    tr = DataLoader(tr_ds, shuffle=True, **dl_kw)
    va = DataLoader(va_ds, shuffle=False, **dl_kw)

    model = AutiLensNet(cfg).to(device)
    crit = build_criterion(cfg, _pos_weight(tr_ds, cfg).to(device)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train"]["epochs"])

    best, best_state, best_ep = -1.0, None, -1
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        for batch in tr:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad()
            loss = crit(model(batch), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sched.step()
        vp, vy = _infer(model, va, device, tta=cfg["eval"].get("tta", False))
        f1 = compute_metrics(vy, vp, cfg["labels"])["macro_f1"]
        if f1 > best:
            best, best_ep = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_ep >= cfg["train"]["early_stop_patience"]:
            break
    model.load_state_dict(best_state)
    return model


def _pos_weight(tr_subset, cfg) -> torch.Tensor:
    base = tr_subset.dataset if isinstance(tr_subset, Subset) else tr_subset
    idx = tr_subset.indices if isinstance(tr_subset, Subset) else range(len(base))
    y = base.meta.iloc[list(idx)][cfg["labels"]].values.astype("float32")
    pos = y.sum(0)
    neg = len(y) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.minimum(w, cfg["train"]["pos_weight_clip"]), dtype=torch.float32)


def run(cfg, tag: str) -> dict:
    device = resolve_device(cfg["train"]["device"])
    k = cfg["cv"]["folds"]
    labels = list(cfg["labels"])
    print(f"device={device}  modality={cfg['model']['modality']}  backbone={cfg['model']['vision_backbone']}  folds={k}")

    # Pool train+val by pointing a dataset at a temporary "trainval" split file.
    splits_dir = cfg.resolve_path("splits_dir")
    ids = []
    for s in ("train", "val"):
        ids += [l.strip() for l in (splits_dir / f"{s}.txt").read_text().splitlines() if l.strip()]
    (splits_dir / "trainval.txt").write_text("\n".join(ids) + "\n")

    pool = AVASDClips(cfg, "trainval", train=True)
    pool_eval = AVASDClips(cfg, "trainval", train=False)
    folds = _fold_assignment(list(pool.meta["subject_id"]), k, cfg["seed"])

    test_ds = AVASDClips(cfg, "test", train=False)
    test_dl = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], collate_fn=_collate)

    models_dir = cfg.resolve_path("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    oof_prob = np.zeros((len(pool), len(labels)), dtype=np.float32)
    oof_true = pool.meta[labels].values.astype("float32")
    test_probs = []
    fold_f1 = []

    for f in range(k):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        model = _train_one(cfg, Subset(pool, tr_idx.tolist()), Subset(pool_eval, va_idx.tolist()), device)

        va_dl = DataLoader(Subset(pool_eval, va_idx.tolist()), batch_size=cfg["train"]["batch_size"],
                           collate_fn=_collate)
        vp, vy = _infer(model, va_dl, device, tta=cfg["eval"].get("tta", False))
        oof_prob[va_idx] = vp
        fold_f1.append(compute_metrics(vy, vp, labels)["macro_f1"])

        tp, _ = _infer(model, test_dl, device, tta=cfg["eval"].get("tta", False))
        test_probs.append(tp)
        torch.save({"model": model.state_dict(), "cfg": json.loads(json.dumps(cfg)), "fold": f},
                   models_dir / f"{tag}_fold{f}.pt")
        print(f"  fold {f}: val_macroF1={fold_f1[-1]:.3f}  (train {len(tr_idx)} / val {len(va_idx)} clips)")

    thr = tune_thresholds(oof_true, oof_prob) if cfg["eval"]["tune_thresholds"] else np.full(len(labels), 0.5)
    oof_metrics = compute_metrics(oof_true, oof_prob, labels, thr)

    ens_prob = np.mean(test_probs, axis=0)
    test_true = test_ds.meta[labels].values.astype("float32")
    ens_metrics = compute_metrics(test_true, ens_prob, labels, thr)
    single_test = [compute_metrics(test_true, tp, labels, thr)["macro_f1"] for tp in test_probs]

    np.save(models_dir / f"{tag}_thresholds.npy", thr)
    result = {
        "tag": tag,
        "modality": cfg["model"]["modality"],
        "backbone": cfg["model"]["vision_backbone"],
        "folds": k,
        "oof_macro_f1": oof_metrics["macro_f1"],
        "oof_macro_auprc": oof_metrics["macro_auprc"],
        "fold_val_macro_f1": {"mean": float(np.mean(fold_f1)), "std": float(np.std(fold_f1)), "per_fold": fold_f1},
        "test_single_model_macro_f1": {"mean": float(np.mean(single_test)), "std": float(np.std(single_test))},
        "test_ensemble_metrics": ens_metrics,
        "thresholds": thr.tolist(),
    }
    reports = cfg.resolve_path("reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{tag}_cv.json").write_text(json.dumps(result, indent=2, default=float))

    print(f"\n== {tag} ==")
    print(f"OOF  macro-F1 {oof_metrics['macro_f1']:.3f} | macro-AUPRC {oof_metrics['macro_auprc']:.3f} | ECE {oof_metrics['ece']:.3f}")
    print(f"fold val macro-F1  {np.mean(fold_f1):.3f} +/- {np.std(fold_f1):.3f}")
    print(f"TEST single-model  macro-F1 {np.mean(single_test):.3f} +/- {np.std(single_test):.3f}")
    print(f"TEST {k}-model ENSEMBLE  macro-F1 {ens_metrics['macro_f1']:.3f} | "
          f"macro-AUPRC {ens_metrics['macro_auprc']:.3f} | macro-AUROC {ens_metrics['macro_auroc']:.3f} | "
          f"ECE {ens_metrics['ece']:.3f}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--modality", choices=["vision", "audio", "av"])
    ap.add_argument("--temporal", choices=["mean", "lstm", "transformer"])
    ap.add_argument("--backbone")
    ap.add_argument("--loss", choices=["bce", "focal"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--folds", type=int)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), a)
    if a.folds:
        cfg["cv"]["folds"] = a.folds
    tag = a.tag or f"{cfg['model']['modality']}_{cfg['model']['vision_backbone']}_cv"
    run(cfg, tag)
