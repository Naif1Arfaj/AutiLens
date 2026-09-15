"""Nested subject-disjoint CV over cached features (seconds per config).

The whole feature cache is a few MB, so every split is loaded into device memory
up front and batched by hand -- no DataLoader, no per-epoch backbone pass.

**Nested CV.** Per OUTER fold, an inner subject-disjoint split is carved from that
fold's TRAINING portion. Early stopping, epoch selection AND the fitting of
thresholds/calibrators all happen on the inner split only; those artefacts are then
frozen before the outer fold is scored. Outer OOF predictions are generated only
after the epoch/configuration is fixed, and are never a training, stopping or
fitting signal.

**Selection uses outer OOF only.** The 92-clip held-out test set is reporting-only
and is evaluated once, after a candidate is locked (`--eval-test`, Phase 4).

Usage:
  python -m src.cv_fast --target families --modality av --tag v3_stage1
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from src.config import (
    TARGET_FAMILIES, TARGET_LABELS, family_truth, load_config, resolve_device,
    resolve_targets, set_seed,
)
from src.cv import _fold_assignment
from src.evaluation.calibration import (
    apply_calibrators, fit_calibrators, save_calibrators,
)
from src.evaluation.metrics import compute_metrics, tune_thresholds
from src.fusion.head import FeatureHead, build_loss, mixup


# --------------------------------------------------------------------------- #
def load_split(cfg, split: str, device: str) -> dict:
    """Load every cached feature for a split into device tensors."""
    proc = cfg.resolve_path("processed_dir")
    # `vision_backbone` may be a list: features from each backbone are concatenated
    # along the feature axis. They share the (window, flip) layout, so the variants
    # stay aligned and one head sees both views. Cheaper and usually as good as
    # averaging two separately-trained heads, and it needs no extra training pass.
    backbones = cfg["features"]["vision_backbone"]
    backbones = [backbones] if isinstance(backbones, str) else list(backbones)
    vdirs = [proc / f"feat_{b}" for b in backbones]
    adir = proc / f"feat_audio_{cfg['features']['audio_encoder']}"
    labels = list(cfg["labels"])
    modality = cfg["model"]["modality"]

    meta = pd.read_csv(cfg.resolve_path("metadata_dir") / "metadata.csv")
    ids = [l.strip() for l in
           (cfg.resolve_path("splits_dir") / f"{split}.txt").read_text().splitlines()
           if l.strip()]
    meta = meta.set_index("video_id").loc[ids].reset_index()

    out = {"ids": list(meta.video_id), "subjects": list(meta.subject_id),
           "y": torch.tensor(meta[labels].values.astype("float32"), device=device)}

    if modality in ("vision", "av"):
        per_backbone = [np.stack([np.load(d / f"{i}.npy") for i in meta.video_id])
                        for d in vdirs]                                     # each (N,W,F,D)
        v = (per_backbone[0] if len(per_backbone) == 1
             else np.concatenate(per_backbone, axis=-1))
        out["vision"] = torch.tensor(v.astype("float32"), device=device)
    if modality in ("audio", "av"):
        a = np.stack([np.load(adir / f"{i}.npy") for i in meta.video_id])   # (N,K,D)
        out["audio"] = torch.tensor(a.astype("float32"), device=device)
    return out


def _train_batch(data: dict, idx: torch.Tensor) -> dict:
    """One random (window, flip) and one random audio variant per clip."""
    b = {"labels": data["y_target"][idx]}
    if "vision" in data:
        v = data["vision"][idx]                                  # (B,W,F,D)
        B, W, F, D = v.shape
        w = torch.randint(W, (B,), device=v.device)
        f = torch.randint(F, (B,), device=v.device)
        b["vision"] = v[torch.arange(B, device=v.device), w, f].unsqueeze(1)
    if "audio" in data:
        a = data["audio"][idx]                                   # (B,K,D)
        B, K, _ = a.shape
        k = torch.randint(K, (B,), device=a.device)
        b["audio"] = a[torch.arange(B, device=a.device), k].unsqueeze(1)
    return b


def _eval_batch(data: dict, idx: torch.Tensor) -> dict:
    """All vision variants (TTA), clean audio variant."""
    b = {"labels": data["y_target"][idx]}
    if "vision" in data:
        v = data["vision"][idx]
        b["vision"] = v.reshape(v.shape[0], -1, v.shape[-1])     # (B,W*F,D)
    if "audio" in data:
        b["audio"] = data["audio"][idx][:, :1]                   # (B,1,D) clean
    return b


@torch.no_grad()
def _predict(model, data, idx, bs=256) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        out.append(torch.sigmoid(model(_eval_batch(data, idx[s:s + bs]))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def train_head(cfg, data, inner_tr, inner_va, device, names) -> tuple[FeatureHead, float, int]:
    """Train one head. **Early stopping and epoch selection use `inner_va` only.**

    `inner_va` is carved from the outer fold's TRAINING portion, so the outer
    validation fold is never read here — not as a stopping signal, not for
    checkpoint selection. The caller generates outer-fold predictions only after
    this returns and the epoch is fixed.
    """
    h = cfg["head"]
    vdim = data["vision"].shape[-1] if "vision" in data else 0
    adim = data["audio"].shape[-1] if "audio" in data else 0
    model = FeatureHead(cfg, vdim, adim, n_out=len(names)).to(device)
    crit = build_loss(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=float(h["lr"]),
                            weight_decay=float(h["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(h["epochs"]))

    bs, alpha = int(h["batch_size"]), float(h["mixup_alpha"])
    y_inner_va = data["y_target"][inner_va].cpu().numpy()
    best, best_state, best_ep = -1.0, None, -1

    for ep in range(1, int(h["epochs"]) + 1):
        model.train()
        perm = inner_tr[torch.randperm(len(inner_tr), device=device)]
        for s in range(0, len(perm), bs):
            batch = _train_batch(data, perm[s:s + bs])
            batch, y = mixup(batch, alpha)
            opt.zero_grad()
            crit(model(batch), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        f1 = compute_metrics(y_inner_va, _predict(model, data, inner_va), names)["macro_f1"]
        if f1 > best:
            best, best_ep = f1, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= int(h["early_stop_patience"]):
            break

    model.load_state_dict(best_state)
    return model, best, best_ep


def _inner_split(subjects, outer_tr_idx, inner_k, seed):
    """Carve a subject-disjoint inner validation split from the OUTER TRAINING
    portion only. Must be subject-disjoint or identity leakage returns."""
    sub_tr = [subjects[i] for i in outer_tr_idx]
    inner_folds = _fold_assignment(sub_tr, inner_k, seed)
    inner_va = outer_tr_idx[inner_folds == 0]
    inner_tr = outer_tr_idx[inner_folds != 0]

    # Invariants that keep outer OOF unbiased. Asserted every run, not spot-checked:
    outer_set = set(outer_tr_idx.tolist())
    assert set(inner_tr.tolist()) <= outer_set and set(inner_va.tolist()) <= outer_set, \
        "inner split escaped the outer TRAINING portion -- outer OOF would be contaminated"
    assert not (set(inner_tr.tolist()) & set(inner_va.tolist())), "inner train/val overlap"
    subj_tr = {subjects[i] for i in inner_tr}
    subj_va = {subjects[i] for i in inner_va}
    assert not (subj_tr & subj_va), \
        f"inner split is not subject-disjoint; shared subjects: {sorted(subj_tr & subj_va)[:5]}"
    assert len(inner_va) > 0, "empty inner validation split"
    return inner_tr, inner_va


def _fit_frozen_artefacts(cfg, y_inner, p_inner, names):
    """Fit calibrators + thresholds on INNER-val predictions, then freeze them.
    They are applied unchanged to the outer fold, which keeps outer OOF unbiased."""
    cals = fit_calibrators(y_inner, p_inner) if cfg["eval"].get("calibrate", True) else None
    p_cal = apply_calibrators(cals, p_inner) if cals else p_inner
    floor = float(cfg["eval"].get("min_precision", 0.0))
    thr = (tune_thresholds(y_inner, p_cal, floor)
           if cfg["eval"]["tune_thresholds"] else np.full(len(names), 0.5))
    return cals, thr


def _run_cv(cfg, pool, train_names, y_train, folds, device, tag_note="", agg=None):
    """Nested-CV loop for one target. Returns unbiased outer-OOF artefacts.

    Per outer fold: carve an inner split from the outer TRAINING portion, early-stop
    on it, fit thresholds/calibrators on it, freeze both, and only then score the
    outer fold. Outer OOF is never used during training, stopping or fitting.

    ``agg=(groups, score_names, y_score)`` runs the *derived* baseline: train on the
    9 labels but aggregate to families **before** fitting thresholds/calibrators, so
    the baseline is nested exactly like the direct arm and the comparison is fair.
    """
    k = int(cfg["cv"]["folds"])
    inner_k = int(cfg["cv"].get("inner_folds", 5))
    pool["y_target"] = torch.tensor(y_train, dtype=torch.float32, device=device)

    groups_, score_names, y_score = agg if agg else (None, train_names, y_train)
    n, c = y_score.shape

    oof_raw = np.zeros((n, c), dtype=np.float32)
    oof_cal = np.zeros((n, c), dtype=np.float32)
    oof_thr = np.zeros((n, c), dtype=np.float32)   # per-fold thresholds, per clip
    states, cals_all, inner_f1, best_eps = [], [], [], []

    for f in range(k):
        outer_tr = np.where(folds != f)[0]
        outer_va = np.where(folds == f)[0]
        inner_tr_np, inner_va_np = _inner_split(pool["subjects"], outer_tr, inner_k,
                                                int(cfg["seed"]) + f)
        inner_tr = torch.tensor(inner_tr_np, device=device)
        inner_va = torch.tensor(inner_va_np, device=device)

        model, f1, ep = train_head(cfg, pool, inner_tr, inner_va, device, train_names)

        # Fit + FREEZE on inner-val, then score the untouched outer fold.
        p_inner = _predict(model, pool, inner_va)
        if groups_:
            p_inner = _noisy_or(p_inner, groups_)
        cals_f, thr_f = _fit_frozen_artefacts(cfg, y_score[inner_va_np], p_inner, score_names)

        p_outer = _predict(model, pool, torch.tensor(outer_va, device=device))
        if groups_:
            p_outer = _noisy_or(p_outer, groups_)
        oof_raw[outer_va] = p_outer
        oof_cal[outer_va] = apply_calibrators(cals_f, p_outer) if cals_f else p_outer
        oof_thr[outer_va] = thr_f

        states.append({kk: vv.cpu() for kk, vv in model.state_dict().items()})
        cals_all.append(cals_f)
        inner_f1.append(f1)
        best_eps.append(ep)
        print(f"  {tag_note}fold {f}: inner-val macro-F1 {f1:.3f} @ep{ep}  "
              f"(inner-train {len(inner_tr_np)} / inner-val {len(inner_va_np)} "
              f"/ outer-val {len(outer_va)})")

    return {"raw": oof_raw, "cal": oof_cal, "thr": oof_thr, "states": states,
            "cals": cals_all, "inner_f1": inner_f1, "best_epochs": best_eps}


def _noisy_or(p, groups):
    """Aggregate 9-class probabilities into families. Baseline only, never served."""
    return np.stack([1.0 - np.prod(1.0 - p[:, g], axis=1) for g in groups], axis=1)


def run(cfg, tag: str, target: str = TARGET_LABELS, eval_test: bool = False) -> dict:
    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    labels = list(cfg["labels"])
    names, groups = resolve_targets(cfg, target)
    k = int(cfg["cv"]["folds"])
    t_start = time.time()

    # Pool train+val; the official test split is never trained on.
    splits_dir = cfg.resolve_path("splits_dir")
    ids = []
    for s in ("train", "val"):
        ids += [l.strip() for l in (splits_dir / f"{s}.txt").read_text().splitlines() if l.strip()]
    (splits_dir / "trainval.txt").write_text("\n".join(ids) + "\n")

    # Features are loaded ONCE and shared across every target in this run.
    pool = load_split(cfg, "trainval", device)
    folds = _fold_assignment(pool["subjects"], k, cfg["seed"])
    y9 = pool["y"].cpu().numpy()
    y_t = family_truth(y9, groups) if groups else y9

    print(f"device={device}  modality={cfg['model']['modality']}  "
          f"backbone={cfg['features']['vision_backbone']}  target={target} ({len(names)} outputs)  "
          f"pool={len(pool['ids'])}  outer folds={k}  inner folds={cfg['cv'].get('inner_folds', 5)}")

    primary = _run_cv(cfg, pool, names, y_t, folds, device)
    oof_metrics = compute_metrics(y_t, primary["cal"], names, primary["thr"])

    # --- derived 9->4 baseline, same folds and inner splits (baseline only) ---
    derived_metrics = None
    if groups:
        print("  [baseline] 9-class head -> noisy-OR aggregation, identical folds/inner splits:")
        base = _run_cv(cfg, pool, labels, y9, folds, device, tag_note="[base] ",
                       agg=(groups, names, y_t))
        derived_metrics = compute_metrics(y_t, base["cal"], names, base["thr"])

    # --- persist --------------------------------------------------------------
    models_dir = cfg.resolve_path("models_dir"); models_dir.mkdir(parents=True, exist_ok=True)
    reports = cfg.resolve_path("reports_dir"); reports.mkdir(parents=True, exist_ok=True)
    torch.save({"folds": primary["states"], "cfg": json.loads(json.dumps(cfg)),
                "target": target, "target_names": names, "member_groups": groups,
                "thresholds": primary["thr"].mean(0).tolist(),
                "fold_thresholds": [primary["thr"][folds == f][0].tolist() for f in range(k)],
                "best_epochs": primary["best_epochs"]},
               models_dir / f"{tag}.pt")
    for i, c in enumerate(primary["cals"]):
        if c:
            save_calibrators(c, models_dir / f"{tag}_calibrators_fold{i}.pkl")
    np.savez(models_dir / f"{tag}_oof.npz", prob=primary["raw"], prob_cal=primary["cal"],
             y=y_t, ids=np.array(pool["ids"]), thresholds=primary["thr"], folds=folds)

    result = {
        "tag": tag, "target": target, "scope": f"{len(names)}-class ({target})",
        "modality": cfg["model"]["modality"],
        "backbone": (cfg["features"]["vision_backbone"]
                     if isinstance(cfg["features"]["vision_backbone"], str)
                     else "+".join(cfg["features"]["vision_backbone"])),
        "audio_encoder": cfg["features"]["audio_encoder"],
        "folds": k, "inner_folds": int(cfg["cv"].get("inner_folds", 5)),
        "nested_cv": True, "seconds": round(time.time() - t_start, 1),
        "min_precision": float(cfg["eval"].get("min_precision", 0.0)),
        "calibrated": bool(primary["cals"][0]),
        "oof_macro_f1": oof_metrics["macro_f1"],
        "oof_macro_auprc": oof_metrics["macro_auprc"],
        "oof_metrics": oof_metrics,
        "inner_val_macro_f1": {"mean": float(np.mean(primary["inner_f1"])),
                               "std": float(np.std(primary["inner_f1"])),
                               "per_fold": primary["inner_f1"]},
        "best_epochs": primary["best_epochs"],
        "derived_baseline_oof_metrics": derived_metrics,
    }

    # --- test set: reporting only, never a selection signal -------------------
    if eval_test:
        result["test_ensemble_metrics"] = _evaluate_test(
            cfg, device, names, primary, tag, models_dir)
        result["test_evaluated"] = True
    else:
        result["test_evaluated"] = False
        result["test_note"] = ("held-out test set deliberately NOT evaluated; it is "
                               "reporting-only and is touched once, after the candidate "
                               "is locked on outer-OOF performance (Phase 4)")

    (reports / f"{tag}_cv.json").write_text(json.dumps(result, indent=2, default=float))

    print(f"\n== {tag} ==  ({result['seconds']}s)  scope: {result['scope']}  [nested CV]")
    print(f"OOF   macro-F1 {oof_metrics['macro_f1']:.3f} | P {oof_metrics['macro_precision']:.3f} "
          f"| R {oof_metrics['macro_recall']:.3f} | AUPRC {oof_metrics['macro_auprc']:.3f} "
          f"| AUROC {oof_metrics['macro_auroc']:.3f} | ECE {oof_metrics['ece']:.3f}")
    if derived_metrics:
        d = derived_metrics
        print(f"BASE  9->4 noisy-OR  macro-F1 {d['macro_f1']:.3f} | P {d['macro_precision']:.3f} "
              f"| AUROC {d['macro_auroc']:.3f}   (baseline only, never served)")
    print(f"{'target':44s} {'sup':>4s} {'F1':>5s} {'P':>5s} {'R':>5s} {'AUPRC':>6s} {'AUROC':>6s}")
    for lab, v in oof_metrics["per_class"].items():
        print(f"{lab:44s} {v['support']:4d} {v['f1']:5.2f} {v['precision']:5.2f} "
              f"{v['recall']:5.2f} {v['auprc']:6.2f} {v['auroc']:6.2f}")
    if not eval_test:
        print("\nTEST: not evaluated (selection uses outer OOF only).")
    return result


def _evaluate_test(cfg, device, names, primary, tag, models_dir) -> dict:
    """Phase 4 only. Ensembles the fold heads, applies the FINAL threshold/calibrator
    refitted on the pooled outer-OOF predictions of all 335 development clips."""
    test = load_split(cfg, "test", device)
    y_test = test["y"].cpu().numpy()
    _, groups = resolve_targets(cfg, "families" if len(names) != len(cfg["labels"]) else "labels")
    if groups and len(names) != len(cfg["labels"]):
        y_test = family_truth(y_test, groups)

    vdim = test["vision"].shape[-1] if "vision" in test else 0
    adim = test["audio"].shape[-1] if "audio" in test else 0
    idx = torch.arange(len(test["ids"]), device=device)
    probs = []
    for sd in primary["states"]:
        m = FeatureHead(cfg, vdim, adim, n_out=len(names)).to(device).eval()
        m.load_state_dict({k: v.to(device) for k, v in sd.items()})
        probs.append(_predict(m, test, idx))
    ens = np.mean(probs, axis=0)

    # Final artefacts refit on the POOLED OUTER-OOF predictions (unbiased: each
    # clip was predicted by a model that never trained on it).
    final_cals, final_thr = _fit_frozen_artefacts(
        cfg, np.load(models_dir / f"{tag}_oof.npz")["y"],
        np.load(models_dir / f"{tag}_oof.npz")["raw"], names)
    save_calibrators(final_cals, models_dir / f"{tag}_calibrators_final.pkl") if final_cals else None
    ens_cal = apply_calibrators(final_cals, ens) if final_cals else ens
    return compute_metrics(y_test, ens_cal, names, final_thr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--modality", choices=["vision", "audio", "av"])
    ap.add_argument("--backbone", help="features.vision_backbone; comma-separate "
                                          "to concatenate cached features, e.g. "
                                          "swin3d_t,r2plus1d_18")
    ap.add_argument("--folds", type=int)
    ap.add_argument("--min-precision", type=float)
    ap.add_argument("--mixup", type=float)
    ap.add_argument("--aggregate", choices=["mean", "attention"])
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--target", choices=[TARGET_LABELS, TARGET_FAMILIES],
                    default=None, help="9 behaviors (default) or the 4 Stage-1 families")
    ap.add_argument("--inner-folds", type=int)
    ap.add_argument("--eval-test", action="store_true",
                    help="PHASE 4 ONLY. The 92-clip test set is reporting-only and must "
                         "never inform model selection; leave this off while comparing "
                         "configurations.")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.modality:      cfg["model"]["modality"] = a.modality
    if a.backbone:
        bb = [b.strip() for b in a.backbone.split(",") if b.strip()]
        cfg["features"]["vision_backbone"] = bb[0] if len(bb) == 1 else bb
    if a.folds:         cfg["cv"]["folds"] = a.folds
    if a.inner_folds:   cfg["cv"]["inner_folds"] = a.inner_folds
    if a.min_precision is not None: cfg["eval"]["min_precision"] = a.min_precision
    if a.mixup is not None:         cfg["head"]["mixup_alpha"] = a.mixup
    if a.aggregate:                 cfg["head"]["aggregate"] = a.aggregate
    if a.no_calibrate:  cfg["eval"]["calibrate"] = False
    if a.seed is not None: cfg["seed"] = a.seed
    target = a.target or cfg["cv"].get("target", TARGET_LABELS)
    _bb = cfg["features"]["vision_backbone"]
    _bb = _bb if isinstance(_bb, str) else "+".join(_bb)
    tag = a.tag or f"{cfg['model']['modality']}_{_bb}_{target}"
    run(cfg, tag, target=target, eval_test=a.eval_test)
