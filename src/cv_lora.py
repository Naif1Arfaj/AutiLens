"""LoRA fine-tuning of the Swin3D-T backbone (Phase 3).

Unlike `src/cv_fast.py`, the backbone weights change, so features cannot be cached
and the backbone runs every forward pass. Raw windows come from
`data/processed/windows/` (or the JPEG-packed tar on Colab).

Same methodology as the frozen pipeline and for the same reasons:
  * nested CV -- inner split from the outer fold's TRAINING portion drives early
    stopping, epoch selection, and the fitting of thresholds/calibrators;
  * those artefacts are frozen before the outer fold is scored;
  * the held-out test set is never touched here.

LoRA is implemented by hand: `peft` is unavailable on Python 3.9, and torchvision's
Swin3D reads `qkv.weight` / `proj.weight` DIRECTLY and hands the tensors to a
functional (`shifted_window_attention_3d`) -- it never calls `Linear.forward`. An
adapter that overrides `forward()` therefore does nothing at all. The adapter here
exposes a merged `weight` property instead, so autograd reaches A and B.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

TARGET_SUFFIXES = ("qkv", "proj")
STAGE_PREFIXES = {
    "last": ["features.6"],
    "all": ["features.0", "features.2", "features.4", "features.6"],
}


class LoRALinear(nn.Module):
    """Low-rank adapter exposing a MERGED weight.

    `B` is zero-initialised so training starts exactly at the pretrained function.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    @property
    def weight(self):                      # read by shifted_window_attention_3d
        return self.base.weight + (self.B @ self.A) * self.scale

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):                  # used if anything does call the module
        return nn.functional.linear(x, self.weight, self.bias)


def inject_lora(model: nn.Module, stages: str = "last", r: int = 8, alpha: int = 16) -> int:
    """Wrap the attention projections in the chosen stages. Returns how many."""
    prefixes = STAGE_PREFIXES[stages]
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if (isinstance(child, nn.Linear) and child_name in TARGET_SUFFIXES
                    and any(full.startswith(p) for p in prefixes)):
                setattr(mod, child_name, LoRALinear(child, r, alpha))
                n += 1
    return n


def build_lora_backbone(stages: str = "last", r: int = 8, alpha: int = 16,
                        device: str = "cpu"):
    """Kinetics-400 Swin3D-T with everything frozen except the LoRA parameters."""
    from torchvision.models import video as V

    net = V.swin3d_t(weights=V.Swin3D_T_Weights.KINETICS400_V1)
    net.head = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False
    n_layers = inject_lora(net, stages, r, alpha)
    net = net.to(device)
    trainable = [p for p in net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)

    # Every trainable tensor must be a LoRA A/B; anything else means the freeze leaked.
    lora_ids = {id(p) for m in net.modules() if isinstance(m, LoRALinear)
                for p in (m.A, m.B)}
    stray = [p.shape for p in trainable if id(p) not in lora_ids]
    assert not stray, f"non-LoRA parameters are trainable: {stray}"
    return net, n_layers, n_params


def audit(model: nn.Module) -> dict:
    """Confirm base weights are frozen and only A/B carry gradients."""
    base = [p for m in model.modules() if isinstance(m, LoRALinear)
            for p in m.base.parameters()]
    lora = [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]
    return {
        "lora_layers": sum(1 for m in model.modules() if isinstance(m, LoRALinear)),
        "base_params_frozen": all(not p.requires_grad for p in base),
        "lora_params_trainable": all(p.requires_grad for p in lora),
        "n_trainable": sum(p.numel() for p in lora),
    }


# --------------------------------------------------------------------------- #
# Window store: raw frames, .npy locally or an extracted image tree on Colab
# --------------------------------------------------------------------------- #
class WindowStore:
    """Raw window frames, from whichever layout is present.

    Local runs have `<processed>/windows/<clip>.npy` of shape (W,T,H,W,3).
    Colab unpacks the WEBP tar to `<processed>/windows_raw/<clip>/<ww>_<tt>.webp`.
    Both yield the same array so the training loop does not care which it got.
    """

    def __init__(self, processed_dir, n_windows: int, num_frames: int, size: int):
        from pathlib import Path

        self.npy_dir = Path(processed_dir) / "windows"
        self.img_dir = Path(processed_dir) / "windows_raw"
        self.shape = (n_windows, num_frames, size, size, 3)
        self.mode = ("npy" if self.npy_dir.exists() and any(self.npy_dir.glob("*.npy"))
                     else "img" if self.img_dir.exists() and any(self.img_dir.iterdir())
                     else None)
        if self.mode is None:
            raise FileNotFoundError(
                f"No window cache. Expected {self.npy_dir}/*.npy or an extracted tree "
                f"at {self.img_dir}/<clip>/<ww>_<tt>.<ext>")

    def get(self, vid: str) -> np.ndarray:
        import cv2

        if self.mode == "npy":
            return np.load(self.npy_dir / f"{vid}.npy")
        d = self.img_dir / vid
        out = np.zeros(self.shape, dtype=np.uint8)
        for w in range(self.shape[0]):
            for t in range(self.shape[1]):
                hits = list(d.glob(f"{w:02d}_{t:02d}.*"))
                if not hits:
                    raise FileNotFoundError(f"missing frame {w:02d}_{t:02d} for {vid}")
                img = cv2.imread(str(hits[0]), cv2.IMREAD_COLOR)
                out[w, t] = img[:, :, ::-1]
            # NOTE: decode is the bottleneck on Colab; it is still far cheaper than
            # the Swin3D-T forward it feeds.
        return out


_KINETICS = ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])


def _to_batch(frames_u8: np.ndarray, device: str, size: int) -> torch.Tensor:
    """(B,T,H,W,3) uint8 -> (B,3,T,H,W) normalised, the layout Swin3D expects."""
    x = torch.from_numpy(frames_u8).to(device).float().div_(255.0)
    b, t = x.shape[0], x.shape[1]
    x = x.permute(0, 1, 4, 2, 3).reshape(b * t, 3, *x.shape[2:4])
    if x.shape[-1] != size:
        x = nn.functional.interpolate(x, size=(size, size), mode="bilinear",
                                      align_corners=False)
    mean = torch.tensor(_KINETICS[0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(_KINETICS[1], device=device).view(1, 3, 1, 1)
    x = ((x - mean) / std).view(b, t, 3, size, size)
    return x.permute(0, 2, 1, 3, 4)


# --------------------------------------------------------------------------- #
# Nested-CV training. Same methodology as src/cv_fast.py -- see its docstring.
# --------------------------------------------------------------------------- #
class LoRAModel(nn.Module):
    """Fine-tuned vision backbone + the frozen cached audio branch + head.

    Only the vision side is fine-tuned. Audio features stay cached and frozen,
    which keeps a fold trainable in minutes instead of hours and matches the
    finding that the mel encoder is the weak part anyway.
    """

    def __init__(self, cfg, n_out, audio_dim, stages, rank, alpha, device):
        super().__init__()
        from src.fusion.head import FeatureHead

        self.backbone, self.n_layers, self.n_lora = build_lora_backbone(
            stages, rank, alpha, device)
        self.head = FeatureHead(cfg, 768, audio_dim, n_out=n_out).to(device)
        self.modality = cfg["model"]["modality"]

    def forward(self, frames, audio=None):
        v = self.backbone(frames)                       # (B, 768)
        batch = {"vision": v.unsqueeze(1)}              # (B,1,768)
        if audio is not None:
            batch["audio"] = audio
        return self.head(batch)

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


def run(cfg, tag, target="labels", stages="last", rank=8, alpha=16,
        epochs=30, batch_size=4, lora_lr=1e-4, head_lr=5e-4, patience=None,
        only_fold=None):
    import json
    import time

    import pandas as pd
    from src.config import family_truth, resolve_device, resolve_targets, set_seed
    from src.cv import _fold_assignment
    from src.cv_fast import _fit_frozen_artefacts, _inner_split
    from src.evaluation.calibration import apply_calibrators, save_calibrators
    from src.evaluation.metrics import compute_metrics
    from src.fusion.head import build_loss

    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    names, groups = resolve_targets(cfg, target)
    k, inner_k = int(cfg["cv"]["folds"]), int(cfg["cv"].get("inner_folds", 5))
    w = cfg["windows"]
    t0 = time.time()

    splits = cfg.resolve_path("splits_dir")
    ids = [l.strip() for sp in ("train", "val")
           for l in (splits / f"{sp}.txt").read_text().splitlines() if l.strip()]
    meta = pd.read_csv(cfg.resolve_path("metadata_dir") / "metadata.csv")
    meta = meta.set_index("video_id").loc[ids].reset_index()
    subjects = list(meta.subject_id)
    y9 = meta[list(cfg["labels"])].values.astype("float32")
    y_t = family_truth(y9, groups) if groups else y9

    proc = cfg.resolve_path("processed_dir")
    store = WindowStore(proc, int(w["n_windows"]), int(w["num_frames"]),
                        int(w["frame_size"]))
    use_audio = cfg["model"]["modality"] in ("audio", "av")
    afeat = None
    if use_audio:
        adir = proc / f"feat_audio_{cfg['features']['audio_encoder']}"
        npz = proc / "mel_f16.npz"
        if adir.exists():
            afeat = np.stack([np.load(adir / f"{v}.npy") for v in meta.video_id])
        elif npz.exists():
            raise SystemExit(
                "Found mel_f16.npz but no audio FEATURE cache. Run:\n"
                "  python -m src.preprocessing.extract_features --only audio")
        else:
            raise SystemExit(f"modality={cfg['model']['modality']} needs {adir}")

    print(f"device={device} target={target} ({len(names)} outputs)  stages={stages} "
          f"r={rank}  window store={store.mode}  pool={len(meta)}  folds={k}")

    folds = _fold_assignment(subjects, k, cfg["seed"])
    n = len(meta)
    oof_raw = np.zeros((n, len(names)), np.float32)
    oof_cal = np.zeros((n, len(names)), np.float32)
    oof_thr = np.zeros((n, len(names)), np.float32)
    states, cals_all, inner_f1, best_eps = [], [], [], []

    def predict(model, idx, all_windows=True):
        model.eval()
        out = np.zeros((len(idx), len(names)), np.float32)
        with torch.no_grad():
            for s in range(0, len(idx), batch_size):
                chunk = idx[s:s + batch_size]
                arrs = np.stack([store.get(meta.video_id[i]) for i in chunk])
                a = None
                if use_audio:
                    a = torch.tensor(afeat[chunk][:, :1].astype("float32"), device=device)
                wins = range(arrs.shape[1]) if all_windows else [0]
                acc = np.zeros((len(chunk), len(names)), np.float32)
                for wi in wins:                       # average over windows = TTA
                    x = _to_batch(arrs[:, wi], device, int(w["frame_size"]))
                    acc += torch.sigmoid(model(x, a)).cpu().numpy()
                out[s:s + len(chunk)] = acc / len(list(wins))
        return out

    # `only_fold` runs a single outer fold of the normal k-fold split. The split
    # geometry is unchanged, so train/val sizes match a full run -- but OOF then
    # covers 1/k of the pool, which is a DIAGNOSTIC, not a comparable result.
    fold_ids = range(k) if only_fold is None else [int(only_fold)]
    for f in fold_ids:
        outer_tr, outer_va = np.where(folds != f)[0], np.where(folds == f)[0]
        inner_tr, inner_va = _inner_split(subjects, outer_tr, inner_k,
                                          int(cfg["seed"]) + f)
        model = LoRAModel(cfg, len(names), afeat.shape[-1] if use_audio else 0,
                          stages, rank, alpha, device)
        crit = build_loss(cfg)
        opt = torch.optim.AdamW([
            {"params": [p for m in model.backbone.modules()
                        if isinstance(m, LoRALinear) for p in (m.A, m.B)], "lr": lora_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ], weight_decay=1e-4)

        best, best_state, best_ep = -1.0, None, -1
        # NOT head.early_stop_patience (25): that is tuned for the frozen pipeline,
        # where epochs cost milliseconds and it runs 120 of them. A LoRA epoch is
        # ~2 min, and the first run peaked at 7-10 across all folds, so 5 is
        # generous here and 25 would never fire under a 30-epoch ceiling.
        pat = int(patience if patience is not None
                  else cfg["head"].get("lora_early_stop_patience", 5))
        for ep in range(1, epochs + 1):
            model.train()
            perm = np.random.permutation(inner_tr)
            for s in range(0, len(perm), batch_size):
                chunk = perm[s:s + batch_size]
                arrs = np.stack([store.get(meta.video_id[i]) for i in chunk])
                wi = np.random.randint(arrs.shape[1])          # random window
                x = _to_batch(arrs[:, wi], device, int(w["frame_size"]))
                a = (torch.tensor(afeat[chunk][:, np.random.randint(afeat.shape[1])]
                                  .astype("float32"), device=device).unsqueeze(1)
                     if use_audio else None)
                yb = torch.tensor(y_t[chunk], device=device)
                opt.zero_grad()
                crit(model(x, a), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.trainable(), 5.0)
                opt.step()

            f1 = compute_metrics(y_t[inner_va], predict(model, inner_va, False),
                                 names)["macro_f1"]
            print(f"  fold {f} ep{ep:02d}: inner-val macro-F1 {f1:.3f}")
            if f1 > best:
                best, best_ep = f1, ep
                best_state = {kk: vv.detach().cpu().clone()
                              for kk, vv in model.state_dict().items()}
            # Early stopping, same as cv_fast. Without it the epoch count has to be
            # guessed; with it, `epochs` is just a generous ceiling. Every fold of
            # the first LoRA run peaked at 7-10 and then declined, so the remaining
            # epochs were pure cost.
            if ep - best_ep >= pat:
                print(f"  fold {f}: early stop at ep{ep} "
                      f"(no gain for {pat} epochs since ep{best_ep})")
                break
        model.load_state_dict({kk: vv.to(device) for kk, vv in best_state.items()})

        # Fit + FREEZE on inner-val, then score the untouched outer fold.
        cals_f, thr_f = _fit_frozen_artefacts(cfg, y_t[inner_va],
                                              predict(model, inner_va), names)
        p_out = predict(model, outer_va)
        oof_raw[outer_va] = p_out
        oof_cal[outer_va] = apply_calibrators(cals_f, p_out) if cals_f else p_out
        oof_thr[outer_va] = thr_f
        states.append({kk: vv for kk, vv in best_state.items() if "lora" in kk.lower()
                       or kk.startswith("head") or ".A" in kk or ".B" in kk})
        cals_all.append(cals_f); inner_f1.append(best); best_eps.append(best_ep)
        print(f"  fold {f}: best inner-val {best:.3f} @ep{best_ep}")

    if only_fold is not None:
        covered = np.where(folds == int(only_fold))[0]
        m = compute_metrics(y_t[covered], oof_cal[covered], names, oof_thr[covered])
    else:
        m = compute_metrics(y_t, oof_cal, names, oof_thr)
    models_dir = cfg.resolve_path("models_dir"); models_dir.mkdir(parents=True, exist_ok=True)
    reports = cfg.resolve_path("reports_dir"); reports.mkdir(parents=True, exist_ok=True)
    torch.save({"folds": states, "cfg": json.loads(json.dumps(cfg)), "target": target,
                "target_names": names, "member_groups": groups, "lora": {
                    "stages": stages, "rank": rank, "alpha": alpha},
                "thresholds": oof_thr.mean(0).tolist(),
                "best_epochs": best_eps}, models_dir / f"{tag}.pt")
    for i, c in enumerate(cals_all):
        if c:
            save_calibrators(c, models_dir / f"{tag}_calibrators_fold{i}.pkl")
    np.savez(models_dir / f"{tag}_oof.npz", prob=oof_raw, prob_cal=oof_cal, y=y_t,
             ids=np.array(list(meta.video_id)), thresholds=oof_thr, folds=folds)

    result = {"tag": tag, "target": target,
              "only_fold": only_fold,
              "diagnostic": only_fold is not None,
              "partial_oof_note": (
                  f"single outer fold ({only_fold}) -- OOF covers {len(np.where(folds == int(only_fold))[0])}"
                  f" of {n} clips. NOT comparable to full k-fold runs; diagnostic only."
                  if only_fold is not None else None), "seed": int(cfg["seed"]), "scope": f"{len(names)}-class ({target})",
              "modality": cfg["model"]["modality"], "backbone": "swin3d_t+lora",
              "audio_encoder": cfg["features"]["audio_encoder"],
              "lora": {"stages": stages, "rank": rank, "alpha": alpha},
              "folds": k, "inner_folds": inner_k, "nested_cv": True,
              "seconds": round(time.time() - t0, 1),
              "min_precision": float(cfg["eval"].get("min_precision", 0.0)),
              "calibrated": bool(cals_all[0]),
              "oof_macro_f1": m["macro_f1"], "oof_macro_auprc": m["macro_auprc"],
              "oof_metrics": m,
              "inner_val_macro_f1": {"mean": float(np.mean(inner_f1)),
                                     "std": float(np.std(inner_f1)),
                                     "per_fold": inner_f1},
              "best_epochs": best_eps, "test_evaluated": False,
              "test_note": "held-out test set deliberately NOT evaluated"}
    (reports / f"{tag}_cv.json").write_text(json.dumps(result, indent=2, default=float))

    print(f"\n== {tag} ==  ({result['seconds']}s)  scope: {result['scope']}  [nested CV]")
    print(f"OOF   macro-F1 {m['macro_f1']:.3f} | P {m['macro_precision']:.3f} "
          f"| R {m['macro_recall']:.3f} | AUPRC {m['macro_auprc']:.3f} "
          f"| AUROC {m['macro_auroc']:.3f} | ECE {m['ece']:.3f}")
    for lab, v in m["per_class"].items():
        print(f"{lab:44s} {v['support']:4d} F1 {v['f1']:.2f}  AUROC {v['auroc']:.2f}")
    print("\nTEST: not evaluated (selection uses outer OOF only).")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LoRA fine-tuning with nested CV")
    ap.add_argument("--config", default=None)
    ap.add_argument("--stages", choices=list(STAGE_PREFIXES), default="last")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--target", choices=["labels", "families"], default="labels")
    ap.add_argument("--epochs", type=int, default=30,
                    help="ceiling, not a target: early stopping decides")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop after N epochs without an inner-val gain "
                         "(default: head.early_stop_patience)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--only-fold", type=int, default=None,
                    help="run ONE outer fold of the k-fold split. Diagnostic: "
                         "OOF then covers 1/k of the pool and is not "
                         "comparable to full runs.")
    ap.add_argument("--seed", type=int,
                    help="overrides cfg.seed; changes both init and the fold\n                          assignment, matching how cv_fast seeds vary")
    ap.add_argument("--modality", choices=["vision", "audio", "av"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--audit-only", action="store_true",
                    help="print the parameter audit and exit; no training")
    a = ap.parse_args()

    if a.audit_only or a.tag is None:
        net, n_layers, n_params = build_lora_backbone(a.stages, a.rank, a.alpha, "cpu")
        print(f"stages={a.stages} r={a.rank}: {n_layers} adapted layers, "
              f"{n_params:,} trainable params")
        print(" audit:", audit(net))
        if a.audit_only:
            raise SystemExit(0)
        raise SystemExit("\nPass --tag to train, e.g. --target families --tag lora_A")

    from src.config import load_config

    cfg = load_config(a.config)
    if a.modality:
        cfg["model"]["modality"] = a.modality
    if a.seed is not None:
        cfg["seed"] = a.seed
    run(cfg, a.tag, a.target, a.stages, a.rank, a.alpha,
        a.epochs, a.batch_size, a.lora_lr, a.head_lr, a.patience,
        a.only_fold)
