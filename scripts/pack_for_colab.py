"""Pack the window + mel caches for upload to a private Google Drive.

LoRA fine-tuning cannot use the frozen-feature cache (the backbone weights change),
so it needs the raw windows: 3.09 GB. JPEG q95 brings that to ~346 MB and mel to
float16 (~82 MB), so the upload is minutes rather than an hour.

**JPEG is lossy.** q95 is high quality, not lossless. `--validate` is the gate: it
compares decoded pixels against the raw arrays AND compares a frozen model's
predictions on both. If any thresholded decision flips, use the raw cache instead --
the 2.6 GB saving is not worth confounding the experiment this phase exists to run.

PRIVACY: this is video of real children from YouTube/Facebook. Upload only to the
user's own private Drive. Never GitHub, never a public dataset host (guide §14).

    python -m scripts.pack_for_colab --out data/packed
    python -m scripts.pack_for_colab --validate --out data/packed
"""
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import load_config

QUALITY = 95


def pack(cfg, out_dir: Path, quality: int = QUALITY) -> None:
    proc = cfg.resolve_path("processed_dir")
    src, mel_dir = proc / "windows", proc / "mel"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    tar_path = out_dir / f"windows_q{quality}.tar"
    n_bytes = 0
    with tarfile.open(tar_path, "w") as tar:
        for f in tqdm(files, desc=f"pack windows q{quality}"):
            arr = np.load(f)                                   # (W,T,H,W,3) uint8
            W, T = arr.shape[0], arr.shape[1]
            for w in range(W):
                for t in range(T):
                    ok, buf = cv2.imencode(
                        ".jpg", arr[w, t][:, :, ::-1],         # RGB -> BGR for cv2
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if not ok:
                        raise RuntimeError(f"JPEG encode failed for {f.name} w{w} t{t}")
                    data = buf.tobytes(); n_bytes += len(data)
                    info = tarfile.TarInfo(f"{f.stem}/{w:02d}_{t:02d}.jpg")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
    # mel: float32 -> float16 halves it with no meaningful loss for a log-mel input
    mel_out = out_dir / "mel_f16.npz"
    np.savez_compressed(mel_out, **{f.stem: np.load(f).astype(np.float16)
                                    for f in tqdm(sorted(mel_dir.glob("*.npy")), desc="pack mel")})
    print(f"\n  {tar_path}  {tar_path.stat().st_size / 1e6:.0f} MB  ({len(files)} clips)")
    print(f"  {mel_out}  {mel_out.stat().st_size / 1e6:.0f} MB")
    print("\nUpload to your OWN PRIVATE Google Drive. Not GitHub, not a public host.")


def _decode(tar: tarfile.TarFile, stem: str, shape) -> np.ndarray:
    W, T = shape[0], shape[1]
    out = np.zeros(shape, dtype=np.uint8)
    for w in range(W):
        for t in range(T):
            m = tar.extractfile(f"{stem}/{w:02d}_{t:02d}.jpg")
            img = cv2.imdecode(np.frombuffer(m.read(), np.uint8), cv2.IMREAD_COLOR)
            out[w, t] = img[:, :, ::-1]
    return out


def validate(cfg, out_dir: Path, n: int = 40, quality: int = QUALITY) -> bool:
    """Gate: decoded pixels AND frozen-model predictions must be materially unchanged."""
    import torch

    from src.inference_fast import FastEnsemblePredictor

    proc = cfg.resolve_path("processed_dir")
    src = proc / "windows"
    tar_path = out_dir / f"windows_q{quality}.tar"
    if not tar_path.exists():
        raise SystemExit(f"{tar_path} not found -- run without --validate first")

    files = sorted(src.glob("*.npy"))
    pick = files[:: max(1, len(files) // n)][:n]               # spread across the set

    diffs, psnrs = [], []
    with tarfile.open(tar_path) as tar:
        decoded = {}
        for f in tqdm(pick, desc="decode + compare"):
            raw = np.load(f)
            dec = _decode(tar, f.stem, raw.shape)
            decoded[f.stem] = dec
            d = np.abs(raw.astype(np.int16) - dec.astype(np.int16))
            diffs.append((d.mean(), d.max()))
            mse = (d.astype(np.float64) ** 2).mean()
            psnrs.append(10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0)

    mean_d = float(np.mean([a for a, _ in diffs])); max_d = int(max(b for _, b in diffs))
    print(f"\n  pixel delta: mean {mean_d:.2f}, max {max_d}   PSNR: "
          f"mean {np.mean(psnrs):.1f} dB, min {np.min(psnrs):.1f} dB")

    # Prediction equivalence on a frozen model -- the check that actually matters.
    from src.model_registry import select_best
    card = select_best(cfg.resolve_path("models_dir"), n_classes=9)
    if card is None:
        print("  no scored checkpoint found; pixel check only")
        return max_d < 32
    pred = FastEnsemblePredictor(card.path)
    vnorm, model, in_size = pred.vnorm, pred.vis_model, pred.in_size

    @torch.no_grad()
    def feats(arr):
        x = torch.from_numpy(arr).to(pred.device).float().div_(255.0)
        w, t = x.shape[0], x.shape[1]
        x = x.permute(0, 1, 4, 2, 3).reshape(w * t, 3, *x.shape[2:4])
        if x.shape[-1] != in_size:
            x = torch.nn.functional.interpolate(x, size=(in_size, in_size),
                                                mode="bilinear", align_corners=False)
        x = vnorm(x).view(w, t, 3, in_size, in_size)
        v = model(x.permute(0, 2, 1, 3, 4))
        b = {"vision": v.unsqueeze(1)[:1]}
        return torch.sigmoid(pred.heads[0](b))[0].cpu().numpy()

    dp, flips = [], 0
    for f in tqdm(pick, desc="model agreement"):
        raw = np.load(f)
        p_raw, p_jpg = feats(raw), feats(decoded[f.stem])
        dp.append(np.abs(p_raw - p_jpg).max())
        flips += int(((p_raw >= pred.thr) != (p_jpg >= pred.thr)).any())

    print(f"  max |Δprobability|: {max(dp):.4f} (mean {np.mean(dp):.4f})")
    print(f"  thresholded decision flips: {flips} / {len(pick)} clips")
    ok = flips == 0
    print(f"\n  GATE: {'PASS -- q95 is safe for LoRA training' if ok else 'FAIL -- use the raw 3.09 GB cache instead'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/packed")
    ap.add_argument("--quality", type=int, default=QUALITY)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    if a.validate:
        raise SystemExit(0 if validate(cfg, out, a.n, a.quality) else 1)
    pack(cfg, out, a.quality)
