"""Pack the window + mel caches for upload to a private Google Drive.

LoRA fine-tuning cannot use the frozen-feature cache (the backbone weights change),
so it needs the raw windows: 3.09 GB. JPEG q95 brings that to ~346 MB and mel to
float16 (~82 MB), so the upload is minutes rather than an hour.

**Default is WEBP lossless (0.86 GB, 3.6x smaller), not JPEG.** JPEG q95 was tried
first at 0.32 GB and FAILED the gate: despite 42.5 dB PSNR it shifted probabilities by
up to 0.218 and flipped a thresholded decision on **4 of 40 clips**. A 10% flip rate
would confound a LoRA experiment whose whole purpose is measuring a small accuracy
delta, so the extra 0.5 GB is worth it. Lossless passes the gate by construction.

`--format jpg --quality 95` reproduces the failing configuration if you want to see it.
`--validate` compares decoded pixels against the raw arrays AND compares a frozen
model's predictions on both; any decision flip fails.

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
#: (extension, cv2 params, lossless?) per format. WEBP quality 101 means lossless.
FORMATS = {
    "webp": (".webp", lambda q: [cv2.IMWRITE_WEBP_QUALITY, 101], True),
    "png": (".png", lambda q: [cv2.IMWRITE_PNG_COMPRESSION, 9], True),
    "jpg": (".jpg", lambda q: [cv2.IMWRITE_JPEG_QUALITY, q], False),
}
DEFAULT_FORMAT = "webp"


def pack(cfg, out_dir: Path, quality: int = QUALITY,
         fmt: str = DEFAULT_FORMAT) -> None:
    proc = cfg.resolve_path("processed_dir")
    src, mel_dir = proc / "windows", proc / "mel"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    ext, params, lossless = FORMATS[fmt]
    tar_path = out_dir / _tar_name(fmt, quality)
    n_bytes = 0
    with tarfile.open(tar_path, "w") as tar:
        for f in tqdm(files, desc=f"pack windows {fmt}"):
            arr = np.load(f)                                   # (W,T,H,W,3) uint8
            W, T = arr.shape[0], arr.shape[1]
            for w in range(W):
                for t in range(T):
                    ok, buf = cv2.imencode(
                        ext, arr[w, t][:, :, ::-1],            # RGB -> BGR for cv2
                        params(quality))
                    if not ok:
                        raise RuntimeError(f"JPEG encode failed for {f.name} w{w} t{t}")
                    data = buf.tobytes(); n_bytes += len(data)
                    info = tarfile.TarInfo(f"{f.stem}/{w:02d}_{t:02d}{ext}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
    # mel: float32 -> float16 halves it with no meaningful loss for a log-mel input
    mel_out = out_dir / "mel_f16.npz"
    np.savez_compressed(mel_out, **{f.stem: np.load(f).astype(np.float16)
                                    for f in tqdm(sorted(mel_dir.glob("*.npy")), desc="pack mel")})
    print(f"\n  {tar_path}  {tar_path.stat().st_size / 1e6:.0f} MB  ({len(files)} clips)")
    print(f"  {mel_out}  {mel_out.stat().st_size / 1e6:.0f} MB")
    print("\nUpload to your OWN PRIVATE Google Drive. Not GitHub, not a public host.")


def _tar_name(fmt: str, quality: int) -> str:
    return f"windows_{fmt}.tar" if FORMATS[fmt][2] else f"windows_{fmt}{quality}.tar"


def _decode(tar: tarfile.TarFile, stem: str, shape, ext: str = ".webp") -> np.ndarray:
    W, T = shape[0], shape[1]
    out = np.zeros(shape, dtype=np.uint8)
    for w in range(W):
        for t in range(T):
            m = tar.extractfile(f"{stem}/{w:02d}_{t:02d}{ext}")
            img = cv2.imdecode(np.frombuffer(m.read(), np.uint8), cv2.IMREAD_COLOR)
            out[w, t] = img[:, :, ::-1]
    return out


def validate(cfg, out_dir: Path, n: int = 40, quality: int = QUALITY,
             fmt: str = DEFAULT_FORMAT) -> bool:
    """Gate: decoded pixels AND frozen-model predictions must be materially unchanged."""
    import torch

    from src.inference_fast import FastEnsemblePredictor

    proc = cfg.resolve_path("processed_dir")
    src = proc / "windows"
    ext, _, lossless = FORMATS[fmt]
    tar_path = out_dir / _tar_name(fmt, quality)
    if not tar_path.exists():
        raise SystemExit(f"{tar_path} not found -- run without --validate first")

    files = sorted(src.glob("*.npy"))
    pick = files[:: max(1, len(files) // n)][:n]               # spread across the set

    diffs, psnrs = [], []
    with tarfile.open(tar_path) as tar:
        decoded = {}
        for f in tqdm(pick, desc="decode + compare"):
            raw = np.load(f)
            dec = _decode(tar, f.stem, raw.shape, ext)
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

    # Audio is untouched by JPEG, so both arms get the SAME cached audio vector.
    # That isolates the compression effect while keeping the comparison end-to-end
    # through the real (av) head rather than stopping at the visual features.
    afeat_dir = proc / f"feat_audio_{cfg['features']['audio_encoder']}"

    @torch.no_grad()
    def feats(arr, stem):
        x = torch.from_numpy(arr).to(pred.device).float().div_(255.0)
        w, t = x.shape[0], x.shape[1]
        x = x.permute(0, 1, 4, 2, 3).reshape(w * t, 3, *x.shape[2:4])
        if x.shape[-1] != in_size:
            x = torch.nn.functional.interpolate(x, size=(in_size, in_size),
                                                mode="bilinear", align_corners=False)
        x = vnorm(x).view(w, t, 3, in_size, in_size)
        v = model(x.permute(0, 2, 1, 3, 4))
        b = {"vision": v.unsqueeze(1)[:1]}
        if pred.aud_model is not None:
            a = np.load(afeat_dir / f"{stem}.npy").astype("float32")[:1]   # clean variant
            b["audio"] = torch.tensor(a, device=pred.device).unsqueeze(0)
        return torch.sigmoid(pred.heads[0](b))[0].cpu().numpy()

    dp, flips = [], 0
    for f in tqdm(pick, desc="model agreement"):
        raw = np.load(f)
        p_raw, p_jpg = feats(raw, f.stem), feats(decoded[f.stem], f.stem)
        dp.append(np.abs(p_raw - p_jpg).max())
        flips += int(((p_raw >= pred.thr) != (p_jpg >= pred.thr)).any())

    print(f"  max |Δprobability|: {max(dp):.4f} (mean {np.mean(dp):.4f})")
    print(f"  thresholded decision flips: {flips} / {len(pick)} clips")
    ok = flips == 0
    label = fmt if lossless else f"{fmt} q{quality}"
    verdict = (f"PASS -- {label} is safe for LoRA training" if ok
               else f"FAIL -- {label} changes predictions; use the raw 3.09 GB cache")
    print(f"\n  GATE: {verdict}")
    return ok


def unpack_mel(cfg, out_dir: Path) -> None:
    """Expand mel_f16.npz back into <processed>/mel/*.npy.

    The packer ships one compressed npz, but `extract_features --only audio`
    (and the rest of the pipeline) expect per-clip .npy files. Without this the
    audio feature cache cannot be rebuilt on a fresh machine, and cv_lora stops
    with "Found mel_f16.npz but no audio FEATURE cache".
    """
    proc = cfg.resolve_path("processed_dir")
    mel_dir = proc / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / "mel_f16.npz"
    if not src.exists():
        src = proc / "mel_f16.npz"          # notebook copies it next to the caches
    if not src.exists():
        raise SystemExit(f"mel_f16.npz not found in {out_dir} or {proc}")
    z = np.load(src)
    for k in tqdm(z.files, desc="unpack mel"):
        np.save(mel_dir / f"{k}.npy", z[k].astype(np.float32))
    print(f"  {len(z.files)} mel arrays -> {mel_dir}")
    print("  next: python -m src.preprocessing.extract_features --only audio")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/packed")
    ap.add_argument("--quality", type=int, default=QUALITY,
                    help="JPEG quality; ignored for lossless formats")
    ap.add_argument("--format", choices=list(FORMATS), default=DEFAULT_FORMAT)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--unpack-mel", action="store_true",
                    help="expand mel_f16.npz -> <processed>/mel/*.npy")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    if a.unpack_mel:
        unpack_mel(cfg, out)
        raise SystemExit(0)
    if a.validate:
        raise SystemExit(0 if validate(cfg, out, a.n, a.quality, a.format) else 1)
    pack(cfg, out, a.quality, a.format)
