"""Frame sampling + audio feature extraction (PDF section 6, steps 4-9).

Two public entry points:
  * ``sample_frames(path, num_frames, size)`` -> uint8 array (T, H, W, 3), RGB
  * ``log_mel(path_or_wav, cfg.audio)``       -> float32 array (n_mels, frames)

``precompute`` caches both for every usable clip so training does not re-decode
video on every epoch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
def sample_frames(path: str | Path, num_frames: int, size: int) -> np.ndarray:
    """Uniformly sample ``num_frames`` RGB frames and centre-crop to ``size``."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fall back to sequential read for containers with no frame count.
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames decoded from {path}")
        idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
        picked = [frames[i] for i in idx]
    else:
        idx = np.linspace(0, total - 1, num_frames).round().astype(int)
        picked = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                ok, fr = cap.read()
            if not ok:
                fr = picked[-1] if picked else np.zeros((size, size, 3), np.uint8)
            picked.append(fr)
        cap.release()

    out = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    for k, fr in enumerate(picked):
        if fr.ndim == 2:
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
        h, w = fr.shape[:2]
        s = min(h, w)
        fr = fr[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
        fr = cv2.resize(fr, (size, size), interpolation=cv2.INTER_AREA)
        out[k] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    return out


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def log_mel(path: str | Path, audio_cfg) -> np.ndarray:
    import librosa

    sr = int(audio_cfg["sample_rate"])
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    max_len = int(audio_cfg["max_seconds"] * sr)
    if len(y) == 0:
        y = np.zeros(max_len, np.float32)
    y = y[:max_len]
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=int(audio_cfg["n_fft"]),
        hop_length=int(audio_cfg["hop_length"]),
        n_mels=int(audio_cfg["n_mels"]),
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def precompute(config_path: str | None = None, overwrite: bool = False) -> None:
    import pandas as pd
    from tqdm import tqdm

    cfg = load_config(config_path)
    meta = pd.read_csv(cfg.resolve_path("metadata_dir") / "metadata.csv")
    meta = meta[meta.has_video == 1]
    video_dir = cfg.resolve_path("clips_video")
    audio_dir = cfg.resolve_path("clips_audio")
    proc = cfg.resolve_path("processed_dir")
    (proc / "frames").mkdir(parents=True, exist_ok=True)
    (proc / "mel").mkdir(parents=True, exist_ok=True)

    nf, sz = int(cfg["video"]["num_frames"]), int(cfg["video"]["frame_size"])
    for vid in tqdm(meta.video_id, desc="precompute"):
        fpath = proc / "frames" / f"{vid}.npy"
        if overwrite or not fpath.exists():
            np.save(fpath, sample_frames(video_dir / f"{vid}.mp4", nf, sz))
        mpath = proc / "mel" / f"{vid}.npy"
        if overwrite or not mpath.exists():
            src = audio_dir / f"{vid}.wav"
            src = src if src.exists() else video_dir / f"{vid}.mp4"
            np.save(mpath, log_mel(src, cfg["audio"]))
    print(f"Cached frames + mel for {len(meta)} clips under {proc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    precompute(a.config, a.overwrite)
