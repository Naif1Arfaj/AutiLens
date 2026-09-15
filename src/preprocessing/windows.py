"""Windowed, fixed-fps frame sampling.

Why this exists
---------------
``features.sample_frames`` spreads ``num_frames`` uniformly across the WHOLE
clip. For a 9 s clip that is fine, but the dataset contains clips up to 887 s,
where 16 uniform frames means one frame every 55 seconds. Autism-related
stereotypies (arm flapping, rocking, spinning) are 2-4 Hz motions, so that
sampling aliases the behaviour away completely -- no architecture can recover
information that was discarded here.

This module instead takes ``n_windows`` windows of ``num_frames`` CONSECUTIVE
frames sampled at ``target_fps``. At the defaults (16 frames @ 10 fps) each
window is 1.6 s of real motion, which resolves a 2-4 Hz behaviour, and the
windows are spread across the clip so long clips are still covered.

Returns ``(n_windows, num_frames, size, size, 3)`` uint8 RGB.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config


def _center_crop_resize(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    s = min(h, w)
    frame = frame[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_windows(
    path: str | Path,
    num_frames: int = 16,
    size: int = 224,
    n_windows: int = 3,
    target_fps: float = 10.0,
) -> np.ndarray:
    """Sample ``n_windows`` fixed-fps windows of consecutive frames."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if not (1.0 <= native_fps <= 240.0):
        native_fps = 25.0  # container lied; assume a sane default

    # Step between kept frames so the window plays back at ~target_fps.
    stride = max(1, int(round(native_fps / max(target_fps, 1e-6))))
    span = (num_frames - 1) * stride + 1  # frames consumed by one window

    out = np.zeros((n_windows, num_frames, size, size, 3), dtype=np.uint8)

    if total <= 0:  # no frame count -> decode everything once
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            return out  # unreadable clip: caller sees an all-zero cache entry
        total = len(frames)
        starts = np.linspace(0, max(0, total - span), n_windows).round().astype(int)
        for wi, st in enumerate(starts):
            idx = np.clip(st + np.arange(num_frames) * stride, 0, total - 1)
            for k, i in enumerate(idx):
                out[wi, k] = _center_crop_resize(frames[int(i)], size)
        return out

    starts = np.linspace(0, max(0, total - span), n_windows).round().astype(int)
    for wi, st in enumerate(starts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(st))
        picked: list[np.ndarray] = []
        # Read `span` frames consecutively (cheap) and keep every `stride`-th.
        for j in range(span):
            ok, fr = cap.read()
            if not ok:
                break
            if j % stride == 0:
                picked.append(fr)
                if len(picked) == num_frames:
                    break
        if not picked:  # seek landed past the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
            if not ok:
                continue
            picked = [fr]
        while len(picked) < num_frames:  # short clip: hold the last frame
            picked.append(picked[-1])
        for k in range(num_frames):
            out[wi, k] = _center_crop_resize(picked[k], size)

    cap.release()
    return out


def window_time_spans(path: str | Path, num_frames: int = 16,
                      n_windows: int = 3, target_fps: float = 10.0) -> list[tuple[float, float]]:
    """Real (start_s, end_s) of each window, so evidence can cite clip timestamps."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    cap.release()
    if not (1.0 <= native_fps <= 240.0):
        native_fps = 25.0
    if total <= 0:
        return [(0.0, 0.0)] * n_windows
    stride = max(1, int(round(native_fps / max(target_fps, 1e-6))))
    span = (num_frames - 1) * stride + 1
    starts = np.linspace(0, max(0, total - span), n_windows).round().astype(int)
    return [(round(float(st) / native_fps, 2),
             round(float(min(st + span, total)) / native_fps, 2)) for st in starts]


def precompute(config_path: str | None = None, overwrite: bool = False) -> None:
    import pandas as pd
    from tqdm import tqdm

    cfg = load_config(config_path)
    w = cfg["windows"]
    meta = pd.read_csv(cfg.resolve_path("metadata_dir") / "metadata.csv")
    meta = meta[meta.has_video == 1]
    video_dir = cfg.resolve_path("clips_video")
    out_dir = cfg.resolve_path("processed_dir") / "windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    expect = (int(w["n_windows"]), int(w["num_frames"]),
              int(w["frame_size"]), int(w["frame_size"]), 3)
    n_done = 0
    for vid in tqdm(meta.video_id, desc="windows"):
        dst = out_dir / f"{vid}.npy"
        if not overwrite and dst.exists():
            try:
                if np.load(dst, mmap_mode="r").shape == expect:
                    continue
            except (ValueError, OSError):
                pass  # corrupt or stale shape -> regenerate
        arr = sample_windows(
            video_dir / f"{vid}.mp4",
            int(w["num_frames"]), int(w["frame_size"]),
            int(w["n_windows"]), float(w["target_fps"]),
        )
        np.save(dst, arr)
        n_done += 1
    print(f"windows cache ready at {out_dir} ({n_done} written, shape {expect})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    precompute(a.config, a.overwrite)
