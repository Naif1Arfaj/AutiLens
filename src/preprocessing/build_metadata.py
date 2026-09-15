"""Step 1 of the data-preparation pipeline (PDF section 6).

Builds a single metadata table describing every annotated clip:
  video_id, subject_id, split, <one column per behavior label>,
  has_video, has_audio, n_frames, fps, duration_s

- ``subject_id`` is the source (YouTube/Facebook) video id, i.e. the clip id
  with the trailing ``_<start>_<end>`` stripped. Splits are made by subject so
  that no clip from the same source video appears in more than one split
  (PDF: "never put clips from the same child/person into both train and test").
- The AV-ASD repository already ships subject-disjoint train/val/test CSVs; we
  keep those assignments and simply verify disjointness.
- Rows whose video file is missing are kept in the table (has_video=0) but are
  excluded from the usable splits written to ``data/splits``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from src.config import load_config


def subject_of(video_id: str) -> str:
    return video_id.rsplit("_", 2)[0]


def _probe(path: Path) -> tuple[int, float, float]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    cap.release()
    dur = n / fps if fps > 0 else 0.0
    return n, round(fps, 3), round(dur, 3)


def build(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    labels = list(cfg["labels"])
    csv_dir = cfg.resolve_path("csv_dir")
    video_dir = cfg.resolve_path("clips_video")
    audio_dir = cfg.resolve_path("clips_audio")

    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(csv_dir / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    full = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("Video_ID")
        .rename(columns={"Video_ID": "video_id"})
        .reset_index(drop=True)
    )
    if "Original_ID" in full.columns:
        # Full download ships the true source-video id directly -- more
        # robust than parsing it back out of the clip id string.
        full["subject_id"] = full["Original_ID"]
    else:
        full["subject_id"] = full["video_id"].map(subject_of)

    # Verify subject-disjoint splits (PDF section 6, "Critical rule").
    by_split = {s: set(full.loc[full.split == s, "subject_id"]) for s in ("train", "val", "test")}
    overlaps = {
        "train_val": sorted(by_split["train"] & by_split["val"]),
        "train_test": sorted(by_split["train"] & by_split["test"]),
        "val_test": sorted(by_split["val"] & by_split["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Splits are NOT subject-disjoint: {overlaps}")

    has_video, has_audio, n_frames, fps_col, dur_col = [], [], [], [], []
    for vid in full["video_id"]:
        vpath = video_dir / f"{vid}.mp4"
        exists = vpath.exists()
        n, fps, dur = _probe(vpath) if exists else (0, 0.0, 0.0)
        has_v = exists and n > 0  # exclude truncated/corrupt files (0 decodable frames)
        has_video.append(int(has_v))
        has_audio.append(int((audio_dir / f"{vid}.wav").exists()))
        n_frames.append(n)
        fps_col.append(fps)
        dur_col.append(dur)

    meta = full[["video_id", "subject_id", "split", *labels]].copy()
    meta["has_video"] = has_video
    meta["has_audio"] = has_audio
    meta["n_frames"] = n_frames
    meta["fps"] = fps_col
    meta["duration_s"] = dur_col

    out_dir = cfg.resolve_path("metadata_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "metadata.csv", index=False)

    splits_dir = cfg.resolve_path("splits_dir")
    splits_dir.mkdir(parents=True, exist_ok=True)
    usable = meta[meta.has_video == 1].copy()
    for split in ("train", "val", "test"):
        usable.loc[usable.split == split, "video_id"].to_csv(
            splits_dir / f"{split}.txt", index=False, header=False
        )

    summary = {
        "labels": labels,
        "n_annotated": int(len(meta)),
        "n_with_video": int(usable.shape[0]),
        "n_with_audio_and_video": int(((meta.has_audio == 1) & (meta.has_video == 1)).sum()),
        "per_split_total": meta.split.value_counts().to_dict(),
        "per_split_usable": usable.split.value_counts().to_dict(),
        "usable_positive_counts": {lab: int(usable[lab].sum()) for lab in labels},
        "subject_disjoint": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'metadata.csv'} and splits under {splits_dir}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    build(ap.parse_args().config)
