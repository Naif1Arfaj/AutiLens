"""Frozen-backbone feature extraction (run once, train heads for seconds).

Nothing in the vision/audio backbones trains, so re-running them every epoch --
40 epochs x 5 folds = 200 identical passes -- was pure waste. Here we run each
backbone ONCE per clip and cache the resulting vectors; head training then reads
float16 vectors and takes seconds instead of ~45 min per config.

Because augmentation normally happens *before* the backbone, the variants we
want must be baked into the cache:

  vision : (n_windows, flips, D)   3 temporal windows x {original, h-flip}
  audio  : (n_variants, D)         1 clean + (n_variants-1) SpecAugment draws

Continuous augmentation is then recovered at train time with feature-space
mixup, which costs nothing.

Outputs
  data/processed/feat_<backbone>/<video_id>.npy
  data/processed/feat_audio_<encoder>/<video_id>.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config, resolve_device

# Kinetics-400 / ImageNet normalisation constants.
_KINETICS = ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
_IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# --------------------------------------------------------------------------- #
# Vision backbones
# --------------------------------------------------------------------------- #
def build_vision_backbone(name: str, device: str):
    """Return (module, feat_dim, input_size). Module maps (B,3,T,H,W) -> (B,D)."""
    from torchvision.models import video as V

    if name == "swin3d_t":
        net = V.swin3d_t(weights=V.Swin3D_T_Weights.KINETICS400_V1)
        net.head = nn.Identity()
        dim, size = 768, 224
        model = net
    elif name in ("r2plus1d_18", "mc3_18", "r3d_18"):
        table = {
            "r2plus1d_18": (V.r2plus1d_18, V.R2Plus1D_18_Weights.KINETICS400_V1),
            "mc3_18": (V.mc3_18, V.MC3_18_Weights.KINETICS400_V1),
            "r3d_18": (V.r3d_18, V.R3D_18_Weights.KINETICS400_V1),
        }
        ctor, w = table[name]
        net = ctor(weights=w)
        model = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
        dim, size = 512, 112
    else:
        raise ValueError(f"unknown vision backbone {name!r}")

    return model.eval().to(device), dim, size


class _Norm:
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def __call__(self, x):  # x: (N,3,H,W) in [0,1]
        return (x - self.mean) / self.std


@torch.no_grad()
def extract_vision(cfg, device: str, overwrite: bool = False) -> None:
    import pandas as pd
    from tqdm import tqdm

    name = cfg["features"]["vision_backbone"]
    flips = int(cfg["features"]["flips"])
    model, dim, in_size = build_vision_backbone(name, device)
    norm = _Norm(*_KINETICS, device)

    proc = cfg.resolve_path("processed_dir")
    src_dir = proc / "windows"
    out_dir = proc / f"feat_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(cfg.resolve_path("metadata_dir") / "metadata.csv")
    meta = meta[meta.has_video == 1]
    n_win = int(cfg["windows"]["n_windows"])

    for vid in tqdm(meta.video_id, desc=f"vision:{name}"):
        dst = out_dir / f"{vid}.npy"
        if not overwrite and dst.exists():
            continue
        src = src_dir / f"{vid}.npy"
        if not src.exists():
            continue

        arr = np.load(src)                      # (W,T,H,W,3) uint8
        x = torch.from_numpy(arr).to(device).float().div_(255.0)
        w, t = x.shape[0], x.shape[1]
        x = x.permute(0, 1, 4, 2, 3).reshape(w * t, 3, *x.shape[2:4])  # (W*T,3,H,W)
        if x.shape[-1] != in_size:              # e.g. 224 cache -> 112 backbone
            x = nn.functional.interpolate(x, size=(in_size, in_size),
                                          mode="bilinear", align_corners=False)
        x = norm(x).view(w, t, 3, in_size, in_size)

        feats = np.empty((n_win, flips, dim), dtype=np.float16)
        for f in range(flips):
            xf = torch.flip(x, dims=[-1]) if f == 1 else x
            out = model(xf.permute(0, 2, 1, 3, 4))   # (W,3,T,H,W) -> (W,D)
            feats[:, f] = out.float().cpu().numpy().astype(np.float16)
        np.save(dst, feats)

    print(f"vision features -> {out_dir}  shape=({n_win},{flips},{dim})")


# --------------------------------------------------------------------------- #
# Audio encoder: ImageNet ResNet18 over the mel spectrogram
# --------------------------------------------------------------------------- #
def build_audio_encoder(name: str, device: str):
    import torchvision

    if name != "resnet18_mel":
        raise ValueError(f"unknown audio encoder {name!r}")
    net = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    model = nn.Sequential(*list(net.children())[:-1], nn.Flatten())
    return model.eval().to(device), 512


def spec_augment(mel: np.ndarray, rng: np.random.Generator, aug_cfg) -> np.ndarray:
    """SpecAugment: frequency + time masking. The audio branch previously had
    NO augmentation at all, which on 246 clips is a memorisation setup."""
    m = mel.copy()
    n_mels, n_t = m.shape
    fill = float(m.min())
    for _ in range(int(aug_cfg["freq_masks"])):
        w = int(rng.integers(0, int(aug_cfg["freq_width"]) + 1))
        if w and n_mels > w:
            f0 = int(rng.integers(0, n_mels - w))
            m[f0:f0 + w, :] = fill
    for _ in range(int(aug_cfg["time_masks"])):
        w = int(rng.integers(0, int(aug_cfg["time_width"]) + 1))
        if w and n_t > w:
            t0 = int(rng.integers(0, n_t - w))
            m[:, t0:t0 + w] = fill
    if rng.random() < 0.5:  # small circular time shift
        m = np.roll(m, int(rng.integers(-n_t // 10, n_t // 10 + 1)), axis=1)
    return m


def _mel_to_image(mel: np.ndarray, device: str) -> torch.Tensor:
    """(n_mels,T) log-mel -> (1,3,n_mels,T) normalised 3-channel 'image'."""
    m = torch.from_numpy(mel).float().to(device)
    m = (m - m.mean()) / (m.std() + 1e-6)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)      # -> [0,1]
    m = m.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # grey -> 3 channels
    mean = torch.tensor(_IMAGENET[0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET[1], device=device).view(1, 3, 1, 1)
    return (m - mean) / std


@torch.no_grad()
def extract_audio(cfg, device: str, overwrite: bool = False) -> None:
    import pandas as pd
    from tqdm import tqdm

    from src.preprocessing.features import log_mel

    name = cfg["features"]["audio_encoder"]
    model, dim = build_audio_encoder(name, device)
    n_var = int(cfg["audio"]["n_variants"])
    aug_cfg = cfg["audio"]["spec_aug"]

    proc = cfg.resolve_path("processed_dir")
    mel_dir = proc / "mel"
    out_dir = proc / f"feat_audio_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = cfg.resolve_path("clips_audio")
    video_dir = cfg.resolve_path("clips_video")

    meta = pd.read_csv(cfg.resolve_path("metadata_dir") / "metadata.csv")
    meta = meta[meta.has_video == 1]

    for vid in tqdm(meta.video_id, desc=f"audio:{name}"):
        dst = out_dir / f"{vid}.npy"
        if not overwrite and dst.exists():
            continue
        cache = mel_dir / f"{vid}.npy"
        if cache.exists():
            mel = np.load(cache)
        else:
            src = audio_dir / f"{vid}.wav"
            src = src if src.exists() else video_dir / f"{vid}.mp4"
            mel = log_mel(src, cfg["audio"])

        rng = np.random.default_rng(abs(hash(vid)) % (2**32))
        feats = np.empty((n_var, dim), dtype=np.float16)
        for v in range(n_var):
            m = mel if v == 0 else spec_augment(mel, rng, aug_cfg)  # variant 0 = clean
            out = model(_mel_to_image(m, device))
            feats[v] = out.float().cpu().numpy().astype(np.float16)
        np.save(dst, feats)

    print(f"audio features -> {out_dir}  shape=({n_var},{dim})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", help="override features.vision_backbone")
    ap.add_argument("--only", choices=["vision", "audio"], default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.backbone:
        cfg["features"]["vision_backbone"] = a.backbone
    dev = resolve_device(cfg["train"]["device"])
    print(f"device={dev}  vision={cfg['features']['vision_backbone']}  "
          f"audio={cfg['features']['audio_encoder']}")
    if a.only != "audio":
        extract_vision(cfg, dev, a.overwrite)
    if a.only != "vision":
        extract_audio(cfg, dev, a.overwrite)
