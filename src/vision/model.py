"""Visual branch (PDF sections 4 & 7).

Two backbone families, selected by ``cfg.model.vision_backbone``:

* 2D frame encoder + temporal head -- ``resnet18`` / ``resnet34`` / ``efficientnet_b0``
  (ImageNet pretrained, applied per frame, then mean / LSTM / Transformer over time).
* 3D video backbone -- ``r2plus1d_18`` / ``mc3_18`` / ``r3d_18``
  (Kinetics-400 pretrained spatiotemporal CNN; the temporal head is bypassed).

Input to ``VisionBranch.forward`` is always ``(B, T, 3, H, W)`` with pixels in
[0, 1]; per-backbone normalisation happens inside the module.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

_IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_KINETICS = ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
VIDEO_BACKBONES = {"r2plus1d_18", "mc3_18", "r3d_18"}


class _Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):  # x: (N, 3, H, W)
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- #
# 2D per-frame encoder
# --------------------------------------------------------------------------- #
class FrameEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet18", freeze: bool = True):
        super().__init__()
        if backbone == "resnet18":
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.feat_dim = 512
        elif backbone == "resnet34":
            net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            self.feat_dim = 512
        elif backbone == "efficientnet_b0":
            net = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            self.feat_dim = 1280
        else:
            raise ValueError(f"unknown 2D backbone {backbone}")

        if backbone.startswith("resnet"):
            self.body = nn.Sequential(*list(net.children())[:-1])
        else:
            self.body = nn.Sequential(net.features, net.avgpool)
        self.norm = _Normalize(*_IMAGENET)

        if freeze:
            for p in self.body.parameters():
                p.requires_grad = False
        self.frozen = freeze

    def forward(self, frames: torch.Tensor) -> torch.Tensor:  # (B,T,3,H,W) [0,1]
        b, t = frames.shape[:2]
        x = self.norm(frames.flatten(0, 1))
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            f = self.body(x).flatten(1)
        return f.view(b, t, -1)


class TemporalHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, kind: str = "transformer",
                 layers: int = 1, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.kind = kind
        self.proj = nn.Linear(in_dim, embed_dim)
        if kind == "mean":
            pass
        elif kind == "lstm":
            self.rnn = nn.LSTM(embed_dim, embed_dim // 2, num_layers=layers, batch_first=True,
                               bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        elif kind == "transformer":
            self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos = nn.Parameter(torch.zeros(1, 512, embed_dim))
            enc = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 2, dropout, batch_first=True)
            self.tr = nn.TransformerEncoder(enc, layers)
        else:
            raise ValueError(kind)
        self.out_dim = embed_dim

    def forward(self, feats: torch.Tensor):
        x = self.proj(feats)
        if self.kind == "mean":
            return x.mean(1), x
        if self.kind == "lstm":
            seq, _ = self.rnn(x)
            return seq.mean(1), seq
        b, t, _ = x.shape
        x = x + self.pos[:, :t]
        x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
        x = self.tr(x)
        return x[:, 0], x[:, 1:]


# --------------------------------------------------------------------------- #
# 3D video backbone
# --------------------------------------------------------------------------- #
class VideoEncoder(nn.Module):
    def __init__(self, backbone: str, embed_dim: int, freeze: bool = True):
        super().__init__()
        from torchvision.models import video as V

        table = {
            "r2plus1d_18": (V.r2plus1d_18, V.R2Plus1D_18_Weights.KINETICS400_V1, 512),
            "mc3_18": (V.mc3_18, V.MC3_18_Weights.KINETICS400_V1, 512),
            "r3d_18": (V.r3d_18, V.R3D_18_Weights.KINETICS400_V1, 512),
        }
        ctor, weights, feat = table[backbone]
        net = ctor(weights=weights)
        self.body = nn.Sequential(*list(net.children())[:-1])  # drop fc -> (B, feat, 1,1,1)
        self.norm = _Normalize(*_KINETICS)
        self.proj = nn.Linear(feat, embed_dim)
        if freeze:
            for p in self.body.parameters():
                p.requires_grad = False
        self.frozen = freeze
        self.out_dim = embed_dim

    def forward(self, frames: torch.Tensor):  # (B,T,3,H,W) [0,1]
        b, t = frames.shape[:2]
        x = self.norm(frames.flatten(0, 1)).view(b, t, 3, *frames.shape[-2:])
        x = x.permute(0, 2, 1, 3, 4)  # (B,3,T,H,W)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            f = self.body(x).flatten(1)
        emb = self.proj(f)
        return emb, None  # no per-frame evidence for 3D backbone


class VisionBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.is_video = m["vision_backbone"] in VIDEO_BACKBONES
        if self.is_video:
            self.encoder = VideoEncoder(m["vision_backbone"], m["embed_dim"], m["freeze_backbone"])
            self.out_dim = self.encoder.out_dim
        else:
            self.encoder = FrameEncoder(m["vision_backbone"], m["freeze_backbone"])
            self.temporal = TemporalHead(self.encoder.feat_dim, m["embed_dim"], m["temporal"],
                                         m["temporal_layers"], m["temporal_heads"], m["dropout"])
            self.out_dim = self.temporal.out_dim

    def forward(self, frames: torch.Tensor):
        if self.is_video:
            return self.encoder(frames)
        return self.temporal(self.encoder(frames))
