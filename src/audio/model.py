"""Audio branch: small log-mel CNN encoder (PDF section 8)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class AudioBranch(nn.Module):
    """(B, 1, n_mels, time) -> (B, embed_dim)."""

    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg["model"]["embed_dim"]
        self.net = nn.Sequential(
            _block(1, 32), _block(32, 64), _block(64, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim), nn.ReLU(inplace=True),
            nn.Dropout(cfg["model"]["dropout"]),
        )
        self.out_dim = embed_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(mel))
