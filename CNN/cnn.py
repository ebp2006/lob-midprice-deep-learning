"""1D/2D CNN over the price-level x time grid.

Input: (batch, 1, T=100, 40), treating the 40 feature columns as a spatial
axis (price-level x {price,volume} x {bid,ask}) and T as the other spatial
axis -- a plain 2D CNN over this grid, simpler than DeepLOB's Inception
design (see models/deeplob.py for that).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),  # pair (price,vol) per level
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),  # merge bid/ask per level
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=(1, 10)),  # collapse remaining 10 price levels
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, T, 40)
        z = self.conv(x)
        z = self.pool(z).flatten(1)
        return self.head(z)
