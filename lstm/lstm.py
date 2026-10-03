"""Plain LSTM over the T x 40 sliding window. Input: (batch, T, 40)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int = 40, hidden: int = 64, num_layers: int = 1,
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, 40)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # final time step's hidden state
        return self.head(last)
