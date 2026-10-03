"""Optional stretch model: a small Transformer encoder over the T x 40 window.

Input: (batch, T=100, 40), same convention as lstm.py. A linear projection
lifts the 40-dim snapshot to d_model, learned positional embeddings encode
tick-time order (event order, not wall-clock time), a standard
pre-norm TransformerEncoder does the sequence mixing, and the final time
step's representation is classified -- mirroring how lstm.py uses only the
last hidden state, for a fair architectural comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, n_features: int = 40, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 num_classes: int = 3, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, 40)
        T = x.shape[1]
        z = self.input_proj(x) + self.pos_embed[:, :T, :]
        z = self.encoder(z)
        last = z[:, -1, :]
        return self.head(last)
