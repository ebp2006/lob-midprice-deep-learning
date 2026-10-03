"""Shape test for the Transformer model.

Run from repo root:
  pytest tests/test_transformer_shape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "transformer"))

from train_transformer import TransformerLOB  # noqa: E402


def test_transformer_output_shape():
    model = TransformerLOB()
    x = torch.randn(4, 100, 40)

    y = model(x)

    assert y.shape == (4, 3)

