"""Shape test for the LSTM model.

Run from repo root:
  pytest tests/test_lstm_shape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lstm"))

from lstm.train_lstm import LSTMLOB  # noqa: E402


def test_lstm_output_shape():
    model = LSTMLOB()
    x = torch.randn(4, 100, 40)

    y = model(x)

    assert y.shape == (4, 3)

