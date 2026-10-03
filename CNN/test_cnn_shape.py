"""Shape test for the CNN model.

This version imports CNNLOB from cnn/train_cnn.py. If your teammate moves the
architecture to cnn/cnn_model.py, update the import accordingly.

Run from repo root:
  pytest tests/test_cnn_shape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cnn"))

from train_cnn import CNNLOB  # noqa: E402


def test_cnn_output_shape_from_sequence_input():
    model = CNNLOB()
    x = torch.randn(4, 100, 40)

    y = model(x)

    assert y.shape == (4, 3)


def test_cnn_output_shape_from_channel_input():
    model = CNNLOB()
    x = torch.randn(4, 1, 100, 40)

    y = model(x)

    assert y.shape == (4, 3)

