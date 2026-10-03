"""Tests for the shared FI-2010 preprocessing pipeline.

Run from repo root:
  pytest tests/test_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.preprocessing import (  # noqa: E402
    LABEL_MAPPING,
    create_sliding_windows,
    create_train_val_split,
    extract_labels,
    extract_raw_lob_features,
)


def test_extract_raw_lob_features_shape():
    matrix = np.zeros((149, 12), dtype=np.float32)
    matrix[:40, :] = np.arange(40 * 12, dtype=np.float32).reshape(40, 12)

    features = extract_raw_lob_features(matrix)

    assert features.shape == (12, 40)
    assert features.dtype == np.float32
    assert features[0, 0] == matrix[0, 0]
    assert features[0, 39] == matrix[39, 0]


def test_extract_labels_maps_fi2010_labels_to_project_labels():
    matrix = np.zeros((149, 6), dtype=np.float32)
    matrix[144, :] = np.array([1, 2, 3, 1, 3, 2])

    labels = extract_labels(matrix)

    expected = np.array([LABEL_MAPPING[1], LABEL_MAPPING[2], LABEL_MAPPING[3], LABEL_MAPPING[1], LABEL_MAPPING[3], LABEL_MAPPING[2]])
    np.testing.assert_array_equal(labels, expected)
    assert labels.dtype == np.int64


def test_chronological_train_val_split_keeps_order():
    x = np.arange(10 * 40, dtype=np.float32).reshape(10, 40)
    y = np.arange(10, dtype=np.int64)

    x_train, y_train, x_val, y_val = create_train_val_split(x, y, val_fraction=0.2)

    assert x_train.shape[0] == 8
    assert x_val.shape[0] == 2
    np.testing.assert_array_equal(y_train, np.arange(8))
    np.testing.assert_array_equal(y_val, np.arange(8, 10))


def test_sliding_windows_use_last_event_label():
    x = np.arange(6 * 40, dtype=np.float32).reshape(6, 40)
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

    windows, labels, end_indices = create_sliding_windows(x, y, window_size=3)

    assert windows.shape == (4, 3, 40)
    np.testing.assert_array_equal(labels, np.array([2, 0, 1, 2]))
    np.testing.assert_array_equal(end_indices, np.array([2, 3, 4, 5]))
    np.testing.assert_array_equal(windows[0], x[0:3])
    np.testing.assert_array_equal(windows[-1], x[3:6])


def test_windows_do_not_cross_split_boundary_when_called_per_split():
    x_train = np.arange(5 * 40, dtype=np.float32).reshape(5, 40)
    y_train = np.zeros(5, dtype=np.int64)
    x_val = np.arange(1000, 1000 + 5 * 40, dtype=np.float32).reshape(5, 40)
    y_val = np.ones(5, dtype=np.int64)

    train_windows, train_labels, _ = create_sliding_windows(x_train, y_train, window_size=3)
    val_windows, val_labels, _ = create_sliding_windows(x_val, y_val, window_size=3)

    assert train_windows.max() < 1000
    assert val_windows.min() >= 1000
    assert set(train_labels.tolist()) == {0}
    assert set(val_labels.tolist()) == {1}

