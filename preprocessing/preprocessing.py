"""Data preprocessing module for FI-2010 Limit Order Book (LOB) dataset.

Pipeline:
1. Load raw FI-2010 Z-Score normalized TXT files.
2. Extract 40 raw LOB features (rows 0-39) -> shape (n_events, 40).
3. Extract k=10 target labels (row 144) -> mapped to {0: UP, 1: STATIONARY, 2: DOWN}.
4. Chronological train/validation split (80/20 of train period).
5. Build 100-event sliding windows separately per split to prevent data leakage.
6. Save preprocessed numpy arrays and metadata JSON.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

# FI-2010 Dataset Constants
N_RAW_FEATURES = 40
N_TOTAL_ROWS = 149
LABEL_ROW_K10 = 144  # Row 144 corresponding to horizon k=10 (index 0 of the 5 label rows 144-148)

# Target label mapping according to prompt:
# Original FI-2010 labels: 1 = DOWN, 2 = STATIONARY, 3 = UP
# Standardized mapping: 0 = UP, 1 = STATIONARY, 2 = DOWN
LABEL_MAPPING = {
    3: 0,  # UP
    2: 1,  # STATIONARY
    1: 2,  # DOWN
}

LABEL_NAMES = {0: "UP", 1: "STATIONARY", 2: "DOWN"}


def read_raw_txt_matrix(file_path: Path) -> np.ndarray:
    """Read whitespace-delimited FI-2010 text file and return feature-major matrix (149, n_events)."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[preprocessing] Reading raw matrix from {file_path.name}...")
    matrix = np.loadtxt(file_path)

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")

    r, c = matrix.shape
    if r == N_TOTAL_ROWS:
        return matrix  # Feature-major (149, n_events)
    elif c == N_TOTAL_ROWS:
        return matrix.T  # Events-major (n_events, 149) -> transpose
    else:
        raise ValueError(
            f"Neither dimension matches expected {N_TOTAL_ROWS} rows. Shape: {matrix.shape}"
        )


def extract_raw_lob_features(matrix: np.ndarray) -> np.ndarray:
    """Extract top 40 raw LOB features (rows 0..39) transposed to shape (n_events, 40)."""
    features = matrix[:N_RAW_FEATURES, :].T  # (n_events, 40)
    return features.astype(np.float32)


def extract_labels(matrix: np.ndarray, label_row: int = LABEL_ROW_K10) -> np.ndarray:
    """Extract horizon label row, map values {1: DOWN, 2: STATIONARY, 3: UP} to {0: UP, 1: STATIONARY, 2: DOWN}."""
    raw_labels = matrix[label_row, :].astype(int)  # (n_events,)

    unique_vals = set(np.unique(raw_labels).tolist())
    if not unique_vals.issubset({1, 2, 3}):
        raise ValueError(f"Unexpected raw label values found: {unique_vals}. Expected subset of {{1, 2, 3}}.")

    # Map labels: 3 -> 0 (UP), 2 -> 1 (STATIONARY), 1 -> 2 (DOWN)
    mapped_labels = np.zeros_like(raw_labels, dtype=np.int64)
    for raw_val, target_val in LABEL_MAPPING.items():
        mapped_labels[raw_labels == raw_val] = target_val

    return mapped_labels


def load_split_file(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single raw FI-2010 file, return (features, labels)."""
    matrix = read_raw_txt_matrix(file_path)
    features = extract_raw_lob_features(matrix)
    labels = extract_labels(matrix, label_row=LABEL_ROW_K10)
    return features, labels


def create_train_val_split(
    train_features: np.ndarray, train_labels: np.ndarray, val_fraction: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split training events into Train (1 - val_fraction) and Val (val_fraction).

    No random shuffling is performed to preserve temporal sequence.
    """
    n_events = train_features.shape[0]
    cut_idx = int(round(n_events * (1.0 - val_fraction)))

    X_train_raw = train_features[:cut_idx]
    y_train_raw = train_labels[:cut_idx]
    X_val_raw = train_features[cut_idx:]
    y_val_raw = train_labels[cut_idx:]

    print(f"[preprocessing] Chronological split at event {cut_idx}/{n_events}:")
    print(f"                Train events: {X_train_raw.shape[0]}, Val events: {X_val_raw.shape[0]}")

    return X_train_raw, y_train_raw, X_val_raw, y_val_raw


def create_sliding_windows(
    features: np.ndarray, labels: np.ndarray, window_size: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build chronological 100-event sliding windows.

    Window i covers events [i : i + window_size].
    Label for window i is the label associated with the LAST event in the window (index i + window_size - 1).

    Returns:
        X: (n_windows, window_size, 40) float32 array
        y: (n_windows,) int64 array
        end_indices: (n_windows,) int64 array tracking ending event indices in original sequence
    """
    n_events, n_features = features.shape
    if n_events < window_size:
        raise ValueError(f"Number of events ({n_events}) < window size ({window_size})")

    n_windows = n_events - window_size + 1
    X = np.zeros((n_windows, window_size, n_features), dtype=np.float32)

    for i in range(n_windows):
        X[i] = features[i : i + window_size]

    y = labels[window_size - 1 :].astype(np.int64)
    end_indices = np.arange(window_size - 1, n_events, dtype=np.int64)

    return X, y, end_indices


def find_raw_files(raw_dir: Path) -> Tuple[Path, list[Path]]:
    """Locate Train_Dst_NoAuction_ZScore_CF_7.txt and Test_Dst_NoAuction_ZScore_CF_7,8,9.txt."""
    all_files = list(raw_dir.rglob("*.txt"))
    if not all_files:
        raise FileNotFoundError(f"No .txt files found under {raw_dir}")

    train_file = None
    test_files = []

    for f in all_files:
        filename = f.name
        if "Train_Dst_NoAuction_ZScore_CF_7" in filename or (
            "Train" in filename and "ZScore" in filename and "_7" in filename
        ):
            train_file = f
        elif "Test" in filename and "ZScore" in filename and any(f"_{k}" in filename for k in (7, 8, 9)):
            test_files.append(f)

    if train_file is None:
        raise FileNotFoundError(f"Train_Dst_NoAuction_ZScore_CF_7.txt not found in {raw_dir}")

    test_files = sorted(test_files, key=lambda f: f.name)
    if len(test_files) < 3:
        # Fallback to any Test files found
        test_files = sorted([f for f in all_files if "Test" in f.name])

    print(f"[preprocessing] Found Train file: {train_file.name}")
    print(f"[preprocessing] Found Test files ({len(test_files)}): {[f.name for f in test_files]}")

    return train_file, test_files


def compute_class_distribution(y: np.ndarray) -> Dict[str, Any]:
    """Compute counts and percentages for classes 0 (UP), 1 (STATIONARY), 2 (DOWN)."""
    unique, counts = np.unique(y, return_counts=True)
    count_dict = {int(k): int(v) for k, v in zip(unique, counts)}
    total = len(y)
    dist = {}
    for c in [0, 1, 2]:
        c_count = count_dict.get(c, 0)
        c_pct = (c_count / total * 100.0) if total > 0 else 0.0
        dist[LABEL_NAMES[c]] = {"code": c, "count": c_count, "percentage": round(c_pct, 2)}
    return dist


def process_and_save_data(
    raw_dir: str | Path,
    output_dir: str | Path,
    window_size: int = 100,
    val_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Complete data preprocessing pipeline."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file, test_files = find_raw_files(raw_dir)

    # 1. Load Train Raw
    train_features, train_labels = load_split_file(train_file)
    print(f"[preprocessing] Raw Train dataset: {train_features.shape[0]} events, {train_features.shape[1]} features.")

    # 2. Chronological Train / Val Split
    X_tr_raw, y_tr_raw, X_va_raw, y_va_raw = create_train_val_split(
        train_features, train_labels, val_fraction=val_fraction
    )

    # 3. Create Sliding Windows for Train and Val
    X_train, y_train, train_end_idx = create_sliding_windows(X_tr_raw, y_tr_raw, window_size=window_size)
    X_val, y_val, val_end_idx = create_sliding_windows(X_va_raw, y_va_raw, window_size=window_size)

    # 4. Load Test Files and Create Windows separately per test file to prevent file-boundary leakage
    test_X_windows = []
    test_y_windows = []
    for tf in test_files:
        tf_feats, tf_labs = load_split_file(tf)
        w_x, w_y, _ = create_sliding_windows(tf_feats, tf_labs, window_size=window_size)
        test_X_windows.append(w_x)
        test_y_windows.append(w_y)

    X_test = np.concatenate(test_X_windows, axis=0)
    y_test = np.concatenate(test_y_windows, axis=0)

    # 5. Save numpy arrays
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    # 6. Compute Metadata
    metadata = {
        "dataset_name": "FI-2010 Limit Order Book",
        "prediction_horizon_k": 10,
        "label_row_index": LABEL_ROW_K10,
        "window_size": window_size,
        "num_features": N_RAW_FEATURES,
        "val_fraction": val_fraction,
        "label_mapping": {"3 (raw)": "0 (UP)", "2 (raw)": "1 (STATIONARY)", "1 (raw)": "2 (DOWN)"},
        "shapes": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
        "class_distributions": {
            "train": compute_class_distribution(y_train),
            "val": compute_class_distribution(y_val),
            "test": compute_class_distribution(y_test),
        },
        "raw_files_used": {
            "train": train_file.name,
            "test": [f.name for f in test_files],
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[preprocessing] Preprocessed datasets saved to {output_dir}:")
    print(f"                X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"                X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"                X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return metadata


if __name__ == "__main__":
    raw_dir = Path("data/raw")
    output_dir = Path("data/processed")
    if raw_dir.exists():
        process_and_save_data(raw_dir, output_dir)
    else:
        print(f"Raw directory {raw_dir} does not exist yet.")
