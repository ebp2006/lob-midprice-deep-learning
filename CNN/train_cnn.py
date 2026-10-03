"""Train CNN on preprocessed FI-2010 LOB windows.

Expected input files:
  data/processed/X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy

Run from repo root:
  python cnn/train_cnn.py --data-dir data/processed --epochs 30 --tag cnn
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["up", "stationary", "down"]


class CNNLOB(nn.Module):
    """2D CNN over 100-event x 40-feature LOB windows."""

    def __init__(self, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (batch, 1, 100, 40)
        return self.classifier(self.features(x))


def load_arrays(data_dir: Path):
    x_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int64)
    x_val = np.load(data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(data_dir / "y_val.npy").astype(np.int64)
    x_test = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(data_dir / "y_test.npy").astype(np.int64)
    return x_train, y_train, x_val, y_val, x_test, y_test


def make_loader(x, y, batch_size: int, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        ps.append(logits.argmax(1).cpu())
        ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def plot_confusion(cm, out_path: Path, title: str):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), CLASSES)
    plt.yticks(range(3), CLASSES)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="cnn/results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="cnn")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = load_arrays(Path(args.data_dir))
    train_loader = make_loader(x_train, y_train, args.bs, True)
    val_loader = make_loader(x_val, y_val, args.bs, False)
    test_loader = make_loader(x_test, y_test, args.bs, False)

    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = torch.tensor(counts.sum() / (3 * counts), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    model = CNNLOB(dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    best, bad = -1.0, 0
    hist = {"train_loss": [], "val_f1": []}
    best_path = out_dir / f"{args.tag}_best.pt"

    for ep in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)

        yv, pv = predict(model, val_loader, device)
        val_f1 = f1_score(yv, pv, average="macro", zero_division=0)
        sched.step(val_f1)
        hist["train_loss"].append(total / n)
        hist["val_f1"].append(val_f1)
        print(f"ep {ep + 1:02d} loss {total / n:.4f} val macroF1 {val_f1:.4f} lr {opt.param_groups[0]['lr']:.1e} ({time.time() - t0:.0f}s)")

        if val_f1 > best:
            best, bad = val_f1, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    yt, pt = predict(model, test_loader, device)
    p, r, f, _ = precision_recall_fscore_support(yt, pt, average="macro", zero_division=0)
    pc = f1_score(yt, pt, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(yt, pt, labels=[0, 1, 2])

    metrics = {
        "model": "CNN",
        "tag": args.tag,
        "config": vars(args),
        "accuracy": accuracy_score(yt, pt),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
        "per_class_f1": dict(zip(CLASSES, pc.tolist())),
        "confusion_matrix": cm.tolist(),
        "best_val_macro_f1": best,
        "params": sum(x.numel() for x in model.parameters()),
    }
    (out_dir / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(cm, out_dir / f"{args.tag}_confusion.png", f"{args.tag} confusion")

    plt.figure(figsize=(6, 3.5))
    plt.plot(hist["train_loss"], label="train loss")
    plt.plot(hist["val_f1"], label="val macro-F1")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.tag}_curves.png", dpi=150)
    plt.close()

    print(f"[{args.tag}] best val macroF1 {best:.4f} | test macroF1 {f:.4f} acc {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()

