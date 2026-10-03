"""
train_mlp.py -- MLP baseline (flattened 100x40 -> FC layers -> 3 classes)

Data comes from preprocessing.py (X_train/X_val/X_test .npy + y_*.npy in data/processed).
      python train_mlp.py --tag base7
Tuning flags: --hidden 512 256 64  --dropout 0.3  --wd 1e-5  --lr 1e-3  --no-bn  --tag myrun
Each run writes <results-dir>/<tag>_metrics.json, _confusion.png, _curves.png, _best.pt
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW, N_FEAT = 100, 40
CLASSES = ["up", "stationary", "down"]      # 0=UP, 1=STATIONARY, 2=DOWN (preprocessing.py mapping)


class NpyWindows(Dataset):
    """Reads windows saved by preprocessing.py (X: (n,100,40) float32, y: (n,) int64).
    mmap keeps RAM low, because the X arrays are several GB."""
    def __init__(self, x_path, y_path, mmap=True):
        self.x = np.load(x_path, mmap_mode="r" if mmap else None)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.x[i])), int(self.y[i])


class MLP(nn.Module):
    def __init__(self, in_dim=WINDOW * N_FEAT, hidden=(512, 256, 64), p=0.3, n_cls=3, use_bn=True):
        super().__init__()
        layers, d = [nn.Flatten()], in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        ps.append(model(xb.to(dev)).argmax(1).cpu())
        ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--no-mmap", action="store_true", help="load X fully into RAM (needs ~7 GB)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", nargs="+", type=int, default=[512, 256, 64])
    ap.add_argument("--no-bn", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="mlp")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.results_dir, exist_ok=True)
    out = lambda s: f"{a.results_dir}/{a.tag}_{s}"

    d = a.data_dir
    tr, va, te = (NpyWindows(f"{d}/X_{s}.npy", f"{d}/y_{s}.npy", not a.no_mmap)
                  for s in ("train", "val", "test"))
    ytr = tr.y
    print(f"windows -> train {len(tr)}, val {len(va)}, test {len(te)} | device {dev}")
    for name, ds in (("train", tr), ("val", va), ("test", te)):
        print(f"{name} class mix [up stat down]:", np.round(np.bincount(ds.y, minlength=3) / len(ds), 3))
    print("train class counts:", np.bincount(ytr, minlength=3))

    cnt = np.bincount(ytr, minlength=3).astype(np.float32)
    w = torch.tensor(cnt.sum() / (3 * cnt), device=dev)       # inverse-frequency weights
    loss_fn = nn.CrossEntropyLoss(weight=w)

    dl = lambda ds, sh: DataLoader(ds, batch_size=a.bs, shuffle=sh)
    tl, vl, sl = dl(tr, True), dl(va, False), dl(te, False)

    model = MLP(hidden=tuple(a.hidden), p=a.dropout, use_bn=not a.no_bn).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    best, bad, hist = -1, 0, {"train_loss": [], "val_f1": []}

    for ep in range(a.epochs):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        yv, pv = predict(model, vl, dev)
        vf1 = f1_score(yv, pv, average="macro")
        sched.step(vf1)
        hist["train_loss"].append(tot / n); hist["val_f1"].append(vf1)
        print(f"ep {ep+1:02d} loss {tot/n:.4f} val macroF1 {vf1:.4f} lr {opt.param_groups[0]['lr']:.1e} ({time.time()-t0:.0f}s)")
        if vf1 > best:
            best, bad = vf1, 0
            torch.save(model.state_dict(), out("best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(out("best.pt"), map_location=dev))
    yt, pt = predict(model, sl, dev)
    p, r, f, _ = precision_recall_fscore_support(yt, pt, average="macro", zero_division=0)
    pc = f1_score(yt, pt, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(yt, pt, labels=[0, 1, 2])
    m = {"model": "MLP", "tag": a.tag, "config": vars(a), "accuracy": accuracy_score(yt, pt),
         "macro_precision": p, "macro_recall": r, "macro_f1": f,
         "per_class_f1": dict(zip(CLASSES, pc.tolist())),
         "confusion_matrix": cm.tolist(), "best_val_macro_f1": best,
         "params": sum(x.numel() for x in model.parameters())}
    json.dump(m, open(out("metrics.json"), "w"), indent=2)
    print(f"[{a.tag}] best val macroF1 {best:.4f} | test macroF1 {f:.4f} acc {m['accuracy']:.4f}")

    plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), CLASSES); plt.yticks(range(3), CLASSES)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(f"{a.tag} confusion (test)")
    plt.tight_layout(); plt.savefig(out("confusion.png"), dpi=150); plt.close()

    plt.figure(figsize=(6, 3.5))
    plt.plot(hist["train_loss"], label="train loss"); plt.plot(hist["val_f1"], label="val macro-F1")
    plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    plt.savefig(out("curves.png"), dpi=150)


if __name__ == "__main__":
    main()
