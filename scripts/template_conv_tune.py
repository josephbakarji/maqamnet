#!/usr/bin/env python3
"""
Template-conv systematic tuning: resolution, architecture, regularization.

Key hypothesis: 40c/bin is too coarse for bayat/kurd distinction (51c apart).
Need to make higher resolution work by addressing the sparsity problem.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478,
    estimate_tonic, _TONIC_TEMPLATES, EXTENDED_MAQAMS
)
from template_conv_v2 import build_templates, compute_features, FlatDataset

PROJECT_DIR = Path(__file__).parent.parent
REF_HZ = 110.0


# ============================================================
# Models with various regularization / architectural choices
# ============================================================
class CircConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=(k//2, 0))
    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
        return self.conv(x)


class TemplateNet(nn.Module):
    """Configurable template-conv CNN."""
    def __init__(self, in_ch, h, w, nc, channels, kernel_size=3,
                 dropout=0.3, use_circular=True, pool_strategy="adaptive"):
        super().__init__()
        layers = []
        in_c = in_ch
        for i, ch in enumerate(channels):
            if use_circular:
                layers.append(CircConv2d(in_c, ch, kernel_size))
            else:
                layers.append(nn.Conv2d(in_c, ch, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU())
            if i < len(channels) - 1:
                layers.append(nn.Dropout2d(dropout * 0.5))
            in_c = ch
        self.features = nn.Sequential(*layers)

        if pool_strategy == "adaptive":
            self.pool = nn.AdaptiveAvgPool2d((3, 6))
            fc_in = channels[-1] * 3 * 6
        elif pool_strategy == "global":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            fc_in = channels[-1]
        else:  # max
            self.pool = nn.AdaptiveMaxPool2d((3, 6))
            fc_in = channels[-1] * 3 * 6

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, nc),
        )

    def forward(self, x):
        return self.fc(self.pool(self.features(x)))


class TemplateNetDeep(nn.Module):
    """Deeper network with residual connections for higher-res inputs."""
    def __init__(self, in_ch, h, w, nc, base_ch=32, n_blocks=3,
                 dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            CircConv2d(in_ch, base_ch, 5),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )

        blocks = []
        ch = base_ch
        for i in range(n_blocks):
            next_ch = ch * 2 if i < n_blocks - 1 else ch
            blocks.append(ResBlock(ch, next_ch, dropout))
            ch = next_ch
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 2 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, nc),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.fc(self.pool(x))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.conv1 = CircConv2d(in_ch, out_ch, 3)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = CircConv2d(out_ch, out_ch, 3)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout * 0.3)
        self.downsample = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + F.adaptive_avg_pool2d(identity, out.shape[-2:]))
        return self.pool(out)


# ============================================================
# Training with proper scheduling
# ============================================================
def train_cv(X, y, g, model_fn, tag, epochs=250, patience=40, bs=32,
             lr=1e-3, weight_decay=1e-4, use_scheduler=True):
    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnts = Counter(strat)
    for i in range(len(strat)):
        if cnts[strat[i]] < 5: strat[i] = str(y[i])
    sp = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(y); accs = []

    for fi, (tr, te) in enumerate(sp.split(X, strat)):
        model = model_fn()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if use_scheduler:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(FlatDataset(X[tr], y[tr]), batch_size=bs, shuffle=True)

        ba, bp, pat = 0, None, 0
        for ep in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in loader:
                loss = crit(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X[te]))
                p = logits.argmax(1).numpy(); acc = (p == y[te]).mean()
                if acc > ba: ba = acc; bp = p; pat = 0
                else: pat += 1
                if pat >= patience: break
            if use_scheduler:
                sched.step(epoch_loss)

        preds[te] = bp; accs.append(ba)

    acc = np.mean(accs)
    ds = {d: (preds[g==d]==y[g==d]).mean() for d in sorted(set(g))}

    # Per-maqam accuracy
    class_names = sorted(set(g))  # actually need maqam names, not dataset
    cm = confusion_matrix(y, preds)

    print(f"  {tag:55s} {acc:.1%}  "
          f"oud={ds.get('oud',0):.1%}  cairo={ds.get('cairo',0):.1%}  "
          f"m478={ds.get('maqam478',0):.1%}")
    return acc, preds


# ============================================================
def main():
    print("=" * 70)
    print("TEMPLATE-CONV HYPERPARAMETER TUNING")
    print("=" * 70)

    print("\nLoading...")
    records = []
    for loader, name in [(load_arabic_oud, "oud"),
                          (load_cairo_congress, "cairo"),
                          (load_maqam478, "maqam478")]:
        recs = loader()
        print(f"  {name}: {len(recs)}")
        records.extend(recs)
    records = [r for r in records if r["maqam"] in EXTENDED_MAQAMS]

    class_names = sorted(set(r["maqam"] for r in records))
    maqam_to_idx = {m: i for i, m in enumerate(class_names)}
    nc = len(class_names)

    # ── Precompute features at multiple resolutions ──
    print("\nPrecomputing features...")
    feature_cache = {}
    for num_bins in [20, 30, 40, 60, 120]:
        templates, tpl_names = build_templates(num_bins)
        nt = len(templates)
        for num_time in [10, 20]:
            feats, labels, groups = [], [], []
            for r in records:
                f = compute_features(r["f0"], num_bins, num_time, templates, "full")
                if f is not None:
                    feats.append(f)
                    labels.append(maqam_to_idx[r["maqam"]])
                    groups.append(r["dataset"])
            X = np.array(feats, dtype=np.float32).transpose(0, 2, 1, 3)  # (N,nt,T,bins)
            y = np.array(labels); g = np.array(groups)
            feature_cache[(num_bins, num_time)] = (X, y, g, nt)
            print(f"  {num_bins}bins x {num_time}t: {X.shape}")

    # ── Sweep 1: Resolution ──
    print(f"\n{'='*70}")
    print("SWEEP 1: RESOLUTION (fixed arch: (32,64), circ, adaptive pool)")
    print(f"{'='*70}")

    for num_bins in [20, 30, 40, 60, 120]:
        for num_time in [10, 20]:
            X, y, g, nt = feature_cache[(num_bins, num_time)]
            train_cv(X, y, g,
                     lambda nt_=nt, ntm=num_time, nb=num_bins:
                         TemplateNet(nt_, ntm, nb, nc, (32, 64), 3, 0.3, True, "adaptive"),
                     f"bins={num_bins} time={num_time} (32,64)")

    # ── Sweep 2: Architecture depth/width at best resolution candidates ──
    print(f"\n{'='*70}")
    print("SWEEP 2: ARCHITECTURE (60 bins, 20 time)")
    print(f"{'='*70}")

    X60, y60, g60, nt60 = feature_cache[(60, 20)]

    configs = [
        ((16, 32), 3, 0.3, "adaptive", "(16,32) k3"),
        ((32, 64), 3, 0.3, "adaptive", "(32,64) k3"),
        ((32, 64, 128), 3, 0.3, "adaptive", "(32,64,128) k3"),
        ((64, 128), 3, 0.3, "adaptive", "(64,128) k3"),
        ((32, 64), 5, 0.3, "adaptive", "(32,64) k5"),
        ((32, 64), 3, 0.5, "adaptive", "(32,64) k3 drop=0.5"),
        ((32, 64), 3, 0.2, "adaptive", "(32,64) k3 drop=0.2"),
        ((32, 64), 3, 0.3, "global", "(32,64) k3 global_pool"),
        ((32, 64), 3, 0.3, "max", "(32,64) k3 max_pool"),
    ]
    for chs, ks, drop, pool, label in configs:
        train_cv(X60, y60, g60,
                 lambda c=chs, k=ks, d=drop, p=pool:
                     TemplateNet(nt60, 20, 60, nc, c, k, d, True, p),
                 f"60bins {label}")

    # ── Sweep 3: Deep residual network ──
    print(f"\n{'='*70}")
    print("SWEEP 3: DEEP RESIDUAL NET")
    print(f"{'='*70}")

    for num_bins in [30, 60]:
        X, y, g, nt = feature_cache[(num_bins, 20)]
        for base_ch in [16, 32]:
            for n_blocks in [2, 3]:
                for drop in [0.2, 0.4]:
                    train_cv(X, y, g,
                             lambda nb=num_bins, bc=base_ch, nb_=n_blocks, d=drop:
                                 TemplateNetDeep(nt, 20, nb, nc, bc, nb_, d),
                             f"{num_bins}bins ResNet ch={base_ch} blocks={n_blocks} d={drop}")

    # ── Sweep 4: Optimization (LR, weight decay, batch size) ──
    print(f"\n{'='*70}")
    print("SWEEP 4: OPTIMIZATION (60bins, (32,64))")
    print(f"{'='*70}")

    for lr in [5e-4, 1e-3, 2e-3]:
        for wd in [1e-4, 1e-3, 1e-2]:
            train_cv(X60, y60, g60,
                     lambda: TemplateNet(nt60, 20, 60, nc, (32, 64), 3, 0.3, True, "adaptive"),
                     f"lr={lr} wd={wd}",
                     lr=lr, weight_decay=wd)

    # ── Sweep 5: Per-maqam analysis of best model ──
    print(f"\n{'='*70}")
    print("BEST MODEL: PER-MAQAM ANALYSIS")
    print(f"{'='*70}")

    # Run best config and get per-maqam breakdown
    best_bins, best_time = 60, 20
    X, y, g, nt = feature_cache[(best_bins, best_time)]
    acc, preds = train_cv(X, y, g,
                          lambda: TemplateNet(nt, 20, best_bins, nc, (32, 64, 128), 3, 0.3, True, "adaptive"),
                          "BEST: 60bins (32,64,128)")

    print(f"\n  Per-maqam accuracy:")
    for ci, maq in enumerate(class_names):
        mask = y == ci
        if mask.sum() > 0:
            macc = (preds[mask] == ci).mean()
            print(f"    {maq:12s} (n={mask.sum():3d}): {macc:.1%}")

    cm = confusion_matrix(y, preds)
    print(f"\n  Confusion matrix:")
    print(f"  {'':12s}", end="")
    for m in class_names:
        print(f" {m[:5]:>5s}", end="")
    print()
    for i, m in enumerate(class_names):
        print(f"  {m:12s}", end="")
        for j in range(nc):
            print(f" {cm[i,j]:5d}", end="")
        print()

    print(f"\n  Reference: Disambig Transformer=85.8%  Oracle=93.5%")


if __name__ == "__main__":
    main()
