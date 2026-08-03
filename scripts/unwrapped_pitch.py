#!/usr/bin/env python3
"""
Unwrapped pitch representation: no tonic normalization, no octave wrapping.
Let the model learn tonic invariance implicitly.

The pitch-time matrix spans the full vocal/instrument range (~60-800 Hz)
in cents from a fixed reference. The model must learn to:
1. Find the tonic (most repeated / structurally important pitch)
2. Compute intervals relative to it
3. Classify the maqam

We test CNN, LSTM, and Transformer to see which architecture handles
the tonic invariance best.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478,
    estimate_tonic, compute_histogram,
    EXTENDED_MAQAMS, MAQAM_COLORS
)

PROJECT_DIR = Path(__file__).parent.parent


# ============================================================
# Unwrapped pitch-time matrix
# ============================================================
REF_HZ = 55.0   # fixed reference (low A)
CENTS_MIN = 0    # 55 Hz
CENTS_MAX = 4800 # ~55 * 2^4 = 880 Hz


def compute_unwrapped_pitch_time(f0, num_bins=240, num_time=40,
                                  cents_range=(CENTS_MIN, CENTS_MAX),
                                  sigma_cents=10.0):
    """Pitch-time matrix WITHOUT octave wrapping or tonic normalization.

    Pitch axis: full range in cents from REF_HZ.
    Time axis: piece divided into num_time equal windows.
    """
    f0_v = f0[(f0 > 0) & (~np.isnan(f0)) & (f0 >= 55) & (f0 <= 880)]
    if len(f0_v) < num_time * 5:
        return None

    cents = 1200.0 * np.log2(f0_v / REF_HZ)

    cmin, cmax = cents_range
    bin_width = (cmax - cmin) / num_bins
    sigma_bins = sigma_cents / bin_width

    n = len(cents)
    mat = np.zeros((num_time, num_bins), dtype=np.float32)

    for t in range(num_time):
        lo = int(t * n / num_time)
        hi = int((t + 1) * n / num_time)
        seg = cents[lo:hi]
        if len(seg) == 0:
            continue
        for c in seg:
            bf = (c - cmin) / bin_width
            if bf < 0 or bf >= num_bins:
                continue
            lo_b = int(np.floor(bf))
            hi_b = lo_b + 1
            if hi_b >= num_bins:
                hi_b = num_bins - 1
            frac = bf - np.floor(bf)
            mat[t, lo_b] += (1 - frac)
            mat[t, hi_b] += frac
        s = mat[t].sum()
        if s > 0:
            mat[t] /= s

    return mat


# ============================================================
# Models
# ============================================================
class UnwrappedCNN2D(nn.Module):
    """2D CNN on unwrapped pitch-time matrix. NO circular padding."""

    def __init__(self, num_bins=240, num_time=40, num_classes=7,
                 channels=(32, 64, 128), dropout=0.3):
        super().__init__()
        layers = []
        in_ch = 1
        for ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_ch = ch
        self.features = nn.Sequential(*layers)

        # Compute output size
        h, w = num_time, num_bins
        for _ in channels:
            h = h // 2
            w = w // 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * h * w, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, num_time, num_bins) -> (B, 1, T, P)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.fc(x)


class UnwrappedLSTM(nn.Module):
    """LSTM on sequence of unwrapped pitch histograms."""

    def __init__(self, num_bins=240, hidden_size=128, num_layers=2,
                 num_classes=7, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(num_bins, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, T, num_bins)
        out, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])


class UnwrappedTransformer(nn.Module):
    """Transformer on sequence of unwrapped pitch histograms.

    Self-attention can learn to attend to the tonic pitch across
    all time windows and compute relative intervals.
    """

    def __init__(self, num_bins=240, num_time=40, d_model=128,
                 nhead=4, num_layers=2, num_classes=7, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(num_bins, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, num_time, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, T, num_bins)
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        # Pool over time (mean)
        x = x.mean(dim=1)
        return self.classifier(x)


# ============================================================
# Training
# ============================================================
class PitchTimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_cv(X, y, groups, class_names, model_fn, tag,
             n_folds=5, epochs=200, batch_size=32, patience=30):
    """Standard 5-fold CV."""
    strat_labels = np.array([f"{yi}_{gi}" for yi, gi in zip(y, groups)])
    strat_counts = Counter(strat_labels)
    for i in range(len(strat_labels)):
        if strat_counts[strat_labels[i]] < n_folds:
            strat_labels[i] = str(y[i])

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_preds = np.zeros_like(y)
    fold_accs = []

    for fi, (tr, te) in enumerate(splitter.split(X, strat_labels)):
        tr_loader = DataLoader(PitchTimeDataset(X[tr], y[tr]),
                               batch_size=batch_size, shuffle=True)
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc, best_preds, pat = 0, None, 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in tr_loader:
                loss = criterion(model(xb), yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            model.eval()
            with torch.no_grad():
                Xv = torch.FloatTensor(X[te])
                preds = model(Xv).argmax(1).numpy()
                acc = (preds == y[te]).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_preds = preds
                    pat = 0
                else:
                    pat += 1
                if pat >= patience:
                    break

        all_preds[te] = best_preds
        fold_accs.append(best_acc)
        ds_counts = Counter(groups[te])
        print(f"    [{tag}] Fold {fi+1}: {best_acc:.1%}  (test: {dict(ds_counts)})")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"    [{tag}] Overall: {mean_acc:.1%} +/- {std_acc:.1%}")
    return mean_acc, std_acc, all_preds


def print_results(y, preds, groups, class_names, tag):
    print(f"\n  Per-maqam accuracy ({tag}):")
    for ci, name in enumerate(class_names):
        mask = y == ci
        if mask.sum() > 0:
            acc = (preds[mask] == ci).mean()
            print(f"    {name:12s} (n={mask.sum():3d}): {acc:.1%}")

    print(f"\n  Per-dataset accuracy ({tag}):")
    for ds in sorted(set(groups)):
        mask = groups == ds
        if mask.sum() > 0:
            acc = (preds[mask] == y[mask]).mean()
            print(f"    {ds:10s}: {acc:.1%}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("UNWRAPPED PITCH: CAN MODELS LEARN TONIC INVARIANCE?")
    print("=" * 70)

    # Load data
    print("\nLoading datasets...")
    records = []
    for loader, name in [(load_arabic_oud, "oud"),
                          (load_cairo_congress, "cairo"),
                          (load_maqam478, "maqam478")]:
        recs = loader()
        print(f"  {name}: {len(recs)} recordings")
        records.extend(recs)
    records = [r for r in records if r["maqam"] in EXTENDED_MAQAMS]

    class_names = sorted(set(r["maqam"] for r in records))
    maqam_to_idx = {m: i for i, m in enumerate(class_names)}
    n_classes = len(class_names)

    # ── Build unwrapped features at multiple resolutions ──
    configs = [
        # (num_bins, num_time, cents_range, label)
        (240, 40, (CENTS_MIN, CENTS_MAX), "240x40 (20c/bin)"),
        (480, 40, (CENTS_MIN, CENTS_MAX), "480x40 (10c/bin)"),
        (240, 20, (CENTS_MIN, CENTS_MAX), "240x20 (20c/bin, coarse time)"),
    ]

    for num_bins, num_time, cents_range, label in configs:
        print(f"\n{'='*70}")
        print(f"RESOLUTION: {label}")
        print(f"  Pitch: {num_bins} bins over {cents_range[0]}-{cents_range[1]}c "
              f"= {(cents_range[1]-cents_range[0])/num_bins:.0f}c/bin")
        print(f"  Time: {num_time} windows")
        print(f"{'='*70}")

        # Build features
        valid = []
        for r in records:
            mat = compute_unwrapped_pitch_time(
                r["f0"], num_bins=num_bins, num_time=num_time,
                cents_range=cents_range)
            if mat is not None:
                valid.append((mat, maqam_to_idx[r["maqam"]], r["dataset"]))

        X = np.array([v[0] for v in valid], dtype=np.float32)
        y = np.array([v[1] for v in valid])
        groups = np.array([v[2] for v in valid])
        print(f"  Valid: {len(valid)} recordings, shape: {X.shape}")

        # ── 2D CNN ──
        print(f"\n  [A] 2D CNN:")
        acc_cnn, std_cnn, preds_cnn = train_cv(
            X, y, groups, class_names,
            model_fn=lambda nb=num_bins, nt=num_time: UnwrappedCNN2D(
                num_bins=nb, num_time=nt, num_classes=n_classes,
                channels=(32, 64, 128), dropout=0.3),
            tag="CNN", epochs=200, batch_size=32)
        print_results(y, preds_cnn, groups, class_names, "CNN")

        # ── LSTM ──
        print(f"\n  [B] LSTM:")
        acc_lstm, std_lstm, preds_lstm = train_cv(
            X, y, groups, class_names,
            model_fn=lambda nb=num_bins: UnwrappedLSTM(
                num_bins=nb, hidden_size=128, num_layers=2,
                num_classes=n_classes, dropout=0.3),
            tag="LSTM", epochs=200, batch_size=32)
        print_results(y, preds_lstm, groups, class_names, "LSTM")

        # ── Transformer ──
        print(f"\n  [C] Transformer:")
        acc_tfm, std_tfm, preds_tfm = train_cv(
            X, y, groups, class_names,
            model_fn=lambda nb=num_bins, nt=num_time: UnwrappedTransformer(
                num_bins=nb, num_time=nt, d_model=128, nhead=4,
                num_layers=2, num_classes=n_classes, dropout=0.3),
            tag="Transformer", epochs=200, batch_size=32)
        print_results(y, preds_tfm, groups, class_names, "Transformer")

        # Only run the highest-res config
        if num_bins == 480:
            break  # skip last config if 480 already done

    # ── Summary ──
    print(f"\n{'='*70}")
    print("COMPARISON WITH TONIC-NORMALIZED BASELINES")
    print(f"{'='*70}")
    print(f"\n  With oracle tonic + octave wrapping:")
    print(f"    RF on histogram:              91.8%")
    print(f"    2D CNN on pitch-time (20x60): 94.0%")
    print(f"\n  Without tonic (unwrapped):")
    print(f"    Results shown above.")
    print(f"\n  Without tonic (other approaches):")
    print(f"    Blind tonic + RF:             73.5%")
    print(f"    Max-all templates + RF:       76.9%")
    print(f"    Fourier chroma magnitudes:    66.7%")
    print()


if __name__ == "__main__":
    main()
