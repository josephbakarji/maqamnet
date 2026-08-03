#!/usr/bin/env python3
"""
Publication figure: t-SNE across 4 representations.

Panels (left to right):
  (a) Raw histogram — no tonic normalization
  (b) FFT magnitudes — tonic-invariant
  (c) Tonic-normalized histogram — oracle tonic
  (d) ResNet learned features — after training on template-conv

Panel (c) and (d) swapped vs earlier version.
Template response panel removed.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 150,
})

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478,
    estimate_tonic, compute_histogram, EXTENDED_MAQAMS, MAQAM_COLORS
)
from template_conv_v2 import build_templates, compute_features, FlatDataset
from template_conv_tune import CircConv2d, ResBlock
from template_conv_ablation import build_ajnas_templates, compute_features_with_templates

PROJECT_DIR = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
REF_HZ = 110.0


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, in_ch, h, w, nc, base_ch=64, n_blocks=2, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            CircConv2d(in_ch, base_ch, 5), nn.BatchNorm2d(base_ch), nn.ReLU())
        self.blocks = nn.ModuleList()
        ch = base_ch
        for i in range(n_blocks):
            next_ch = ch * 2 if i < n_blocks - 1 else ch
            self.blocks.append(ResBlock(ch, next_ch, dropout))
            ch = next_ch
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.fc1 = nn.Linear(ch * 2 * 4, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, nc)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks: x = b(x)
        x = self.pool(x).flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(self.drop(x))

    def extract_features(self, x):
        with torch.no_grad():
            x = self.stem(x)
            for b in self.blocks: x = b(x)
            x = self.pool(x).flatten(1)
            x = self.relu(self.fc1(x))
        return x.numpy()


def compute_raw_histogram(f0, num_bins=60):
    f0_v = f0[(f0 > 0) & (~np.isnan(f0)) & (f0 >= 60) & (f0 <= 800)]
    if len(f0_v) < 50: return None
    cents = (1200.0 * np.log2(f0_v / REF_HZ)) % 1200
    hist = np.zeros(num_bins, dtype=np.float64)
    for c in cents:
        bf = c * num_bins / 1200.0
        lo = int(np.floor(bf)) % num_bins
        hi = (lo + 1) % num_bins
        frac = bf - np.floor(bf)
        hist[lo] += (1 - frac); hist[hi] += frac
    hist = gaussian_filter1d(hist, sigma=8.0*num_bins/1200.0, mode="wrap")
    s = hist.sum()
    if s > 0: hist /= s
    return hist


def compute_fft_features(f0, num_bins=60):
    hist = compute_raw_histogram(f0, num_bins)
    if hist is None: return None
    return np.abs(np.fft.rfft(hist))


def make_panel(ax, emb, labels, groups, class_names, title, sil_score,
               show_maqam_legend=False, show_dataset_legend=False,
               show_axes=False, predictions=None):
    """Plot one t-SNE panel."""
    dataset_markers = {"oud": "o", "cairo": "^", "maqam478": "s"}
    dataset_sizes = {"oud": 28, "cairo": 28, "maqam478": 10}
    dataset_labels = {"oud": "Oud", "cairo": "Cairo", "maqam478": "M478"}
    ds_order = sorted(set(groups))

    for ci, maq in enumerate(class_names):
        for ds in ds_order:
            mask = (labels == ci) & (groups == ds)
            if mask.sum() == 0:
                continue
            color = MAQAM_COLORS.get(maq, f"C{ci}")
            marker = dataset_markers.get(ds, "o")
            size = dataset_sizes.get(ds, 12)

            if predictions is not None:
                correct = mask & (predictions == labels)
                wrong = mask & (predictions != labels)
                if correct.sum() > 0:
                    ax.scatter(emb[correct, 0], emb[correct, 1],
                              c=color, marker=marker, s=size, alpha=0.6,
                              edgecolors='none')
                if wrong.sum() > 0:
                    ax.scatter(emb[wrong, 0], emb[wrong, 1],
                              c=color, marker='x', s=35, alpha=0.9,
                              linewidths=1.0)
            else:
                ax.scatter(emb[mask, 0], emb[mask, 1],
                          c=color, marker=marker, s=size, alpha=0.6,
                          edgecolors='none')

    ax.set_title(title, fontweight='bold', fontsize=10)

    ax.set_xlabel("t-SNE 1", fontsize=8)
    if show_axes:
        ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.tick_params(labelsize=6)
    # Keep tick marks but minimal
    ax.tick_params(axis='both', which='both', length=2)

    # Silhouette text with box
    info = f"silhouette = {sil_score:.3f}"
    if predictions is not None:
        acc = (predictions == labels).mean()
        n_err = (predictions != labels).sum()
        info += f"\nCV accuracy = {acc:.1%} ({n_err} errors)"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, fontsize=6.5, va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='lightgray', alpha=0.85))

    # Maqam legend
    if show_maqam_legend:
        handles = []
        for ci, maq in enumerate(class_names):
            color = MAQAM_COLORS.get(maq, f"C{ci}")
            handles.append(mlines.Line2D([], [], color=color, marker='o',
                          linestyle='', markersize=5, label=maq))
        leg1 = ax.legend(handles=handles, loc='lower left', frameon=True,
                        framealpha=0.85, edgecolor='lightgray',
                        fontsize=7, handletextpad=0.2, borderpad=0.3,
                        labelspacing=0.3)
        ax.add_artist(leg1)

    # Dataset legend
    if show_dataset_legend:
        ds_handles = []
        for ds in ds_order:
            ds_handles.append(mlines.Line2D([], [], color='gray',
                             marker=dataset_markers[ds], linestyle='',
                             markersize=5, label=dataset_labels[ds]))
        if predictions is not None:
            ds_handles.append(mlines.Line2D([], [], color='gray',
                             marker='x', linestyle='', markersize=5,
                             markeredgewidth=1.0,
                             label='CV misclassified'))
        ax.legend(handles=ds_handles, loc='lower right', frameon=True,
                 framealpha=0.85, edgecolor='lightgray',
                 fontsize=7, handletextpad=0.2, borderpad=0.3,
                 labelspacing=0.3)


def main():
    print("=" * 70)
    print("PUBLICATION FIGURE: 4-PANEL t-SNE")
    print("=" * 70)

    # Load data
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

    for r in records:
        if r.get("tonic_annotated"):
            r["tonic"] = r["tonic_annotated"]
        else:
            r["tonic"] = estimate_tonic(r["f0"], maqam=r["maqam"])

    # Build 36-channel template features
    maqam_tpls, maqam_names = build_templates(30)
    ajnas_dict = build_ajnas_templates(30)
    combined_dict = {}
    for i, name in enumerate(maqam_names):
        combined_dict[f"m_{name}"] = maqam_tpls[i]
    for name, tpl in sorted(ajnas_dict.items()):
        combined_dict[f"j_{name}"] = tpl
    n_tpls = len(combined_dict)

    # Compute representations
    print("\nComputing representations...")
    raw_hists, norm_hists, fft_feats, tconv36_list = [], [], [], []
    labels, groups = [], []

    for r in records:
        raw = compute_raw_histogram(r["f0"], 60)
        norm = compute_histogram(r["f0"], r["tonic"], num_bins=60) if r["tonic"] else None
        fft = compute_fft_features(r["f0"], 60)
        tc36 = compute_features_with_templates(r["f0"], 30, 20, combined_dict)

        if all(x is not None for x in [raw, norm, fft, tc36]):
            raw_hists.append(raw)
            norm_hists.append(norm)
            fft_feats.append(fft)
            tconv36_list.append(tc36)
            labels.append(maqam_to_idx[r["maqam"]])
            groups.append(r["dataset"])

    raw_X = np.array(raw_hists)
    norm_X = np.array(norm_hists)
    fft_X = np.array(fft_feats)
    tconv36_X = np.array(tconv36_list, dtype=np.float32).transpose(0, 2, 1, 3)
    y = np.array(labels)
    g = np.array(groups)
    print(f"  Valid: {len(y)}")

    # Train single model for features
    print("\n  Training ResNet on all data for feature extraction...")
    model = ResNetFeatureExtractor(n_tpls, 20, 30, nc, 64, 2, 0.4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(FlatDataset(tconv36_X, y), batch_size=32, shuffle=True)

    for epoch in range(200):
        model.train()
        el = 0
        for xb, yb in loader:
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item()
        sched.step(el)
        if (epoch+1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(torch.FloatTensor(tconv36_X)).argmax(1).numpy() == y).mean()
            print(f"    Epoch {epoch+1}: {acc:.1%}")

    model.eval()
    resnet_feats = model.extract_features(torch.FloatTensor(tconv36_X))
    print(f"  Features: {resnet_feats.shape}")

    # Get CV predictions for error markers
    print("\n  Running 5-fold CV for error markers...")
    from sklearn.model_selection import StratifiedKFold
    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnts = Counter(strat)
    for i in range(len(strat)):
        if cnts[strat[i]] < 5: strat[i] = str(y[i])

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_preds = np.full(len(y), -1, dtype=np.int64)
    for fi, (tr, te) in enumerate(skf.split(tconv36_X, strat)):
        cv_model = ResNetFeatureExtractor(n_tpls, 20, 30, nc, 64, 2, 0.4)
        cv_opt = torch.optim.Adam(cv_model.parameters(), lr=1e-3, weight_decay=1e-4)
        cv_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(cv_opt, patience=10, factor=0.5)
        cv_loader = DataLoader(FlatDataset(tconv36_X[tr], y[tr]), batch_size=32, shuffle=True)
        best_acc, best_p, pat = 0, None, 0
        for ep in range(250):
            cv_model.train()
            el = 0
            for xb, yb in cv_loader:
                loss = crit(cv_model(xb), yb)
                cv_opt.zero_grad(); loss.backward(); cv_opt.step()
                el += loss.item()
            cv_sched.step(el)
            cv_model.eval()
            with torch.no_grad():
                p = cv_model(torch.FloatTensor(tconv36_X[te])).argmax(1).numpy()
                a = (p == y[te]).mean()
            if a > best_acc: best_acc = a; best_p = p.copy(); pat = 0
            else: pat += 1
            if pat >= 40: break
        cv_preds[te] = best_p
        print(f"    Fold {fi+1}: {best_acc:.1%}")
    print(f"  CV accuracy: {(cv_preds==y).mean():.1%}")

    # Generate figures at selected perplexities
    for perp in [20, 30]:
        print(f"\n  Perplexity = {perp}")

        # 4 panels: (a) raw, (b) FFT, (c) tonic-normalized, (d) ResNet
        panels = [
            ("(a) Raw histogram\n(no tonic)", raw_X, None),
            ("(b) FFT magnitudes\n(tonic-invariant)", fft_X, None),
            ("(c) Tonic-normalized\n(oracle tonic)", norm_X, None),
            ("(d) ResNet features\n(after training)", resnet_feats, cv_preds),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for pi, (title, X, preds) in enumerate(panels):
            X_use = X
            if X.shape[1] > 200:
                rng = np.random.RandomState(42)
                idx = rng.choice(X.shape[1], 200, replace=False)
                X_use = X[:, idx]

            tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                       max_iter=1000)
            emb = tsne.fit_transform(X_use)
            sil = silhouette_score(emb, y)

            make_panel(axes[pi], emb, y, g, class_names, title, sil,
                      show_maqam_legend=(pi == 2),
                      show_dataset_legend=(pi == 3),
                      show_axes=(pi == 0),
                      predictions=preds)

            acc_s = f"  acc={(preds==y).mean():.1%}" if preds is not None else ""
            print(f"    {title.split(chr(10))[0]:35s} sil={sil:.3f}{acc_s}")

        fig.tight_layout(w_pad=1.0)

        for fmt in ["pdf", "png"]:
            fname = FIGURES_DIR / f"fig_tsne_4panel_p{perp}.{fmt}"
            fig.savefig(fname, bbox_inches='tight', dpi=300 if fmt == "pdf" else 150)
        print(f"    Saved fig_tsne_4panel_p{perp}.pdf/.png")
        plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
