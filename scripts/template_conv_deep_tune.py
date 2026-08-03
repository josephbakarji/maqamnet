#!/usr/bin/env python3
"""
Combined template-conv 36-channel deep tuning + full ResNet interpretability.

1. Tune 36-channel (maqam+ajnas) ResNet: depth (2-4 blocks), width (32-64),
   dropout, with focus on oud/cairo accuracy.
2. Full ResNet flow: Grad-CAM + saliency at EVERY block, t-SNE flow,
   class separability, maqamic peak analysis (1-4-5).
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter, defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cosine as cosine_dist
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478,
    estimate_tonic, _TONIC_TEMPLATES, EXTENDED_MAQAMS, MAQAM_COLORS,
    load_diarmaqar
)
from template_conv_v2 import build_templates, compute_features, FlatDataset
from template_conv_ablation import build_ajnas_templates, compute_features_with_templates
from template_conv_tune import CircConv2d, ResBlock

PROJECT_DIR = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
REF_HZ = 110.0


# ============================================================
# ResNet with full instrumentation (hooks at every block)
# ============================================================
class InstrumentedResNet(nn.Module):
    """ResNet with Grad-CAM hooks and activation storage at every stage."""

    def __init__(self, in_ch, h, w, nc, base_ch=32, n_blocks=2,
                 dropout=0.4):
        super().__init__()
        self.n_blocks = n_blocks

        self.stem = nn.Sequential(
            CircConv2d(in_ch, base_ch, 5),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        ch = base_ch
        for i in range(n_blocks):
            next_ch = ch * 2 if i < n_blocks - 1 else ch
            self.blocks.append(ResBlock(ch, next_ch, dropout))
            ch = next_ch

        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 2 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, nc),
        )
        self._final_ch = ch
        self._activations = {}
        self._gradients = {}

    def forward(self, x, store=False):
        if store:
            self._activations["input"] = x.detach().clone()

        x = self.stem(x)
        if store:
            self._activations["stem"] = x.detach().clone()

        for i, block in enumerate(self.blocks):
            x = block(x)
            if store:
                self._activations[f"block_{i}"] = x.detach().clone()

        pooled = self.pool(x)
        if store:
            self._activations["pooled"] = pooled.detach().clone()

        return self.fc(pooled)

    def grad_cam_at_block(self, x, block_idx, target_class=None):
        """Grad-CAM on a specific block's output."""
        # Register hooks on the target block
        target = self.blocks[block_idx] if block_idx >= 0 else self.stem
        acts = {}
        grads = {}

        def save_act(m, inp, out):
            acts["val"] = out

        def save_grad(m, gi, go):
            grads["val"] = go[0]

        h1 = target.register_forward_hook(save_act)
        h2 = target.register_full_backward_hook(save_grad)

        self.eval()
        x = x.clone().requires_grad_(True)
        out = self.forward(x)
        if target_class is None:
            target_class = out.argmax(1).item()

        self.zero_grad()
        out[0, target_class].backward()

        a = acts["val"]   # (1, C, H, W)
        g = grads["val"]  # (1, C, H, W)
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = (w * a).sum(1, keepdim=True)
        cam = F.relu(cam).squeeze().detach().numpy()

        h1.remove(); h2.remove()
        return cam

    def input_saliency(self, x, target_class=None):
        """Gradient * input saliency on the input tensor."""
        self.eval()
        x = x.clone().requires_grad_(True)
        out = self.forward(x)
        if target_class is None:
            target_class = out.argmax(1).item()
        self.zero_grad()
        out[0, target_class].backward()
        sal = (x.grad * x).detach().squeeze(0).numpy()
        return sal  # (in_ch, T, bins)


# ============================================================
# Build combined (maqam + ajnas) templates
# ============================================================
def build_combined_templates(num_bins=30):
    """Build maqam + ajnas templates, return (dict, sorted_names)."""
    maqam_templates, maqam_names = build_templates(num_bins)
    ajnas_templates = build_ajnas_templates(num_bins)

    combined = {}
    for i, name in enumerate(maqam_names):
        combined[f"m_{name}"] = maqam_templates[i]
    for name, tpl in sorted(ajnas_templates.items()):
        combined[f"j_{name}"] = tpl

    names = sorted(combined.keys())
    tpls = [combined[n] for n in names]
    return tpls, names


# ============================================================
# Training with per-dataset + per-maqam reporting
# ============================================================
def train_cv_detailed(X, y, g, maqam_names_list, model_fn, tag,
                      epochs=250, patience=40, bs=32, lr=1e-3,
                      weight_decay=1e-4):
    """Train with 5-fold CV, return detailed per-dataset/per-maqam accuracy."""
    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnts = Counter(strat)
    for i in range(len(strat)):
        if cnts[strat[i]] < 5:
            strat[i] = str(y[i])

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(y)
    best_models = []

    for fi, (tr, te) in enumerate(skf.split(X, strat)):
        model = model_fn()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(FlatDataset(X[tr], y[tr]), batch_size=bs, shuffle=True)

        ba, bp, pat, best_state = 0, None, 0, None
        for ep in range(epochs):
            model.train()
            el = 0
            for xb, yb in loader:
                loss = crit(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item()
            sched.step(el)

            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X[te]))
                p = logits.argmax(1).numpy()
                acc = (p == y[te]).mean()
                if acc > ba:
                    ba = acc; bp = p; pat = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    pat += 1
                if pat >= patience:
                    break

        preds[te] = bp
        best_models.append(best_state)

    # Overall accuracy
    acc = (preds == y).mean()
    ds = {d: (preds[g == d] == y[g == d]).mean() for d in sorted(set(g))}

    # Per-maqam accuracy
    per_maqam = {}
    nc = len(maqam_names_list)
    for ci, maq in enumerate(maqam_names_list):
        mask = y == ci
        if mask.sum() > 0:
            per_maqam[maq] = (preds[mask] == ci).mean()

    # Print summary
    oud_acc = ds.get('oud', 0)
    cairo_acc = ds.get('cairo', 0)
    m478_acc = ds.get('maqam478', 0)
    # Balanced = mean of oud and cairo (what we're trying to improve)
    oud_cairo_mean = (oud_acc + cairo_acc) / 2

    print(f"  {tag:50s} {acc:.1%}  oud={oud_acc:.1%}  cairo={cairo_acc:.1%}  "
          f"m478={m478_acc:.1%}  oud+cairo={oud_cairo_mean:.1%}")

    return acc, preds, ds, per_maqam, best_models


# ============================================================
# Maqamic peak analysis: check if Grad-CAM peaks at 1-4-5
# ============================================================
def analyze_maqamic_peaks(model, X, y, g, class_names, tpl_names,
                          num_bins=30, num_time=20, tag=""):
    """For each maqam: compute mean saliency, check if peaks at 1-4-5 degrees."""
    model.eval()

    # Tonic positions per maqam from DiArMaqAr
    jins_templates, note_cents, _ = load_diarmaqar()

    # Known scale degree positions (in fraction of octave)
    # 1st = 0 (tonic), 4th ≈ 498c (just fourth), 5th ≈ 702c (just fifth)
    degree_positions = {
        "1st (tonic)": 0,
        "4th": 498 * num_bins / 1200,
        "5th": 702 * num_bins / 1200,
    }

    print(f"\n  === Maqamic Peak Analysis ({tag}) ===")

    # Compute input saliency for each sample
    saliency_by_class = defaultdict(list)
    for i in range(len(X)):
        x = torch.FloatTensor(X[i]).unsqueeze(0)
        sal = model.input_saliency(x, target_class=int(y[i]))
        saliency_by_class[int(y[i])].append(sal)

    # For each maqam: mean saliency averaged over templates and time → pitch profile
    results = {}
    for ci, maq in enumerate(class_names):
        sals = saliency_by_class.get(ci)
        if not sals:
            continue
        mean_sal = np.mean(sals, axis=0)  # (n_ch, T, bins)
        # Average over time and channels → pitch profile
        pitch_profile = np.abs(mean_sal).mean(axis=(0, 1))  # (bins,)
        # Normalize
        if pitch_profile.max() > 0:
            pitch_profile /= pitch_profile.max()

        # Find top peaks
        from scipy.signal import find_peaks
        # Use circular extension for peak finding
        extended = np.tile(pitch_profile, 3)
        peaks, props = find_peaks(extended, height=0.3, distance=2)
        peaks = peaks[(peaks >= num_bins) & (peaks < 2 * num_bins)] - num_bins
        peak_cents = peaks * 1200 / num_bins

        # Check if peaks align with 1-4-5
        near_1 = any(abs(p) < 1.5 or abs(p - num_bins) < 1.5 for p in peaks)
        near_4 = any(abs(p - degree_positions["4th"]) < 1.5 for p in peaks)
        near_5 = any(abs(p - degree_positions["5th"]) < 1.5 for p in peaks)

        print(f"    {maq:12s}: peaks at {peak_cents} cents  "
              f"1st={'Y' if near_1 else 'n'}  "
              f"4th={'Y' if near_4 else 'n'}  "
              f"5th={'Y' if near_5 else 'n'}")

        results[maq] = {
            "profile": pitch_profile,
            "peaks": peak_cents,
            "near_145": (near_1, near_4, near_5),
            "mean_saliency": mean_sal,
        }

    return results


# ============================================================
# Full ResNet flow visualization
# ============================================================
def visualize_resnet_flow(model, X, y, g, class_names, tpl_names,
                          num_bins=30, tag=""):
    """Comprehensive ResNet flow: activations, separability, t-SNE at every block."""
    model.eval()
    n_blocks = model.n_blocks

    # Collect activations at every stage
    with torch.no_grad():
        model(torch.FloatTensor(X), store=True)

    stage_keys = ["input", "stem"] + [f"block_{i}" for i in range(n_blocks)]
    stage_labels = ["Input\n(templates×time×shift)"] + ["Stem"] + \
                   [f"ResBlock {i}" for i in range(n_blocks)]

    # 1. Class separability at each stage
    print(f"\n  === Class Separability ({tag}) ===")
    separabilities = []
    for key, label in zip(stage_keys, stage_labels):
        act = model._activations[key].numpy()  # (N, C, H, W)
        flat = act.reshape(len(act), -1)  # (N, D)

        # Inter-class variance / intra-class variance
        global_mean = flat.mean(axis=0)
        inter, intra = 0, 0
        nc = len(class_names)
        for ci in range(nc):
            mask = y == ci
            if mask.sum() == 0:
                continue
            class_mean = flat[mask].mean(axis=0)
            inter += mask.sum() * np.sum((class_mean - global_mean) ** 2)
            intra += np.sum((flat[mask] - class_mean) ** 2)
        inter /= len(y)
        intra /= len(y)
        ratio = inter / (intra + 1e-8)
        separabilities.append((label.replace('\n', ' '), inter, intra, ratio))
        print(f"    {label.replace(chr(10),' '):25s}: inter={inter:.1f}  "
              f"intra={intra:.1f}  ratio={ratio:.3f}")

    # 2. t-SNE at each stage
    n_stages = len(stage_keys)
    fig_tsne, axes_tsne = plt.subplots(1, n_stages, figsize=(5 * n_stages, 5))
    if n_stages == 1:
        axes_tsne = [axes_tsne]

    for si, (key, label) in enumerate(zip(stage_keys, stage_labels)):
        act = model._activations[key].numpy()
        flat = act.reshape(len(act), -1)

        # Subsample if too many features
        if flat.shape[1] > 500:
            rng = np.random.RandomState(42)
            idx = rng.choice(flat.shape[1], 500, replace=False)
            flat = flat[:, idx]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb = tsne.fit_transform(flat)

        ax = axes_tsne[si]
        for ci, maq in enumerate(class_names):
            mask = y == ci
            color = MAQAM_COLORS.get(maq, f"C{ci}")
            ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=maq,
                      alpha=0.5, s=15, edgecolors='none')
        ax.set_title(f"{label}\nratio={separabilities[si][3]:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if si == 0:
            ax.legend(fontsize=7, loc='best')

    fig_tsne.suptitle(f"ResNet Flow: t-SNE at Each Stage ({tag})", fontsize=13, y=1.02)
    fig_tsne.tight_layout()
    fname = FIGURES_DIR / f"fig_resnet_flow_{tag.replace(' ', '_').lower()}.png"
    fig_tsne.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved {fname.name}")

    # 3. Separability bar chart
    fig_sep, ax_sep = plt.subplots(figsize=(8, 4))
    labels_s = [s[0] for s in separabilities]
    ratios = [s[3] for s in separabilities]
    bars = ax_sep.bar(range(len(ratios)), ratios,
                      color=['#3498db'] + ['#2ecc71'] + ['#e74c3c'] * n_blocks)
    ax_sep.set_xticks(range(len(ratios)))
    ax_sep.set_xticklabels(labels_s, fontsize=9)
    ax_sep.set_ylabel("Inter/Intra Class Variance Ratio", fontsize=11)
    ax_sep.set_title(f"Class Separability Through ResNet ({tag})", fontsize=12)
    for b, r in zip(bars, ratios):
        ax_sep.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                   f"{r:.3f}", ha='center', fontsize=9)
    fig_sep.tight_layout()
    fname = FIGURES_DIR / f"fig_separability_{tag.replace(' ', '_').lower()}.png"
    fig_sep.savefig(fname, dpi=150)
    print(f"  Saved {fname.name}")

    return separabilities


# ============================================================
# Grad-CAM flow: per-maqam at every block
# ============================================================
def gradcam_flow(model, X, y, class_names, num_bins=30, tag=""):
    """Grad-CAM at each ResNet block for each maqam class."""
    model.eval()
    n_blocks = model.n_blocks
    nc = len(class_names)

    # For each block, compute mean Grad-CAM per class
    # Grad-CAM gives (H, W) at that block's spatial resolution
    # We'll focus on the shift (W) axis to look for maqamic peaks

    fig, axes = plt.subplots(nc, n_blocks + 1, figsize=(4 * (n_blocks + 1), 2.5 * nc))

    block_labels = ["Stem"] + [f"Block {i}" for i in range(n_blocks)]

    for bi in range(n_blocks + 1):
        block_idx = bi - 1  # -1 = stem, 0 = block0, etc.

        for ci, maq in enumerate(class_names):
            mask = y == ci
            indices = np.where(mask)[0]
            # Sample up to 30 per class for speed
            if len(indices) > 30:
                indices = np.random.RandomState(42).choice(indices, 30, replace=False)

            cams = []
            for idx in indices:
                x = torch.FloatTensor(X[idx]).unsqueeze(0)
                try:
                    cam = model.grad_cam_at_block(x, block_idx, target_class=ci)
                    if cam.ndim == 2:
                        cams.append(cam)
                except:
                    pass

            if not cams:
                continue

            # Average and normalize
            mean_cam = np.mean(cams, axis=0)
            if mean_cam.max() > 0:
                mean_cam /= mean_cam.max()

            ax = axes[ci, bi]
            ax.imshow(mean_cam, aspect='auto', cmap='hot', origin='lower',
                     vmin=0, vmax=1)
            if ci == 0:
                ax.set_title(block_labels[bi], fontsize=10, fontweight='bold')
            if bi == 0:
                ax.set_ylabel(maq, fontsize=10, fontweight='bold',
                            color=MAQAM_COLORS.get(maq, 'black'))
            ax.tick_params(labelsize=6)
            if ci < nc - 1:
                ax.set_xticklabels([])

    fig.suptitle(f"Grad-CAM Flow Through ResNet ({tag})", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = FIGURES_DIR / f"fig_gradcam_flow_{tag.replace(' ', '_').lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved {fname.name}")


# ============================================================
# Saliency pitch profiles with maqamic peak overlay
# ============================================================
def saliency_with_145(model, X, y, class_names, num_bins=30, tag=""):
    """Input saliency averaged to pitch profile, with 1-4-5 degree markers."""
    model.eval()
    nc = len(class_names)

    # 1-4-5 positions in bins
    tonic_bin = 0
    fourth_bin = int(round(498 * num_bins / 1200))
    fifth_bin = int(round(702 * num_bins / 1200))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    bin_cents = np.arange(num_bins) * 1200 / num_bins

    for ci, maq in enumerate(class_names):
        if ci >= 8:
            break
        mask = y == ci
        indices = np.where(mask)[0]
        if len(indices) > 40:
            indices = np.random.RandomState(42).choice(indices, 40, replace=False)

        sals = []
        for idx in indices:
            x = torch.FloatTensor(X[idx]).unsqueeze(0)
            sal = model.input_saliency(x, target_class=ci)
            sals.append(sal)

        if not sals:
            continue

        mean_sal = np.mean(sals, axis=0)  # (n_ch, T, bins)
        # Average over channels and time → pitch profile
        pitch_profile = np.abs(mean_sal).mean(axis=(0, 1))
        if pitch_profile.max() > 0:
            pitch_profile /= pitch_profile.max()

        ax = axes[ci]
        color = MAQAM_COLORS.get(maq, f"C{ci}")
        ax.plot(bin_cents, pitch_profile, color=color, linewidth=2, label='Saliency')
        ax.fill_between(bin_cents, pitch_profile, alpha=0.2, color=color)

        # Mark 1-4-5 degrees
        for deg_name, deg_bin, ls in [("1st", tonic_bin, '-'),
                                       ("4th", fourth_bin, '--'),
                                       ("5th", fifth_bin, ':')]:
            deg_cents = deg_bin * 1200 / num_bins
            ax.axvline(deg_cents, color='gray', linestyle=ls, alpha=0.7, linewidth=1.5)
            ax.text(deg_cents + 10, 0.95, deg_name, fontsize=7, color='gray',
                   va='top', ha='left')

        ax.set_title(maq, fontsize=11, fontweight='bold', color=color)
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Shift (cents)" if ci >= 4 else "", fontsize=9)
        ax.set_ylabel("Norm. saliency" if ci % 4 == 0 else "", fontsize=9)
        ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
        ax.tick_params(labelsize=7)

    # Hide extra axes
    for i in range(nc, 8):
        axes[i].set_visible(False)

    fig.suptitle(f"Input Saliency Pitch Profiles with 1-4-5 Degree Markers ({tag})",
                fontsize=13)
    fig.tight_layout()
    fname = FIGURES_DIR / f"fig_saliency_145_{tag.replace(' ', '_').lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved {fname.name}")


# ============================================================
# Per-dataset saliency: compare oud vs cairo vs maqam478
# ============================================================
def saliency_per_dataset(model, X, y, g, class_names, num_bins=30, tag=""):
    """Show saliency broken down by dataset to understand oud/cairo failures."""
    model.eval()
    nc = len(class_names)
    datasets = sorted(set(g))

    fig, axes = plt.subplots(nc, len(datasets), figsize=(5 * len(datasets), 2.5 * nc))

    bin_cents = np.arange(num_bins) * 1200 / num_bins
    fourth_cents = 498
    fifth_cents = 702

    for di, ds in enumerate(datasets):
        for ci, maq in enumerate(class_names):
            mask = (y == ci) & (g == ds)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                axes[ci, di].set_visible(False)
                continue
            if len(indices) > 30:
                indices = np.random.RandomState(42).choice(indices, 30, replace=False)

            # Separate correct vs incorrect predictions
            with torch.no_grad():
                logits = model(torch.FloatTensor(X[indices]))
                pred = logits.argmax(1).numpy()

            correct = pred == ci
            n_correct = correct.sum()
            n_total = len(indices)

            sals = []
            for idx in indices:
                x = torch.FloatTensor(X[idx]).unsqueeze(0)
                sal = model.input_saliency(x, target_class=ci)
                sals.append(np.abs(sal).mean(axis=(0, 1)))

            mean_profile = np.mean(sals, axis=0)
            if mean_profile.max() > 0:
                mean_profile /= mean_profile.max()

            ax = axes[ci, di]
            color = MAQAM_COLORS.get(maq, f"C{ci}")
            ax.plot(bin_cents, mean_profile, color=color, linewidth=2)
            ax.fill_between(bin_cents, mean_profile, alpha=0.15, color=color)

            # 1-4-5 markers
            for dc, ls in [(0, '-'), (fourth_cents, '--'), (fifth_cents, ':')]:
                ax.axvline(dc, color='gray', linestyle=ls, alpha=0.5, linewidth=1)

            ax.set_xlim(0, 1200)
            ax.set_ylim(0, 1.1)
            ax.text(0.98, 0.95, f"{n_correct}/{n_total}",
                   transform=ax.transAxes, fontsize=8, ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

            if ci == 0:
                ax.set_title(ds, fontsize=11, fontweight='bold')
            if di == 0:
                ax.set_ylabel(maq, fontsize=10, color=color, fontweight='bold')
            ax.tick_params(labelsize=6)

    fig.suptitle(f"Saliency per Dataset per Maqam ({tag})\n"
                f"Gray lines: tonic (solid), 4th (dashed), 5th (dotted)",
                fontsize=12)
    fig.tight_layout()
    fname = FIGURES_DIR / f"fig_saliency_dataset_{tag.replace(' ', '_').lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved {fname.name}")


# ============================================================
def main():
    print("=" * 70)
    print("36-CHANNEL DEEP TUNING + FULL RESNET INTERPRETABILITY")
    print("=" * 70)

    # ── Load data ──
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

    # ── Build combined templates ──
    num_bins = 30
    num_time = 20

    # Maqam templates (8)
    maqam_tpls, maqam_names = build_templates(num_bins)
    # Ajnas templates (28)
    ajnas_dict = build_ajnas_templates(num_bins)
    # Combined (36)
    combined_dict = {}
    for i, name in enumerate(maqam_names):
        combined_dict[f"m_{name}"] = maqam_tpls[i]
    for name, tpl in sorted(ajnas_dict.items()):
        combined_dict[f"j_{name}"] = tpl
    combined_names = sorted(combined_dict.keys())
    n_tpls = len(combined_names)

    print(f"\nBuilding 36-channel features...")
    feats, labels, groups = [], [], []
    for r in records:
        f = compute_features_with_templates(r["f0"], num_bins, num_time, combined_dict)
        if f is not None:
            feats.append(f)
            labels.append(maqam_to_idx[r["maqam"]])
            groups.append(r["dataset"])
    X = np.array(feats, dtype=np.float32).transpose(0, 2, 1, 3)  # (N, 36, T, bins)
    y = np.array(labels); g = np.array(groups)
    print(f"  Data: {X.shape}")

    # ============================================================
    # PART 1: Deep Tuning Sweep
    # ============================================================
    print(f"\n{'='*70}")
    print("PART 1: DEPTH / WIDTH / REGULARIZATION SWEEP")
    print(f"{'='*70}")
    print(f"\n  {'Config':50s} {'Overall':>7s}  {'oud':>7s}  {'cairo':>7s}  "
          f"{'m478':>7s}  {'oud+cairo':>10s}")
    print(f"  {'-'*50} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}")

    best_acc = 0
    best_config = None
    best_preds = None
    best_models = None

    configs = [
        # (base_ch, n_blocks, dropout, tag)
        (32, 2, 0.4, "ch32 b2 d0.4 (baseline)"),
        (32, 3, 0.4, "ch32 b3 d0.4"),
        (32, 4, 0.4, "ch32 b4 d0.4"),
        (48, 2, 0.4, "ch48 b2 d0.4"),
        (48, 3, 0.4, "ch48 b3 d0.4"),
        (64, 2, 0.4, "ch64 b2 d0.4"),
        (64, 3, 0.4, "ch64 b3 d0.4"),
        (32, 3, 0.3, "ch32 b3 d0.3"),
        (32, 3, 0.5, "ch32 b3 d0.5"),
        (48, 3, 0.3, "ch48 b3 d0.3"),
        (48, 3, 0.5, "ch48 b3 d0.5"),
    ]

    for base_ch, n_blocks, dropout, ctag in configs:
        acc, preds, ds, per_maqam, models = train_cv_detailed(
            X, y, g, class_names,
            lambda bc=base_ch, nb=n_blocks, d=dropout:
                InstrumentedResNet(n_tpls, num_time, num_bins, nc, bc, nb, d),
            f"36ch {ctag}",
        )
        oud_cairo = (ds.get('oud', 0) + ds.get('cairo', 0)) / 2
        if oud_cairo > best_acc:
            best_acc = oud_cairo
            best_config = (base_ch, n_blocks, dropout, ctag)
            best_preds = preds
            best_models = models

    print(f"\n  ** Best oud+cairo config: {best_config[3]} "
          f"(oud+cairo mean = {best_acc:.1%})")

    # Per-maqam breakdown of best
    print(f"\n  Per-maqam accuracy (best config):")
    for ci, maq in enumerate(class_names):
        mask = y == ci
        if mask.sum() > 0:
            macc = (best_preds[mask] == ci).mean()
            # Per-dataset within maqam
            ds_parts = []
            for ds in sorted(set(g)):
                dm = (y == ci) & (g == ds)
                if dm.sum() > 0:
                    ds_parts.append(f"{ds}={(best_preds[dm]==ci).mean():.0%}")
            print(f"    {maq:12s} (n={mask.sum():3d}): {macc:.1%}  "
                  f"[{', '.join(ds_parts)}]")

    # Confusion matrix
    cm = confusion_matrix(y, best_preds)
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

    # ============================================================
    # PART 2: Full Interpretability on Best Model
    # ============================================================
    print(f"\n{'='*70}")
    print("PART 2: FULL INTERPRETABILITY")
    print(f"{'='*70}")

    # Train a single model on all data for interpretability
    base_ch, n_blocks, dropout, ctag = best_config
    print(f"\n  Training full model: {ctag}")
    model = InstrumentedResNet(n_tpls, num_time, num_bins, nc,
                               base_ch, n_blocks, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(FlatDataset(X, y), batch_size=32, shuffle=True)

    for epoch in range(250):
        model.train()
        el = 0
        for xb, yb in loader:
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item()
        sched.step(el)
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(torch.FloatTensor(X)).argmax(1).numpy() == y).mean()
            print(f"    Epoch {epoch+1}: {acc:.1%}")

    # 2a. ResNet flow: t-SNE + separability at every block
    print("\n  Computing ResNet flow...")
    visualize_resnet_flow(model, X, y, g, class_names, combined_names,
                         num_bins, tag=ctag)

    # 2b. Saliency with 1-4-5 degree markers
    print("\n  Computing saliency pitch profiles...")
    saliency_with_145(model, X, y, class_names, num_bins, tag=ctag)

    # 2c. Per-dataset saliency
    print("\n  Computing per-dataset saliency...")
    saliency_per_dataset(model, X, y, g, class_names, num_bins, tag=ctag)

    # 2d. Grad-CAM at every block
    print("\n  Computing Grad-CAM flow...")
    gradcam_flow(model, X, y, class_names, num_bins, tag=ctag)

    # 2e. Maqamic peak analysis
    peak_results = analyze_maqamic_peaks(model, X, y, g, class_names,
                                         combined_names, num_bins, num_time,
                                         tag=ctag)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
