#!/usr/bin/env python3
"""
Template-conv ablation and extensions:

1. Learned conv (random init) vs template conv — does DiArMaqAr help?
2. More templates: add all 28 ajnas from DiArMaqAr
3. Ensemble: TemplateConv ResNet + Disambig Transformer
4. Interpretability: Grad-CAM on best ResNet
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478,
    estimate_tonic, compute_histogram, compute_pitch_time,
    _TONIC_TEMPLATES, EXTENDED_MAQAMS, MAQAM_COLORS,
    PitchTimeCNN2D_CAM, FeatDataset, load_diarmaqar
)
from template_conv_v2 import build_templates, compute_features
from template_conv_tune import TemplateNetDeep, CircConv2d, ResBlock

PROJECT_DIR = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
REF_HZ = 110.0


# ============================================================
# Build ajnas templates (28 templates instead of 8)
# ============================================================
def build_ajnas_templates(num_bins=30):
    """Build templates from ALL ajnas in DiArMaqAr (not just maqam scales)."""
    jins_templates, note_cents, _ = load_diarmaqar()
    sigma_cents = 10.0
    sigma_bins = sigma_cents * num_bins / 1200.0

    templates = {}
    for name, degrees in jins_templates.items():
        tpl = np.zeros(num_bins, dtype=np.float64)
        for deg in degrees:
            center = deg * num_bins / 1200.0
            for b in range(num_bins):
                dist = min(abs(b - center), num_bins - abs(b - center))
                tpl[b] += np.exp(-0.5 * (dist / sigma_bins) ** 2)
        s = tpl.sum()
        if s > 0:
            tpl /= s
        templates[name] = tpl

    return templates


def compute_features_with_templates(f0, num_bins, num_time, templates_dict):
    """Compute template response using arbitrary template dict."""
    f0_v = f0[(f0 > 0) & (~np.isnan(f0)) & (f0 >= 60) & (f0 <= 800)]
    if len(f0_v) < num_time * 10:
        return None
    cents = (1200.0 * np.log2(f0_v / REF_HZ)) % 1200
    n = len(cents)
    sigma_bins = 8.0 * num_bins / 1200.0

    tpl_list = [v for _, v in sorted(templates_dict.items())]
    nt = len(tpl_list)
    tensor = np.zeros((num_time, nt, num_bins), dtype=np.float32)

    for t in range(num_time):
        lo = int(t * n / num_time); hi = int((t + 1) * n / num_time)
        seg = cents[lo:hi]
        if len(seg) < 5: continue
        hist = np.zeros(num_bins, dtype=np.float64)
        for c in seg:
            bf = c * num_bins / 1200.0
            lo_b = int(np.floor(bf)) % num_bins; hi_b = (lo_b + 1) % num_bins
            frac = bf - np.floor(bf)
            hist[lo_b] += (1 - frac); hist[hi_b] += frac
        hist = gaussian_filter1d(hist, sigma=sigma_bins, mode="wrap")
        s = hist.sum()
        if s > 0: hist /= s

        hist_fft = np.fft.rfft(hist)
        for ti, tpl in enumerate(tpl_list):
            tpl_fft = np.fft.rfft(tpl)
            xcorr = np.real(np.fft.irfft(np.conj(hist_fft) * tpl_fft, n=num_bins))
            tensor[t, ti, :] = xcorr.astype(np.float32)

    return tensor


# ============================================================
# Learned-conv baseline (random filters instead of templates)
# ============================================================
def compute_raw_histograms(f0, num_bins=30, num_time=20):
    """Just the raw pitch-time histogram (no template conv)."""
    f0_v = f0[(f0 > 0) & (~np.isnan(f0)) & (f0 >= 60) & (f0 <= 800)]
    if len(f0_v) < num_time * 10:
        return None
    cents = (1200.0 * np.log2(f0_v / REF_HZ)) % 1200
    n = len(cents)
    sigma_bins = 8.0 * num_bins / 1200.0

    mat = np.zeros((num_time, num_bins), dtype=np.float32)
    for t in range(num_time):
        lo = int(t * n / num_time); hi = int((t + 1) * n / num_time)
        seg = cents[lo:hi]
        if len(seg) < 5: continue
        for c in seg:
            bf = c * num_bins / 1200.0
            lo_b = int(np.floor(bf)) % num_bins; hi_b = (lo_b + 1) % num_bins
            frac = bf - np.floor(bf)
            mat[t, lo_b] += (1 - frac); mat[t, hi_b] += frac
        mat[t] = gaussian_filter1d(mat[t], sigma=sigma_bins, mode="wrap")
        s = mat[t].sum()
        if s > 0: mat[t] /= s
    return mat


class LearnedConvResNet(nn.Module):
    """Same ResNet architecture but with LEARNED first-layer convolution
    instead of pre-computed template cross-correlation."""
    def __init__(self, num_bins=30, num_time=20, n_classes=7,
                 n_filters=8, base_ch=32, n_blocks=2, dropout=0.4):
        super().__init__()
        # First layer: learned circular conv (replaces template xcorr)
        self.conv0_pad = num_bins // 2
        self.conv0 = nn.Conv2d(1, n_filters, (3, num_bins),
                               padding=(1, 0))  # full-width on pitch axis

        self.stem = nn.Sequential(
            CircConv2d(n_filters, base_ch, 5),
            nn.BatchNorm2d(base_ch), nn.ReLU())

        blocks = []
        ch = base_ch
        for i in range(n_blocks):
            next_ch = ch * 2 if i < n_blocks - 1 else ch
            blocks.append(ResBlock(ch, next_ch, dropout))
            ch = next_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(ch * 2 * 4, 128),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        # x: (B, 1, T, bins)
        # Circular pad on pitch axis before first conv
        x = F.pad(x, (self.conv0_pad, self.conv0_pad, 0, 0), mode='circular')
        x = F.relu(self.conv0(x))  # (B, n_filters, T, 1) — squeezes pitch dim
        # Wait, this collapses the pitch dim. We need a different approach.
        # Actually for fair comparison: the template-conv has (B, 8, T, 30) input.
        # The learned version should start from (B, 1, T, 30) raw histogram
        # and learn 8 circular-conv filters of width ~30 to match.
        # Let me use circular conv1d per time step instead.
        return self.fc(self.pool(self.blocks(self.stem(x))))


class LearnedConvResNetV2(nn.Module):
    """Fair comparison: raw histogram → learned circular convolutions.
    Input: (B, 1, T, bins) raw pitch-time histogram.
    Applies circular conv on the pitch axis to learn filter responses,
    then same ResNet architecture."""
    def __init__(self, num_bins=30, num_time=20, n_classes=7,
                 n_filters=8, base_ch=32, n_blocks=2, dropout=0.4):
        super().__init__()
        # Learned circular convolutions on pitch axis (like template xcorr)
        self.circ_filters = nn.Sequential(
            CircConv2d(1, n_filters, 5),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        )
        # Same ResNet as template-conv
        self.stem = nn.Sequential(
            CircConv2d(n_filters, base_ch, 5),
            nn.BatchNorm2d(base_ch), nn.ReLU())
        blocks = []
        ch = base_ch
        for i in range(n_blocks):
            next_ch = ch * 2 if i < n_blocks - 1 else ch
            blocks.append(ResBlock(ch, next_ch, dropout))
            ch = next_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(ch * 2 * 4, 128),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.circ_filters(x)  # (B, n_filters, T, bins)
        x = self.stem(x)
        x = self.blocks(x)
        return self.fc(self.pool(x))


# ============================================================
# Training
# ============================================================
class FlatDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_cv(X, y, g, model_fn, tag, epochs=250, patience=40, bs=32):
    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnts = Counter(strat)
    for i in range(len(strat)):
        if cnts[strat[i]] < 5: strat[i] = str(y[i])
    sp = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(y); accs = []
    probs_all = np.zeros((len(y), max(y) + 1))

    for fi, (tr, te) in enumerate(sp.split(X, strat)):
        model = model_fn()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(FlatDS(X[tr], y[tr]), batch_size=bs, shuffle=True)

        ba, bp, pat, best_probs = 0, None, 0, None
        for ep in range(epochs):
            model.train()
            el = 0
            for xb, yb in loader:
                loss = crit(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item()
            sched.step(el)
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X[te]))
                p = logits.argmax(1).numpy(); acc = (p == y[te]).mean()
                if acc > ba:
                    ba = acc; bp = p; pat = 0
                    best_probs = torch.softmax(logits, dim=1).numpy()
                else: pat += 1
                if pat >= patience: break
        preds[te] = bp; accs.append(ba)
        if best_probs is not None:
            probs_all[te] = best_probs

    acc = np.mean(accs)
    ds = {d: (preds[g == d] == y[g == d]).mean() for d in sorted(set(g))}
    print(f"  {tag:55s} {acc:.1%}  "
          f"oud={ds.get('oud', 0):.1%}  cairo={ds.get('cairo', 0):.1%}  "
          f"m478={ds.get('maqam478', 0):.1%}")
    return acc, preds, probs_all


# ============================================================
def main():
    print("=" * 70)
    print("TEMPLATE-CONV ABLATION + EXTENSIONS")
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
    for r in records:
        if r.get("tonic_annotated"):
            r["tonic"] = r["tonic_annotated"]
        else:
            r["tonic"] = estimate_tonic(r["f0"], maqam=r["maqam"])

    class_names = sorted(set(r["maqam"] for r in records))
    maqam_to_idx = {m: i for i, m in enumerate(class_names)}
    nc = len(class_names)
    n = len(records)
    num_bins = 30

    # ── Build features ──
    print("\nBuilding features...")

    # A: Template-conv with 8 maqam templates (our best)
    templates_8, tpl_names_8 = build_templates(num_bins)
    nt8 = len(templates_8)
    feats_tpl8, labels, groups = [], [], []
    for r in records:
        f = compute_features(r["f0"], num_bins, 20, templates_8, "full")
        if f is not None:
            feats_tpl8.append(f); labels.append(maqam_to_idx[r["maqam"]])
            groups.append(r["dataset"])
    X_tpl8 = np.array(feats_tpl8, dtype=np.float32).transpose(0, 2, 1, 3)
    y = np.array(labels); g = np.array(groups)
    print(f"  8 maqam templates: {X_tpl8.shape}")

    # B: Template-conv with 28 ajnas templates
    ajnas_tpls = build_ajnas_templates(num_bins)
    nt_ajnas = len(ajnas_tpls)
    feats_ajnas = []
    for i, r in enumerate(records):
        if i < len(feats_tpl8):  # same valid set
            f = compute_features_with_templates(r["f0"], num_bins, 20, ajnas_tpls)
            if f is not None:
                feats_ajnas.append(f)
            else:
                feats_ajnas.append(np.zeros((20, nt_ajnas, num_bins), dtype=np.float32))
    # Align with valid set
    X_ajnas = np.array(feats_ajnas[:len(y)], dtype=np.float32).transpose(0, 2, 1, 3)
    print(f"  {nt_ajnas} ajnas templates: {X_ajnas.shape}")

    # C: Combined (8 maqam + 28 ajnas = 36 templates)
    all_tpls = {}
    for name, tpl in sorted(_TONIC_TEMPLATES.items()):
        # Downsample to num_bins
        if len(tpl) != num_bins:
            all_tpls[f"maq_{name}"] = tpl.reshape(num_bins, len(tpl) // num_bins).sum(1)
        else:
            all_tpls[f"maq_{name}"] = tpl
    for name, tpl in sorted(ajnas_tpls.items()):
        all_tpls[f"jins_{name}"] = tpl
    nt_all = len(all_tpls)
    feats_all = []
    for r in records:
        f = compute_features_with_templates(r["f0"], num_bins, 20, all_tpls)
        if f is not None:
            feats_all.append(f)
        else:
            feats_all.append(np.zeros((20, nt_all, num_bins), dtype=np.float32))
    X_all = np.array(feats_all[:len(y)], dtype=np.float32).transpose(0, 2, 1, 3)
    print(f"  {nt_all} combined templates: {X_all.shape}")

    # D: Raw histogram (for learned conv baseline)
    feats_raw = []
    for r in records:
        f = compute_raw_histograms(r["f0"], num_bins, 20)
        if f is not None:
            feats_raw.append(f)
        else:
            feats_raw.append(np.zeros((20, num_bins), dtype=np.float32))
    X_raw = np.array(feats_raw[:len(y)], dtype=np.float32)
    X_raw = X_raw[:, np.newaxis, :, :]  # (N, 1, T, bins) for CNN input
    print(f"  Raw histogram: {X_raw.shape}")

    # ── Experiment 1: Template-conv vs Learned-conv ──
    print(f"\n{'=' * 70}")
    print("EXP 1: TEMPLATE CONV vs LEARNED CONV (does DiArMaqAr help?)")
    print(f"{'=' * 70}")

    # Template-conv ResNet (our best)
    train_cv(X_tpl8, y, g,
             lambda: TemplateNetDeep(nt8, 20, num_bins, nc, 32, 2, 0.4),
             "TemplateConv ResNet (8 maqam tpls)")

    # Learned-conv ResNet (same architecture, random init filters)
    train_cv(X_raw, y, g,
             lambda: LearnedConvResNetV2(num_bins, 20, nc, 8, 32, 2, 0.4),
             "LearnedConv ResNet (8 random filters)")

    # Learned with MORE filters
    train_cv(X_raw, y, g,
             lambda: LearnedConvResNetV2(num_bins, 20, nc, 16, 32, 2, 0.4),
             "LearnedConv ResNet (16 random filters)")

    train_cv(X_raw, y, g,
             lambda: LearnedConvResNetV2(num_bins, 20, nc, 32, 32, 2, 0.4),
             "LearnedConv ResNet (32 random filters)")

    # ── Experiment 2: More templates from DiArMaqAr ──
    print(f"\n{'=' * 70}")
    print("EXP 2: MORE TEMPLATES (ajnas + combined)")
    print(f"{'=' * 70}")

    # 28 ajnas templates
    train_cv(X_ajnas, y, g,
             lambda: TemplateNetDeep(nt_ajnas, 20, num_bins, nc, 32, 2, 0.4),
             f"TemplateConv ResNet ({nt_ajnas} ajnas tpls)")

    # Combined 36 templates
    train_cv(X_all, y, g,
             lambda: TemplateNetDeep(nt_all, 20, num_bins, nc, 32, 2, 0.4),
             f"TemplateConv ResNet ({nt_all} combined tpls)")

    # (Ensemble experiment omitted from the public release; not used in the paper.)


if __name__ == "__main__":
    main()
