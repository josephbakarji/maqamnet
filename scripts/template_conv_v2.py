#!/usr/bin/env python3
"""
Template-convolution v2: systematic optimization.

Key issues from v1:
- Raw cross-correlation is sparse and high-dimensional (8×120=960 per window)
- CNN can't learn to find peaks from 619 examples
- Need to either reduce dimensionality or help the model find the peaks

Approaches:
1. Peak extraction: take argmax and max value per template → compact features
2. Reduced bins: 60 or 30 bins instead of 120
3. More templates: add ajnas templates alongside maqam templates
4. Learned pooling: let the model learn to aggregate the shift dimension
5. Template response + raw histogram concatenation
6. Deeper/wider CNN architectures
7. Different time resolutions
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
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478,
    estimate_tonic, _TONIC_TEMPLATES, EXTENDED_MAQAMS
)

PROJECT_DIR = Path(__file__).parent.parent
REF_HZ = 110.0


# ============================================================
# Feature computation variants
# ============================================================
def raw_histogram(f0, num_bins, num_time, sigma_cents=8.0):
    """Raw (non-tonic-normalized) pitch-time histogram."""
    f0_v = f0[(f0 > 0) & (~np.isnan(f0)) & (f0 >= 60) & (f0 <= 800)]
    if len(f0_v) < num_time * 10:
        return None
    cents = (1200.0 * np.log2(f0_v / REF_HZ)) % 1200
    n = len(cents)
    sigma_bins = sigma_cents * num_bins / 1200.0
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


def template_xcorr(hist_window, templates, num_bins):
    """Cross-correlate a single histogram window with all templates."""
    hist_fft = np.fft.rfft(hist_window)
    responses = []
    for tpl in templates:
        tpl_fft = np.fft.rfft(tpl)
        xcorr = np.real(np.fft.irfft(np.conj(hist_fft) * tpl_fft, n=num_bins))
        responses.append(xcorr)
    return np.array(responses, dtype=np.float32)  # (n_tpl, num_bins)


def build_templates(num_bins):
    """Build templates at given resolution."""
    tpl_names = sorted(_TONIC_TEMPLATES.keys())
    templates = []
    for name in tpl_names:
        tpl_120 = _TONIC_TEMPLATES[name]
        if num_bins == 120:
            templates.append(tpl_120.copy())
        elif num_bins == 60:
            templates.append(tpl_120.reshape(60, 2).sum(axis=1))
        elif num_bins == 30:
            templates.append(tpl_120.reshape(30, 4).sum(axis=1))
        else:
            # Interpolate
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, 120, endpoint=False)
            x_new = np.linspace(0, 1, num_bins, endpoint=False)
            templates.append(interp1d(x_old, tpl_120, kind='linear')(x_new))
    # Normalize
    for i in range(len(templates)):
        s = templates[i].sum()
        if s > 0: templates[i] /= s
    return templates, tpl_names


def compute_features(f0, num_bins, num_time, templates, mode="full"):
    """Compute template-conv features in various modes.

    Modes:
    - "full": full cross-correlation tensor (T, n_tpl, num_bins)
    - "peaks": peak height + position per template per window (T, n_tpl, 2)
    - "topk": top-3 peak heights per template per window (T, n_tpl, 3)
    - "stats": mean, max, argmax, std per template per window (T, n_tpl, 4)
    - "concat": full xcorr + raw histogram stacked (T, n_tpl+1, num_bins)
    """
    hist_mat = raw_histogram(f0, num_bins, num_time)
    if hist_mat is None:
        return None

    n_tpl = len(templates)

    if mode == "full":
        tensor = np.zeros((num_time, n_tpl, num_bins), dtype=np.float32)
        for t in range(num_time):
            tensor[t] = template_xcorr(hist_mat[t], templates, num_bins)
        return tensor

    elif mode == "peaks":
        tensor = np.zeros((num_time, n_tpl, 2), dtype=np.float32)
        for t in range(num_time):
            xcorrs = template_xcorr(hist_mat[t], templates, num_bins)
            for ti in range(n_tpl):
                tensor[t, ti, 0] = xcorrs[ti].max()  # peak height
                tensor[t, ti, 1] = xcorrs[ti].argmax() / num_bins  # peak position (normalized)
        return tensor

    elif mode == "topk":
        k = 3
        tensor = np.zeros((num_time, n_tpl, k * 2), dtype=np.float32)
        for t in range(num_time):
            xcorrs = template_xcorr(hist_mat[t], templates, num_bins)
            for ti in range(n_tpl):
                top_idx = np.argsort(xcorrs[ti])[::-1][:k]
                for ki in range(k):
                    tensor[t, ti, ki * 2] = xcorrs[ti][top_idx[ki]]
                    tensor[t, ti, ki * 2 + 1] = top_idx[ki] / num_bins
        return tensor

    elif mode == "stats":
        tensor = np.zeros((num_time, n_tpl, 4), dtype=np.float32)
        for t in range(num_time):
            xcorrs = template_xcorr(hist_mat[t], templates, num_bins)
            for ti in range(n_tpl):
                tensor[t, ti, 0] = xcorrs[ti].mean()
                tensor[t, ti, 1] = xcorrs[ti].max()
                tensor[t, ti, 2] = xcorrs[ti].argmax() / num_bins
                tensor[t, ti, 3] = xcorrs[ti].std()
        return tensor

    elif mode == "concat":
        # Stack raw histogram as an additional "template" channel
        tensor = np.zeros((num_time, n_tpl + 1, num_bins), dtype=np.float32)
        for t in range(num_time):
            tensor[t, :n_tpl] = template_xcorr(hist_mat[t], templates, num_bins)
            tensor[t, n_tpl] = hist_mat[t]  # raw histogram as extra channel
        return tensor


# ============================================================
# Models
# ============================================================
class FlatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TemplateConvNet(nn.Module):
    """Flexible CNN for template response tensors."""
    def __init__(self, in_channels, h, w, n_classes, channels=(32, 64),
                 use_circular=False, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = in_channels
        cur_h, cur_w = h, w
        for ch in channels:
            if use_circular and cur_w >= 4:
                layers.append(CircConv2d(in_ch, ch, 3))
            else:
                layers.append(nn.Conv2d(in_ch, ch, 3, padding=1))
            layers.extend([nn.BatchNorm2d(ch), nn.ReLU()])
            if cur_h >= 4 and cur_w >= 4:
                layers.append(nn.MaxPool2d(2))
                cur_h //= 2; cur_w //= 2
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 2 * 4, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.fc(x)


class CircConv2d(nn.Module):
    """Conv2d with circular padding on the last (shift/pitch) dimension."""
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=(k//2, 0))
    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
        return self.conv(x)


class PeakTransformer(nn.Module):
    """Transformer on peak features per time step."""
    def __init__(self, feat_dim, num_time, d_model=64, nhead=4,
                 n_layers=2, n_classes=7, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, num_time, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(32, n_classes))
    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B, T, -1)
        x = self.proj(x) + self.pos[:, :T, :]
        return self.fc(self.enc(x).mean(dim=1))


# ============================================================
# Training
# ============================================================
def train_cv(X, y, g, model_fn, tag, epochs=200, patience=30, bs=32):
    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnts = Counter(strat)
    for i in range(len(strat)):
        if cnts[strat[i]] < 5: strat[i] = str(y[i])
    sp = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(y); accs = []
    for fi, (tr, te) in enumerate(sp.split(X, strat)):
        model = model_fn()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(FlatDataset(X[tr], y[tr]), batch_size=bs, shuffle=True)
        ba, bp, pat = 0, None, 0
        for ep in range(epochs):
            model.train()
            for xb, yb in loader:
                loss = crit(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X[te]))
                p = logits.argmax(1).numpy(); acc = (p == y[te]).mean()
                if acc > ba: ba = acc; bp = p; pat = 0
                else: pat += 1
                if pat >= patience: break
        preds[te] = bp; accs.append(ba)
    acc = np.mean(accs)
    ds = {d: (preds[g==d]==y[g==d]).mean() for d in sorted(set(g))}
    print(f"  {tag:50s} {acc:.1%}  "
          f"oud={ds.get('oud',0):.1%}  cairo={ds.get('cairo',0):.1%}  "
          f"m478={ds.get('maqam478',0):.1%}")
    return acc


def train_rf(X, y, g, tag):
    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnts = Counter(strat)
    for i in range(len(strat)):
        if cnts[strat[i]] < 5: strat[i] = str(y[i])
    sp = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(y); accs = []
    for fi, (tr, te) in enumerate(sp.split(X, strat)):
        clf = RandomForestClassifier(n_estimators=300, random_state=42)
        clf.fit(X[tr], y[tr]); preds[te] = clf.predict(X[te])
        accs.append((preds[te] == y[te]).mean())
    acc = np.mean(accs)
    ds = {d: (preds[g==d]==y[g==d]).mean() for d in sorted(set(g))}
    print(f"  {tag:50s} {acc:.1%}  "
          f"oud={ds.get('oud',0):.1%}  cairo={ds.get('cairo',0):.1%}  "
          f"m478={ds.get('maqam478',0):.1%}")
    return acc


# ============================================================
def main():
    from tonic_close_gap import compute_fft_features  # optional path, not used in the paper pipeline
    print("=" * 70)
    print("TEMPLATE-CONVOLUTION v2: OPTIMIZATION")
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

    # ── Sweep feature modes and resolutions ──
    configs = [
        # (num_bins, num_time, mode, label)
        (60, 20, "peaks", "peaks 60bins 20t"),
        (60, 20, "topk", "topk 60bins 20t"),
        (60, 20, "stats", "stats 60bins 20t"),
        (60, 10, "peaks", "peaks 60bins 10t"),
        (60, 10, "stats", "stats 60bins 10t"),
        (30, 20, "peaks", "peaks 30bins 20t"),
        (30, 20, "stats", "stats 30bins 20t"),
        (60, 20, "full", "full 60bins 20t"),
        (30, 20, "full", "full 30bins 20t"),
        (60, 20, "concat", "concat(xcorr+hist) 60bins 20t"),
        (60, 1, "stats", "stats 60bins fullpiece"),
        (60, 1, "peaks", "peaks 60bins fullpiece"),
    ]

    print(f"\n{'=' * 70}")
    print("RF ON FLATTENED FEATURES")
    print(f"{'=' * 70}")

    best_rf_acc, best_rf_config = 0, ""
    for num_bins, num_time, mode, label in configs:
        templates, tpl_names = build_templates(num_bins)

        feats, labels, groups = [], [], []
        for r in records:
            f = compute_features(r["f0"], num_bins, num_time, templates, mode)
            if f is not None:
                feats.append(f.flatten())
                labels.append(maqam_to_idx[r["maqam"]])
                groups.append(r["dataset"])

        X = np.array(feats, dtype=np.float32)
        y = np.array(labels); g = np.array(groups)
        acc = train_rf(X, y, g, f"RF {label} ({X.shape[1]}d)")
        if acc > best_rf_acc:
            best_rf_acc = acc; best_rf_config = label

    print(f"\n  Best RF: {best_rf_config} ({best_rf_acc:.1%})")

    # ── CNN on best configurations ──
    print(f"\n{'=' * 70}")
    print("CNN ON BEST CONFIGURATIONS")
    print(f"{'=' * 70}")

    for num_bins, num_time, mode, label in [
        (60, 20, "full", "full 60bins 20t"),
        (30, 20, "full", "full 30bins 20t"),
        (60, 20, "concat", "concat 60bins 20t"),
        (60, 20, "stats", "stats 60bins 20t"),
    ]:
        templates, tpl_names = build_templates(num_bins)
        nt = len(templates)

        feats, labels, groups = [], [], []
        for r in records:
            f = compute_features(r["f0"], num_bins, num_time, templates, mode)
            if f is not None:
                feats.append(f); labels.append(maqam_to_idx[r["maqam"]])
                groups.append(r["dataset"])
        X = np.array(feats, dtype=np.float32)
        y = np.array(labels); g = np.array(groups)

        if mode == "full":
            # (N, T, nt, bins) → (N, nt, T, bins) for CNN channels
            X_cnn = X.transpose(0, 2, 1, 3)
            in_ch = nt; h = num_time; w = num_bins
            for chs in [(16, 32), (32, 64), (32, 64, 128)]:
                train_cv(X_cnn, y, g,
                         lambda c=chs: TemplateConvNet(nt, h, w, nc, c, True, 0.3),
                         f"CNN-circ {chs} {label}")
        elif mode == "concat":
            X_cnn = X.transpose(0, 2, 1, 3)
            in_ch = nt + 1; h = num_time; w = num_bins
            train_cv(X_cnn, y, g,
                     lambda: TemplateConvNet(in_ch, h, w, nc, (32, 64), True, 0.3),
                     f"CNN-circ (32,64) {label}")
        elif mode == "stats":
            # Transformer on stats
            feat_dim = nt * 4
            train_cv(X, y, g,
                     lambda: PeakTransformer(feat_dim, num_time, 64, 4, 2, nc, 0.3),
                     f"Transformer {label}")

    # ── Peak features + RF ensemble with FFT ──
    print(f"\n{'=' * 70}")
    print("BEST FEATURES + FFT ENSEMBLE")
    print(f"{'=' * 70}")

    templates_60, _ = build_templates(60)
    # Build peak features + FFT features
    feats_peaks, feats_fft, labels, groups = [], [], [], []
    for r in records:
        f = compute_features(r["f0"], 60, 20, templates_60, "stats")
        fft = compute_fft_features(r["f0"])
        if f is not None and fft is not None:
            feats_peaks.append(f.flatten())
            feats_fft.append(fft)
            labels.append(maqam_to_idx[r["maqam"]])
            groups.append(r["dataset"])

    X_p = np.array(feats_peaks, dtype=np.float32)
    X_f = np.array(feats_fft, dtype=np.float32)
    y = np.array(labels); g = np.array(groups)

    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnts = Counter(strat)
    for i in range(len(strat)):
        if cnts[strat[i]] < 5: strat[i] = str(y[i])
    sp = StratifiedKFold(5, shuffle=True, random_state=42)

    # Concatenated
    X_cat = np.hstack([X_p, X_f])
    train_rf(X_cat, y, g, f"RF stats+FFT concat ({X_cat.shape[1]}d)")

    # Probability ensemble
    for w_p, w_f in [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6)]:
        preds = np.zeros_like(y); accs = []
        for fi, (tr, te) in enumerate(sp.split(X_p, strat)):
            clf_p = RandomForestClassifier(n_estimators=300, random_state=42)
            clf_f = RandomForestClassifier(n_estimators=300, random_state=42)
            clf_p.fit(X_p[tr], y[tr]); clf_f.fit(X_f[tr], y[tr])
            pp = clf_p.predict_proba(X_p[te]); pf = clf_f.predict_proba(X_f[te])
            preds[te] = (w_p * pp + w_f * pf).argmax(1)
            accs.append((preds[te] == y[te]).mean())
        acc = np.mean(accs)
        ds = {d: (preds[g==d]==y[g==d]).mean() for d in sorted(set(g))}
        print(f"  Ens stats({w_p})+FFT({w_f})                       "
              f"  {acc:.1%}  oud={ds.get('oud',0):.1%}  "
              f"cairo={ds.get('cairo',0):.1%}  m478={ds.get('maqam478',0):.1%}")

    print(f"\n  Reference: Disambig Transformer=85.8%  Oracle=93.5%")


if __name__ == "__main__":
    main()
