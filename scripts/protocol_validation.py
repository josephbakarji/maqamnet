#!/usr/bin/env python3
"""
Protocol validation: stricter evaluation of the two key models.

Re-runs MaqamNet (36 channels, base width 64, 2 residual blocks, dropout
0.4) and the unwrapped-pitch CNN baseline under a validation-based
protocol: the same stratified 5-fold splits as the paper, but with 15% of
each training fold held out for validation, early stopping driven by
validation accuracy only, and the test fold evaluated once, at the
best-validation epoch. For reference, the max-over-epochs test accuracy
(the protocol used for the headline tables) is recorded in the same run.

Also includes: (a) a Maqam-478-only run quantifying the train-test gap on
that subset alone, and (b) per-dataset pitch-class separability statistics
(entropy, effective classes, active bins).

Reported in Section 5.3 and the Discussion of the paper.
Outputs: results/protocol_validation.{json,log}
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478, EXTENDED_MAQAMS
)
from template_conv_v2 import build_templates, FlatDataset
from template_conv_ablation import build_ajnas_templates, compute_features_with_templates
from template_conv_deep_tune import InstrumentedResNet
from unwrapped_pitch import UnwrappedCNN2D, compute_unwrapped_pitch_time

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_DIR / "results" / "protocol_validation.json"


# ============================================================
# Corrected-protocol training
# ============================================================
def make_strat_labels(y, g, min_count):
    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnts = Counter(strat)
    for i in range(len(strat)):
        if cnts[strat[i]] < min_count:
            strat[i] = str(y[i])
    return strat


def train_proper_cv(X, y, g, model_fn, tag, epochs=250, patience=40,
                    bs=32, lr=1e-3, weight_decay=1e-4, val_frac=0.15):
    """5-fold CV with val-based early stopping; test evaluated at best-val epoch.

    Returns dict with corrected and old-protocol (max test) accuracies.
    """
    strat = make_strat_labels(y, g, min_count=5)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)

    preds = np.zeros_like(y)          # corrected protocol predictions
    fold_rows = []

    for fi, (tr_all, te) in enumerate(skf.split(X, strat)):
        torch.manual_seed(fi)
        np.random.seed(fi)

        # Stratified val split within the training fold only
        strat_tr = make_strat_labels(y[tr_all], g[tr_all], min_count=2)
        tr, va = train_test_split(
            np.arange(len(tr_all)), test_size=val_frac,
            random_state=42, stratify=strat_tr)
        tr, va = tr_all[tr], tr_all[va]

        model = model_fn()
        opt = torch.optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=10, factor=0.5)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(FlatDataset(X[tr], y[tr]), batch_size=bs,
                            shuffle=True)

        Xva = torch.FloatTensor(X[va])
        Xte = torch.FloatTensor(X[te])
        Xtr = torch.FloatTensor(X[tr])

        best_val, best_state, pat = 0.0, None, 0
        max_test = 0.0   # what the OLD protocol would have reported
        for ep in range(epochs):
            model.train()
            el = 0.0
            for xb, yb in loader:
                loss = crit(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
                el += loss.item()
            sched.step(el)

            model.eval()
            with torch.no_grad():
                val_acc = (model(Xva).argmax(1).numpy() == y[va]).mean()
                test_acc = (model(Xte).argmax(1).numpy() == y[te]).mean()
            max_test = max(max_test, test_acc)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
            if pat >= patience:
                break

        # Evaluate ONCE at the best-validation epoch
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p_te = model(Xte).argmax(1).numpy()
            train_acc = (model(Xtr).argmax(1).numpy() == y[tr]).mean()
        test_at_val = (p_te == y[te]).mean()
        preds[te] = p_te

        fold_rows.append({
            "fold": fi + 1,
            "train_acc": float(train_acc),
            "val_acc": float(best_val),
            "test_at_best_val": float(test_at_val),
            "max_test_over_epochs_OLD": float(max_test),
        })
        print(f"    [{tag}] fold {fi+1}: train={train_acc:.1%} "
              f"val={best_val:.1%} test@bestval={test_at_val:.1%} "
              f"(old-protocol max test={max_test:.1%})")

    overall = float((preds == y).mean())
    per_ds = {d: float((preds[g == d] == y[g == d]).mean())
              for d in sorted(set(g))}
    old_mean = float(np.mean([r["max_test_over_epochs_OLD"] for r in fold_rows]))
    tr_mean = float(np.mean([r["train_acc"] for r in fold_rows]))

    print(f"  [{tag}] CORRECTED: {overall:.1%}  per-dataset: "
          + "  ".join(f"{d}={a:.1%}" for d, a in per_ds.items()))
    print(f"  [{tag}] old-protocol mean-of-max: {old_mean:.1%}  |  "
          f"train mean: {tr_mean:.1%}  (train-test gap "
          f"{tr_mean - overall:+.1%})")

    return {"tag": tag, "overall_corrected": overall, "per_dataset": per_ds,
            "old_protocol_mean_max_test": old_mean,
            "mean_train_acc": tr_mean, "folds": fold_rows}


# ============================================================
# Separability statistics
# ============================================================
def pitch_class_stats(records, num_bins=120, ref_hz=110.0):
    """Per-dataset octave-wrapped pitch-class entropy and active-bin stats."""
    out = {}
    per_ds = {}
    for r in records:
        f0 = r["f0"]
        f0 = f0[(f0 > 0) & np.isfinite(f0) & (f0 >= 55) & (f0 <= 880)]
        if len(f0) < 100:
            continue
        cents = (1200.0 * np.log2(f0 / ref_hz)) % 1200.0
        hist, _ = np.histogram(cents, bins=num_bins, range=(0, 1200))
        hist = gaussian_filter1d(hist.astype(float), sigma=1.0, mode="wrap")
        p = hist / hist.sum()
        nz = p[p > 0]
        H = float(-(nz * np.log2(nz)).sum())              # bits
        eff = float(2.0 ** H)                             # effective classes
        active = int((hist > 0.2 * hist.max()).sum())     # bins > 20% of max
        per_ds.setdefault(r["dataset"], []).append((H, eff, active))

    print("\n  Pitch-class separability statistics (octave-wrapped, 10c bins):")
    print(f"    {'dataset':10s} {'n':>4s} {'entropy(bits)':>14s} "
          f"{'eff. classes':>13s} {'active bins':>12s}")
    for ds, rows in sorted(per_ds.items()):
        a = np.array(rows)
        out[ds] = {
            "n": len(rows),
            "entropy_bits_mean": float(a[:, 0].mean()),
            "entropy_bits_std": float(a[:, 0].std()),
            "effective_classes_mean": float(a[:, 1].mean()),
            "effective_classes_std": float(a[:, 1].std()),
            "active_bins_mean": float(a[:, 2].mean()),
            "active_bins_std": float(a[:, 2].std()),
        }
        print(f"    {ds:10s} {len(rows):4d} "
              f"{a[:,0].mean():7.2f}±{a[:,0].std():4.2f} "
              f"{a[:,1].mean():8.1f}±{a[:,1].std():4.1f} "
              f"{a[:,2].mean():7.1f}±{a[:,2].std():4.1f}")
    return out


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("PROTOCOL VALIDATION: val-based early stopping, no test peeking")
    print("=" * 70)

    print("\nLoading datasets...")
    records = []
    for loader in (load_arabic_oud, load_cairo_congress, load_maqam478):
        records.extend(loader())
    records = [r for r in records if r["maqam"] in EXTENDED_MAQAMS]

    class_names = sorted(set(r["maqam"] for r in records))
    maqam_to_idx = {m: i for i, m in enumerate(class_names)}
    nc = len(class_names)
    print(f"  {len(records)} recordings, {nc} maqams")

    results = {}

    # ── Separability stats first (cheap) ──
    print(f"\n{'='*70}\nPART A: DATASET SEPARABILITY\n{'='*70}")
    results["separability"] = pitch_class_stats(records)

    # ── 36-channel template features (paper's MaqamNet input) ──
    num_bins, num_time = 30, 20
    maqam_tpls, maqam_names = build_templates(num_bins)
    ajnas_dict = build_ajnas_templates(num_bins)
    combined = {f"m_{n}": maqam_tpls[i] for i, n in enumerate(maqam_names)}
    for n, t in sorted(ajnas_dict.items()):
        combined[f"j_{n}"] = t
    n_tpls = len(combined)

    print(f"\nBuilding {n_tpls}-channel template features...")
    feats, labels, groups = [], [], []
    for r in records:
        f = compute_features_with_templates(r["f0"], num_bins, num_time, combined)
        if f is not None:
            feats.append(f)
            labels.append(maqam_to_idx[r["maqam"]])
            groups.append(r["dataset"])
    Xt = np.array(feats, dtype=np.float32).transpose(0, 2, 1, 3)
    yt = np.array(labels); gt = np.array(groups)
    print(f"  Template tensor: {Xt.shape}")

    # ── Unwrapped 240x40 features (paper's lower-bound baseline) ──
    print("Building unwrapped 240x40 features...")
    feats_u, labels_u, groups_u = [], [], []
    for r in records:
        m = compute_unwrapped_pitch_time(r["f0"], num_bins=240, num_time=40)
        if m is not None:
            feats_u.append(m)
            labels_u.append(maqam_to_idx[r["maqam"]])
            groups_u.append(r["dataset"])
    Xu = np.array(feats_u, dtype=np.float32)
    yu = np.array(labels_u); gu = np.array(groups_u)
    print(f"  Unwrapped tensor: {Xu.shape}")

    # ── Part B: corrected-protocol runs on the unified corpus ──
    print(f"\n{'='*70}\nPART B: VALIDATION-BASED PROTOCOL, UNIFIED CORPUS\n{'='*70}")

    print("\n[B1] MaqamNet 36ch ch64 b2 d0.4 (paper Table 2 best):")
    results["maqamnet36_corrected"] = train_proper_cv(
        Xt, yt, gt,
        lambda: InstrumentedResNet(n_tpls, num_time, num_bins, nc,
                                   base_ch=64, n_blocks=2, dropout=0.4),
        "MaqamNet36-ch64b2")

    print("\n[B2] Unwrapped 2D CNN 240x40 (paper Table 2 lower bound):")
    results["unwrapped_cnn_corrected"] = train_proper_cv(
        Xu, yu, gu,
        lambda: UnwrappedCNN2D(num_bins=240, num_time=40, num_classes=nc,
                               channels=(32, 64, 128), dropout=0.3),
        "UnwrappedCNN-240x40")

    # ── Part C: Maqam-478 alone, validation-based protocol ──
    print(f"\n{'='*70}\nPART C: MAQAM-478 ALONE, VALIDATION-BASED PROTOCOL\n{'='*70}")
    m478 = gt == "maqam478"
    X4, y4, g4 = Xt[m478], yt[m478], gt[m478]
    print(f"  {len(y4)} recordings, {len(set(y4))} maqam classes")
    results["maqamnet36_m478_only"] = train_proper_cv(
        X4, y4, g4,
        lambda: InstrumentedResNet(n_tpls, num_time, num_bins, nc,
                                   base_ch=64, n_blocks=2, dropout=0.4),
        "MaqamNet36-M478only")

    # ── Save ──
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {RESULTS_PATH}")

    # ── Summary vs paper ──
    print(f"\n{'='*70}\nSUMMARY vs PAPER TABLE NUMBERS\n{'='*70}")
    mn = results["maqamnet36_corrected"]
    un = results["unwrapped_cnn_corrected"]
    print(f"  MaqamNet 36ch : paper 89.0 (old protocol) | "
          f"corrected {mn['overall_corrected']:.1%} | "
          f"old-protocol replica {mn['old_protocol_mean_max_test']:.1%}")
    print(f"  Unwrapped CNN : paper 68.2 (old protocol) | "
          f"corrected {un['overall_corrected']:.1%} | "
          f"old-protocol replica {un['old_protocol_mean_max_test']:.1%}")
    print(f"  Per-dataset corrected: MaqamNet "
          + " ".join(f"{d}={a:.1%}" for d, a in mn["per_dataset"].items()))
    print(f"                         Unwrapped "
          + " ".join(f"{d}={a:.1%}" for d, a in un["per_dataset"].items()))


if __name__ == "__main__":
    main()
