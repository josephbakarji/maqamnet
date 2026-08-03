#!/usr/bin/env python3
"""
Table 1 and the learning-free baseline of Table 2.

1. Tonic-noise sensitivity (Table 1): Gaussian noise (in cents) applied to
   oracle tonics, Random Forest on the 60-bin tonic-normalized histogram,
   stratified 5-fold CV; 5 noise seeds per sigma, mean +/- std reported.

2. "Raw xcorr argmax (no ML)" row of Table 2: each recording is scored
   against the 7 maqam-family DiArMaqAr templates by the max over circular
   shifts of the dot product with its raw 240-bin pitch-class histogram;
   the argmax template is the prediction. No learning involved.

Outputs: results/tonic_noise_and_xcorr.{json,log}
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478,
    estimate_tonic, compute_histogram, _TONIC_TEMPLATES, EXTENDED_MAQAMS
)

PROJECT_DIR = Path(__file__).parent.parent
OUT = PROJECT_DIR / "results" / "tonic_noise_and_xcorr.json"

SIGMAS = [0, 10, 25, 50, 100, 200]
N_NOISE_SEEDS = 5


def raw_pc_hist(f0, pc_bins=240):
    f0_v = f0[(f0 > 0) & np.isfinite(f0) & (f0 >= 60) & (f0 <= 800)]
    if len(f0_v) < 100:
        return None
    cents = (1200.0 * np.log2(f0_v / 110.0)) % 1200
    h = np.zeros(pc_bins)
    for c in cents:
        bf = c * pc_bins / 1200.0
        lo = int(np.floor(bf)) % pc_bins
        hi = (lo + 1) % pc_bins
        frac = bf - np.floor(bf)
        h[lo] += 1 - frac
        h[hi] += frac
    return gaussian_filter1d(h, sigma=2, mode="wrap")


def template_score(pc_hist, template):
    """Max over circular shifts of dot(downsampled shifted hist, template)."""
    pc_bins = len(pc_hist)
    nb = len(template)
    ratio = pc_bins // nb
    best = -1.0
    for shift in range(pc_bins):
        d = np.roll(pc_hist, -shift).reshape(nb, ratio).sum(axis=1)
        s = d.sum()
        if s > 0:
            d = d / s
        best = max(best, float(np.dot(d, template)))
    return best


def rf_cv_accuracy(X, y, strat, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    preds = np.zeros_like(y)
    for tr, te in skf.split(X, strat):
        rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=2)
        rf.fit(X[tr], y[tr])
        preds[te] = rf.predict(X[te])
    return float((preds == y).mean())


def main():
    print("=" * 70)
    print("TABLE 1 (tonic noise) + RAW XCORR ARGMAX BASELINE")
    print("=" * 70)

    records = []
    for loader in (load_arabic_oud, load_cairo_congress, load_maqam478):
        records.extend(loader())
    records = [r for r in records if r["maqam"] in EXTENDED_MAQAMS]
    class_names = sorted(set(r["maqam"] for r in records))
    cls_idx = {m: i for i, m in enumerate(class_names)}
    print(f"  {len(records)} recordings, classes: {class_names}")

    # Oracle tonics (uses maqam label, as in the paper)
    print("\nEstimating oracle tonics...")
    tonics = np.array([estimate_tonic(r["f0"], maqam=r["maqam"]) for r in records])

    y = np.array([cls_idx[r["maqam"]] for r in records])
    g = np.array([r["dataset"] for r in records])
    strat = np.array([f"{yi}_{gi}" for yi, gi in zip(y, g)])
    cnt = Counter(strat)
    for i in range(len(strat)):
        if cnt[strat[i]] < 5:
            strat[i] = str(y[i])

    results = {"table1": {}, "xcorr_argmax": {}}

    # ── Part 1: Table 1 noise curve ──
    print("\nPART 1: tonic-noise sensitivity (RF, 60-bin histogram)")
    print(f"  {'sigma':>6s}  {'mean':>6s}  {'std':>5s}   paper")
    paper = {0: 91.8, 10: 89.9, 25: 84.1, 50: 76.7, 100: 61.0, 200: 46.2}
    for sigma in SIGMAS:
        accs = []
        seeds = [0] if sigma == 0 else range(N_NOISE_SEEDS)
        for ns in seeds:
            rng = np.random.default_rng(ns)
            noisy = tonics * 2 ** (rng.normal(0, sigma, len(tonics)) / 1200.0)
            X, yy, ss = [], [], []
            for r, t, yi, si in zip(records, noisy, y, strat):
                h = compute_histogram(r["f0"], t, num_bins=60)
                if h is not None:
                    X.append(h); yy.append(yi); ss.append(si)
            X = np.array(X); yy = np.array(yy); ss = np.array(ss)
            accs.append(rf_cv_accuracy(X, yy, ss, seed=ns))
        m, s = 100 * np.mean(accs), 100 * np.std(accs)
        results["table1"][sigma] = {"mean": m, "std": s, "runs": [100 * a for a in accs]}
        print(f"  {sigma:6d}  {m:6.1f}  {s:5.1f}   {paper[sigma]}")

    # ── Part 2: raw xcorr argmax ──
    print("\nPART 2: raw xcorr argmax (no ML)")
    templates = {m: np.asarray(_TONIC_TEMPLATES[m]) for m in class_names
                 if m in _TONIC_TEMPLATES}
    preds, ys, gs = [], [], []
    for r in records:
        h = raw_pc_hist(r["f0"])
        if h is None:
            continue
        scores = {m: template_score(h, t) for m, t in templates.items()}
        preds.append(cls_idx[max(scores, key=scores.get)])
        ys.append(cls_idx[r["maqam"]]); gs.append(r["dataset"])
    preds, ys, gs = np.array(preds), np.array(ys), np.array(gs)
    acc = float((preds == ys).mean())
    per_ds = {d: float((preds[gs == d] == ys[gs == d]).mean()) for d in sorted(set(gs))}
    results["xcorr_argmax"] = {"overall": 100 * acc, "per_dataset":
                               {d: 100 * a for d, a in per_ds.items()},
                               "n": int(len(ys))}
    print(f"  overall: {100*acc:.1f}%  (paper: 36.3)")
    for d, a in per_ds.items():
        print(f"    {d:10s}: {100*a:.1f}%")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
