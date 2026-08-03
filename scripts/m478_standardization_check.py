#!/usr/bin/env python3
"""
Why is Maqam-478 accurately classified WITHOUT tonic normalization?

Hypothesis A (smaller pitch inventory): rejected; per-recording pitch-class
entropy is nearly identical across datasets (see protocol_validation.py,
Part A).

Hypothesis B (tested here): recitations are standardized in ABSOLUTE pitch.
Within a maqam class, different recordings place their material at nearly
the same absolute frequencies, so a no-tonic model can classify from raw
histograms; oud performers transpose freely, so their raw histograms are
inconsistent within a class.

Test: mean pairwise cosine similarity of raw unwrapped pitch histograms
(240 bins, 0-4800c above 55 Hz, whole piece), within class vs between
classes, per dataset. Reported in the Discussion of the paper.
"""

import sys
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478, EXTENDED_MAQAMS
)

REF_HZ = 55.0


def unwrapped_histogram(f0, num_bins=240, cmin=0, cmax=4800):
    f0 = f0[(f0 > 0) & np.isfinite(f0) & (f0 >= 55) & (f0 <= 880)]
    if len(f0) < 100:
        return None
    cents = 1200.0 * np.log2(f0 / REF_HZ)
    hist, _ = np.histogram(cents, bins=num_bins, range=(cmin, cmax))
    hist = hist.astype(float)
    n = np.linalg.norm(hist)
    return hist / n if n > 0 else None


def main():
    print("=" * 70)
    print("M478 STANDARDIZATION CHECK: within-class raw-histogram similarity")
    print("=" * 70)

    records = []
    for loader in (load_arabic_oud, load_cairo_congress, load_maqam478):
        records.extend(loader())
    records = [r for r in records if r["maqam"] in EXTENDED_MAQAMS]

    by_ds = {}
    for r in records:
        h = unwrapped_histogram(r["f0"])
        if h is not None:
            by_ds.setdefault(r["dataset"], []).append((r["maqam"], h))

    print(f"\n  {'dataset':10s} {'within-class':>13s} {'between-class':>14s} "
          f"{'contrast':>9s}")
    for ds, items in sorted(by_ds.items()):
        within, between = [], []
        for (m1, h1), (m2, h2) in combinations(items, 2):
            sim = float(h1 @ h2)
            (within if m1 == m2 else between).append(sim)
        w, b = np.mean(within), np.mean(between)
        print(f"  {ds:10s} {w:13.3f} {b:14.3f} {w - b:+9.3f}")

    print("\n  Interpretation: high within-class similarity with a large")
    print("  within-between contrast means raw (un-normalized) histograms are")
    print("  already class-consistent, i.e., absolute pitch placement is")
    print("  standardized within each maqam class of that dataset.")


if __name__ == "__main__":
    main()
