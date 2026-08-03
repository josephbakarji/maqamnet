#!/usr/bin/env python3
"""
Seed-variance study for the ablation (Table 4).

Quantifies run-to-run variability by retraining each configuration over
multiple random seeds under the headline-table protocol (full training
fold, best-test-epoch selection):

  A. TemplateNetDeep on 8 maqam-template features   (ch32, b2, d0.4) x 5 seeds
  B. LearnedConvResNetV2 on raw hist, 8 rand filters (ch32, b2, d0.4) x 5 seeds
  C. InstrumentedResNet on 36-ch combined features   (ch64, b2, d0.4) x 3 seeds

Table 4 reports the resulting mean +/- std for A and B; the caption reports
the seed mean for C.

Output: results/seed_variance_ablation.{json,log}
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import template_conv_ablation as abl
from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478, EXTENDED_MAQAMS
)
from template_conv_deep_tune import InstrumentedResNet, build_combined_templates

PROJECT_DIR = Path(__file__).parent.parent
OUT = PROJECT_DIR / "results" / "seed_variance_ablation.json"


def main():
    print("=" * 70)
    print("SEED-VARIANCE ABLATION (original protocol, matching Table 4)")
    print("=" * 70)

    records = []
    for loader in (load_arabic_oud, load_cairo_congress, load_maqam478):
        records.extend(loader())
    records = [r for r in records if r["maqam"] in EXTENDED_MAQAMS]
    class_names = sorted(set(r["maqam"] for r in records))
    maqam_to_idx = {m: i for i, m in enumerate(class_names)}
    nc = len(class_names)
    num_bins, num_time = 30, 20

    # Features exactly as in template_conv_ablation.main / deep_tune.main
    print("\nBuilding features...")
    templates_8, _ = abl.build_templates(num_bins)
    nt8 = len(templates_8)
    feats, labels, groups = [], [], []
    for r in records:
        f = abl.compute_features(r["f0"], num_bins, num_time, templates_8, "full")
        if f is not None:
            feats.append(f); labels.append(maqam_to_idx[r["maqam"]])
            groups.append(r["dataset"])
    X_tpl8 = np.array(feats, dtype=np.float32).transpose(0, 2, 1, 3)
    y = np.array(labels); g = np.array(groups)

    feats_raw = []
    for r in records:
        f = abl.compute_raw_histograms(r["f0"], num_bins, num_time)
        feats_raw.append(f if f is not None
                         else np.zeros((num_time, num_bins), dtype=np.float32))
    X_raw = np.array(feats_raw[:len(y)], dtype=np.float32)[:, np.newaxis, :, :]

    tpls36, names36 = build_combined_templates(num_bins)
    d36 = dict(zip(names36, tpls36))
    feats36 = []
    for r in records:
        f = abl.compute_features_with_templates(r["f0"], num_bins, num_time, d36)
        feats36.append(f if f is not None
                       else np.zeros((num_time, len(names36), num_bins),
                                     dtype=np.float32))
    X_36 = np.array(feats36[:len(y)], dtype=np.float32).transpose(0, 2, 1, 3)
    print(f"  tpl8 {X_tpl8.shape}, raw {X_raw.shape}, comb36 {X_36.shape}")

    configs = [
        ("template8_ch32", X_tpl8, lambda: abl.TemplateNetDeep(
            nt8, num_time, num_bins, nc, 32, 2, 0.4), 5),
        ("random8_ch32", X_raw, lambda: abl.LearnedConvResNetV2(
            num_bins, num_time, nc, 8, 32, 2, 0.4), 5),
        ("combined36_ch64", X_36, lambda: InstrumentedResNet(
            len(names36), num_time, num_bins, nc, 64, 2, 0.4), 3),
    ]

    results = {}
    for tag, X, model_fn, n_seeds in configs:
        print(f"\n[{tag}] {n_seeds} seeds (old protocol):")
        rows = []
        for seed in range(n_seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            acc, preds, _ = abl.train_cv(X, y, g, model_fn,
                                         f"{tag} seed {seed}")
            oud = float((preds[g == "oud"] == y[g == "oud"]).mean())
            rows.append({"seed": seed, "acc": float(acc), "oud": oud})
        accs = [r["acc"] for r in rows]
        ouds = [r["oud"] for r in rows]
        results[tag] = {
            "runs": rows,
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "oud_mean": float(np.mean(ouds)), "oud_std": float(np.std(ouds)),
        }
        print(f"  [{tag}] acc {100*np.mean(accs):.1f} ± {100*np.std(accs):.1f} "
              f"| oud {100*np.mean(ouds):.1f} ± {100*np.std(ouds):.1f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT}")
    t8, r8 = results["template8_ch32"], results["random8_ch32"]
    print("\nSUMMARY (old protocol, mean ± std over seeds):")
    print(f"  template-init 8ch : {100*t8['acc_mean']:.1f} ± {100*t8['acc_std']:.1f} "
          f"(paper Table 4: 86.3, Table 2: 87.2)")
    print(f"  random-init 8ch   : {100*r8['acc_mean']:.1f} ± {100*r8['acc_std']:.1f} "
          f"(paper Table 4: 87.2)")
    c36 = results["combined36_ch64"]
    print(f"  combined 36 ch64  : {100*c36['acc_mean']:.1f} ± {100*c36['acc_std']:.1f} "
          f"(paper: 89.0)")


if __name__ == "__main__":
    main()
