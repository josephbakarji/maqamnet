#!/usr/bin/env python3
"""
Publication figure: Combined Grad-CAM comparison.

Left column: Oracle-tonic CNN Grad-CAM (pitch axis = cents from tonic)
Right column: Template-conv ResNet STEM Grad-CAM (pitch axis = shift bins ~ candidate tonic)

7 rows = 7 maqams. Colorbar. Proper axis labels. PDF output.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
})

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_arabic_analysis import (
    load_arabic_oud, load_cairo_congress, load_maqam478,
    estimate_tonic, compute_histogram, compute_pitch_time,
    _TONIC_TEMPLATES, EXTENDED_MAQAMS, MAQAM_COLORS,
    PitchTimeCNN2D_CAM, FeatDataset
)
from template_conv_v2 import build_templates, compute_features, FlatDataset
from template_conv_tune import CircConv2d, ResBlock
from template_conv_ablation import build_ajnas_templates, compute_features_with_templates
from template_conv_deep_tune import InstrumentedResNet

PROJECT_DIR = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
REF_HZ = 110.0


def main():
    print("=" * 70)
    print("COMBINED GRAD-CAM FIGURE")
    print("=" * 70)

    # Load data
    print("\nLoading...")
    records = []
    for loader, name in [(load_arabic_oud, "oud"),
                          (load_cairo_congress, "cairo"),
                          (load_maqam478, "maqam478")]:
        recs = loader()
        records.extend(recs)
    records = [r for r in records if r["maqam"] in EXTENDED_MAQAMS]

    class_names = sorted(set(r["maqam"] for r in records))
    maqam_to_idx = {m: i for i, m in enumerate(class_names)}
    nc = len(class_names)

    # Compute oracle tonics
    for r in records:
        if r.get("tonic_annotated"):
            r["tonic"] = r["tonic_annotated"]
        else:
            r["tonic"] = estimate_tonic(r["f0"], maqam=r["maqam"])

    # ── Prepare oracle-tonic CNN data (20×60 pitch-time) ──
    print("\nBuilding oracle-tonic features...")
    oracle_feats, oracle_labels = [], []
    oracle_records = []
    for r in records:
        pt = compute_pitch_time(r["f0"], r["tonic"], num_time=20, num_bins=60)
        if pt is not None:
            oracle_feats.append(pt)
            oracle_labels.append(maqam_to_idx[r["maqam"]])
            oracle_records.append(r)
    X_oracle = np.array(oracle_feats, dtype=np.float32)[:, np.newaxis, :, :]
    y_oracle = np.array(oracle_labels)
    print(f"  Oracle data: {X_oracle.shape}")

    # ── Prepare template-conv data (36 channels) ──
    print("Building template-conv features...")
    maqam_tpls, maqam_names = build_templates(30)
    ajnas_dict = build_ajnas_templates(30)
    combined_dict = {}
    for i, name in enumerate(maqam_names):
        combined_dict[f"m_{name}"] = maqam_tpls[i]
    for name, tpl in sorted(ajnas_dict.items()):
        combined_dict[f"j_{name}"] = tpl
    n_tpls = len(combined_dict)

    tc_feats, tc_labels = [], []
    for r in records:
        f = compute_features_with_templates(r["f0"], 30, 20, combined_dict)
        if f is not None:
            tc_feats.append(f)
            tc_labels.append(maqam_to_idx[r["maqam"]])
    X_tc = np.array(tc_feats, dtype=np.float32).transpose(0, 2, 1, 3)
    y_tc = np.array(tc_labels)
    print(f"  Template-conv data: {X_tc.shape}")

    # ── Train oracle CNN ──
    print("\nTraining oracle-tonic CNN...")
    cnn = PitchTimeCNN2D_CAM(num_bins=60, num_time=20, num_classes=nc)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(FeatDataset(X_oracle, y_oracle), batch_size=32, shuffle=True)
    for ep in range(150):
        cnn.train()
        for xb, yb in loader:
            logits, _ = cnn(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 50 == 0:
            cnn.eval()
            with torch.no_grad():
                logits, _ = cnn(torch.FloatTensor(X_oracle))
                acc = (logits.argmax(1).numpy() == y_oracle).mean()
            print(f"  Epoch {ep+1}: {acc:.1%}")

    # ── Train template-conv ResNet ──
    print("\nTraining template-conv ResNet...")
    resnet = InstrumentedResNet(n_tpls, 20, 30, nc, 64, 2, 0.4)
    opt2 = torch.optim.Adam(resnet.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=10, factor=0.5)
    loader2 = DataLoader(FlatDataset(X_tc, y_tc), batch_size=32, shuffle=True)
    for ep in range(200):
        resnet.train()
        el = 0
        for xb, yb in loader2:
            loss = crit(resnet(xb), yb)
            opt2.zero_grad(); loss.backward(); opt2.step()
            el += loss.item()
        sched.step(el)
        if (ep+1) % 50 == 0:
            resnet.eval()
            with torch.no_grad():
                acc = (resnet(torch.FloatTensor(X_tc)).argmax(1).numpy() == y_tc).mean()
            print(f"  Epoch {ep+1}: {acc:.1%}")

    # ── Compute Grad-CAMs ──
    print("\nComputing Grad-CAMs...")
    cnn.eval()
    resnet.eval()

    # Per-class mean Grad-CAM
    oracle_cams = {}  # class_idx -> mean cam (H, W) in pitch-time space
    stem_cams = {}    # class_idx -> mean cam (H, W) in stem space

    for ci, maq in enumerate(class_names):
        # Oracle CNN
        mask = y_oracle == ci
        indices = np.where(mask)[0]
        if len(indices) > 40:
            indices = np.random.RandomState(42).choice(indices, 40, replace=False)

        cams = []
        for idx in indices:
            x = torch.FloatTensor(X_oracle[idx])
            cam = cnn.grad_cam(x, target_class=ci)
            if cam is not None and cam.ndim == 2:
                cams.append(cam)
        if cams:
            oracle_cams[ci] = np.mean(cams, axis=0)

        # Template-conv ResNet stem
        mask2 = y_tc == ci
        indices2 = np.where(mask2)[0]
        if len(indices2) > 40:
            indices2 = np.random.RandomState(42).choice(indices2, 40, replace=False)

        stem_list = []
        for idx in indices2:
            x = torch.FloatTensor(X_tc[idx]).unsqueeze(0)
            try:
                cam = resnet.grad_cam_at_block(x, -1, target_class=ci)  # -1 = stem
                if cam is not None and cam.ndim == 2:
                    stem_list.append(cam)
            except:
                pass
        if stem_list:
            stem_cams[ci] = np.mean(stem_list, axis=0)

        print(f"  {maq}: oracle={len(cams)} cams, stem={len(stem_list)} cams")

    # ── Build figure ──
    print("\nGenerating figure...")

    fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(nc, 3, width_ratios=[1, 1, 0.05],
                           hspace=0.3, wspace=0.25)

    # Shared colormap
    cmap = 'inferno'

    for ci, maq in enumerate(class_names):
        color = MAQAM_COLORS.get(maq, 'black')

        # Left: Oracle CNN Grad-CAM
        ax_oracle = fig.add_subplot(gs[ci, 0])
        if ci in oracle_cams:
            cam = oracle_cams[ci]
            if cam.max() > 0:
                cam = cam / cam.max()
            im1 = ax_oracle.imshow(cam, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                                    origin='lower',
                                    extent=[0, 1200, 0, 20])
            ax_oracle.set_xlim(0, 1200)
        if ci == 0:
            ax_oracle.set_title("Oracle-tonic CNN\n(pitch in cents from tonic)",
                               fontsize=9, fontweight='bold')
        ax_oracle.set_ylabel(maq, fontsize=10, fontweight='bold', color=color,
                            rotation=0, labelpad=55, va='center')
        if ci == nc - 1:
            ax_oracle.set_xlabel("Pitch (cents from tonic)", fontsize=8)
            ax_oracle.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
        else:
            ax_oracle.set_xticklabels([])
        ax_oracle.set_yticks([0, 5, 10, 15, 20])
        if ci == nc // 2:
            ax_oracle.set_ylabel(maq + "\n\nTime window", fontsize=9,
                                fontweight='bold', color=color,
                                rotation=0, labelpad=55, va='center')

        # Right: Template-conv ResNet stem
        ax_stem = fig.add_subplot(gs[ci, 1])
        if ci in stem_cams:
            cam = stem_cams[ci]
            if cam.max() > 0:
                cam = cam / cam.max()
            im2 = ax_stem.imshow(cam, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                                  origin='lower',
                                  extent=[0, 1200, 0, 20])
            ax_stem.set_xlim(0, 1200)
        if ci == 0:
            ax_stem.set_title("Template-conv ResNet (stem)\n(shift = candidate tonic)",
                             fontsize=9, fontweight='bold')
        if ci == nc - 1:
            ax_stem.set_xlabel("Pitch shift (cents)", fontsize=8)
            ax_stem.set_xticks([0, 200, 400, 600, 800, 1000, 1200])
        else:
            ax_stem.set_xticklabels([])
        ax_stem.set_yticks([0, 5, 10, 15, 20])
        ax_stem.set_yticklabels([])

    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im2, cax=cbar_ax)
    cb.set_label("Normalized Grad-CAM activation", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle("Grad-CAM: What the Model Attends To", fontsize=12,
                fontweight='bold', y=0.98)

    for fmt in ["pdf", "png"]:
        fname = FIGURES_DIR / f"fig_gradcam_combined.{fmt}"
        fig.savefig(fname, bbox_inches='tight', dpi=300 if fmt == "pdf" else 150)
    print(f"  Saved fig_gradcam_combined.pdf/.png")
    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
