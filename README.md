# MaqamNet — Musicologically Interpretable Arabic Maqam Identification

Code and derived data for the paper:

> J. Bakarji, K. Allami, M. El Asal, and D. Abou Jaoude,
> **"Musicologically Interpretable Deep Learning Framework for Arabic Maqam
> Identification and Analysis"**, in *Proc. of the 27th Int. Society for
> Music Information Retrieval Conf. (ISMIR)*, Abu Dhabi, UAE, 2026.

MaqamNet bypasses explicit tonic estimation by encoding the tonic as a
latent shift variable: raw octave-wrapped pitch histograms are circularly
cross-correlated with maqam and jins templates from the
[DiArMaqAr](https://github.com/Music-Intelligence-Lab/DiArMaqAr) archive,
and a ResNet with circular padding classifies the resulting response
tensor end-to-end.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is sufficient. All experiments run on CPU (no GPU required);
approximate runtimes below are for a recent laptop CPU.

## Data

The `data/` directory contains everything needed to reproduce the paper —
**derived pitch tracks only, no audio**:

| Dataset | Contents here | Source |
|---|---|---|
| Arabic Oud Taqasim (104) | recording list + per-piece **pitch distributions** (whole-piece and windowed histograms); no audio or pitch contours | commercial recordings |
| Cairo Congress 1932 (94 used) | pitch tracks (pickles) + metadata with annotated tonics | ORD-CC32 corpus (Bozkurt et al.) |
| Maqam-478 (421 used) | pYIN pitch caches (`.npy`) | Quranic recitation dataset (Shahriar et al.) |
| DiArMaqAr | tuning systems, maqamat, ajnas (JSON) | [Music-Intelligence-Lab/DiArMaqAr](https://github.com/Music-Intelligence-Lab/DiArMaqAr) |

See `data/README.md` for details and licensing notes. The unified corpus
after restricting to the 7 maqam families shared across datasets is
**619 recordings** (104 oud + 94 Cairo + 421 Maqam-478). Cairo Congress
and Maqam-478 experiments reproduce fully from this repository alone;
oud-dependent rows additionally require regenerating pYIN pitch tracks
from the recordings listed in `data/arabic_oud/metadata.csv` (the provided
oud histograms are the distribution-level inputs the features are built
from). All scripts use identical stratified 5-fold splits (fixed seed).

## Reproducing the paper

| Paper result | Command | Runtime |
|---|---|---|
| Table 1 (tonic-noise sensitivity) + Table 2 "Raw xcorr argmax" row | `python scripts/tonic_noise_and_xcorr.py` | ~15 min |
| Table 2 oracle rows (RF, 2D CNN) | `python scripts/unified_arabic_analysis.py` | ~1 h |
| Table 2 "MaqamNet (8 tpls)" + Table 4 ablation rows | `python scripts/template_conv_ablation.py` | ~1.5 h |
| Table 2 "MaqamNet (36 tpls)" (ch64 sweep) | `python scripts/template_conv_deep_tune.py` | ~3 h |
| Table 2 unwrapped-pitch baseline (per-subset) | `python scripts/unwrapped_pitch.py` | ~1 h |
| Table 4 seed means (±std) | `python scripts/seed_variance_ablation.py` | ~2 h |
| Sec. 5.3 validation-based protocol + Maqam-478-only + separability stats | `python scripts/protocol_validation.py` | ~1 h |
| Discussion: absolute-pitch standardization analysis | `python scripts/m478_standardization_check.py` | ~2 min |
| Figure 2 (t-SNE of four representations) | `python figure-code/fig_tsne_representations.py` | ~45 min |
| Figure 3 (Grad-CAM / saliency grid) | `python figure-code/fig_gradcam_combined.py` | ~1 h |

Outputs are written to `results/` and `figures/`. Neural results vary by
±1 point (overall) across seeds; Table 4 quantifies this
(`seed_variance_ablation.py`).

## Model and training configuration

**MaqamNet (best configuration, "MaqamNet (36 tpls)" in Table 2):**

- **Input**: `(N, 36, 20, 30)` — 36 template channels (8 maqam + 28 jins
  DiArMaqAr templates, Ronzevalle 1904 tuning), 20 proportional temporal
  windows, 30 pitch-class bins (40 cents/bin). Each channel is the
  circular cross-correlation of the window's raw octave-wrapped pitch
  histogram with one template.
- **Architecture** (~716K parameters): circular-padded conv stem
  (5×5, 36→64) with BatchNorm/ReLU; 2 residual blocks (64→128) with
  circular padding on the pitch axis, BatchNorm, dropout, MaxPool2d(2);
  AdaptiveAvgPool2d((2,4)); FC 128 → 7 classes.
- **Optimization**: Adam, lr 1e-3, weight decay 1e-4, batch size 32,
  up to 250 epochs, `ReduceLROnPlateau(patience=10, factor=0.5)` on the
  training loss, early-stopping patience 40, dropout 0.4.
- **Evaluation protocols**: (a) *headline tables*: stratified 5-fold CV
  (composite maqam-dataset stratification, seed 42), best test-fold epoch;
  (b) *validation-based* (Sec. 5.3): identical folds, 15% of each training
  fold held out for validation, early stopping on validation only, test
  fold evaluated once at the best-validation epoch.

The unwrapped-pitch baseline CNN operates on `(N, 40, 240)` pitch-time
matrices (0–4800 cents above 55 Hz, no octave wrapping, no tonic
information), channels (32, 64, 128), dropout 0.3, same optimizer.

## Citation

```bibtex
@inproceedings{bakarji2026maqamnet,
  author    = {Bakarji, Joseph and Allami, Khyam and El Asal, Mohammad and Abou Jaoude, Dany},
  title     = {Musicologically Interpretable Deep Learning Framework for {A}rabic Maqam Identification and Analysis},
  booktitle = {Proc. of the Int. Society for Music Information Retrieval Conf.},
  year      = {2026}
}
```

For the DiArMaqAr archive, please also cite:

```bibtex
@inproceedings{allami2026diarmaqar,
  author    = {Allami, Khyam and El Asal, Mohammad and El Khansa, Lina and Bakarji, Joseph},
  title     = {Culture-Specific Computational Musicology: A Case Study of {DiArMaqAr} (Digital {A}rabic Maqam Archive)},
  booktitle = {Proc. of the Int. Society for Music Information Retrieval Conf.},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The derived pitch
data and DiArMaqAr JSON files carry the terms described in
`data/README.md`.
