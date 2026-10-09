#!/usr/bin/env python3
"""
Unified Arabic Maqam Analysis — Combined datasets with interpretability.

Loads all 3 Arabic datasets (oud taqasim, Cairo Congress 1932, Maqam-478 Quranic),
trains a 2D CNN, generates Grad-CAM maps, runs K-means/HMM ajnas segmentation,
labels with DiArMaqAr, and produces publication figures.

Common maqams across datasets: bayat, rast, hijaz, nahawand, saba, (segah/sikah/seka).
Kurd: oud + maqam478 only.
"""

import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from collections import Counter, defaultdict
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_pitch_track(pitch_path):
    """Load a .pitch file (single column of F0 values in Hz)."""
    return np.loadtxt(pitch_path, dtype=np.float64)


def _normalize_maqam(name):
    """Normalize maqam name variants."""
    import unicodedata
    name = name.lower().strip()
    if "[" in name:
        name = name.split("[")[0].strip()
    for prefix in ("maqām ", "maqam "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = unicodedata.normalize("NFC", name)

    variants = {
        "bayyati": "bayat", "bayati": "bayat", "bayyat": "bayat",
        "sikah": "segah", "sika": "segah",
        "hijazkar": "hijaz", "hijaz kar": "hijaz",
    }
    for variant, canonical in variants.items():
        if variant in name:
            return canonical
    return name.split()[0] if " " in name else name

# ============================================================
# Constants
# ============================================================
PROJECT_DIR = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Maqam name normalization across datasets
MAQAM_MAP = {
    # Arabic oud names
    "bayat": "bayat", "bayati": "bayat",
    "hijaz": "hijaz", "hijazkar": "hijaz",
    "kurd": "kurd",
    "nahawand": "nahawand",
    "rast": "rast",
    "saba": "saba",
    "segah": "segah", "sikah": "segah", "seka": "segah",
    # Cairo Congress names
    "husayni": "husayni",
    "ushshaq": "ushshaq",
}

# 6 core maqams present in all 3 datasets (kurd only in 2)
CORE_MAQAMS = ["bayat", "hijaz", "nahawand", "rast", "saba", "segah"]
EXTENDED_MAQAMS = CORE_MAQAMS + ["kurd"]

MAQAM_COLORS = {
    "bayat": "#3B7DD8", "hijaz": "#E74C3C", "kurd": "#27AE60",
    "nahawand": "#F39C12", "rast": "#9B59B6", "saba": "#1ABC9C",
    "segah": "#E67E22",
}


# ============================================================
# Feature extraction
# ============================================================

# Precompute DiArMaqAr scale templates for tonic estimation
# Template construction (see README, "Known issue: template construction"). The paper's code resolves note
# names through the tuning system's one-octave list, which drops names from other octaves (e.g. husayni, awj,
# kurdan), so maqam templates keep 3 to 6 of their 7 degrees. Set MAQAMNET_FIXED_TEMPLATES=1 to resolve every
# note across octaves (scripts/fixed_templates.py). The default reproduces the published numbers.
FIXED_TEMPLATES = os.environ.get("MAQAMNET_FIXED_TEMPLATES") == "1"


def _build_tonic_templates():
    """Build maqam scale templates from DiArMaqAr for tonic estimation."""
    from scipy.ndimage import gaussian_filter1d as gf1d
    data_dir = PROJECT_DIR / "data" / "diarmaqar" / "data"
    try:
        with open(data_dir / "tuningSystems.json") as f:
            ts_list = json.load(f)
        with open(data_dir / "maqamat.json") as f:
            maqamat_data = json.load(f)
    except FileNotFoundError:
        return {}

    ts = next((t for t in ts_list if t["id"] == "ronzevalle_1904"), None)
    if ts is None:
        return {}

    note_names = ts["noteNames"][0]
    pcs = [float(p) for p in ts["tuningSystemPitchClasses"]]
    ref = pcs[0]
    note_to_cents = {}
    for name, pc in zip(note_names, pcs):
        if pc > 0:
            note_to_cents[name] = 1200 * np.log2(ref / pc)

    maqam_scales = {}
    for m in maqamat_data:
        name = m["idName"].replace("maqam_", "")
        asc = m.get("ascendingNoteNames", [])
        if not asc:
            continue
        tc = note_to_cents.get(asc[0], 0)
        degrees = sorted(set((note_to_cents[n] - tc) % 1200
                              for n in asc if n in note_to_cents))
        if degrees:
            maqam_scales[name] = degrees
    if FIXED_TEMPLATES:
        import fixed_templates as _ft
        for name in list(maqam_scales):
            try:
                maqam_scales[name] = _ft.degrees("maqam", name)
            except KeyError:
                pass

    our_to_dia = {
        "bayat": "bayyat", "hijaz": "hijaz", "kurd": "kurd",
        "nahawand": "nahawand", "rast": "rast", "saba": "saba",
        "segah": "segah", "ajam": "ajam_ushayran",
    }

    templates = {}
    num_bins = 120
    sigma = 10.0
    for our, dia in our_to_dia.items():
        if dia not in maqam_scales:
            continue
        tpl = np.zeros(num_bins)
        for deg in maqam_scales[dia]:
            center = deg * num_bins / 1200.0
            for b in range(num_bins):
                dist = min(abs(b - center), num_bins - abs(b - center))
                tpl[b] += np.exp(-0.5 * (dist * (1200.0 / num_bins) / sigma) ** 2)
        s = tpl.sum()
        if s > 0:
            tpl /= s
        templates[our] = tpl
    return templates


# Build once at import time
_TONIC_TEMPLATES = _build_tonic_templates()


def estimate_tonic(f0, maqam=None):
    """Estimate tonic from pitch track using DiArMaqAr template matching.

    If maqam is provided and a template exists, uses shift-and-compare:
    circularly shifts the pitch-class histogram across candidate tonics and
    picks the shift with highest dot-product against the maqam's theoretical
    scale template. Validated at 70% within 50 cents on Cairo Congress
    annotated data (vs 36% for the simple heuristic).

    Falls back to lowest-prominent-peak heuristic if no template available.
    """
    from scipy.ndimage import gaussian_filter1d

    f0_v = f0[(f0 > 0) & (~np.isnan(f0))]
    if len(f0_v) < 100:
        return 220.0

    f0_bounded = f0_v[(f0_v >= 60) & (f0_v <= 800)]
    if len(f0_bounded) < 100:
        return 220.0

    template = _TONIC_TEMPLATES.get(maqam) if maqam else None

    if template is not None:
        # --- Template-based estimation ---
        ref_hz = 110.0
        cents_from_ref = (1200.0 * np.log2(f0_bounded / ref_hz)) % 1200

        # Fine-grained pitch-class histogram (5-cent bins)
        pc_bins = 240
        pc_hist = np.zeros(pc_bins)
        for c in cents_from_ref:
            bf = c * pc_bins / 1200.0
            lo = int(np.floor(bf)) % pc_bins
            hi = (lo + 1) % pc_bins
            frac = bf - np.floor(bf)
            pc_hist[lo] += (1 - frac)
            pc_hist[hi] += frac
        pc_hist = gaussian_filter1d(pc_hist, sigma=2, mode="wrap")

        # Shift and score
        num_bins = len(template)  # 120
        best_score, best_pc = -1, 0
        ratio = pc_bins // num_bins  # 2

        for shift in range(pc_bins):
            shifted = np.roll(pc_hist, -shift)
            downsampled = shifted.reshape(num_bins, ratio).sum(axis=1)
            s = downsampled.sum()
            if s > 0:
                downsampled /= s
            score = np.dot(downsampled, template)
            if score > best_score:
                best_score = score
                best_pc = shift * 1200.0 / pc_bins

        # Convert pitch class to Hz and resolve octave
        tonic_pc_hz = 110.0 * 2 ** (best_pc / 1200)
        best_octave_hz = tonic_pc_hz
        best_energy = 0
        for octave_shift in [-1, 0, 1, 2]:
            candidate = tonic_pc_hz * (2 ** octave_shift)
            if candidate < 60 or candidate > 500:
                continue
            energy = np.sum(np.abs(1200 * np.log2(f0_bounded / candidate)) < 150)
            if energy > best_energy:
                best_energy = energy
                best_octave_hz = candidate
        return float(best_octave_hz)

    else:
        # --- Fallback: lowest prominent peak ---
        f0_b = f0_bounded[(f0_bounded >= 80) & (f0_bounded <= 600)]
        if len(f0_b) < 100:
            return 220.0
        cents = 1200.0 * np.log2(f0_b / 80.0)
        max_c = 1200 * np.log2(600 / 80.0)
        nb = int(max_c / 5)
        hist, edges = np.histogram(cents, bins=nb, range=(0, max_c))
        top5 = np.argsort(hist)[::-1][:5]
        return float(np.min(80.0 * 2 ** ((edges[top5] + 2.5) / 1200)))


def compute_histogram(f0, tonic, num_bins=120):
    """Octave-wrapped pitch histogram relative to tonic."""
    f0_v = f0[(f0 > 0) & (~np.isnan(f0))]
    if len(f0_v) < 100:
        return None
    cents = (1200.0 * np.log2(f0_v / tonic)) % 1200
    hist = np.zeros(num_bins, dtype=np.float32)
    for c in cents:
        bf = c * num_bins / 1200.0
        lo = int(np.floor(bf)) % num_bins
        hi = (lo + 1) % num_bins
        frac = bf - np.floor(bf)
        hist[lo] += (1 - frac)
        hist[hi] += frac
    s = hist.sum()
    return (hist / s) if s > 0 else None


def compute_pitch_time(f0, tonic, num_bins=60, num_time=20):
    """Pitch-time matrix: num_time windows x num_bins pitch bins."""
    f0_v = f0[(f0 > 0) & (~np.isnan(f0))]
    if len(f0_v) < num_time * 10:
        return None
    cents = (1200.0 * np.log2(f0_v / tonic)) % 1200
    n = len(cents)
    mat = np.zeros((num_time, num_bins), dtype=np.float32)
    for t in range(num_time):
        lo = int(t * n / num_time)
        hi = int((t + 1) * n / num_time)
        seg = cents[lo:hi]
        if len(seg) == 0:
            continue
        for c in seg:
            bf = c * num_bins / 1200.0
            lo_b = int(np.floor(bf)) % num_bins
            hi_b = (lo_b + 1) % num_bins
            frac = bf - np.floor(bf)
            mat[t, lo_b] += (1 - frac)
            mat[t, hi_b] += frac
        s = mat[t].sum()
        if s > 0:
            mat[t] /= s
    return mat


def compute_windowed_histograms(f0, tonic, window_sec=15.0, hop_sec=7.5,
                                sr=86.13, num_bins=60):
    """Sliding-window pitch histograms for temporal analysis.

    sr = frames per second (pYIN default: 22050/256 ~ 86.13 fps).
    Returns (windows, window_times) or (None, None).
    """
    f0_v = f0.copy()
    # Keep zeros (unvoiced) for timing but ignore in histogram
    n_frames = len(f0_v)
    window_frames = int(window_sec * sr)
    hop_frames = int(hop_sec * sr)

    if n_frames < window_frames:
        return None, None

    windows = []
    times = []
    pos = 0
    while pos + window_frames <= n_frames:
        chunk = f0_v[pos:pos + window_frames]
        voiced = chunk[(chunk > 0) & (~np.isnan(chunk))]
        if len(voiced) < 50:
            pos += hop_frames
            continue

        cents = (1200.0 * np.log2(voiced / tonic)) % 1200
        hist = np.zeros(num_bins, dtype=np.float32)
        for c in cents:
            bf = c * num_bins / 1200.0
            lo_b = int(np.floor(bf)) % num_bins
            hi_b = (lo_b + 1) % num_bins
            frac = bf - np.floor(bf)
            hist[lo_b] += (1 - frac)
            hist[hi_b] += frac
        s = hist.sum()
        if s > 0:
            hist /= s
        windows.append(hist)
        times.append((pos + window_frames // 2) / sr)  # center time in seconds
        pos += hop_frames

    if len(windows) < 3:
        return None, None
    return np.array(windows), np.array(times)


# ============================================================
# Dataset loaders
# ============================================================
def load_arabic_oud():
    """Load Arabic oud taqasim pitch tracks."""
    pitch_dir = PROJECT_DIR / "data" / "arabic_oud" / "pitch"
    if not pitch_dir.exists():
        print("  Arabic Oud: raw pitch tracks are not distributed with this "
              "release (see data/README.md); pitch distributions are provided "
              "in data/arabic_oud/histograms/. Oud-dependent experiments "
              "require regenerating pitch tracks from the recordings listed "
              "in data/arabic_oud/metadata.csv.")
        return []
    records = []
    for pf in sorted(pitch_dir.glob("*.pitch")):
        maqam_raw = pf.stem.split("--")[0]
        maqam = _normalize_maqam(maqam_raw)
        maqam = MAQAM_MAP.get(maqam, maqam)
        f0 = load_pitch_track(pf)
        if f0 is not None and len(f0) > 200:
            records.append({
                "f0": f0,
                "maqam": maqam,
                "dataset": "oud",
                "filename": pf.stem,
                "tonic_annotated": None,
            })
    return records


def load_cairo_congress():
    """Load Cairo Congress 1932 recordings."""
    data_dir = PROJECT_DIR / "data" / "cairo_congress"
    metadata = pd.read_csv(data_dir / "allfiles_metadata.csv")

    records = []
    for _, row in metadata.iterrows():
        mode = str(row.get("mode", "")).strip().lower()
        if mode in ("na", "", "nan"):
            continue

        # Normalize maqam name
        maqam = MAQAM_MAP.get(mode, mode)

        # Load pickle
        path = data_dir / f"{row['path']}.pickle"
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue

        # Get pitch track
        f0 = None
        for key in ["pitch_makam", "pitch_pyin"]:
            if key in data and data[key] is not None:
                f0 = np.asarray(data[key], dtype=np.float64)
                break
        if f0 is None or len(f0) < 200:
            continue

        # Get annotated tonic if available
        tonic_ann = None
        tonic_hz = row.get("tonic_Hz")
        if pd.notna(tonic_hz):
            try:
                tonic_ann = float(tonic_hz)
            except (ValueError, TypeError):
                pass

        records.append({
            "f0": f0,
            "maqam": maqam,
            "dataset": "cairo",
            "filename": row["path"],
            "tonic_annotated": tonic_ann,
        })
    return records


def load_maqam478():
    """Load Maqam-478 Quranic recitations with pitch caching.

    First run: extracts F0 via pYIN and saves to .npy cache files.
    Subsequent runs: loads from cache (~100x faster).
    """
    import librosa
    data_dir = PROJECT_DIR / "data" / "maqam478"
    cache_dir = data_dir / "pitch_cache"
    cache_dir.mkdir(exist_ok=True)

    audio_files = sorted(data_dir.rglob("*.wav"))
    if not audio_files:
        # Audio not distributed with this release: load cached pitch tracks.
        cache_files = sorted(cache_dir.glob("*.npy"))
        if not cache_files:
            print("  Maqam-478: no WAV files or cached pitch tracks found")
            return []
        print(f"  Maqam-478: loading {len(cache_files)} cached pitch tracks")
        records = []
        for cf in cache_files:
            parts = cf.stem.split("_", 1)
            maqam = MAQAM_MAP.get(parts[0].lower(), parts[0].lower())
            f0 = np.load(cf)
            if len(f0) < 200:
                continue
            records.append({"f0": f0, "maqam": maqam, "dataset": "maqam478",
                            "filename": parts[1] if len(parts) > 1 else cf.stem,
                            "tonic_annotated": None})
        return records

    # Check how many are cached
    cached = sum(1 for af in audio_files
                 if (cache_dir / f"{af.parent.name}_{af.stem}.npy").exists())
    if cached == len(audio_files):
        print(f"  Maqam-478: loading {cached} cached pitch tracks")
    else:
        print(f"  Maqam-478: {cached}/{len(audio_files)} cached, "
              f"extracting {len(audio_files) - cached} with pYIN...")

    records = []
    extracted = 0
    for i, af in enumerate(audio_files):
        maqam_raw = af.parent.name.lower()
        maqam = MAQAM_MAP.get(maqam_raw, maqam_raw)
        cache_path = cache_dir / f"{af.parent.name}_{af.stem}.npy"

        if cache_path.exists():
            f0 = np.load(cache_path)
        else:
            extracted += 1
            if extracted % 50 == 0:
                print(f"  Maqam-478: extracted [{extracted}]...")
            try:
                y, sr = librosa.load(af, sr=22050, mono=True)
                f0, _, _ = librosa.pyin(y, sr=sr, fmin=80, fmax=800,
                                         frame_length=2048, hop_length=256)
                f0 = np.where(np.isnan(f0), 0.0, f0)
                np.save(cache_path, f0)
            except Exception:
                continue

        if len(f0) < 200:
            continue

        records.append({
            "f0": f0,
            "maqam": maqam,
            "dataset": "maqam478",
            "filename": af.stem,
            "tonic_annotated": None,
        })
    if extracted > 0:
        print(f"  Maqam-478: extracted and cached {extracted} new pitch tracks")
    return records


def load_all_datasets(maqam_filter=None, skip_maqam478=False):
    """Load and combine all Arabic datasets.

    Returns list of record dicts with f0, maqam, dataset, tonic, features.
    """
    print("\n" + "=" * 80)
    print("LOADING ALL ARABIC DATASETS")
    print("=" * 80)

    print("\n[1/3] Arabic oud taqasim...")
    records = load_arabic_oud()
    print(f"  Loaded {len(records)} oud recordings")

    print("\n[2/3] Cairo Congress 1932...")
    cairo = load_cairo_congress()
    print(f"  Loaded {len(cairo)} Cairo recordings")
    records.extend(cairo)

    if not skip_maqam478:
        print("\n[3/3] Maqam-478 Quranic recitations...")
        m478 = load_maqam478()
        print(f"  Loaded {len(m478)} Maqam-478 recordings")
        records.extend(m478)
    else:
        print("\n[3/3] Skipping Maqam-478 (skip_maqam478=True)")

    # Filter to target maqams
    if maqam_filter:
        records = [r for r in records if r["maqam"] in maqam_filter]

    # Estimate tonics using DiArMaqAr template matching
    print("\nEstimating tonics (DiArMaqAr template matching)...")
    n_annotated = 0
    for r in records:
        if r["tonic_annotated"] is not None:
            r["tonic"] = r["tonic_annotated"]
            n_annotated += 1
        else:
            r["tonic"] = estimate_tonic(r["f0"], maqam=r["maqam"])
    print(f"  {n_annotated} annotated, {len(records) - n_annotated} estimated via templates")

    # Compute features
    print("Computing features...")
    valid_records = []
    for r in records:
        hist = compute_histogram(r["f0"], r["tonic"], num_bins=120)
        pt = compute_pitch_time(r["f0"], r["tonic"], num_bins=60, num_time=20)
        if hist is not None and pt is not None:
            r["histogram"] = hist
            r["pitch_time"] = pt
            valid_records.append(r)

    records = valid_records

    # Summary
    print(f"\nTotal valid recordings: {len(records)}")
    by_dataset = Counter(r["dataset"] for r in records)
    by_maqam = Counter(r["maqam"] for r in records)
    print(f"  By dataset: {dict(by_dataset)}")
    print(f"  By maqam:   {dict(sorted(by_maqam.items()))}")

    # Cross-table
    print(f"\n  {'':15s}", end="")
    datasets = sorted(by_dataset.keys())
    for d in datasets:
        print(f" {d:>8s}", end="")
    print(f" {'total':>8s}")
    for m in sorted(by_maqam.keys()):
        print(f"  {m:15s}", end="")
        for d in datasets:
            n = sum(1 for r in records if r["maqam"] == m and r["dataset"] == d)
            print(f" {n:8d}", end="")
        print(f" {by_maqam[m]:8d}")

    return records


# ============================================================
# DiArMaqAr theory integration
# ============================================================
def load_diarmaqar():
    """Load DiArMaqAr templates: jins → cents intervals."""
    data_dir = PROJECT_DIR / "data" / "diarmaqar" / "data"

    with open(data_dir / "tuningSystems.json") as f:
        tuning_systems = json.load(f)
    with open(data_dir / "maqamat.json") as f:
        maqamat = json.load(f)
    with open(data_dir / "ajnas.json") as f:
        ajnas = json.load(f)

    # Use Ronzevalle 1904 tuning system (modernist Arabic)
    ts = next((t for t in tuning_systems if t["id"] == "ronzevalle_1904"), None)
    if ts is None:
        return {}, {}, {}

    note_names = ts["noteNames"][0] if ts["noteNames"] else []
    pitch_classes = ts["tuningSystemPitchClasses"]

    # Convert to cents from reference
    pcs_float = []
    for pc in pitch_classes:
        try:
            pcs_float.append(float(pc))
        except (ValueError, TypeError):
            pcs_float.append(1.0)

    ref = pcs_float[0]
    note_to_cents = {}
    for name, pc_val in zip(note_names, pcs_float):
        if pc_val > 0:
            note_to_cents[name] = 1200 * np.log2(ref / pc_val)

    # Build jins templates: name → [cents from tonic]
    jins_templates = {}
    for j in ajnas:
        name = j["idName"].replace("jins_", "")
        notes = j.get("noteNames", [])
        if not notes:
            continue
        tonic_cents = note_to_cents.get(notes[0], 0)
        degrees = []
        for note in notes:
            if note in note_to_cents:
                degrees.append((note_to_cents[note] - tonic_cents) % 1200)
        if degrees:
            jins_templates[name] = sorted(set(degrees))
    if FIXED_TEMPLATES:
        import fixed_templates as _ft
        jins_templates = {name: _ft.degrees("jins", name) for name in jins_templates}

    # Build suyur descriptions
    suyur = {}
    for m in maqamat:
        name = m["idName"].replace("maqam_", "")
        suyur_list = m.get("suyur", [])
        if suyur_list:
            suyur[name] = suyur_list

    return jins_templates, note_to_cents, suyur


def jins_to_histogram(degrees, num_bins=60, sigma=12.0):
    """Convert jins interval list (cents) to Gaussian-smoothed histogram template."""
    template = np.zeros(num_bins, dtype=np.float32)
    for deg in degrees:
        bin_center = deg * num_bins / 1200.0
        for b in range(num_bins):
            dist = min(abs(b - bin_center), num_bins - abs(b - bin_center))
            template[b] += np.exp(-0.5 * (dist * (1200.0 / num_bins) / sigma) ** 2)
    s = template.sum()
    if s > 0:
        template /= s
    return template


# ============================================================
# Models
# ============================================================
class PitchTimeCNN2D_CAM(nn.Module):
    """2D CNN with Grad-CAM support for pitch-time matrices."""

    def __init__(self, num_bins=60, num_time=20, num_classes=7,
                 channels=(16, 32, 64), dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, channels[0], (3, 7), padding=(1, 0), padding_mode="circular"),
            nn.BatchNorm2d(channels[0]), nn.ReLU(),
            nn.MaxPool2d((1, 2)))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], (3, 7), padding=(1, 0),
                      padding_mode="circular"),
            nn.BatchNorm2d(channels[1]), nn.ReLU(),
            nn.MaxPool2d((2, 2)))
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], (3, 5), padding=(1, 0),
                      padding_mode="circular"),
            nn.BatchNorm2d(channels[2]), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(channels[2] * 4 * 4, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes))

        # Grad-CAM hooks
        self._features = None
        self._gradients = None

    def _save_features(self, module, input, output):
        self._features = output

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        features = x
        x = self.gap(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits, features

    def grad_cam(self, x, target_class=None):
        """Compute Grad-CAM activation map."""
        self.eval()
        x = x.unsqueeze(0) if x.dim() == 3 else x
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x.requires_grad_(True)
        logits, features = self.forward(x)
        features.retain_grad()

        if target_class is None:
            target_class = logits.argmax(1).item()

        logits[0, target_class].backward(retain_graph=True)
        grads = features.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


class FeatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================================================
# Training utilities
# ============================================================
def train_and_evaluate(X, y, groups, class_names, model_fn, tag="",
                       n_folds=5, epochs=200, batch_size=32, patience=30):
    """Stratified K-fold CV ensuring each fold has all datasets represented.

    Uses a composite stratification variable (maqam_dataset) so that each
    fold gets proportional samples from each maqam×dataset combination.
    """
    if groups is not None:
        # Create composite strat variable: "maqam_dataset"
        strat_labels = np.array([f"{yi}_{gi}" for yi, gi in zip(y, groups)])
        # Filter out strata with < n_folds samples (can't split them)
        strat_counts = Counter(strat_labels)
        for i in range(len(strat_labels)):
            if strat_counts[strat_labels[i]] < n_folds:
                strat_labels[i] = str(y[i])  # fall back to maqam-only
    else:
        strat_labels = y

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(splitter.split(X, strat_labels))

    all_preds = np.zeros_like(y)
    fold_accs = []
    best_model = None
    best_model_acc = 0

    for fold_idx, (tr, te) in enumerate(splits):
        Xt, yt = X[tr], y[tr]
        Xv, yv = X[te], y[te]
        tr_loader = DataLoader(FeatDataset(Xt, yt), batch_size=batch_size,
                               shuffle=True, drop_last=False)
        vl_loader = DataLoader(FeatDataset(Xv, yv), batch_size=batch_size)

        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc, best_preds, pat = 0, None, 0
        for epoch in range(epochs):
            model.train()
            for xb, yb in tr_loader:
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                correct, preds = 0, []
                for xb, yb in vl_loader:
                    out = model(xb)
                    logits = out[0] if isinstance(out, tuple) else out
                    p = logits.argmax(1)
                    correct += (p == yb).sum().item()
                    preds.extend(p.numpy())
                acc = correct / len(yv)
                if acc > best_acc:
                    best_acc = acc
                    best_preds = np.array(preds)
                    pat = 0
                    if acc > best_model_acc:
                        best_model_acc = acc
                        best_model = model
                else:
                    pat += 1
                if pat >= patience:
                    break

        all_preds[te] = best_preds
        fold_accs.append(best_acc)
        # Show dataset breakdown in fold
        te_datasets = groups[te] if groups is not None else None
        fold_str = f"  [{tag}] Fold {fold_idx+1}: {best_acc:.1%}"
        if te_datasets is not None:
            ds_counts = Counter(te_datasets)
            fold_str += f"  (test: {dict(ds_counts)})"
        print(fold_str)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"  [{tag}] Overall: {mean_acc:.1%} +/- {std_acc:.1%}")
    return mean_acc, std_acc, all_preds, best_model


# ============================================================
# Grad-CAM analysis
# ============================================================
def compute_gradcam_maps(model, X, y, class_names):
    """Compute per-class mean Grad-CAM activation maps."""
    model.eval()
    n_classes = len(class_names)
    cam_by_class = defaultdict(list)

    for i in range(len(X)):
        x_tensor = torch.FloatTensor(X[i]).unsqueeze(0).unsqueeze(0)
        cam = model.grad_cam(x_tensor, target_class=int(y[i]))
        cam_by_class[int(y[i])].append(cam)

    mean_cams = {}
    for cls_idx in range(n_classes):
        cams = cam_by_class.get(cls_idx, [])
        if cams:
            # Resize all to same shape and average
            shapes = [c.shape for c in cams]
            target_shape = max(shapes, key=lambda s: s[0] * s[1])
            resized = []
            for c in cams:
                if c.shape != target_shape:
                    from scipy.ndimage import zoom
                    factors = (target_shape[0] / c.shape[0],
                              target_shape[1] / c.shape[1])
                    c = zoom(c, factors, order=1)
                resized.append(c)
            mean_cams[cls_idx] = np.mean(resized, axis=0)

    return mean_cams


# ============================================================
# HMM segmentation
# ============================================================
def fit_hmm_segmentation(windowed_hists, n_states=8):
    """Fit Gaussian HMM to windowed histograms for ajnas segmentation.

    Uses hmmlearn if available, otherwise implements a simple Viterbi-like
    approach using K-means + transition counting.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
        use_hmmlearn = True
    except ImportError:
        use_hmmlearn = False

    if use_hmmlearn:
        # Concatenate all sequences
        all_obs = np.vstack(windowed_hists)
        lengths = [len(w) for w in windowed_hists]

        hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                          n_iter=100, random_state=42, verbose=False)
        try:
            hmm.fit(all_obs, lengths)
        except Exception as e:
            print(f"  HMM fit warning: {e}")

        # Decode each sequence
        state_seqs = []
        for seq in windowed_hists:
            try:
                states = hmm.predict(seq, [len(seq)])
                state_seqs.append(states)
            except Exception:
                # Fallback: assign each window to nearest HMM mean
                dists = np.linalg.norm(seq[:, None, :] - hmm.means_[None, :, :], axis=2)
                state_seqs.append(dists.argmin(axis=1))

        return hmm, state_seqs
    else:
        print("  hmmlearn not available, using K-means + transition matrix fallback")
        # Fallback: K-means clustering + empirical transition matrix
        all_obs = np.vstack(windowed_hists)
        kmeans = KMeans(n_clusters=n_states, n_init=10, random_state=42)
        all_labels = kmeans.fit_predict(all_obs)

        # Build transition matrix
        T = np.zeros((n_states, n_states))
        idx = 0
        state_seqs = []
        for seq in windowed_hists:
            n = len(seq)
            labels = all_labels[idx:idx + n]
            state_seqs.append(labels)
            for t in range(n - 1):
                T[labels[t], labels[t + 1]] += 1
            idx += n

        # Normalize
        row_sums = T.sum(axis=1, keepdims=True)
        T_norm = np.where(row_sums > 0, T / row_sums, 1.0 / n_states)

        return (kmeans, T_norm), state_seqs


# ============================================================
# Main analysis pipeline
# ============================================================
def main():
    print("=" * 80)
    print("UNIFIED ARABIC MAQAM ANALYSIS")
    print("=" * 80)

    # ----------------------------------------------------------
    # 1. Load all datasets
    # ----------------------------------------------------------
    records = load_all_datasets(maqam_filter=EXTENDED_MAQAMS)

    if len(records) == 0:
        print("No records loaded! Check data paths.")
        return

    # Build arrays
    class_to_idx = {m: i for i, m in enumerate(sorted(set(r["maqam"] for r in records)))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    n_classes = len(class_names)

    X_hist = np.array([r["histogram"] for r in records])
    X_pt = np.array([r["pitch_time"] for r in records])
    y = np.array([class_to_idx[r["maqam"]] for r in records])
    datasets = np.array([r["dataset"] for r in records])

    print(f"\nClasses ({n_classes}): {class_names}")

    # ----------------------------------------------------------
    # 2. Classification: 2D CNN on combined data
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 1: CLASSIFICATION ON COMBINED ARABIC DATA")
    print("=" * 80)

    # 2D CNN on pitch-time matrix
    print("\n[A] 2D CNN on pitch-time (20x60), stratified-group 5-fold CV:")
    acc_cnn, std_cnn, preds_cnn, best_cnn = train_and_evaluate(
        X_pt, y, datasets, class_names,
        model_fn=lambda: PitchTimeCNN2D_CAM(
            num_bins=60, num_time=20, num_classes=n_classes,
            channels=(16, 32, 64), dropout=0.3),
        tag="2D-CNN", n_folds=5, epochs=200, batch_size=32)

    # Per-maqam and per-dataset accuracy
    print("\nPer-maqam accuracy (2D CNN):")
    for cls_idx, name in enumerate(class_names):
        mask = y == cls_idx
        if mask.sum() > 0:
            acc = (preds_cnn[mask] == cls_idx).mean()
            print(f"  {name:12s} (n={mask.sum():3d}): {acc:.1%}")

    print("\nPer-dataset accuracy (2D CNN):")
    for ds in sorted(set(datasets)):
        mask = datasets == ds
        if mask.sum() > 0:
            acc = (preds_cnn[mask] == y[mask]).mean()
            print(f"  {ds:10s} (n={mask.sum():3d}): {acc:.1%}")

    # Confusion matrix
    cm = confusion_matrix(y, preds_cnn)
    print("\nConfusion matrix:")
    print(f"  {'':12s}", end="")
    for name in class_names:
        print(f" {name[:5]:>5s}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"  {name:12s}", end="")
        for j in range(n_classes):
            print(f" {cm[i,j]:5d}", end="")
        print()

    # ----------------------------------------------------------
    # 3. Grad-CAM interpretability
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 2: GRAD-CAM INTERPRETABILITY")
    print("=" * 80)

    # Retrain on full data for Grad-CAM (not CV — we want the best model)
    print("\nTraining full-data model for Grad-CAM...")
    full_model = PitchTimeCNN2D_CAM(num_bins=60, num_time=20,
                                     num_classes=n_classes,
                                     channels=(16, 32, 64), dropout=0.3)
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    full_loader = DataLoader(FeatDataset(X_pt, y), batch_size=32, shuffle=True)

    for epoch in range(100):
        full_model.train()
        for xb, yb in full_loader:
            out = full_model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            full_model.eval()
            with torch.no_grad():
                correct = 0
                for xb, yb in full_loader:
                    out = full_model(xb)
                    logits = out[0] if isinstance(out, tuple) else out
                    correct += (logits.argmax(1) == yb).sum().item()
            print(f"  Epoch {epoch+1}: train acc={correct/len(y):.1%}")

    # Compute Grad-CAM maps
    print("\nComputing Grad-CAM activation maps...")
    mean_cams = compute_gradcam_maps(full_model, X_pt, y, class_names)

    # ----------------------------------------------------------
    # 4. Sayr analysis: windowed clustering + HMM
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 3: SAYR ANALYSIS (WINDOWED CLUSTERING + HMM)")
    print("=" * 80)

    # Compute windowed histograms at 15s windows
    print("\nComputing 15s windowed histograms...")
    all_windowed = []
    all_window_times = []
    windowed_records = []

    for r in records:
        wins, times = compute_windowed_histograms(
            r["f0"], r["tonic"], window_sec=15.0, hop_sec=7.5, num_bins=60)
        if wins is not None:
            all_windowed.append(wins)
            all_window_times.append(times)
            windowed_records.append(r)

    print(f"  Pieces with windows: {len(all_windowed)}")
    total_windows = sum(len(w) for w in all_windowed)
    print(f"  Total windows: {total_windows}")

    # K-means clustering
    print("\nK-means clustering (K=8)...")
    all_windows_flat = np.vstack(all_windowed)
    kmeans = KMeans(n_clusters=8, n_init=20, random_state=42)
    all_labels = kmeans.fit_predict(all_windows_flat)
    sil = silhouette_score(all_windows_flat, all_labels,
                           sample_size=min(5000, len(all_labels)))
    print(f"  Silhouette score: {sil:.3f}")

    # Assign labels back to sequences
    idx = 0
    kmeans_seqs = []
    for wins in all_windowed:
        n = len(wins)
        kmeans_seqs.append(all_labels[idx:idx + n])
        idx += n

    # Self-transition rate
    self_trans = 0
    total_trans = 0
    for seq in kmeans_seqs:
        for t in range(len(seq) - 1):
            total_trans += 1
            if seq[t] == seq[t + 1]:
                self_trans += 1
    print(f"  Self-transition rate: {self_trans/total_trans:.1%}")

    # K-means transition matrix
    T_kmeans = np.zeros((8, 8))
    for seq in kmeans_seqs:
        for t in range(len(seq) - 1):
            T_kmeans[seq[t], seq[t + 1]] += 1
    row_sums = T_kmeans.sum(axis=1, keepdims=True)
    T_kmeans_norm = np.where(row_sums > 0, T_kmeans / row_sums, 0)

    # Code usage per maqam and dataset
    print("\nCode usage per maqam (K=8):")
    print(f"  {'Maqam':12s} {'Dataset':8s} |", end="")
    for k in range(8):
        print(f" C{k}", end="")
    print(" | Dom")
    print("  " + "-" * 75)

    code_usage = defaultdict(lambda: defaultdict(lambda: np.zeros(8)))
    for rec, seq in zip(windowed_records, kmeans_seqs):
        m = rec["maqam"]
        d = rec["dataset"]
        counts = np.bincount(seq, minlength=8).astype(float)
        code_usage[m][d] += counts

    for m in sorted(code_usage.keys()):
        for d in sorted(code_usage[m].keys()):
            counts = code_usage[m][d]
            total = counts.sum()
            if total > 0:
                fracs = counts / total
                dom_idx = np.argmax(fracs)
                print(f"  {m:12s} {d:8s} |", end="")
                for f in fracs:
                    print(f" {f:.2f}", end="")
                print(f" | C{dom_idx}({fracs[dom_idx]:.0%})")

    # HMM segmentation
    print("\nHMM segmentation (8 states)...")
    hmm_result, hmm_seqs = fit_hmm_segmentation(all_windowed, n_states=8)

    hmm_self_trans = 0
    hmm_total_trans = 0
    for seq in hmm_seqs:
        for t in range(len(seq) - 1):
            hmm_total_trans += 1
            if seq[t] == seq[t + 1]:
                hmm_self_trans += 1
    print(f"  HMM self-transition rate: {hmm_self_trans/hmm_total_trans:.1%}")

    # ----------------------------------------------------------
    # 5. DiArMaqAr labeling of discovered states
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 4: DiArMaqAr LABELING OF DISCOVERED STATES")
    print("=" * 80)

    jins_templates, note_cents, suyur = load_diarmaqar()
    print(f"  Loaded {len(jins_templates)} jins templates")

    # Convert jins templates to histograms
    jins_hist = {name: jins_to_histogram(degrees, num_bins=60)
                 for name, degrees in jins_templates.items()}

    # Score each K-means centroid against all jins templates
    print("\nK-means centroids vs DiArMaqAr jins:")
    print(f"  {'Code':6s} | {'Best jins':25s} | {'Score':6s} | {'2nd best':25s} | {'Score':6s}")
    print("  " + "-" * 80)

    centroid_labels = {}
    for k in range(8):
        centroid = kmeans.cluster_centers_[k]
        scores = {}
        for jname, jhist in jins_hist.items():
            scores[jname] = float(np.dot(centroid, jhist))
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        centroid_labels[k] = ranked[0][0]
        print(f"  C{k:4d} | {ranked[0][0]:25s} | {ranked[0][1]:.4f} | "
              f"{ranked[1][0]:25s} | {ranked[1][1]:.4f}")

    # Score HMM states too (if hmmlearn was used)
    if hasattr(hmm_result, 'means_'):
        print("\nHMM states vs DiArMaqAr jins:")
        hmm_labels = {}
        for s in range(8):
            state_hist = hmm_result.means_[s]
            scores = {}
            for jname, jhist in jins_hist.items():
                scores[jname] = float(np.dot(state_hist, jhist))
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            hmm_labels[s] = ranked[0][0]
            print(f"  S{s:4d} | {ranked[0][0]:25s} | {ranked[0][1]:.4f}")

    # ----------------------------------------------------------
    # 6. Generate figures
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 5: GENERATING FIGURES")
    print("=" * 80)

    # --- Figure A: Classification results bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Per-maqam accuracy
    ax = axes[0]
    maqam_accs = {}
    for cls_idx, name in enumerate(class_names):
        mask = y == cls_idx
        if mask.sum() > 0:
            maqam_accs[name] = (preds_cnn[mask] == cls_idx).mean()
    bars = ax.barh(list(maqam_accs.keys()), list(maqam_accs.values()),
                   color=[MAQAM_COLORS.get(m, "#888") for m in maqam_accs.keys()])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Per-maqam accuracy\n(2D CNN, combined, {acc_cnn:.1%} overall)")
    ax.axvline(1.0 / n_classes, color="gray", ls="--", alpha=0.5, label="chance")

    # Per-dataset accuracy
    ax = axes[1]
    ds_accs = {}
    for ds in sorted(set(datasets)):
        mask = datasets == ds
        ds_accs[ds] = (preds_cnn[mask] == y[mask]).mean()
    ax.barh(list(ds_accs.keys()), list(ds_accs.values()), color=["#3498db", "#e74c3c", "#2ecc71"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-dataset accuracy")

    # Confusion matrix
    ax = axes[2]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (normalized)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Combined Arabic Dataset Classification", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_unified_classification.png", dpi=150,
                bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_unified_classification.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_unified_classification")

    # --- Figure B: Grad-CAM per maqam ---
    n_show = min(n_classes, 7)
    fig, axes = plt.subplots(n_show, 3, figsize=(14, 3 * n_show))

    for i, cls_idx in enumerate(range(n_show)):
        name = class_names[cls_idx]
        color = MAQAM_COLORS.get(name, "#888")

        # Mean input
        mask = y == cls_idx
        mean_input = X_pt[mask].mean(axis=0)
        axes[i, 0].imshow(mean_input.T, aspect="auto", origin="lower", cmap="Blues")
        axes[i, 0].set_ylabel(name, fontsize=11, fontweight="bold", color=color)
        if i == 0:
            axes[i, 0].set_title("Mean pitch-time")

        # Grad-CAM
        if cls_idx in mean_cams:
            cam = mean_cams[cls_idx]
            axes[i, 1].imshow(cam.T if cam.ndim == 2 else cam,
                             aspect="auto", origin="lower", cmap="hot")
        if i == 0:
            axes[i, 1].set_title("Grad-CAM activation")

        # Time marginal of activation
        if cls_idx in mean_cams:
            cam = mean_cams[cls_idx]
            time_marginal = cam.mean(axis=-1) if cam.ndim == 2 else cam.mean()
            if hasattr(time_marginal, '__len__'):
                axes[i, 2].barh(range(len(time_marginal)), time_marginal,
                               color=color, alpha=0.7)
                axes[i, 2].set_ylim(-0.5, len(time_marginal) - 0.5)
        if i == 0:
            axes[i, 2].set_title("Time activation")

    plt.suptitle("2D CNN Grad-CAM: What the model attends to per maqam\n"
                 "(combined Arabic datasets)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_unified_gradcam.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_unified_gradcam.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_unified_gradcam")

    # --- Figure C: K-means codebook + transition + timeline ---
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Codebook
    ax = fig.add_subplot(gs[0, 0])
    x_cents = np.linspace(0, 1200, 60, endpoint=False)
    for k in range(8):
        label = f"C{k} ({centroid_labels.get(k, '?')})"
        ax.plot(x_cents, kmeans.cluster_centers_[k], label=label, alpha=0.8, lw=1.5)
    ax.set_xlabel("Cents from tonic")
    ax.set_ylabel("Density")
    ax.set_title("K-means codebook (K=8, 15s windows)")
    ax.legend(fontsize=7, ncol=2)

    # Transition matrix
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(T_kmeans_norm, cmap="Blues", vmin=0, vmax=0.5)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title(f"Transition matrix\n(self-trans={self_trans/total_trans:.0%})")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Code usage per maqam (heatmap)
    ax = fig.add_subplot(gs[0, 2])
    maqam_list = sorted(code_usage.keys())
    usage_matrix = np.zeros((len(maqam_list), 8))
    for i, m in enumerate(maqam_list):
        total_counts = np.zeros(8)
        for d in code_usage[m]:
            total_counts += code_usage[m][d]
        total = total_counts.sum()
        if total > 0:
            usage_matrix[i] = total_counts / total
    im = ax.imshow(usage_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"C{k}" for k in range(8)])
    ax.set_yticks(range(len(maqam_list)))
    ax.set_yticklabels(maqam_list)
    ax.set_title("Code usage per maqam")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Ajnas timeline (3 example pieces)
    ax = fig.add_subplot(gs[1, :])
    # Pick one piece per maqam with enough windows
    example_pieces = []
    seen_maqams = set()
    for rec, seq, times in zip(windowed_records, kmeans_seqs, all_window_times):
        m = rec["maqam"]
        if m not in seen_maqams and len(seq) >= 5:
            example_pieces.append((rec, seq, times))
            seen_maqams.add(m)
        if len(example_pieces) >= 7:
            break

    cmap = plt.cm.Set2
    for i, (rec, seq, times) in enumerate(example_pieces):
        for t_idx in range(len(seq)):
            ax.barh(i, times[min(t_idx + 1, len(times) - 1)] - times[t_idx]
                    if t_idx < len(times) - 1 else 5,
                    left=times[t_idx] - times[0],
                    color=cmap(seq[t_idx] / 8), edgecolor="none", height=0.8)
    ax.set_yticks(range(len(example_pieces)))
    ax.set_yticklabels([f"{r['maqam']} ({r['dataset']})" for r, _, _ in example_pieces],
                       fontsize=9)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Ajnas segmentation timeline (K-means codes, 15s windows)")

    # Color legend for codes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap(k / 8),
                             label=f"C{k}: {centroid_labels.get(k, '?')}")
                       for k in range(8)]
    ax.legend(handles=legend_elements, loc="upper right", ncol=4, fontsize=7)

    plt.suptitle("Unsupervised Ajnas Discovery on Combined Arabic Data", fontsize=14)
    plt.savefig(FIGURES_DIR / "fig_unified_ajnas.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_unified_ajnas.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_unified_ajnas")

    # --- Figure D: Cross-dataset consistency ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Oud vs others correlation per maqam
    ax = axes[0]
    maqam_corrs = defaultdict(dict)
    for m in sorted(code_usage.keys()):
        if "oud" in code_usage[m]:
            oud_counts = code_usage[m]["oud"]
            oud_total = oud_counts.sum()
            if oud_total == 0:
                continue
            oud_frac = oud_counts / oud_total
            for d in ["cairo", "maqam478"]:
                if d in code_usage[m]:
                    d_counts = code_usage[m][d]
                    d_total = d_counts.sum()
                    if d_total > 0:
                        d_frac = d_counts / d_total
                        corr = np.corrcoef(oud_frac, d_frac)[0, 1]
                        maqam_corrs[m][d] = corr

    maqam_order = sorted(maqam_corrs.keys())
    x_pos = np.arange(len(maqam_order))
    width = 0.35
    for i, d in enumerate(["cairo", "maqam478"]):
        corrs = [maqam_corrs[m].get(d, 0) for m in maqam_order]
        colors = [MAQAM_COLORS.get(m, "#888") for m in maqam_order]
        ax.bar(x_pos + i * width, corrs, width, label=f"oud vs {d}",
               alpha=0.7 + 0.2 * i)

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(maqam_order, rotation=45, ha="right")
    ax.set_ylabel("Pearson correlation")
    ax.set_title("Cross-dataset code usage correlation")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", alpha=0.3)

    # Per-dataset code usage heatmap
    ax = axes[1]
    ds_names = sorted(set(datasets))
    ds_usage = np.zeros((len(ds_names), 8))
    for i, d in enumerate(ds_names):
        total = np.zeros(8)
        for m in code_usage:
            if d in code_usage[m]:
                total += code_usage[m][d]
        s = total.sum()
        if s > 0:
            ds_usage[i] = total / s
    im = ax.imshow(ds_usage, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"C{k}" for k in range(8)])
    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(ds_names)
    ax.set_title("Code usage by dataset")
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Cross-Dataset Ajnas Consistency", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_unified_cross_dataset.png", dpi=150,
                bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_unified_cross_dataset.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_unified_cross_dataset")

    # --- Figure E: Sayr trajectories in t-SNE ---
    print("\nComputing t-SNE on windowed embeddings...")
    from sklearn.manifold import TSNE

    # Use K-means labels + original histograms for t-SNE
    # Subsample for speed
    max_tsne = 8000
    if len(all_windows_flat) > max_tsne:
        rng = np.random.RandomState(42)
        tsne_idx = rng.choice(len(all_windows_flat), max_tsne, replace=False)
        tsne_data = all_windows_flat[tsne_idx]
    else:
        tsne_idx = np.arange(len(all_windows_flat))
        tsne_data = all_windows_flat

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(tsne_data)

    # Map back to sequences for trajectory plotting
    # Build reverse index: flat_idx → (piece_idx, window_idx)
    flat_to_piece = []
    idx = 0
    for pi, wins in enumerate(all_windowed):
        for wi in range(len(wins)):
            flat_to_piece.append((pi, wi))
            idx += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT: all trajectories colored by maqam
    ax = axes[0]
    # Plot each piece's trajectory
    piece_trajs = defaultdict(list)
    for flat_i, tsne_i in enumerate(tsne_idx):
        pi, wi = flat_to_piece[tsne_i]
        piece_trajs[pi].append((wi, emb_2d[flat_i]))

    for pi, points in piece_trajs.items():
        if len(points) < 3:
            continue
        points.sort(key=lambda x: x[0])
        coords = np.array([p[1] for p in points])
        m = windowed_records[pi]["maqam"]
        color = MAQAM_COLORS.get(m, "#888")
        ax.plot(coords[:, 0], coords[:, 1], color=color, alpha=0.2, lw=0.8)

    # Mean trajectories per maqam
    maqam_trajectories = defaultdict(list)
    for pi, points in piece_trajs.items():
        m = windowed_records[pi]["maqam"]
        points.sort(key=lambda x: x[0])
        coords = np.array([p[1] for p in points])
        maqam_trajectories[m].append(coords)

    for maqam, trajs in sorted(maqam_trajectories.items()):
        color = MAQAM_COLORS.get(maqam, "#888")
        # Interpolate to common length for averaging
        n_interp = 20
        interped = []
        for traj in trajs:
            if len(traj) >= 3:
                from scipy.interpolate import interp1d
                t_orig = np.linspace(0, 1, len(traj))
                t_new = np.linspace(0, 1, n_interp)
                f_x = interp1d(t_orig, traj[:, 0], kind="linear")
                f_y = interp1d(t_orig, traj[:, 1], kind="linear")
                interped.append(np.column_stack([f_x(t_new), f_y(t_new)]))
        if interped:
            mean_traj = np.mean(interped, axis=0)
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=color, lw=3, alpha=0.9)
            ax.plot(mean_traj[0, 0], mean_traj[0, 1], "o", color=color, ms=10,
                    markeredgecolor="black", markeredgewidth=0.8)
            ax.plot(mean_traj[-1, 0], mean_traj[-1, 1], "s", color=color, ms=10,
                    markeredgecolor="black", markeredgewidth=0.8)
            mid = mean_traj[n_interp // 2]
            ax.text(mid[0], mid[1], maqam, fontsize=9, ha="center", color=color,
                    fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor=color, alpha=0.85,
                              boxstyle="round,pad=0.3"))

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Mean sayr trajectory per maqam\n(circle=start, square=end)")
    ax.grid(True, alpha=0.3)

    # RIGHT: colored by K-means code
    ax = axes[1]
    kmeans_labels_tsne = all_labels[tsne_idx]
    cmap = plt.cm.Set2
    for k in range(8):
        mask = kmeans_labels_tsne == k
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap(k / 8)], alpha=0.3, s=10,
                   label=f"C{k}: {centroid_labels.get(k, '?')}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Windows colored by K-means ajnas code")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Sayr Trajectories in t-SNE Space (combined Arabic data, 15s windows)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_unified_tsne_sayr.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_unified_tsne_sayr.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_unified_tsne_sayr")

    # --- Figure F: DiArMaqAr theory overlay ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axes_flat = axes.flatten()

    for k in range(8):
        ax = axes_flat[k]
        centroid = kmeans.cluster_centers_[k]
        x_cents = np.linspace(0, 1200, 60, endpoint=False)

        # Plot centroid
        ax.fill_between(x_cents, centroid, alpha=0.3, color="steelblue")
        ax.plot(x_cents, centroid, color="steelblue", lw=2, label="K-means centroid")

        # Overlay best-matching jins template
        best_jins = centroid_labels.get(k, "")
        if best_jins in jins_templates:
            degrees = jins_templates[best_jins]
            for deg in degrees:
                ax.axvline(deg, color="red", ls="--", alpha=0.7, lw=1)
                ax.text(deg, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.1,
                        f"{deg:.0f}c", fontsize=7, color="red", ha="center",
                        rotation=90, va="top")

        ax.set_title(f"C{k}: {best_jins}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Cents from tonic", fontsize=8)
        if k % 4 == 0:
            ax.set_ylabel("Density", fontsize=8)

    plt.suptitle("K-Means Centroids vs DiArMaqAr Jins Templates\n"
                 "(blue = discovered, red lines = theoretical intervals)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig_unified_theory_overlay.png", dpi=150,
                bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "fig_unified_theory_overlay.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig_unified_theory_overlay")

    # ----------------------------------------------------------
    # Save summary
    # ----------------------------------------------------------
    summary = {
        "total_recordings": len(records),
        "n_classes": n_classes,
        "class_names": class_names,
        "cnn_accuracy": float(acc_cnn),
        "cnn_std": float(std_cnn),
        "kmeans_silhouette": float(sil),
        "kmeans_self_transition": float(self_trans / total_trans),
        "centroid_labels": centroid_labels,
        "per_dataset_accuracy": {ds: float((preds_cnn[datasets == ds] == y[datasets == ds]).mean())
                                 for ds in sorted(set(datasets))},
        "per_maqam_accuracy": {class_names[i]: float((preds_cnn[y == i] == i).mean())
                               for i in range(n_classes) if (y == i).sum() > 0},
    }

    with open(RESULTS_DIR / "unified_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results/unified_analysis.json")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nKey results:")
    print(f"  Combined 2D CNN accuracy: {acc_cnn:.1%} +/- {std_cnn:.1%}")
    print(f"  K-means silhouette (15s): {sil:.3f}")
    print(f"  Self-transition rate:     {self_trans/total_trans:.1%}")
    print(f"  Ajnas labeled with DiArMaqAr: {len(centroid_labels)}/8 centroids")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
