#!/usr/bin/env python3
"""Correct maqam and jins templates from DiArMaqAr (post-ISMIR fix, Oct 2026).

The templates used in the ISMIR 2026 paper (unified_arabic_analysis._build_tonic_templates and
template_conv_ablation.load_diarmaqar) look note names up in the tuning system's one-octave list
noteNames[0]. Names from other octaves (ḥusaynī, awj, kurdān, ʿajam, muḥayyar, nīm ḥusaynī, qarār ḥiṣār, ...)
are skipped without warning, so maqam templates keep 3 to 6 of their 7 degrees and 7 of the 28 jins
templates lose notes.

Here every note name is resolved with DiArMaqAr's own octave tables (src/models/NoteName.ts, vendored as data/diarmaqar/data/noteNameOctaves.json):
name -> (octave, slot); a tuning-system pitch class with the same slot gives the cents, shifted by
1200 x the octave difference. Unresolvable names raise instead of being dropped.

API
  degrees(kind, name, tuning="ronzevalle_1904") -> sorted pitch-class degrees (cents above the first note)
  build(kind, num_bins=30, tuning=..., sigma_cents=10.0) -> {name: template (sum-normalised)}
     kind = "maqam" | "jins"; same Gaussian smoothing as the paper code.
  audit() -> prints old vs fixed degree counts
"""
import json, math, re
from functools import lru_cache
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
D = PROJECT / "data" / "diarmaqar"


@lru_cache(None)
def _octave_tables():
    raw = json.load(open(D / "data/noteNameOctaves.json", encoding="utf-8"))["tables"]
    tables = [raw[f"octave{w}"] for w in ["Zero", "One", "Two", "Three", "Four"]]
    where = {}
    for o, names in enumerate(tables):
        for i, full in enumerate(names):
            for alias in [full] + full.split("/"):
                where.setdefault(alias.strip(), (o, i))
    return tables, where


@lru_cache(None)
def _tuning(tuning_id):
    ts = next(t for t in json.load(open(D / "data/tuningSystems.json")) if t["id"] == tuning_id)
    pcs = ts["tuningSystemPitchClasses"]
    if any("/" in p for p in pcs):
        r = [eval(p) for p in pcs]; cents = [1200 * math.log2(x / r[0]) for x in r]
    else:
        v = [float(p) for p in pcs]
        cents = [1200 * math.log2(v[0] / x) for x in v] if v[0] > v[-1] else [x - v[0] for x in v]
    _, where = _octave_tables()
    slot_cents = {}                                     # slot -> (octave, cents) for this tuning's pitch classes
    for name, c in zip(ts["noteNames"][0], cents):
        for alias in [name] + name.split("/"):
            if alias.strip() in where:
                o, i = where[alias.strip()]; slot_cents.setdefault(i, (o, c)); break
    return slot_cents


FALLBACKS = set()


def note_cents(name, tuning="ronzevalle_1904"):
    _, where = _octave_tables()
    for alias in [name] + name.split("/"):
        if alias.strip() in where:
            o, i = where[alias.strip()]; break
    else:
        raise KeyError(f"note name not in DiArMaqAr octave tables: {name}")
    sc = _tuning(tuning)
    if i not in sc:   # this tuning has no pitch class named for the slot: use the nearest named slot, and report it
        tables, _ = _octave_tables(); n = len(tables[1])
        j = min(sc, key=lambda k: (min(abs(k - i), n - abs(k - i)), k > i))   # ties go to the lower slot ('nīm' = half-flat)
        FALLBACKS.add((name, tuning, tables[1][j] if j < len(tables[1]) else j))
        o0, c = sc[j]; return c + 1200 * (o - o0) + (1200 / n) * 0 * (i - j)
    o0, c = sc[i]
    return c + 1200 * (o - o0)


@lru_cache(None)
def _defs(kind):
    f = "maqamat.json" if kind == "maqam" else "ajnas.json"
    out = {}
    for x in json.load(open(D / "data" / f)):
        key = x["idName"].replace("maqam_", "").replace("jins_", "")
        out[key] = x["ascendingNoteNames"] if kind == "maqam" else x["noteNames"]
    return out


def degrees(kind, name, tuning="ronzevalle_1904"):
    notes = _defs(kind)[name]
    c = [note_cents(n, tuning) for n in notes]
    return sorted({round((x - c[0]) % 1200, 1) for x in c})


def _gauss(deg, num_bins, sigma_cents):
    b = np.arange(num_bins); t = np.zeros(num_bins); width = 1200.0 / num_bins
    for d in deg:
        ctr = d / width; dist = np.minimum(np.abs(b - ctr), num_bins - np.abs(b - ctr))
        t += np.exp(-0.5 * (dist * width / sigma_cents) ** 2)
    return t / t.sum()


# the paper's 8 maqam template keys -> DiArMaqAr ids
PAPER_MAQAMS = {"ajam": "ajam_ushayran", "bayat": "bayyat", "hijaz": "hijaz", "kurd": "kurd",
                "nahawand": "nahawand", "rast": "rast", "saba": "saba", "segah": "segah"}


def build(kind, num_bins=30, tuning="ronzevalle_1904", sigma_cents=10.0, names=None):
    if kind == "maqam":
        names = names or PAPER_MAQAMS
        return {k: _gauss(degrees("maqam", v, tuning), num_bins, sigma_cents) for k, v in names.items()}
    names = names or list(_defs("jins"))
    return {k: _gauss(degrees("jins", k, tuning), num_bins, sigma_cents) for k in names}


def audit():
    print("maqam degrees (cents above the tonic):")
    for k, v in PAPER_MAQAMS.items():
        print(f"  {k:9s} {len(degrees('maqam', v))}  {degrees('maqam', v)}")
    counts = sorted({len(degrees("jins", k)) for k in _defs("jins")})
    print("jins degree counts:", counts)
    print("nearest-slot substitutions:", sorted(FALLBACKS) or "none")
    assert all(len(degrees("maqam", v)) == 7 for v in PAPER_MAQAMS.values()) and counts[0] >= 3 and counts[-1] <= 5


if __name__ == "__main__":
    audit()
