# Data

This directory distributes **derived pitch features only — no audio**.

## arabic_oud/

The 104 solo oud taqasim come from commercial recordings; neither the
audio nor the time-resolved pitch contours are redistributed (a pitch
contour would allow reconstructing the melody). Instead we provide:

- `metadata.csv` — recording identifier (`<maqam>--<artist>--<album>--<track>`),
  maqam label, duration, and voiced-frame count for each of the 104 pieces.
- `histograms/<id>.npz` — **pitch distributions** per recording:
  - `pc_hist_240`: whole-piece octave-wrapped pitch-class histogram
    (240 bins, 5 cents/bin, reference 110 Hz);
  - `window_hist_wrapped`: 20 proportional-window octave-wrapped
    histograms (20 × 240);
  - `window_hist_unwrapped`: 40 proportional-window unwrapped histograms
    (40 × 240, 20 cents/bin over 0–4800 cents above 55 Hz).

These are the distribution-level inputs from which the paper's features
are built (template cross-correlations operate on window histograms).
Rerunning the oud-dependent experiments end-to-end requires regenerating
pYIN pitch tracks (hop 256 at 22.05 kHz) from the recordings listed in
`metadata.csv` and placing them under `arabic_oud/pitch/` as
single-column `.pitch` files; the Cairo Congress and Maqam-478
experiments reproduce fully from this repository alone.

## cairo_congress/

Pitch tracks (pickled numpy arrays under the original corpus paths) and
`allfiles_metadata.csv` (maqam labels and, for 56 recordings, annotated
tonic frequencies) for the Cairo Congress 1932 recordings, derived from
the ORD-CC32 corpus (Bozkurt et al.). The pickles here are reduced to the
pitch arrays used by the loaders.

## maqam478/pitch_cache/

478 pYIN pitch caches (`.npy`, same frame parameters as above) for the
Maqam-478 Quranic recitation dataset (Shahriar et al.). Filenames encode
the maqam label (`<Maqam>_<Maqam>_<NN>.npy`). The paper uses the 421
recordings in the 7 maqam families shared with the other datasets.

## diarmaqar/data/

`tuningSystems.json`, `maqamat.json`, and `ajnas.json` from the
[DiArMaqAr](https://github.com/Music-Intelligence-Lab/DiArMaqAr) archive
(Digital Arabic Maqam Archive), vendored here for convenience; see the
DiArMaqAr repository for the full archive, documentation, and license.
The paper uses the Ronzevalle 1904 tuning system.

## Note on audio

To rerun pitch extraction from audio (not required for reproduction),
place WAV files under `maqam478/<Maqam>/*.wav`; `load_maqam478()` in
`scripts/unified_arabic_analysis.py` will extract and cache pYIN pitch
tracks automatically (requires `librosa`).
