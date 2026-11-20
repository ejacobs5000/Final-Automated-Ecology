#!/usr/bin/env python3
"""
Rainforest recording denoiser  ->  cleaner bird audio for BirdNET fine-tuning.

Idea: noise (insects, cicadas, rain) is steady over the clip; bird calls are brief
spikes. So per file, per frequency, estimate the "quiet floor" and turn down cells
near it while leaving loud bird syllables alone. No frequency band is hard-notched.
One parameter set adapts to all conditions (quiet / tonal insects / cicada wall /
broadband rain) because the noise template is derived from each file itself.

Pipeline:
  A   transient clamp  - remove broadband "knocks" (only runs if the file has them)
  HP  high-pass 120 Hz - remove <500 Hz rain rumble / wind / thumps
  C   stationary-noise suppression (core) - per-bin noise floor + Wiener gain +
      frequency-dependent gain floor (gentle in the bird band, aggressive elsewhere)
  D   normalize to a fixed RMS (measured before C) + soft-limit -> no clipping
  +   per-file usability flag: bird-band dynamic range too low -> "low confidence"
      (bird buried under rain; nothing recoverable)

Usage:
    python denoise_pipeline.py --input_dir raw_flac --output_dir cleaned \
        --report_csv report.csv

Deps: numpy, scipy, soundfile
"""

import argparse
import csv
import glob
import os
import time
from pathlib import Path

import numpy as np
from scipy.signal import stft, istft, butter, sosfiltfilt
from scipy.ndimage import median_filter, maximum_filter1d, uniform_filter


try:
    import soundfile as sf
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Denoising requires SoundFile. Install the package dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc


# ----------------------------- parameters -----------------------------

STFT_NPERSEG = 2048           # ~23.4 Hz/bin @ 48 kHz
STFT_HOP = 512                # 75% overlap
STFT_WINDOW = "hann"

# A - transient clamp (self-triggered)
KNOCK_GATE = 2                # run the clamp only if >= this many strong knocks are found
DECLICK_SHORT_MS = 3.0        # short window: instantaneous loudness envelope
DECLICK_BASELINE_MS = 300.0   # long window: "normal" loudness baseline
DECLICK_ALLOW_RATIO = 2.2     # allow instantaneous loudness up to this x baseline
DECLICK_SMOOTH_MS = 1.5       # smooth the gain curve
DECLICK_DECIM = 64            # decimation factor for the fast long-window median

# HP - high-pass
HP_CUTOFF_HZ = 120.0
HP_ORDER = 2

# C - stationary-noise suppression (Wiener gain: ~untouched for high-SNR bird
#     syllables, pushed to the floor for cells near the steady noise level)
NOISE_PERCENTILE = 40.0       # per bin, this time-axis percentile = the noise floor N(f)
NOISE_OVEREST = 1.0           # noise overestimate factor (>1 = more aggressive)
GAIN_KNEE = 1.0              # G = xi / (xi + KNEE); 1.0 = standard Wiener
GAIN_SMOOTH = (3, 1)          # gain-mask smoothing (freq bins, time frames); no time
                             # smoothing so syllable onsets/offsets are not eroded

# Gain floor is frequency-dependent: high (gentle) inside the bird band, deep
# (aggressive) elsewhere. Insects/cicadas/rain sit at different frequencies than
# the birds, so this is both aggressive and safe.
GAIN_FLOOR = 0.09            # outside the bird band  (~ -21 dB)
GAIN_FLOOR_BIRD = 0.32       # inside the bird band   (~ -10 dB)
BIRD_PROTECT_HZ = (1000.0, 3800.0)
BIRD_PROTECT_RAMP_HZ = 300.0

# D - normalization
TARGET_RMS_DBFS = -33.0       # target RMS in the 1-8 kHz band
NORM_BAND = (1000.0, 8000.0)
GAIN_LIMIT_DB = 6.0           # max make-up gain (don't lift quiet files' noise floor)
GAIN_FLOOR_DB = -30.0         # max attenuation allowed by normalization
SOFT_LIMIT_PEAK = 0.99

# usability flag
USABILITY_BIRD_BAND = (1500.0, 4500.0)
USABILITY_MIN_DR_DB = 4.0     # bird-band p90/p10 below this -> low confidence


# ----------------------------- helpers -----------------------------

def load_mono(path):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float64), sr


def _stft(x, sr):
    return stft(x, fs=sr, nperseg=STFT_NPERSEG, noverlap=STFT_NPERSEG - STFT_HOP,
                window=STFT_WINDOW, boundary="zeros", padded=True)


def _istft(Z, sr, n):
    _, y = istft(Z, fs=sr, nperseg=STFT_NPERSEG, noverlap=STFT_NPERSEG - STFT_HOP,
                 window=STFT_WINDOW)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


def band_rms(x, sr, lo, hi, order=4):
    """Time-domain RMS after band-pass (measurement only, not in the signal path)."""
    nyq = sr / 2
    hi = min(hi, nyq - 1)
    sos = butter(order, [lo, hi], btype="bandpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, x)
    return float(np.sqrt(np.mean(y ** 2)))


# ----------------------------- A - transient clamp -----------------------------

def detect_broadband_knocks(audio, sr):
    """Find short, near-full-band impulses. Used only to decide whether to run the
    clamp; does not need to be precise. Returns (all events, strong events)."""
    f, t, Z = stft(audio, fs=sr, nperseg=1024, noverlap=768, window="hann")
    mag = np.abs(Z)
    bb = (f >= 500) & (f <= 18000)
    hop = 256
    fe = np.sqrt((mag[bb] ** 2).mean(axis=0))
    base = median_filter(fe, size=201)                       # ~1.3 s local median
    ratio = fe / (base + 1e-12)
    binref = np.median(mag[bb], axis=1, keepdims=True) * (10 ** (6 / 20))
    occ = (mag[bb] > binref).mean(axis=0)                    # fraction of bins lit up
    hot = (ratio > 4.0) & (occ > 0.5)

    events, i = [], 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j < len(hot) and hot[j]:
                j += 1
            dur_ms = (j - i) * hop / sr * 1000
            if dur_ms <= 60:
                events.append((float(t[i]), float(ratio[i:j].max())))
            i = j
        else:
            i += 1
    strong = [e for e in events if e[1] > 6.0]
    return events, strong


def transient_clamp(audio, sr):
    """Short-window envelope vs long-window baseline; pull down the gain wherever the
    envelope clearly exceeds the baseline. The long-window baseline is computed on a
    decimated envelope for speed."""
    n = len(audio)
    short_win = max(1, int(sr * DECLICK_SHORT_MS / 1000))
    env = maximum_filter1d(np.abs(audio), size=short_win)

    d = DECLICK_DECIM
    env_d = env[::d]
    base_win = max(1, int((sr * DECLICK_BASELINE_MS / 1000) / d) | 1)
    base_d = median_filter(env_d, size=base_win)
    base = np.interp(np.arange(n), np.arange(len(base_d)) * d, base_d)

    allowed = base * DECLICK_ALLOW_RATIO + 1e-6
    gain = np.minimum(1.0, allowed / (env + 1e-9))
    sw = max(1, int(sr * DECLICK_SMOOTH_MS / 1000))
    gain = np.convolve(gain, np.ones(sw) / sw, mode="same")
    return audio * gain


# ----------------------------- HP - high-pass -----------------------------

def highpass(audio, sr):
    sos = butter(HP_ORDER, HP_CUTOFF_HZ, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio)


# ----------------------------- C - stationary-noise suppression -----------------------------

def spectral_suppress(audio, sr):
    """Per-bin noise floor N(f) = time-axis percentile of |STFT| (brief birds barely
    move it). Wiener gain G = xi/(xi+KNEE), xi = max(|X|^2/N^2 - 1, 0): high SNR ->
    G~1 (untouched), near the floor -> G->0 (clamped to the frequency-dependent
    floor). Then smooth the gain mask and resynthesize with the original phase."""
    f, t, Z = _stft(audio, sr)
    mag = np.abs(Z)
    N = NOISE_OVEREST * np.percentile(mag, NOISE_PERCENTILE, axis=1, keepdims=True)

    snr_post = (mag ** 2) / (N ** 2 + 1e-20)
    xi = np.maximum(snr_post - 1.0, 0.0)
    gain = xi / (xi + GAIN_KNEE)

    lo, hi = BIRD_PROTECT_HZ
    r = BIRD_PROTECT_RAMP_HZ
    protect = np.clip((f - (lo - r)) / r, 0, 1) * np.clip(((hi + r) - f) / r, 0, 1)
    floor_vec = (GAIN_FLOOR + protect * (GAIN_FLOOR_BIRD - GAIN_FLOOR))[:, None]
    gain = np.maximum(gain, floor_vec)
    gain = np.minimum(gain, 1.0)
    if GAIN_SMOOTH:
        gain = uniform_filter(gain, size=GAIN_SMOOTH, mode="nearest")

    Zs = Z * gain
    y = _istft(Zs, sr, len(audio))

    # measurements for the report
    magS = np.abs(Zs)
    before_db = 20 * np.log10(mag.mean(axis=1) + 1e-12)
    after_db = 20 * np.log10(magS.mean(axis=1) + 1e-12)
    red = {}
    for name, (blo, bhi) in [("low_150_1k", (150, 1000)), ("bird_1_3k", (1000, 3000)),
                             ("mid_3.7_6k", (3700, 6000)), ("insect_6_8.5k", (6000, 8500)),
                             ("high_9_16k", (9000, 16000))]:
        m = (f >= blo) & (f <= bhi)
        red[name] = float((before_db[m] - after_db[m]).mean()) if m.any() else 0.0

    bm = (f >= 1000) & (f <= 3000)
    e0 = np.sqrt((mag[bm] ** 2).mean(axis=0))
    e1 = np.sqrt((magS[bm] ** 2).mean(axis=0))
    red["bird_syllable_change_db"] = float(
        20 * np.log10((np.percentile(e0, 92) + 1e-12) / (np.percentile(e1, 92) + 1e-12)))
    return y, red


# ----------------------------- D - normalization -----------------------------

def normalize_fixed_rms(audio_post, ref_pre, sr):
    """Reference level is measured on the pre-C signal, so denoising does not change
    overall loudness. Scale to a fixed 1-8 kHz RMS target with bounded make-up gain,
    then soft-limit so no sample clips."""
    ref = band_rms(ref_pre, sr, *NORM_BAND)
    if ref < 1e-9:
        return audio_post, 0.0, 0.0
    g = (10 ** (TARGET_RMS_DBFS / 20)) / ref
    g = min(g, 10 ** (GAIN_LIMIT_DB / 20))
    g = max(g, 10 ** (GAIN_FLOOR_DB / 20))
    out = audio_post * g

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    limited = 0.0
    if peak > SOFT_LIMIT_PEAK:
        limited = 20 * np.log10(peak / SOFT_LIMIT_PEAK)
        out = out * (SOFT_LIMIT_PEAK / peak)
    return out, 20 * np.log10(g), limited


# ----------------------------- usability flag -----------------------------

def bird_usability(audio, sr):
    f, t, Z = _stft(audio, sr)
    m = (f >= USABILITY_BIRD_BAND[0]) & (f <= USABILITY_BIRD_BAND[1])
    fe = np.sqrt((np.abs(Z[m]) ** 2).mean(axis=0))
    p90, p10 = np.percentile(fe, [90, 10])
    dr = float(20 * np.log10(p90 / (p10 + 1e-12)))
    return dr, dr >= USABILITY_MIN_DR_DB


# ----------------------------- per-file processing -----------------------------

def process_file(path, output_dir):
    t0 = time.time()
    fname = os.path.basename(path)
    raw, sr = load_mono(path)

    # A - self-triggered transient clamp
    events, strong = detect_broadband_knocks(raw, sr)
    declick_applied = len(strong) >= KNOCK_GATE
    x = transient_clamp(raw, sr) if declick_applied else raw.copy()

    # HP
    x = highpass(x, sr)
    pre_c = x.copy()

    # C - stationary-noise suppression
    y, red = spectral_suppress(x, sr)

    # D - normalization
    out, makeup_db, limited_db = normalize_fixed_rms(y, pre_c, sr)

    # usability is measured on the RAW input (does a recoverable bird signal exist?)
    dr_raw, usable = bird_usability(raw, sr)
    dr_out, _ = bird_usability(out, sr)

    out_path = os.path.join(output_dir, fname.replace(".flac", "_v4.flac"))
    sf.write(out_path, out.astype(np.float32), sr, format="FLAC")

    clipped = float(np.mean(np.abs(out) >= 0.999)) * 100

    return {
        "filename": fname,
        "knocks_total": len(events),
        "knocks_strong": len(strong),
        "declick_applied": int(declick_applied),
        "reduce_low_db": round(red["low_150_1k"], 2),
        "reduce_birdband_bg_db": round(red["bird_1_3k"], 2),
        "bird_syllable_change_db": round(red["bird_syllable_change_db"], 2),
        "reduce_mid_db": round(red["mid_3.7_6k"], 2),
        "reduce_insect_db": round(red["insect_6_8.5k"], 2),
        "reduce_high_db": round(red["high_9_16k"], 2),
        "norm_makeup_db": round(makeup_db, 2),
        "soft_limited_db": round(limited_db, 2),
        "out_rms_dbfs": round(20 * np.log10(np.sqrt(np.mean(out ** 2)) + 1e-12), 2),
        "out_peak": round(float(np.max(np.abs(out))), 4),
        "clipped_pct": round(clipped, 4),
        "birdDR_raw_db": round(dr_raw, 2),
        "birdDR_out_db": round(dr_out, 2),
        "usable": int(usable),
        "proc_s": round(time.time() - t0, 2),
        "output_file": os.path.basename(out_path),
    }


# ----------------------------- main -----------------------------

FIELDNAMES = ["filename", "knocks_total", "knocks_strong", "declick_applied",
              "reduce_low_db", "reduce_birdband_bg_db", "bird_syllable_change_db",
              "reduce_mid_db", "reduce_insect_db", "reduce_high_db", "norm_makeup_db",
              "soft_limited_db", "out_rms_dbfs", "out_peak", "clipped_pct",
              "birdDR_raw_db", "birdDR_out_db", "usable", "proc_s", "output_file"]


def main():
    ap = argparse.ArgumentParser(description="Rainforest recording denoiser")
    ap.add_argument("--input_dir", required=True, help="folder of input .flac files")
    ap.add_argument("--output_dir", required=True, help="folder for cleaned .flac files")
    ap.add_argument("--report_csv", default="report.csv", help="per-file metrics CSV")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.flac")))
    if not files:
        print(f"[warn] no .flac files in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(path)}")
        try:
            r = process_file(path, args.output_dir)
            rows.append(r)
            print(f"    knocks {r['knocks_strong']}/{r['knocks_total']} "
                  f"declick={'Y' if r['declick_applied'] else 'n'} | "
                  f"reduce mid/insect/high = {r['reduce_mid_db']}/"
                  f"{r['reduce_insect_db']}/{r['reduce_high_db']} dB | "
                  f"birdband bg -{r['reduce_birdband_bg_db']} syll d{r['bird_syllable_change_db']} dB | "
                  f"makeup={r['norm_makeup_db']} dB | clip%={r['clipped_pct']} | "
                  f"birdDR raw->out {r['birdDR_raw_db']}->{r['birdDR_out_db']} "
                  f"{'[usable]' if r['usable'] else '[low confidence]'} | {r['proc_s']}s")
        except Exception as e:
            print(f"    [FAILED] {e}", file=sys.stderr)
            rows.append({"filename": os.path.basename(path), "output_file": f"ERROR: {e}"})

    with open(args.report_csv, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nreport -> {args.report_csv}")

    ok = [r for r in rows if "reduce_mid_db" in r]
    if ok:
        print(f"\n===== summary ({len(ok)}/{len(files)}) =====")
        for k, lbl in [("reduce_mid_db", "reduce mid   "), ("reduce_insect_db", "reduce insect"),
                       ("reduce_high_db", "reduce high  "), ("reduce_birdband_bg_db", "birdband bg  "),
                       ("bird_syllable_change_db", "syllable d   "),
                       ("clipped_pct", "clipped %    "), ("proc_s", "proc s       ")]:
            v = np.array([r[k] for r in ok], float)
            print(f"  {lbl}  mean={v.mean():7.2f}  median={np.median(v):7.2f}  "
                  f"min={v.min():7.2f}  max={v.max():7.2f}")
        low = [r["filename"] for r in ok if not r["usable"]]
        print(f"  low-confidence files: {len(low)}/{len(ok)} -> {low}")


if __name__ == "__main__":
    main()
