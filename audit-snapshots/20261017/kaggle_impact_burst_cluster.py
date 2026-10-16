#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Impact-by-Impact Acoustic DSP Analysis — BURST-CLUSTERING Redesign (Kaggle cell)
═══════════════════════════════════════════════════════════════════════════════

APPROVED DESIGN (Option A):
- Explicit BURST CLUSTERING (NOT a min-distance re-tune).
- Detect fine envelope peaks, then group all peaks within a 50 ms merge window
  into ONE physical event (collapses internal ringing/resonance/oscillations).
- Final event timestamp = STRONGEST (maximum-envelope) peak in each cluster.
- Ambiguity/QC flags for borderline or multiple-strong-peak clusters.
- A/B/C/D are anonymous experimental labels. No ML. Raw WAV = source of truth.
- Segment window (research spec): 0.5 s pre + event center + 0.5 s post = 1.0 s.

Detection preprocessing (faithful to reference backend/manual_stop_interactive_recorder(1).py):
  bandpass 1-20 kHz (4th-order Butterworth, filtfilt) -> Hilbert envelope
  -> 1 ms boxcar smoothing -> normalize.

CRITICAL RULES:
- Load WAVs at NATIVE sample rate (preserve high-frequency content)
- Impact #1..#N LOCAL to each file, chronological by event timestamp
- Segment_Duration_s = extraction window (1.0 s here); Duration_s = DSP-derived physical duration (NaN if unreliable)
- Per-file error handling: one bad WAV must not stop the batch
- Reproducibility: print all parameters before processing
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import soundfile as sf

# ═══════════════════════════════════════════════════════════════════════════
# DETECTION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# --- Reference preprocessing (detection envelope) ---
IMPACT_DETECT_HP_HZ = 1000.0
IMPACT_DETECT_LP_HZ = 20000.0
IMPACT_ENVELOPE_SMOOTH_MS = 1.0

# --- Fine peak finding (captures ALL candidate peaks incl. ringing) ---
FINE_MIN_DISTANCE_MS = 3.0      # close enough to see intra-burst ringing
FINE_PROMINENCE = 0.01
FINE_HEIGHT = 0.02

# --- Burst clustering (the redesign) ---
BURST_MERGE_MS = 50.0            # peaks within 50 ms -> one physical event (APPROVED Option A)
# --- Ambiguity thresholds ---
BORDERLINE_GAP_MIN_MS = 60.0     # inter-event gap in [60,100] ms -> BORDERLINE
BORDERLINE_GAP_MAX_MS = 100.0
MULTI_STRONG_REL = 0.30          # >=2 peaks within 30% of cluster max -> MULTI_STRONG

# --- Segment window (research spec: 0.5 s pre + center + 0.5 s post = 1.0 s) ---
SEG_PRE_S = 0.5
SEG_POST_S = 0.5

# --- DSP / FFT ---
N_FFT_PARAM = 8192
HOP_LENGTH_PARAM = 256

MIN_SAMPLES_FOR_DETECTION = 32

print("═" * 80)
print("BURST-CLUSTERING IMPACT DETECTION — PARAMETERS")
print("═" * 80)
print(f"Reference preprocessing:")
print(f"  Bandpass              = {IMPACT_DETECT_HP_HZ}-{IMPACT_DETECT_LP_HZ} Hz (4th-order Butterworth, filtfilt)")
print(f"  Envelope              = Hilbert, {IMPACT_ENVELOPE_SMOOTH_MS} ms boxcar smooth, normalized")
print(f"Fine peak finding:")
print(f"  Min distance          = {FINE_MIN_DISTANCE_MS} ms | prominence={FINE_PROMINENCE} | height={FINE_HEIGHT}")
print(f"Burst clustering (redesign):")
print(f"  BURST_MERGE_MS        = {BURST_MERGE_MS} ms  (peaks within window -> one event)")
print(f"  Event timestamp       = strongest (max-envelope) peak in cluster")
print(f"Ambiguity flags:")
print(f"  BORDERLINE gap        = [{BORDERLINE_GAP_MIN_MS},{BORDERLINE_GAP_MAX_MS}] ms")
print(f"  MULTI_STRONG rel      = {MULTI_STRONG_REL}")
print(f"Segment window (research spec):")
print(f"  PRE/POST              = {SEG_PRE_S} s / {SEG_POST_S} s  -> total {SEG_PRE_S+SEG_POST_S:.1f} s")
print("═" * 80)
print()

# ═══════════════════════════════════════════════════════════════════════════
# IO CONFIG
# ═══════════════════════════════════════════════════════════════════════════

# Kaggle defaults; override via env/local variant if needed.
KAGGLE_WAV_DIR = Path("/kaggle/input/datasets/kokyawnandar/wav-first/untitled folder/")
OUTPUT_DIR = Path("/kaggle/working")
# For local dry-run override (set by wrapper):
WAV_DIR_OVERRIDE = None
OUT_DIR_OVERRIDE = None
if "KAGGLE_WAV_DIR_OVERRIDE" in os.environ:
    KAGGLE_WAV_DIR = Path(os.environ["KAGGLE_WAV_DIR_OVERRIDE"])
if "KAGGLE_OUT_DIR_OVERRIDE" in os.environ:
    OUTPUT_DIR = Path(os.environ["KAGGLE_OUT_DIR_OVERRIDE"])

WAV_LABEL_MAP = {
    "197.wav": "A", "198.wav": "D", "199.wav": "D", "200.wav": "B",
    "201.wav": "D", "202.wav": "A", "203.wav": "D", "204.wav": "C",
    "205.wav": "A", "206.wav": "C", "207.wav": "D", "208.wav": "B",
    "209.wav": "C", "210.wav": "D", "211.wav": "B", "212.wav": "C",
    "213.wav": "A", "214.wav": "B"
}
EXPECTED_FILES = set(WAV_LABEL_MAP.keys())

# ═══════════════════════════════════════════════════════════════════════════
# DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detection_envelope(audio: np.ndarray, sr: float) -> np.ndarray:
    """Reference-style envelope: bandpass -> Hilbert -> boxcar smooth -> normalize."""
    audio = np.asarray(audio, dtype=np.float64).flatten()
    nyq = sr / 2.0
    hp_norm = np.clip(IMPACT_DETECT_HP_HZ / nyq, 1e-5, 0.95)
    lp_norm = np.clip(IMPACT_DETECT_LP_HZ / nyq, hp_norm + 1e-4, 0.99)
    b, a = signal.butter(4, [hp_norm, lp_norm], btype='bandpass')
    x = signal.filtfilt(b, a, audio)
    env = np.abs(signal.hilbert(x))
    sn = max(3, int(sr * IMPACT_ENVELOPE_SMOOTH_MS / 1000.0))
    env = np.convolve(env, np.ones(sn) / sn, mode='same')
    mx = np.max(env)
    if mx > 0:
        env = env / mx
    return env


def fine_peaks(env: np.ndarray, sr: float) -> np.ndarray:
    """All candidate peaks (including ringing)."""
    d = max(1, int(sr * FINE_MIN_DISTANCE_MS / 1000.0))
    peaks, _ = signal.find_peaks(env, distance=d, prominence=FINE_PROMINENCE, height=FINE_HEIGHT)
    return np.asarray(peaks, dtype=int)


def burst_cluster(peaks: np.ndarray, env: np.ndarray, sr: float):
    """Group fine peaks into physical events with a BURST_MERGE_MS window.

    Returns list of dicts: {rep, rep_env, n_peaks, peaks, start, end, peak_times_ms}
    rep = index (sample) of the strongest (max-envelope) peak in the cluster.
    """
    if len(peaks) == 0:
        return []
    merge = max(1, int(sr * BURST_MERGE_MS / 1000.0))
    clusters = []
    cur = [peaks[0]]
    for p in peaks[1:]:
        if (p - cur[-1]) > merge:
            clusters.append(cur)
            cur = [p]
        else:
            cur.append(p)
    clusters.append(cur)

    events = []
    for c in clusters:
        c = np.asarray(c, dtype=int)
        rep = c[np.argmax(env[c])]            # strongest (max-envelope) peak
        rep_env = float(env[rep])
        events.append({
            'rep': int(rep),
            'rep_env': rep_env,
            'n_peaks': int(len(c)),
            'peaks': c,
        })
    return events


def ambiguity_flag(ev: dict, prev_gap_ms: float, next_gap_ms: float) -> Tuple[str, str]:
    """Return (flag, reason). Flags: NONE / MULTI_STRONG / BORDERLINE / MULTI_STRONG+BORDERLINE."""
    # MULTI_STRONG: >=2 peaks within MULTI_STRONG_REL of cluster max
    flags = []
    peaks = ev['peaks']
    env_vals = ev['cluster_env'] if 'cluster_env' in ev else None
    if env_vals is not None and len(env_vals) > 1:
        mx = np.max(env_vals)
        strong = np.sum(env_vals >= (mx * (1.0 - MULTI_STRONG_REL)))
        ev['n_strong'] = int(strong)
        if strong >= 2:
            flags.append("MULTI_STRONG")
    else:
        ev['n_strong'] = int(len(peaks))

    # BORDERLINE: inter-event gap near the cluster boundary
    gap = None
    if prev_gap_ms is not None and BORDERLINE_GAP_MIN_MS <= prev_gap_ms <= BORDERLINE_GAP_MAX_MS:
        gap = prev_gap_ms
    if next_gap_ms is not None and BORDERLINE_GAP_MIN_MS <= next_gap_ms <= BORDERLINE_GAP_MAX_MS:
        gap = next_gap_ms if gap is None else min(gap, next_gap_ms)
    if gap is not None:
        ev['borderline_gap_ms'] = float(gap)
        flags.append("BORDERLINE")
    else:
        ev['borderline_gap_ms'] = np.nan

    flag = "+".join(flags) if flags else "NONE"
    reason = f"gap={gap:.1f}ms" if gap is not None else ""
    if "MULTI_STRONG" in flags:
        reason = (reason + " " + f"n_strong={ev['n_strong']}").strip()
    return flag, reason


# ═══════════════════════════════════════════════════════════════════════════
# SEGMENT + DSP FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def extract_segment(audio: np.ndarray, center_idx: int, sr: float):
    pre = int(sr * SEG_PRE_S)
    post = int(sr * SEG_POST_S)
    start = max(0, center_idx - pre)
    end = min(len(audio), center_idx + post)
    return audio[start:end], start, end


def estimate_physical_duration(segment: np.ndarray, sr: float, center_rel: int) -> float:
    """DSP-derived physical duration via post-peak -20dB decay. NaN if unreliable."""
    if len(segment) < 64 or center_rel >= len(segment) - 10:
        return np.nan
    post = segment[center_rel:]
    env = np.abs(signal.hilbert(post.astype(np.float64)))
    sn = max(3, int(sr * 0.5 / 1000.0))
    if sn < len(env):
        env = np.convolve(env, np.ones(sn) / sn, mode='same')
    peak_val = np.max(env[:min(len(env), int(sr * 0.010))])
    if peak_val <= 0:
        return np.nan
    below = np.where(env < (peak_val * 0.1))[0]
    if len(below) == 0:
        return np.nan
    dur = below[0] / sr
    if dur < 0.005 or dur > 0.500:
        return np.nan
    return dur


def compute_dsp_features(segment: np.ndarray, sr: float, center_rel: int) -> Tuple[Dict, int]:
    features = {}
    features['Peak_Amplitude'] = float(np.max(np.abs(segment)))
    features['RMS'] = float(np.sqrt(np.mean(segment ** 2)))
    features['Energy'] = float(np.sum(segment ** 2))
    features['Peak_Time_s'] = float(center_rel / sr)
    features['Zero_Crossing_Rate'] = float(np.sum(np.abs(np.diff(np.sign(segment)))) / (2.0 * len(segment)))
    features['Duration_s'] = estimate_physical_duration(segment, sr, center_rel)

    n_fft = min(N_FFT_PARAM, len(segment))
    if n_fft < 64:
        n_fft = 64
    if n_fft % 2 != 0:
        n_fft -= 1
    hop = min(HOP_LENGTH_PARAM, n_fft // 4)

    f, t, Zxx = signal.stft(segment, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    power = np.mean(np.abs(Zxx) ** 2, axis=1)
    freq_bins = f

    if np.sum(power) > 0:
        w = power / np.sum(power)
        features['Centroid_Hz'] = float(np.sum(freq_bins * w))
        features['Bandwidth_Hz'] = float(np.sqrt(np.sum(((freq_bins - features['Centroid_Hz']) ** 2) * w)))
        cs = np.cumsum(w)
        ri = np.where(cs >= 0.95)[0]
        features['Rolloff_Hz'] = float(freq_bins[ri[0]]) if len(ri) > 0 else float(freq_bins[-1])
        di = np.argmax(power)
        features['Impact_Freq_Hz'] = float(freq_bins[di])
        hf = freq_bins > 3500.0
        features['HFSR'] = float(np.sum(power[hf]) / np.sum(power)) if np.any(hf) else 0.0
        features['Spectral_Entropy'] = float(entropy(w + 1e-10))
    else:
        features['Centroid_Hz'] = features['Bandwidth_Hz'] = 0.0
        features['Rolloff_Hz'] = features['Impact_Freq_Hz'] = 0.0
        features['HFSR'] = features['Spectral_Entropy'] = 0.0

    try:
        features['Mel_Mean_dB'] = float(10 * np.log10(np.mean(power) + 1e-10))
    except Exception:
        features['Mel_Mean_dB'] = np.nan

    try:
        env = np.abs(signal.hilbert(segment.astype(np.float64)))
        post = env[center_rel:]
        if len(post) > 10:
            t_arr = np.arange(len(post)) / sr
            log_env = np.log(post + 1e-10)
            k = min(len(t_arr), int(sr * 0.050))
            features['Laplace_Sigma'] = float(-np.polyfit(t_arr[:k], log_env[:k], 1)[0])
        else:
            features['Laplace_Sigma'] = np.nan
    except Exception:
        features['Laplace_Sigma'] = np.nan

    return features, n_fft


# ═══════════════════════════════════════════════════════════════════════════
# PER-FILE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_wav_file(wav_path: Path, label: str) -> Tuple[List[Dict], Dict]:
    records = []
    info = {'file': wav_path.name, 'label': label, 'sr': np.nan, 'duration_s': np.nan,
            'detected': 0, 'extracted': 0}
    try:
        audio, sr = sf.read(str(wav_path), dtype='float64')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        info['sr'] = sr
        info['duration_s'] = len(audio) / sr

        env = detection_envelope(audio, sr)
        peaks = fine_peaks(env, sr)
        events = burst_cluster(peaks, env, sr)
        info['detected'] = len(events)

        # Compute per-event cluster env values for MULTI_STRONG
        for ev in events:
            ev['cluster_env'] = env[ev['peaks']]

        # Event times + gaps
        times_ms = [ev['rep'] / sr * 1000.0 for ev in events]
        prev_gaps = [None] + [times_ms[i] - times_ms[i - 1] for i in range(1, len(times_ms))]
        next_gaps = [times_ms[i + 1] - times_ms[i] for i in range(len(times_ms) - 1)] + [None]

        intervals_s = []
        for i, ev in enumerate(events):
            rep = ev['rep']
            flag, reason = ambiguity_flag(ev, prev_gaps[i], next_gaps[i])
            segment, s0, s1 = extract_segment(audio, rep, sr)
            if len(segment) < max(64, int(sr * 0.005)):
                continue
            center_rel = rep - s0
            feats, nfft = compute_dsp_features(segment, sr, center_rel)
            interval = (times_ms[i] - times_ms[i - 1]) / 1000.0 if i > 0 else np.nan
            intervals_s.append(interval)
            record = {
                'File_Name': wav_path.name,
                'Result': label,
                'Impact_Number': i + 1,
                'Impact_Time_s': rep / sr,
                'Segment_Start_s': s0 / sr,
                'Segment_End_s': s1 / sr,
                'Segment_Duration_s': len(segment) / sr,
                'Sample_Rate_Hz': sr,
                'N_FFT': nfft,
                'Frequency_Resolution_Hz': sr / nfft,
                'Cluster_N_Peaks': ev['n_peaks'],
                'Cluster_N_Strong': ev['n_strong'],
                'Ambiguous_Flag': flag,
                'Ambiguous_Reason': reason,
                'Impact_Interval_s': interval,
            }
            record.update(feats)
            records.append(record)
        info['extracted'] = len(records)
    except Exception as e:
        info['error'] = f"{str(e)}\n{traceback.format_exc()}"
    return records, info


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if not KAGGLE_WAV_DIR.exists():
        print(f"✗ ERROR: directory does not exist: {KAGGLE_WAV_DIR}")
        sys.exit(1)
    wav_files = sorted(KAGGLE_WAV_DIR.glob("*.wav"))
    discovered = {f.name for f in wav_files}
    missing = EXPECTED_FILES - discovered
    unexpected = discovered - EXPECTED_FILES
    if missing:
        print(f"⚠️ Missing expected: {sorted(missing)}")
    if unexpected:
        print(f"⚠️ Unexpected (ignored): {sorted(unexpected)}")
    wav_files = [f for f in wav_files if f.name in EXPECTED_FILES]
    print(f"✓ Processing {len(wav_files)} expected files\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records, infos, failed = [], [], []
    for w in wav_files:
        label = WAV_LABEL_MAP[w.name]
        print(f"[{w.name}] ({label}) ...", end=" ", flush=True)
        recs, info = process_wav_file(w, label)
        if 'error' in info:
            failed.append((w.name, info['error']))
            print("FAILED")
        else:
            all_records.extend(recs)
            infos.append(info)
            print(f"{info['detected']} events / {info['extracted']} extracted")

    # Main CSV
    df_main = pd.DataFrame(all_records)
    df_main.to_csv(OUTPUT_DIR / "impact_by_impact_dsp_results.csv", index=False, float_format='%.6f')

    # Summary CSV
    rows = []
    for info in infos:
        recs = [r for r in all_records if r['File_Name'] == info['file']]
        t = [r['Impact_Time_s'] for r in recs]
        iv = [r['Impact_Interval_s'] for r in recs if not np.isnan(r['Impact_Interval_s'])]
        namb = sum(1 for r in recs if r['Ambiguous_Flag'] != 'NONE')
        rows.append({
            'File_Name': info['file'], 'Result': info['label'],
            'Sample_Rate_Hz': info['sr'], 'Duration_s': info['duration_s'],
            'Impact_Count': len(recs),
            'First_Impact_Time_s': t[0] if t else np.nan,
            'Last_Impact_Time_s': t[-1] if t else np.nan,
            'Mean_Impact_Interval_s': np.mean(iv) if iv else np.nan,
            'Median_Impact_Interval_s': np.median(iv) if iv else np.nan,
            'Min_Impact_Interval_s': np.min(iv) if iv else np.nan,
            'Ambiguous_Events': namb,
        })
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(OUTPUT_DIR / "impact_detection_summary.csv", index=False, float_format='%.6f')

    # QC print
    print("\n" + "═" * 80)
    print("QC SUMMARY (burst clustering)")
    print("═" * 80)
    print(f"1. WAV discovered: {len(discovered)}")
    print(f"2. Processed: {len(infos)}  3. Failed: {len(failed)}")
    for fn, e in failed:
        print(f"   FAILED {fn}: {e.splitlines()[0]}")
    print(f"4. Total detected events: {len(df_main)}")
    if len(df_main) > 0:
        per_file = df_main.groupby('File_Name').size()
        print(f"5. Events per WAV: min={per_file.min()} median={per_file.median():.0f} max={per_file.max()}")
        print("   per file:")
        for fn, c in per_file.items():
            lab = df_main[df_main['File_Name'] == fn]['Result'].iloc[0]
            print(f"     {fn} ({lab}): {c}")
        print(f"6. Events per class:")
        for lab in sorted(df_main['Result'].unique()):
            print(f"     {lab}: {len(df_main[df_main['Result'] == lab])}")
        iv = df_main['Impact_Interval_s'].dropna()
        if len(iv) > 0:
            print(f"7. Min interval: {iv.min()*1000:.1f} ms")
            print(f"8. Median interval: {iv.median()*1000:.1f} ms")
            print(f"9. Max interval: {iv.max()*1000:.1f} ms")
        print(f"10. Sample rates: {sorted(df_main['Sample_Rate_Hz'].unique())}")
        namb = (df_main['Ambiguous_Flag'] != 'NONE').sum()
        print(f"11. Ambiguous events: {namb} ({namb/len(df_main)*100:.1f}%)")
        if namb:
            print("    breakdown:")
            print(df_main[df_main['Ambiguous_Flag'] != 'NONE']['Ambiguous_Flag'].value_counts().to_string())
        print(f"12. Over-detection potential: {'YES - review flagged files' if (per_file > 20).any() else 'NO - counts now reflect physical bursts'}")

    print("\nOutputs:")
    print(f"  {OUTPUT_DIR/'impact_by_impact_dsp_results.csv'}")
    print(f"  {OUTPUT_DIR/'impact_detection_summary.csv'}")


import os
if __name__ == "__main__":
    main()
