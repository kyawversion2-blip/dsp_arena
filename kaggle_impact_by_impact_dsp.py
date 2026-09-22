#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Impact-by-Impact Acoustic DSP Analysis — Kaggle Notebook (single cell)
═══════════════════════════════════════════════════════════════════════════════

AUTHORITATIVE REFERENCE: backend/manual_stop_interactive_recorder(1).py
Verified parameters extracted from detect_impact_events() and extract_impact_segment()

FIX (v2): N_FFT and Frequency_Resolution_Hz now reflect the ACTUAL per-segment
          n_fft used (min(8192, len(segment))), not the constant 8192.

CRITICAL RULES:
- A/B/C/D are ANONYMOUS experimental class labels (NEVER use animal names)
- Load WAVs at NATIVE sample rate (preserve high-frequency content)
- Impact numbering: #1, #2, #3... LOCAL to each WAV file, chronologically
- Segment_Duration_s = extraction window duration (NOT physical impact duration)
- Duration_s = DSP-derived acoustic event duration (NaN if unreliable)
- Per-file error handling: one bad WAV must not stop the batch
- Reproducibility: print all reference parameters before processing
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
# AUTHORITATIVE DETECTION PARAMETERS (from backend/manual_stop_interactive_recorder(1).py)
# ═══════════════════════════════════════════════════════════════════════════

HIGHRES_TARGET_SR = 384000  # Target sample rate for high-res processing
FMAX_PARAM = 192000         # Maximum frequency for analysis (Nyquist at 384kHz)
N_FFT_PARAM = 8192          # FFT window size (upper cap; actual per-segment n_fft may be smaller)
HOP_LENGTH_PARAM = 256      # FFT hop length

IMPACT_DETECT_HP_HZ = 1000.0          # High-pass filter cutoff (Hz)
IMPACT_DETECT_LP_HZ = 20000.0         # Low-pass filter cutoff (Hz)
IMPACT_ENVELOPE_SMOOTH_MS = 1.0       # Envelope smoothing window (milliseconds)
IMPACT_MIN_DISTANCE_MS = 12.0         # Minimum distance between peaks (ms)
IMPACT_PROMINENCE_RATIO = 0.035       # Peak prominence threshold (normalized)
IMPACT_HEIGHT_RATIO = 0.025           # Peak height threshold (normalized)
IMPACT_PRE_MS = 8.0                   # Pre-impact window (ms)
IMPACT_POST_MS = 120.0                # Post-impact window (ms)
IMPACT_MERGE_MS = 10.0                # Merge threshold for nearby peaks (ms)
IMPACT_MAX_EVENTS = 50                # Maximum events per file

MIN_SEGMENT_DURATION_S = 0.005        # Minimum segment duration (5ms)
MIN_SAMPLES_FOR_DETECTION = 32        # Minimum audio samples required

print("═" * 80)
print("AUTHORITATIVE DETECTION PARAMETERS")
print("═" * 80)
print(f"Reference: backend/manual_stop_interactive_recorder(1).py")
print(f"DSP Processing:")
print(f"  N_FFT_PARAM (cap)     = {N_FFT_PARAM}")
print(f"  HOP_LENGTH_PARAM      = {HOP_LENGTH_PARAM}")
print(f"  FMAX_PARAM            = {FMAX_PARAM} Hz")
print(f"Impact Detection:")
print(f"  Bandpass Filter       = {IMPACT_DETECT_HP_HZ} - {IMPACT_DETECT_LP_HZ} Hz")
print(f"  Envelope Smoothing    = {IMPACT_ENVELOPE_SMOOTH_MS} ms (boxcar moving average)")
print(f"  Min Distance          = {IMPACT_MIN_DISTANCE_MS} ms")
print(f"  Prominence Threshold  = {IMPACT_PROMINENCE_RATIO}")
print(f"  Height Threshold      = {IMPACT_HEIGHT_RATIO}")
print(f"  Merge Window          = {IMPACT_MERGE_MS} ms")
print(f"  Max Events/File       = {IMPACT_MAX_EVENTS}")
print(f"Segment Extraction:")
print(f"  Pre-impact window     = {IMPACT_PRE_MS} ms")
print(f"  Post-impact window    = {IMPACT_POST_MS} ms")
print(f"  Min segment duration  = {MIN_SEGMENT_DURATION_S * 1000} ms")
print("═" * 80)
print()

# ═══════════════════════════════════════════════════════════════════════════
# KAGGLE DATASET CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

KAGGLE_WAV_DIR = Path("/kaggle/input/datasets/kokyawnandar/wav-first/untitled folder/")
OUTPUT_DIR = Path("/kaggle/working")

WAV_LABEL_MAP = {
    "197.wav": "A", "198.wav": "D", "199.wav": "D", "200.wav": "B",
    "201.wav": "D", "202.wav": "A", "203.wav": "D", "204.wav": "C",
    "205.wav": "A", "206.wav": "C", "207.wav": "D", "208.wav": "B",
    "209.wav": "C", "210.wav": "D", "211.wav": "B", "212.wav": "C",
    "213.wav": "A", "214.wav": "B"
}

EXPECTED_FILES = set(WAV_LABEL_MAP.keys())

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def detect_impact_events(audio: np.ndarray, sr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    AUTHORITATIVE impact detection logic from backend/manual_stop_interactive_recorder(1).py
    Reproduces detect_impact_events() exactly:
    1. float64, flatten
    2. Bandpass 1-20 kHz, 4th-order Butterworth (filtfilt)
    3. Hilbert envelope
    4. Boxcar smoothing (1 ms)
    5. Normalize envelope
    6. find_peaks (distance=12ms, prominence=0.035, height=0.025)
    7. Merge nearby peaks (10 ms, keep stronger)
    8. Limit to 50 strongest, restore chronological order
    """
    audio = np.asarray(audio, dtype=np.float64).flatten()
    if len(audio) < MIN_SAMPLES_FOR_DETECTION:
        return np.array([], dtype=int), np.array([], dtype=np.float64)

    nyq = sr / 2.0
    hp_norm = np.clip(IMPACT_DETECT_HP_HZ / nyq, 1e-5, 0.95)
    lp_norm = np.clip(IMPACT_DETECT_LP_HZ / nyq, hp_norm + 1e-4, 0.99)
    b, a = signal.butter(4, [hp_norm, lp_norm], btype='bandpass')
    filtered = signal.filtfilt(b, a, audio)

    envelope = np.abs(signal.hilbert(filtered))

    smooth_n = max(3, int(sr * IMPACT_ENVELOPE_SMOOTH_MS / 1000.0))
    kernel = np.ones(smooth_n) / smooth_n
    envelope = np.convolve(envelope, kernel, mode='same')

    mx = np.max(envelope)
    if mx <= 0:
        return np.array([], dtype=int), envelope
    env_normalized = envelope / mx

    distance = max(1, int(sr * IMPACT_MIN_DISTANCE_MS / 1000.0))
    peaks, _ = signal.find_peaks(
        env_normalized,
        distance=distance,
        prominence=IMPACT_PROMINENCE_RATIO,
        height=IMPACT_HEIGHT_RATIO
    )
    if len(peaks) == 0:
        return np.array([], dtype=int), env_normalized

    merge_samples = max(1, int(sr * IMPACT_MERGE_MS / 1000.0))
    merged = []
    for p in peaks:
        if len(merged) == 0 or (p - merged[-1]) > merge_samples:
            merged.append(p)
        else:
            if envelope[p] > envelope[merged[-1]]:
                merged[-1] = p
    merged = np.array(merged, dtype=int)

    if len(merged) > IMPACT_MAX_EVENTS:
        strengths = envelope[merged]
        strongest_indices = np.argsort(strengths)[-IMPACT_MAX_EVENTS:]
        merged = np.sort(merged[strongest_indices])

    return merged, env_normalized


def extract_impact_segment(audio: np.ndarray, peak_idx: int, sr: float) -> Tuple[np.ndarray, int, int]:
    pre_samples = int(sr * IMPACT_PRE_MS / 1000.0)
    post_samples = int(sr * IMPACT_POST_MS / 1000.0)
    start_idx = max(0, peak_idx - pre_samples)
    end_idx = min(len(audio), peak_idx + post_samples)
    segment = audio[start_idx:end_idx]
    return segment, start_idx, end_idx


def estimate_physical_duration(segment: np.ndarray, sr: float, peak_relative_idx: int) -> float:
    """DSP-derived physical duration via post-peak -20dB decay. NaN if unreliable."""
    if len(segment) < 64 or peak_relative_idx >= len(segment) - 10:
        return np.nan
    post_peak = segment[peak_relative_idx:]
    envelope = np.abs(signal.hilbert(post_peak.astype(np.float64)))
    smooth_n = max(3, int(sr * 0.5 / 1000.0))
    if smooth_n < len(envelope):
        kernel = np.ones(smooth_n) / smooth_n
        envelope = np.convolve(envelope, kernel, mode='same')
    peak_val = np.max(envelope[:min(len(envelope), int(sr * 0.010))])
    if peak_val <= 0:
        return np.nan
    threshold = peak_val * 0.1  # -20dB
    below_threshold = np.where(envelope < threshold)[0]
    if len(below_threshold) == 0:
        return np.nan
    duration_s = below_threshold[0] / sr
    if duration_s < 0.005 or duration_s > 0.100:
        return np.nan
    return duration_s


def compute_dsp_features(segment: np.ndarray, sr: float, peak_relative_idx: int) -> Tuple[Dict, int]:
    """Compute DSP features for one impact segment.

    Returns (features, actual_n_fft): the dict of features plus the ACTUAL n_fft
    used for the STFT, so N_FFT / Frequency_Resolution_Hz reflect the real window.
    """
    features = {}

    features['Peak_Amplitude'] = np.max(np.abs(segment))
    features['RMS'] = np.sqrt(np.mean(segment ** 2))
    features['Energy'] = np.sum(segment ** 2)
    features['Peak_Time_s'] = peak_relative_idx / sr
    features['Zero_Crossing_Rate'] = np.sum(np.abs(np.diff(np.sign(segment)))) / (2.0 * len(segment))

    features['Duration_s'] = estimate_physical_duration(segment, sr, peak_relative_idx)

    # Actual per-segment n_fft (capped by N_FFT_PARAM and segment length)
    n_fft = min(N_FFT_PARAM, len(segment))
    if n_fft < 64:
        n_fft = 64
    if n_fft % 2 != 0:
        n_fft -= 1
    hop_length = min(HOP_LENGTH_PARAM, n_fft // 4)

    f, t, Zxx = signal.stft(segment, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx)
    power_spectrum = np.mean(magnitude ** 2, axis=1)

    if np.sum(power_spectrum) > 0:
        freq_bins = f
        weights = power_spectrum / np.sum(power_spectrum)
        features['Centroid_Hz'] = np.sum(freq_bins * weights)
        features['Bandwidth_Hz'] = np.sqrt(np.sum(((freq_bins - features['Centroid_Hz']) ** 2) * weights))
        cumsum = np.cumsum(weights)
        rolloff_idx = np.where(cumsum >= 0.95)[0]
        features['Rolloff_Hz'] = freq_bins[rolloff_idx[0]] if len(rolloff_idx) > 0 else freq_bins[-1]
        dominant_idx = np.argmax(power_spectrum)
        features['Impact_Freq_Hz'] = freq_bins[dominant_idx]
        hf_threshold = 3500.0
        hf_mask = freq_bins > hf_threshold
        features['HFSR'] = (np.sum(power_spectrum[hf_mask]) / np.sum(power_spectrum)) if np.any(hf_mask) else 0.0
        features['Spectral_Entropy'] = entropy(weights + 1e-10)
    else:
        features['Centroid_Hz'] = features['Bandwidth_Hz'] = features['Rolloff_Hz'] = 0.0
        features['Impact_Freq_Hz'] = features['HFSR'] = features['Spectral_Entropy'] = 0.0

    try:
        features['Mel_Mean_dB'] = 10 * np.log10(np.mean(power_spectrum) + 1e-10)
    except Exception:
        features['Mel_Mean_dB'] = np.nan

    try:
        envelope = np.abs(signal.hilbert(segment.astype(np.float64)))
        post_peak = envelope[peak_relative_idx:]
        if len(post_peak) > 10:
            t_arr = np.arange(len(post_peak)) / sr
            log_env = np.log(post_peak + 1e-10)
            k = min(len(t_arr), int(sr * 0.050))
            coeffs = np.polyfit(t_arr[:k], log_env[:k], 1)
            features['Laplace_Sigma'] = -coeffs[0]
        else:
            features['Laplace_Sigma'] = np.nan
    except Exception:
        features['Laplace_Sigma'] = np.nan

    try:
        autocorr = np.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
        peaks, _ = signal.find_peaks(autocorr, height=0.3)
        if len(peaks) > 0:
            features['Autocorr_First_Peak_Lag_s'] = peaks[0] / sr
            features['Autocorr_First_Peak_Value'] = autocorr[peaks[0]]
        else:
            features['Autocorr_First_Peak_Lag_s'] = np.nan
            features['Autocorr_First_Peak_Value'] = np.nan
    except Exception:
        features['Autocorr_First_Peak_Lag_s'] = np.nan
        features['Autocorr_First_Peak_Value'] = np.nan

    return features, n_fft


def process_wav_file(wav_path: Path, result_label: str) -> List[Dict]:
    records = []
    try:
        audio, sr = sf.read(str(wav_path), dtype='float64')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        print(f"  Loaded: {len(audio)} samples at {sr} Hz (Nyquist: {sr/2} Hz)")

        peak_indices, envelope = detect_impact_events(audio, sr)
        if len(peak_indices) == 0:
            print(f"  ⚠️  No impacts detected")
            return []
        print(f"  ✓ Detected {len(peak_indices)} impacts")

        intervals = np.diff(peak_indices) / sr if len(peak_indices) > 1 else []

        for impact_num, peak_idx in enumerate(peak_indices, start=1):
            segment, start_idx, end_idx = extract_impact_segment(audio, peak_idx, sr)
            min_samples = max(64, int(sr * MIN_SEGMENT_DURATION_S))
            if len(segment) < min_samples:
                print(f"  ⚠️  Impact #{impact_num}: segment too short, skipping")
                continue
            peak_relative_idx = peak_idx - start_idx
            features, actual_n_fft = compute_dsp_features(segment, sr, peak_relative_idx)
            record = {
                'File_Name': wav_path.name,
                'Result': result_label,
                'Impact_Number': impact_num,
                'Impact_Time_s': peak_idx / sr,
                'Segment_Start_s': start_idx / sr,
                'Segment_End_s': end_idx / sr,
                'Segment_Duration_s': len(segment) / sr,
                'Sample_Rate_Hz': sr,
                'N_FFT': actual_n_fft,
                'Frequency_Resolution_Hz': sr / actual_n_fft,
            }
            record.update(features)
            record['Impact_Interval_s'] = intervals[impact_num - 2] if impact_num > 1 else np.nan
            records.append(record)

        print(f"  ✓ Extracted {len(records)} valid impact segments")
    except Exception as e:
        print(f"  ✗ ERROR processing {wav_path.name}: {str(e)}")
        traceback.print_exc()
    return records


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("KAGGLE WAV DIRECTORY VERIFICATION")
print("═" * 80)

if not KAGGLE_WAV_DIR.exists():
    print(f"✗ ERROR: Directory does not exist: {KAGGLE_WAV_DIR}")
    print(f"  Please verify the Kaggle dataset is mounted correctly.")
    sys.exit(1)

print(f"✓ Directory exists: {KAGGLE_WAV_DIR}")

wav_files = sorted(KAGGLE_WAV_DIR.glob("*.wav"))
discovered_names = {f.name for f in wav_files}
print(f"✓ Discovered {len(wav_files)} WAV files (expected {len(EXPECTED_FILES)})")

missing = EXPECTED_FILES - discovered_names
unexpected = discovered_names - EXPECTED_FILES
if missing:
    print(f"  ⚠️  Missing expected files: {sorted(missing)}")
if unexpected:
    print(f"  ⚠️  Unexpected files (ignored): {sorted(unexpected)}")

wav_files = [f for f in wav_files if f.name in EXPECTED_FILES]
print(f"\n✓ Processing {len(wav_files)} expected WAV files")
print("═" * 80)

all_records = []
failed_files = []
file_summaries = []

for idx, wav_path in enumerate(wav_files, start=1):
    result_label = WAV_LABEL_MAP[wav_path.name]
    print(f"\n[{idx}/{len(wav_files)}] Processing: {wav_path.name} (Class: {result_label})")
    try:
        records = process_wav_file(wav_path, result_label)
        if records:
            all_records.extend(records)
            first_impact = records[0]['Impact_Time_s']
            last_impact = records[-1]['Impact_Time_s']
            intervals = [r['Impact_Interval_s'] for r in records[1:] if not np.isnan(r['Impact_Interval_s'])]
            file_summaries.append({
                'File_Name': wav_path.name, 'Result': result_label,
                'Sample_Rate_Hz': records[0]['Sample_Rate_Hz'],
                'Duration_s': last_impact + IMPACT_POST_MS / 1000.0,
                'Impact_Count': len(records),
                'First_Impact_Time_s': first_impact, 'Last_Impact_Time_s': last_impact,
                'Mean_Impact_Interval_s': np.mean(intervals) if intervals else np.nan,
                'Median_Impact_Interval_s': np.median(intervals) if intervals else np.nan,
            })
        else:
            file_summaries.append({'File_Name': wav_path.name, 'Result': result_label,
                'Sample_Rate_Hz': np.nan, 'Duration_s': np.nan, 'Impact_Count': 0,
                'First_Impact_Time_s': np.nan, 'Last_Impact_Time_s': np.nan,
                'Mean_Impact_Interval_s': np.nan, 'Median_Impact_Interval_s': np.nan})
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)}")
        failed_files.append((wav_path.name, str(e)))
        file_summaries.append({'File_Name': wav_path.name, 'Result': result_label,
            'Sample_Rate_Hz': np.nan, 'Duration_s': np.nan, 'Impact_Count': 0,
            'First_Impact_Time_s': np.nan, 'Last_Impact_Time_s': np.nan,
            'Mean_Impact_Interval_s': np.nan, 'Median_Impact_Interval_s': np.nan})

# ═══════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("SAVING OUTPUTS")
print("═" * 80)

if all_records:
    df_impacts = pd.DataFrame(all_records)
    required_cols = [
        'File_Name', 'Result', 'Impact_Number', 'Impact_Time_s',
        'Segment_Start_s', 'Segment_End_s', 'Segment_Duration_s',
        'Peak_Amplitude', 'RMS', 'Energy', 'Peak_Time_s', 'Duration_s',
        'Impact_Freq_Hz', 'Centroid_Hz', 'Bandwidth_Hz', 'Rolloff_Hz',
        'Zero_Crossing_Rate', 'HFSR', 'Mel_Mean_dB', 'Laplace_Sigma',
        'Impact_Interval_s', 'Sample_Rate_Hz', 'N_FFT', 'Frequency_Resolution_Hz'
    ]
    all_cols = required_cols + [c for c in df_impacts.columns if c not in required_cols]
    df_impacts = df_impacts[all_cols]
    output_path = OUTPUT_DIR / "impact_by_impact_dsp_results.csv"
    df_impacts.to_csv(output_path, index=False, float_format='%.6f')
    print(f"✓ Saved main output: {output_path}  (Total impacts: {len(df_impacts)})")
else:
    print("⚠️  No impacts detected across all files - no main output generated")

df_summary = pd.DataFrame(file_summaries)
summary_path = OUTPUT_DIR / "impact_detection_summary.csv"
df_summary.to_csv(summary_path, index=False, float_format='%.6f')
print(f"✓ Saved summary: {summary_path}")

# ═══════════════════════════════════════════════════════════════════════════
# QC DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 80)
print("QC SUMMARY")
print("═" * 80)
print(f"1. WAV files discovered:      {len(discovered_names)}")
print(f"2. Successfully processed:    {len(wav_files) - len(failed_files)}")
print(f"3. Failed:                    {len(failed_files)}")
if failed_files:
    print(f"   Failed files:")
    for fname, error in failed_files:
        print(f"     - {fname}: {error}")

if all_records:
    df = pd.DataFrame(all_records)
    print(f"\n4. Total detected impacts:    {len(df)}")
    print(f"\n5. Impacts per WAV:")
    for fname in sorted(df['File_Name'].unique()):
        sub = df[df['File_Name'] == fname]
        print(f"     {fname} ({sub['Result'].iloc[0]}): {len(sub)} impacts")
    print(f"\n6. Impacts per experimental class:")
    for label in sorted(df['Result'].unique()):
        print(f"     Class {label}: {len(df[df['Result'] == label])} impacts")
    intervals = df[df['Impact_Interval_s'].notna()]['Impact_Interval_s']
    if len(intervals) > 0:
        print(f"\n7. Min impact interval:       {intervals.min():.4f} s ({intervals.min()*1000:.1f} ms)")
        print(f"8. Median impact interval:    {intervals.median():.4f} s ({intervals.median()*1000:.1f} ms)")
        print(f"9. Max impact interval:       {intervals.max():.4f} s ({intervals.max()*1000:.1f} ms)")
    else:
        print(f"\n7-9. Impact intervals:        N/A (no multi-impact files)")
    print(f"\n10. Sample rate distribution:")
    for sr in sorted(df['Sample_Rate_Hz'].unique()):
        count = len(df[df['Sample_Rate_Hz'] == sr])
        print(f"      {int(sr)} Hz: {count} impacts ({count/len(df)*100:.1f}%)")

    print(f"\n11. Suspicious files (potential over-detection):")
    suspicious_threshold = 20
    suspicious_files = df.groupby('File_Name').size()
    suspicious_files = suspicious_files[suspicious_files > suspicious_threshold]
    if len(suspicious_files) > 0:
        for fname, count in suspicious_files.items():
            label = df[df['File_Name'] == fname]['Result'].iloc[0]
            ifc = df[(df['File_Name'] == fname) & df['Impact_Interval_s'].notna()]['Impact_Interval_s']
            min_iv = f"{ifc.min()*1000:.1f} ms" if len(ifc) > 0 else "N/A"
            print(f"        {fname} ({label}): {count} impacts, min interval {min_iv}")
    else:
        print(f"     ✓ No suspicious files detected")

    print(f"\n12. Potential over-detection warnings:")
    if len(suspicious_files) > 0:
        print(f"     ⚠️  WARNING: {len(suspicious_files)} files may have over-detection")
        print(f"        Review listed files; consider tuning min-distance/prominence/merge")
    else:
        print(f"     ✓ No over-detection warnings")
else:
    print("\n⚠️  No impacts detected - cannot generate QC statistics")

print("\n" + "═" * 80)
print("PROCESSING COMPLETE")
print("═" * 80)
print(f"\nOutputs saved to:")
print(f"  {OUTPUT_DIR / 'impact_by_impact_dsp_results.csv'}")
print(f"  {OUTPUT_DIR / 'impact_detection_summary.csv'}")
