#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Impact-by-Impact Acoustic DSP Analysis Pipeline (Dual Environment: Local & Kaggle)
═══════════════════════════════════════════════════════════════════════════════

AUTHORITATIVE REFERENCE: backend/manual_stop_interactive_recorder(1).py
Verified parameters extracted from detect_impact_events() and extract_impact_segment():
  - IMPACT_DETECT_HP_HZ       = 1000.0 Hz
  - IMPACT_DETECT_LP_HZ       = 20000.0 Hz
  - IMPACT_ENVELOPE_SMOOTH_MS = 1.0 ms (boxcar moving average)
  - IMPACT_MIN_DISTANCE_MS    = 12.0 ms
  - IMPACT_PROMINENCE_RATIO   = 0.035
  - IMPACT_HEIGHT_RATIO       = 0.025
  - IMPACT_PRE_MS             = 8.0 ms
  - IMPACT_POST_MS            = 120.0 ms
  - IMPACT_MERGE_MS           = 10.0 ms
  - IMPACT_MAX_EVENTS         = 50

CRITICAL CONSTRAINTS:
  - Anonymous classes A/B/C/D (No animal names)
  - Native sample rate loaded (preserve high frequencies / ultrasound)
  - Impact numbering #1..N local to each WAV file chronologically
  - Distinguish extraction window (Segment_Duration_s) vs physical event (Duration_s)
  - Report exact actual FFT bin spacing (sr / n_fft) alongside Rayleigh resolution
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import soundfile as sf

# ═══════════════════════════════════════════════════════════════════════════
# AUTHORITATIVE DETECTION PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

HIGHRES_TARGET_SR = 384000  # High-res target sample rate
FMAX_PARAM = 192000         # Max frequency (Nyquist at 384kHz)
N_FFT_PARAM = 8192          # Nominal FFT size
HOP_LENGTH_PARAM = 256      # STFT hop length

IMPACT_DETECT_HP_HZ = 1000.0          # Bandpass HP cutoff
IMPACT_DETECT_LP_HZ = 20000.0         # Bandpass LP cutoff
IMPACT_ENVELOPE_SMOOTH_MS = 1.0       # 1 ms boxcar filter
IMPACT_MIN_DISTANCE_MS = 12.0         # Minimum distance between candidate peaks
IMPACT_PROMINENCE_RATIO = 0.035       # Normalized prominence floor
IMPACT_HEIGHT_RATIO = 0.025           # Normalized height floor
IMPACT_PRE_MS = 8.0                   # Pre-impact lead window
IMPACT_POST_MS = 120.0                # Post-impact decay window
IMPACT_MERGE_MS = 10.0                # Merge window for ringing / sub-pulses
IMPACT_MAX_EVENTS = 50                # Max retained events per recording

MIN_SEGMENT_DURATION_S = 0.005        # 5ms minimum extraction floor
MIN_SAMPLES_FOR_DETECTION = 32

# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & DIRECTORY DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

IS_KAGGLE = Path("/kaggle/working").exists() or Path("/kaggle/input").exists()

if IS_KAGGLE:
    KAGGLE_WAV_DIR = Path("/kaggle/input/datasets/kokyawnandar/wav-first/untitled folder")
    OUTPUT_DIR = Path("/kaggle/working")
else:
    # Local fallback paths
    LOCAL_CANDIDATES = [
        Path("/Users/kyawnandar/Documents/interactive-v2/gamestudio-subagents/projects/dsp_arena/wavs"),
        Path("/Users/kyawnandar/Downloads/dataset"),
        Path("./wavs")
    ]
    KAGGLE_WAV_DIR = next((p for p in LOCAL_CANDIDATES if p.exists()), Path("./wavs"))
    OUTPUT_DIR = Path("./dsp_out")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WAV_LABEL_MAP = {
    "197.wav": "A", "198.wav": "D", "199.wav": "D", "200.wav": "B",
    "201.wav": "D", "202.wav": "A", "203.wav": "D", "204.wav": "C",
    "205.wav": "A", "206.wav": "C", "207.wav": "D", "208.wav": "B",
    "209.wav": "C", "210.wav": "D", "211.wav": "B", "212.wav": "C",
    "213.wav": "A", "214.wav": "B"
}
EXPECTED_FILES = set(WAV_LABEL_MAP.keys())

# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def detect_impact_events(audio: np.ndarray, sr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reproduces backend/manual_stop_interactive_recorder(1).py detect_impact_events():
    1. Float64 conversion and 1D flattening
    2. 4th-order Butterworth bandpass [1000, 20000] Hz (filtfilt)
    3. Hilbert analytic envelope
    4. Boxcar smoothing (1 ms)
    5. Normalization
    6. find_peaks (dist=12ms, prom=0.035, height=0.025)
    7. Merge peaks within 10ms (keep stronger)
    8. Limit to 50 strongest events, restored to chronological order
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
    kernel = np.ones(smooth_n, dtype=np.float64) / smooth_n
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
    return audio[start_idx:end_idx], start_idx, end_idx


def estimate_physical_duration(segment: np.ndarray, sr: float, peak_relative_idx: int) -> float:
    """DSP-derived physical duration via post-peak -20dB envelope decay."""
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
    return float(duration_s)


def compute_dsp_features(segment: np.ndarray, sr: float, peak_relative_idx: int) -> Dict:
    features = {}

    features['Peak_Amplitude'] = float(np.max(np.abs(segment)))
    features['RMS'] = float(np.sqrt(np.mean(segment ** 2)))
    features['Energy'] = float(np.sum(segment ** 2))
    features['Peak_Time_s'] = float(peak_relative_idx / sr)
    features['Zero_Crossing_Rate'] = float(np.sum(np.abs(np.diff(np.sign(segment)))) / (2.0 * len(segment)))
    features['Duration_s'] = estimate_physical_duration(segment, sr, peak_relative_idx)

    # Actual FFT size clamped to segment length
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
        features['Centroid_Hz'] = float(np.sum(freq_bins * weights))
        features['Bandwidth_Hz'] = float(np.sqrt(np.sum(((freq_bins - features['Centroid_Hz']) ** 2) * weights)))
        cumsum = np.cumsum(weights)
        rolloff_idx = np.where(cumsum >= 0.95)[0]
        features['Rolloff_Hz'] = float(freq_bins[rolloff_idx[0]] if len(rolloff_idx) > 0 else freq_bins[-1])
        dominant_idx = int(np.argmax(power_spectrum))
        features['Impact_Freq_Hz'] = float(freq_bins[dominant_idx])
        hf_threshold = 3500.0
        hf_mask = freq_bins > hf_threshold
        features['HFSR'] = float(np.sum(power_spectrum[hf_mask]) / np.sum(power_spectrum)) if np.any(hf_mask) else 0.0
        features['Spectral_Entropy'] = float(entropy(weights + 1e-10))
    else:
        features['Centroid_Hz'] = features['Bandwidth_Hz'] = features['Rolloff_Hz'] = 0.0
        features['Impact_Freq_Hz'] = features['HFSR'] = features['Spectral_Entropy'] = 0.0

    try:
        features['Mel_Mean_dB'] = float(10 * np.log10(np.mean(power_spectrum) + 1e-10))
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
            features['Laplace_Sigma'] = float(-coeffs[0])
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
            features['Autocorr_First_Peak_Lag_s'] = float(peaks[0] / sr)
            features['Autocorr_First_Peak_Value'] = float(autocorr[peaks[0]])
        else:
            features['Autocorr_First_Peak_Lag_s'] = np.nan
            features['Autocorr_First_Peak_Value'] = np.nan
    except Exception:
        features['Autocorr_First_Peak_Lag_s'] = np.nan
        features['Autocorr_First_Peak_Value'] = np.nan

    features['Actual_N_FFT'] = int(n_fft)
    features['Frequency_Resolution_Hz'] = float(sr / n_fft)
    features['Rayleigh_Resolution_Hz'] = float(1.0 / (len(segment) / sr))

    return features


def process_wav_file(wav_path: Path, result_label: str) -> Tuple[List[Dict], float]:
    records = []
    audio_dur = np.nan
    try:
        audio, sr = sf.read(str(wav_path), dtype='float64')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio_dur = len(audio) / sr
        print(f"  Loaded: {len(audio)} samples ({audio_dur:.3f}s) at {sr} Hz (Nyquist: {sr/2} Hz)")

        peak_indices, envelope = detect_impact_events(audio, sr)
        if len(peak_indices) == 0:
            print(f"  ⚠️  No impacts detected")
            return [], audio_dur
        print(f"  ✓ Detected {len(peak_indices)} impacts")

        intervals = np.diff(peak_indices) / sr if len(peak_indices) > 1 else []

        for impact_num, peak_idx in enumerate(peak_indices, start=1):
            segment, start_idx, end_idx = extract_impact_segment(audio, peak_idx, sr)
            min_samples = max(64, int(sr * MIN_SEGMENT_DURATION_S))
            if len(segment) < min_samples:
                print(f"  ⚠️  Impact #{impact_num}: segment too short ({len(segment)} samples), skipping")
                continue
            peak_relative_idx = peak_idx - start_idx
            features = compute_dsp_features(segment, sr, peak_relative_idx)
            record = {
                'File_Name': wav_path.name,
                'Result': result_label,
                'Impact_Number': impact_num,
                'Impact_Time_s': float(peak_idx / sr),
                'Segment_Start_s': float(start_idx / sr),
                'Segment_End_s': float(end_idx / sr),
                'Segment_Duration_s': float(len(segment) / sr),
                'Sample_Rate_Hz': float(sr),
                'Nominal_N_FFT': int(N_FFT_PARAM),
            }
            record.update(features)
            record['Impact_Interval_s'] = float(intervals[impact_num - 2]) if impact_num > 1 else np.nan
            records.append(record)

        print(f"  ✓ Extracted {len(records)} valid impact segments")
    except Exception as e:
        print(f"  ✗ ERROR processing {wav_path.name}: {str(e)}")
        traceback.print_exc()
    return records, audio_dur


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline():
    print("═" * 80)
    print("IMPACT-BY-IMPACT ACOUSTIC DSP PIPELINE")
    print(f"Authoritative Reference: backend/manual_stop_interactive_recorder(1).py")
    print("═" * 80)

    if not KAGGLE_WAV_DIR.exists():
        print(f"Directory not found: {KAGGLE_WAV_DIR}")
        return

    wav_files = sorted(KAGGLE_WAV_DIR.glob("*.wav"))
    discovered = {f.name for f in wav_files}
    print(f"Discovered {len(wav_files)} WAV files at {KAGGLE_WAV_DIR}")

    wav_files = [f for f in wav_files if f.name in EXPECTED_FILES]
    if not wav_files:
        wav_files = sorted(KAGGLE_WAV_DIR.glob("*.wav"))[:18]

    all_records = []
    file_summaries = []
    failed_files = []

    for idx, wav_path in enumerate(wav_files, start=1):
        label = WAV_LABEL_MAP.get(wav_path.name, "A")
        print(f"\n[{idx}/{len(wav_files)}] {wav_path.name} (Class: {label})")
        records, audio_dur = process_wav_file(wav_path, label)
        if records:
            all_records.extend(records)
            first_impact = records[0]['Impact_Time_s']
            last_impact = records[-1]['Impact_Time_s']
            intervals = [r['Impact_Interval_s'] for r in records[1:] if not np.isnan(r['Impact_Interval_s'])]
            file_summaries.append({
                'File_Name': wav_path.name,
                'Result': label,
                'Sample_Rate_Hz': records[0]['Sample_Rate_Hz'],
                'File_Duration_s': audio_dur,
                'Span_To_Last_Impact_s': last_impact + IMPACT_POST_MS / 1000.0,
                'Impact_Count': len(records),
                'First_Impact_Time_s': first_impact,
                'Last_Impact_Time_s': last_impact,
                'Mean_Impact_Interval_s': np.mean(intervals) if intervals else np.nan,
                'Median_Impact_Interval_s': np.median(intervals) if intervals else np.nan,
            })
        else:
            file_summaries.append({
                'File_Name': wav_path.name,
                'Result': label,
                'Sample_Rate_Hz': np.nan,
                'File_Duration_s': audio_dur,
                'Span_To_Last_Impact_s': np.nan,
                'Impact_Count': 0,
                'First_Impact_Time_s': np.nan,
                'Last_Impact_Time_s': np.nan,
                'Mean_Impact_Interval_s': np.nan,
                'Median_Impact_Interval_s': np.nan
            })

    if all_records:
        df_impacts = pd.DataFrame(all_records)
        out_main = OUTPUT_DIR / "impact_by_impact_dsp_results.csv"
        df_impacts.to_csv(out_main, index=False, float_format='%.6f')
        print(f"\n✓ Saved: {out_main} ({len(df_impacts)} impacts)")

    df_summary = pd.DataFrame(file_summaries)
    out_summary = OUTPUT_DIR / "impact_detection_summary.csv"
    df_summary.to_csv(out_summary, index=False, float_format='%.6f')
    print(f"✓ Saved: {out_summary}")


if __name__ == "__main__":
    run_pipeline()
