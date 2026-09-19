#!/usr/bin/env python3
"""
# ==============================================================================
# EVENT-CENTRIC ACOUSTIC IMPACT PIPELINE
# WAV → PHYSICAL IMPACT #1, #2, #3... → EVENT-CENTRIC DSP FEATURES → CSV
# ==============================================================================
#
# Rules Enforced:
#   1. 1 kHz High-Pass Filter (Butterworth 4th-order SOS filtfilt, 1000 Hz cutoff)
#   2. 1.0s Rejection Ceiling (transients > 1.0s marked REJECTED as ambient noise)
#   3. Impact-by-Impact Calculus & Spectral Features (dX/dt, d2X/dt2, RMS, Crest Factor)
#   4. Laplace Damping (Sigma) & High-Frequency Spectral Ratio (HFSR >= 5kHz / >= 1kHz)
#   5. Consecutive Markov Prior Rule (>= 75% Repeat Match, < 75% 3/4 Elimination)
#   6. 2x2 Orthogonal Ruler (Sigma split 25.0 vs HFSR split 0.180)
#   7. Dual-Environment: Runs seamlessly on Kaggle or Local Mac
# ==============================================================================
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, savgol_filter

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. ENVIRONMENT & PATH AUTO-DETECTION (Kaggle vs Local Mac)
# ==============================================================================

IS_KAGGLE = os.path.exists("/kaggle/working") or os.path.exists("/kaggle/input")

if IS_KAGGLE:
    WAV_ROOT = "/kaggle/input/datasets/kokyawnandar/wav-first/untitled folder"
    LABEL_CANDIDATES = glob.glob("/kaggle/input/**/*.csv", recursive=True)
    OUTPUT_CSV = "/kaggle/working/event_centric_impact_dsp_results.csv"
else:
    WAV_ROOT = "/Users/kyawnandar/Downloads/dataset"
    LABEL_CANDIDATES = [
        "/Users/kyawnandar/Downloads/Dataset_Full_Colab_Features_v2.csv",
        "/Users/kyawnandar/Downloads/dataset/dataset_records.csv",
    ]
    OUTPUT_CSV = "/Users/kyawnandar/Downloads/event_centric_impact_dsp_results.csv"

# ==============================================================================
# 2. EVENT-CENTRIC PARAMETERS
# ==============================================================================

# Filtering: Strict >= 1000 Hz high-pass
LOW_FREQ_HP = 1000.0
HIGH_FREQ_LP = 20000.0

# Physical Duration Ceiling
MAX_PHYSICAL_DURATION_S = 1.0  # > 1.0s rejected as ambient noise

# Impact Detector Settings (Calibrated for Physical Macro Impacts)
SMOOTH_MS = 20.0          # 20ms envelope smoothing removes sub-carrier wave oscillations
MIN_DISTANCE_MS = 60.0    # 60ms min interval prevents intra-burst decay cycle over-triggering
PROMINENCE_RATIO = 0.15   # 15% prominence isolates distinct physical bounces/strikes
HEIGHT_RATIO = 0.10       # 10% height floor eliminates quiet ambient noise
MERGE_MS = 25.0
MAX_EVENTS_PER_FILE = 20  # Physical bounce ceiling

# Impact Extraction Segment Window (centered around peak)
PRE_MS = 10.0    # 10ms attack lead
POST_MS = 80.0   # 80ms decay tail (prevents overlapping into next bounce)

# Ruler Calibration Boundaries
SIGMA_SPLIT = 25.0
HFSR_SPLIT = 0.180
CONSECUTIVE_SIMILARITY_THRESHOLD = 0.75  # 75% threshold


# ==============================================================================
# 3. DSP & FILTERING ENGINE
# ==============================================================================

def apply_1khz_highpass(x, sr, low=1000.0, high=20000.0):
    """Event-Centric 1kHz High-Pass Butterworth 4th-order SOS filter."""
    nyquist = sr / 2.0
    high = min(high, nyquist * 0.95)
    if low >= high:
        return x.copy()
    sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def compute_smooth_envelope(x, sr, smooth_ms=2.0):
    """Analytic Hilbert envelope smoothed by Savitzky-Golay."""
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    
    window = int(sr * smooth_ms / 1000.0)
    window = max(window, 5)
    if window % 2 == 0:
        window += 1
    if window >= len(envelope):
        window = len(envelope) - 1
        if window % 2 == 0:
            window -= 1
    if window < 5:
        return envelope
    return savgol_filter(envelope, window_length=window, polyorder=2)


def detect_physical_impacts(x, sr):
    """
    Detects physical acoustic impact events using 1kHz High-Pass + Hilbert Envelope.
    Returns: (peaks, filtered_signal, envelope)
    """
    filtered = apply_1khz_highpass(x, sr, LOW_FREQ_HP, HIGH_FREQ_LP)
    envelope = compute_smooth_envelope(filtered, sr, SMOOTH_MS)

    env_max = np.max(envelope)
    if env_max <= 1e-12:
        return [], filtered, envelope

    norm_env = envelope / env_max
    distance = max(1, int(sr * MIN_DISTANCE_MS / 1000.0))

    peaks, _ = find_peaks(
        norm_env,
        distance=distance,
        prominence=PROMINENCE_RATIO,
        height=HEIGHT_RATIO
    )

    # Merge nearby peaks within MERGE_MS
    merge_samples = int(sr * MERGE_MS / 1000.0)
    merged = []
    for p in peaks:
        if not merged:
            merged.append(p)
        else:
            if p - merged[-1] <= merge_samples:
                if envelope[p] > envelope[merged[-1]]:
                    merged[-1] = p
            else:
                merged.append(p)

    if len(merged) > MAX_EVENTS_PER_FILE:
        strongest = sorted(merged, key=lambda i: envelope[i], reverse=True)[:MAX_EVENTS_PER_FILE]
        merged = sorted(strongest)

    return merged, filtered, envelope


# ==============================================================================
# 4. SPECTRAL & DAMPING FEATURES (Laplace Sigma, HFSR, Centroid)
# ==============================================================================

def compute_laplace_damping(segment, sr):
    """
    Fits Laplace exponential decay rate: A * exp(-sigma * t)
    on the Hilbert envelope decay from peak to tail.
    """
    if len(segment) < 32:
        return 0.0
    env = np.abs(hilbert(segment))
    pk = np.argmax(env)
    tail = env[pk:]
    if len(tail) < 16:
        return 0.0

    t = np.arange(len(tail)) / sr
    tail_norm = tail / (tail[0] + 1e-12)
    mask = tail_norm > 0.05
    if np.sum(mask) < 8:
        return 0.0

    log_tail = np.log(tail_norm[mask] + 1e-12)
    t_sub = t[mask]
    poly = np.polyfit(t_sub, log_tail, 1)
    sigma = float(-poly[0])
    return max(0.0, sigma)


def compute_spectral_metrics(segment, sr):
    """
    Computes FFT Peak Freq, Spectral Centroid, and High-Frequency Spectral Ratio (HFSR).
    HFSR = Energy(f >= 5000 Hz) / Total_Energy(f >= 1000 Hz).
    """
    n = len(segment)
    if n < 16:
        return 0.0, 0.0, 0.0

    n_fft = min(4096, 2 ** int(np.floor(np.log2(n))))
    n_fft = max(256, n_fft)
    n_fft = min(n_fft, n)

    spec = np.abs(np.fft.rfft(segment, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Exclude DC
    if len(spec) > 1:
        spec[0] = 0.0

    # Dominant Frequency
    dom_freq = float(freqs[np.argmax(spec)])

    # Centroid
    tot_mag = np.sum(spec)
    centroid = float(np.sum(freqs * spec) / (tot_mag + 1e-12))

    # HFSR: >= 5kHz energy over >= 1kHz energy
    mask_1k = freqs >= 1000.0
    mask_5k = freqs >= 5000.0
    e_1k = np.sum(spec[mask_1k] ** 2)
    e_5k = np.sum(spec[mask_5k] ** 2)
    hfsr = float(e_5k / (e_1k + 1e-12))

    return dom_freq, centroid, hfsr


# ==============================================================================
# 5. CONSECUTIVE MARKOV PRIOR & 2x2 ORTHOGONAL RULER
# ==============================================================================

def classify_2x2_ruler(sigma, hfsr, allowed_candidates=None):
    """
    2x2 Orthogonal Ruler:
      Low Sigma (<= 25), High HFSR (> 0.18) -> Group B
      High Sigma (> 25), High HFSR (> 0.18) -> Group A
      Low Sigma (<= 25), Low HFSR (<= 0.18) -> Group C
      High Sigma (> 25), Low HFSR (<= 0.18) -> Group D
    """
    if allowed_candidates is None:
        allowed_candidates = {"A", "B", "C", "D"}

    is_high_sigma = (sigma > SIGMA_SPLIT)
    is_high_hfsr = (hfsr > HFSR_SPLIT)

    if is_high_sigma and is_high_hfsr:
        primary = "A"
    elif (not is_high_sigma) and is_high_hfsr:
        primary = "B"
    elif (not is_high_sigma) and (not is_high_hfsr):
        primary = "C"
    else:
        primary = "D"

    if primary in allowed_candidates:
        return primary

    # Fallback to closest available candidate in 2D normalized space
    quadrant_centers = {
        "A": (35.0, 0.35),
        "B": (10.0, 0.35),
        "C": (10.0, 0.10),
        "D": (35.0, 0.10),
    }
    best_cand = None
    min_dist = float("inf")
    for cand in allowed_candidates:
        if cand in quadrant_centers:
            c_sig, c_hfsr = quadrant_centers[cand]
            d = ((sigma - c_sig) / 25.0) ** 2 + ((hfsr - c_hfsr) / 0.2) ** 2
            if d < min_dist:
                min_dist = d
                best_cand = cand

    return best_cand or primary


def evaluate_consecutive_transition(curr_sigma, curr_hfsr, prev_event):
    """
    Consecutive Transition Rule:
      - If Similarity >= 75%: REPEAT previous result.
      - If Similarity < 75%: ELIMINATE previous result with 3/4 confidence,
        then classify from remaining 3 candidates using 2x2 Ruler.
    """
    if prev_event is None:
        pred = classify_2x2_ruler(curr_sigma, curr_hfsr)
        return pred, 0.0, "STANDALONE_RULER_BASELINE"

    prev_group = prev_event["group"]
    prev_sigma = prev_event["sigma"]
    prev_hfsr = prev_event["hfsr"]

    denom_sig = max(curr_sigma, prev_sigma, 1e-6)
    diff_sig = abs(curr_sigma - prev_sigma) / denom_sig

    denom_hfsr = max(curr_hfsr, prev_hfsr, 1e-6)
    diff_hfsr = abs(curr_hfsr - prev_hfsr) / denom_hfsr

    similarity = max(0.0, 1.0 - 0.5 * (diff_sig + diff_hfsr))

    if similarity >= CONSECUTIVE_SIMILARITY_THRESHOLD:
        pred = prev_group
        path = "CONSECUTIVE_REPEAT_MATCH"
    else:
        # Eliminate previous group with 3/4 confidence
        remaining = {"A", "B", "C", "D"} - {prev_group}
        pred = classify_2x2_ruler(curr_sigma, curr_hfsr, allowed_candidates=remaining)
        path = f"CONSECUTIVE_ELIMINATION_3_OF_4 (Eliminated: {prev_group})"

    return pred, similarity, path


# ==============================================================================
# 6. EXTRACT SINGLE IMPACT FEATURES
# ==============================================================================

def extract_impact_features(original, filtered, sr, peak_idx, impact_number, prev_event=None):
    """Extracts calculus, spectral, damping, and event-centric classification for 1 impact."""
    pre_samples = int(sr * PRE_MS / 1000.0)
    post_samples = int(sr * POST_MS / 1000.0)

    start = max(0, peak_idx - pre_samples)
    end = min(len(original), peak_idx + post_samples)
    segment = original[start:end]

    if len(segment) < 16:
        return None, prev_event

    duration_s = len(segment) / sr

    # Physical Duration Guard (Event-Centric 1.0s ceiling)
    is_rejected = duration_s > MAX_PHYSICAL_DURATION_S
    physical_status = "REJECTED_CEILING" if is_rejected else "ACCEPTED"

    # Calculus Metrics
    peak_val = float(np.max(np.abs(segment)))
    rms_val = float(np.sqrt(np.mean(segment ** 2)))
    energy_val = float(np.trapezoid(segment ** 2, dx=1.0 / sr) if hasattr(np, "trapezoid") else np.trapz(segment ** 2, dx=1.0 / sr))
    abs_area = float(np.trapezoid(np.abs(segment), dx=1.0 / sr) if hasattr(np, "trapezoid") else np.trapz(np.abs(segment), dx=1.0 / sr))

    dx = np.gradient(segment, 1.0 / sr)
    d2x = np.gradient(dx, 1.0 / sr)

    dx_max = float(np.max(np.abs(dx)))
    dx_rms = float(np.sqrt(np.mean(dx ** 2)))
    dx_energy = float(np.trapezoid(dx ** 2, dx=1.0 / sr) if hasattr(np, "trapezoid") else np.trapz(dx ** 2, dx=1.0 / sr))

    d2x_max = float(np.max(np.abs(d2x)))
    d2x_rms = float(np.sqrt(np.mean(d2x ** 2)))
    d2x_energy = float(np.trapezoid(d2x ** 2, dx=1.0 / sr) if hasattr(np, "trapezoid") else np.trapz(d2x ** 2, dx=1.0 / sr))

    crest_factor = float(peak_val / (rms_val + 1e-12))

    # Event-Centric Spectral & Damping Features
    dom_freq, centroid, hfsr = compute_spectral_metrics(segment, sr)
    sigma = compute_laplace_damping(segment, sr)

    # Event-Centric Consecutive Evaluation
    if is_rejected:
        pred_group = "REJECTED"
        sim_pct = 0.0
        decision_path = "PHYSICAL_DURATION_CEILING_REJECT"
    else:
        pred_group, sim, decision_path = evaluate_consecutive_transition(sigma, hfsr, prev_event)
        sim_pct = round(sim * 100.0, 1)
        prev_event = {"group": pred_group, "sigma": sigma, "hfsr": hfsr}

    ruler_baseline = classify_2x2_ruler(sigma, hfsr)

    peak_time_s = peak_idx / sr

    row = {
        "Impact": f"Impact #{impact_number}",
        "Impact_Number": impact_number,
        "Impact_Time_sec": round(peak_time_s, 6),
        "Segment_Start_sec": round(start / sr, 6),
        "Segment_End_sec": round(end / sr, 6),
        "Duration_sec": round(duration_s, 4),
        "Physical_Status": physical_status,
        "Peak": round(peak_val, 6),
        "RMS": round(rms_val, 6),
        "Energy": round(energy_val, 8),
        "Abs_Area": round(abs_area, 6),
        "dXdt_Max": round(dx_max, 2),
        "dXdt_RMS": round(dx_rms, 2),
        "dXdt_Energy": round(dx_energy, 4),
        "d2Xdt2_Max": round(d2x_max, 2),
        "d2Xdt2_RMS": round(d2x_rms, 2),
        "d2Xdt2_Energy": round(d2x_energy, 4),
        "Crest_Factor": round(crest_factor, 3),
        "Dominant_Frequency_Hz": round(dom_freq, 1),
        "Centroid_Hz": round(centroid, 1),
        "Laplace_Sigma": round(sigma, 3),
        "HFSR": round(hfsr, 5),
        "2x2_Ruler_Baseline": ruler_baseline,
        "Consecutive_Sim_Pct": sim_pct,
        "Decision_Path": decision_path,
        "Event_Centric_Prediction": pred_group,
    }

    return row, prev_event


# ==============================================================================
# 7. MAIN PROCESSING PIPELINE
# ==============================================================================

def main():
    print("=" * 80)
    print(" 🎯 EVENT-CENTRIC ACOUSTIC IMPACT PIPELINE (WAV → IMPACT #1..N → CSV)")
    print("=" * 80)
    print(f" Environment : {'Kaggle' if IS_KAGGLE else 'Local Mac'}")
    print(f" WAV Root    : {WAV_ROOT}")
    print(f" Output CSV  : {OUTPUT_CSV}\n")

    # 1. Load Ground Truth Labels if available
    label_map = {}
    for csv_candidate in LABEL_CANDIDATES:
        if os.path.exists(csv_candidate):
            try:
                tdf = pd.read_csv(csv_candidate)
                col_name = next((c for c in tdf.columns if c.lower() in ["file_name", "file"]), None)
                col_group = next((c for c in tdf.columns if c.lower() in ["result", "group"]), None)
                if col_name and col_group:
                    for _, r in tdf.iterrows():
                        fn = os.path.basename(str(r[col_name]).strip())
                        label_map[fn] = str(r[col_group]).strip().upper()
                    print(f" Loaded {len(label_map)} labels from: {csv_candidate}")
                    break
            except Exception:
                pass

    # 2. Find WAV files
    all_wavs = sorted(glob.glob(os.path.join(WAV_ROOT, "*.wav")))
    if not all_wavs:
        print(f"❌ No WAV files found in: {WAV_ROOT}")
        return

    # Support optional CLI args (e.g., --range 192-214 or 192.wav)
    range_tag = None
    if len(sys.argv) > 1:
        custom_targets = set()
        for arg in sys.argv[1:]:
            if "-" in arg and not arg.startswith("--"):
                try:
                    s_str, e_str = arg.split("-")
                    range_tag = f"{s_str}_to_{e_str}"
                    custom_targets.update([f"{i}.wav" for i in range(int(s_str), int(e_str) + 1)])
                except Exception:
                    pass
            elif arg.endswith(".wav"):
                custom_targets.add(os.path.basename(arg))
            elif arg.isdigit():
                custom_targets.add(f"{arg}.wav")
        if custom_targets:
            wav_files = [w for w in all_wavs if os.path.basename(w) in custom_targets]
        else:
            wav_files = all_wavs
    else:
        wav_files = all_wavs

    print(f" Found {len(wav_files)} WAV files to process.\n")

    import librosa
    all_results = []
    file_summaries = []

    for fidx, wav_path in enumerate(wav_files, 1):
        filename = os.path.basename(wav_path)
        gt_label = label_map.get(filename, "Unknown")

        print("-" * 80)
        print(f" 📍 [{fidx:02d}/{len(wav_files)}] {filename} (Ground Truth: {gt_label})")

        try:
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            total_dur = len(y) / sr
            print(f"    SR: {sr:,} Hz | Raw Duration: {total_dur:.4f}s")

            peaks, filtered, envelope = detect_physical_impacts(y, sr)
            print(f"    🥁 Physical Impacts Detected: {len(peaks)}")

            if len(peaks) == 0:
                print("    ⚠️ No physical impacts detected above 1kHz threshold.")
                continue

            prev_event = None
            file_impact_rows = []

            for imp_no, peak_idx in enumerate(peaks, 1):
                feat, prev_event = extract_impact_features(
                    y, filtered, sr, peak_idx, imp_no, prev_event=prev_event
                )
                if feat is None:
                    continue

                feat["File_Name"] = filename
                feat["Ground_Truth"] = gt_label
                feat["Sample_Rate_Hz"] = sr
                feat["Recording_Duration_sec"] = round(total_dur, 4)

                all_results.append(feat)
                file_impact_rows.append(feat)

                # Brief live log for first 5 and every 10th impact
                if imp_no <= 5 or imp_no % 10 == 0 or imp_no == len(peaks):
                    print(f"       Impact #{imp_no:02d} @ {feat['Impact_Time_sec']:.4f}s | "
                          f"Freq={feat['Dominant_Frequency_Hz']:6.1f}Hz | "
                          f"Sig={feat['Laplace_Sigma']:5.2f} | "
                          f"HFSR={feat['HFSR']:.4f} | "
                          f"Pred=【 {feat['Event_Centric_Prediction']} 】 "
                          f"(Sim={feat['Consecutive_Sim_Pct']}%)")

            # File-level Consensus Summary
            if file_impact_rows:
                preds = [r["Event_Centric_Prediction"] for r in file_impact_rows if r["Physical_Status"] == "ACCEPTED"]
                majority_pred = max(set(preds), key=preds.count) if preds else "REJECTED"
                file_summaries.append({
                    "File_Name": filename,
                    "Ground_Truth": gt_label,
                    "Impact_Count": len(file_impact_rows),
                    "Majority_Prediction": majority_pred,
                    "First_Impact_Time_s": file_impact_rows[0]["Impact_Time_sec"],
                    "Last_Impact_Time_s": file_impact_rows[-1]["Impact_Time_sec"],
                })

        except Exception as e:
            print(f"    ❌ ERROR processing {filename}: {e}")

    # 3. Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df = df.sort_values(["File_Name", "Impact_Number"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)

        dedicated_csv = None
        if range_tag:
            dedicated_csv = os.path.join(os.path.dirname(OUTPUT_CSV), f"event_centric_impact_results_{range_tag}.csv")
            df.to_csv(dedicated_csv, index=False)

        print("\n" + "=" * 80)
        print(" 🎉 COMPLETE! Event-Centric Impact Extraction finished.")
        print(f" Total Files   : {len(wav_files)}")
        print(f" Total Impacts : {len(df):,}")
        print(f" Output CSV    : {OUTPUT_CSV}")
        if dedicated_csv:
            print(f" Dedicated CSV : {dedicated_csv}")
        print("=" * 80)

        # Print quick summary
        print("\n📋 FILE-LEVEL MAJORITY SUMMARY:")
        sum_df = pd.DataFrame(file_summaries)
        print(sum_df.to_string(index=False))

    else:
        print("❌ No impact features generated.")


if __name__ == "__main__":
    main()
