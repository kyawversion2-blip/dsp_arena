#!/usr/bin/env python3
"""analyze_2612_repeated_impacts.py — Repeated-Impact Sequence Analysis for Large-Size Teetotum (2612).

Ground truth: Observed repeated impacts as Teetotum decelerates and chatters on hard ceramic/marble.
Extracts:
1. Per-event candidate impact features (timing, energy, decay, centroid, spectral cosine similarity, aligned waveform correlation).
2. Per-file sequence descriptors (event count, interval distribution, chatter slope, sequence decay rate, early vs late spectral shift).
3. Statistical comparison between Group A and Group C on verified labels (N=18).
4. Diagnostic plots and sensitivity checks across threshold variations.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, correlate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/kyawnandar/Documents/interactive-v2/gamestudio-subagents/projects/dsp_arena")
sys.path.insert(0, str(ROOT))

import dsp_inverse

DATA_DIR = Path("/Users/kyawnandar/Downloads/2612")
OUT_DIR = ROOT / "dsp_out" / "large_size_2612_impact_analysis"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_verified_labels() -> dict[str, str]:
    labels_p = ROOT / "labels.csv"
    run3_p = ROOT / "dsp_out" / "run3" / "features.csv"
    run1_p = ROOT / "dsp_out" / "run1" / "features.csv"

    labels = {}
    if labels_p.exists():
        df = pd.read_csv(labels_p)
        for _, r in df.iterrows():
            labels[str(r["File_Name"]).strip()] = str(r["Result"]).strip()
    if run3_p.exists():
        df = pd.read_csv(run3_p)
        for _, r in df.iterrows():
            fn = str(r["file"]).strip()
            if fn not in labels:
                labels[fn] = str(r["group"]).strip()
    if run1_p.exists():
        df = pd.read_csv(run1_p)
        for _, r in df.iterrows():
            fn = str(r["file"]).strip()
            if fn not in labels:
                labels[fn] = str(r["group"]).strip()
    return labels


def detect_and_extract_events(
    y: np.ndarray,
    sr: int,
    height_ratio: float = 0.08,
    prominence_ratio: float = 0.04,
    min_dist_ms: float = 3.0,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Detects candidate impacts and extracts per-event acoustic descriptors."""
    hop = int(sr * 0.00025)  # 0.25 ms bins = 96 samples at 384 kHz
    n_blocks = len(y) // hop
    env = np.array([np.max(np.abs(y[i * hop : (i + 1) * hop])) for i in range(n_blocks)])
    t_env = np.arange(n_blocks) * (hop / sr)

    max_env = float(np.max(env)) if len(env) > 0 else 1e-9
    min_dist_bins = max(1, int((min_dist_ms / 1000.0) / (hop / sr)))

    peaks, props = find_peaks(
        env,
        height=height_ratio * max_env,
        distance=min_dist_bins,
        prominence=prominence_ratio * max_env,
    )

    events = []
    fft_size = 1024
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

    for idx, p in enumerate(peaks):
        t_p = float(t_env[p])
        amp = float(env[p])
        samp_p = int(t_p * sr)

        # Pre-impact (1.0 ms) and Ringdown (up to 5.0 ms or next impact onset)
        start = max(0, samp_p - int(sr * 0.001))
        if idx < len(peaks) - 1:
            next_samp = int(t_env[peaks[idx + 1]] * sr)
            end = min(len(y), samp_p + int(sr * 0.005), next_samp)
        else:
            end = min(len(y), samp_p + int(sr * 0.005))

        seg = y[start:end]
        if len(seg) < 32:
            continue

        eff_n = min(len(seg), fft_size)
        w = np.hanning(eff_n)
        spec = np.abs(np.fft.rfft(seg[:eff_n] * w, n=fft_size))
        spec_sum = float(np.sum(spec)) + 1e-12
        centroid = float(np.sum(freqs * spec) / spec_sum)
        bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / spec_sum))

        dom_bin = int(np.argmax(spec))
        dom_freq = float(freqs[dom_bin])

        # High frequency energy ratio (> 2 kHz)
        hf_mask = freqs > 2000.0
        hfsr = float(np.sum(spec[hf_mask]) / spec_sum)

        rms = float(np.sqrt(np.mean(seg ** 2)))

        # Fixed 3 ms segment for waveform correlation (1152 samples at 384 kHz)
        fixed_len = min(int(sr * 0.003), len(y) - samp_p)
        wave_snippet = y[samp_p : samp_p + fixed_len] if fixed_len > 32 else np.zeros(32)

        events.append({
            "event_idx": idx,
            "time_s": t_p,
            "peak_amp": amp,
            "rms": rms,
            "centroid_hz": centroid,
            "bandwidth_hz": bw,
            "dominant_freq_hz": dom_freq,
            "hfsr": hfsr,
            "duration_ms": round((end - start) * 1000.0 / sr, 3),
            "spec_norm": spec / (np.linalg.norm(spec) + 1e-12),
            "wave_snippet": wave_snippet,
        })

    # Successive similarities
    for i in range(len(events)):
        if i < len(events) - 1:
            events[i]["delta_t_ms"] = (events[i + 1]["time_s"] - events[i]["time_s"]) * 1000.0

            # Spectral cosine similarity
            s1 = events[i]["spec_norm"]
            s2 = events[i + 1]["spec_norm"]
            events[i]["succ_spec_sim"] = float(np.dot(s1, s2))

            # Aligned waveform cross-correlation
            w1 = events[i]["wave_snippet"]
            w2 = events[i + 1]["wave_snippet"]
            min_l = min(len(w1), len(w2))
            if min_l >= 64:
                w1_c = w1[:min_l] - np.mean(w1[:min_l])
                w2_c = w2[:min_l] - np.mean(w2[:min_l])
                norm = np.sqrt(np.sum(w1_c ** 2) * np.sum(w2_c ** 2) + 1e-12)
                corr = correlate(w1_c, w2_c, mode="full")
                events[i]["succ_wave_corr"] = float(np.max(np.abs(corr)) / norm)
            else:
                events[i]["succ_wave_corr"] = 0.0
        else:
            events[i]["delta_t_ms"] = np.nan
            events[i]["succ_spec_sim"] = np.nan
            events[i]["succ_wave_corr"] = np.nan

    return events, env, t_env, peaks


def aggregate_file_descriptors(events: list[dict], total_dur_s: float) -> dict:
    """Aggregates per-event features into sequence-level descriptors."""
    if len(events) < 3:
        return {
            "n_impacts": len(events),
            "impact_rate_hz": len(events) / max(total_dur_s, 1e-3),
            "sequence_duration_s": 0.0,
            "mean_delta_t_ms": 0.0,
            "std_delta_t_ms": 0.0,
            "cv_delta_t": 0.0,
            "interval_trend_slope": 0.0,
            "mean_spec_sim": 0.0,
            "std_spec_sim": 0.0,
            "mean_wave_corr": 0.0,
            "std_wave_corr": 0.0,
            "seq_decay_slope": 0.0,
            "early_late_centroid_shift_hz": 0.0,
            "early_late_energy_ratio": 0.0,
            "mean_event_centroid_hz": 0.0,
            "mean_event_hfsr": 0.0,
            "peak_dominance_ratio": 0.0,
        }

    times = np.array([e["time_s"] for e in events])
    amps = np.array([e["peak_amp"] for e in events])
    rms = np.array([e["rms"] for e in events])
    centroids = np.array([e["centroid_hz"] for e in events])
    hfsrs = np.array([e["hfsr"] for e in events])

    intervals = np.array([e["delta_t_ms"] for e in events[:-1]])
    spec_sims = np.array([e["succ_spec_sim"] for e in events[:-1]])
    wave_corrs = np.array([e["succ_wave_corr"] for e in events[:-1]])

    seq_dur = float(times[-1] - times[0])
    mean_dt = float(np.mean(intervals))
    std_dt = float(np.std(intervals))
    cv_dt = float(std_dt / (mean_dt + 1e-9))

    # Linear trend of delta_t over event indices
    indices = np.arange(len(intervals))
    if len(indices) > 2 and np.var(indices) > 0:
        poly_dt = np.polyfit(indices, intervals, 1)
        slope_dt = float(poly_dt[0])
    else:
        slope_dt = 0.0

    # Sequence decay rate: slope of log(peak_amp) over time_s
    log_amps = np.log(np.maximum(amps, 1e-7))
    t_rel = times - times[0]
    if len(t_rel) > 2 and np.var(t_rel) > 0:
        poly_decay = np.polyfit(t_rel, log_amps, 1)
        decay_slope = float(poly_decay[0])  # negative slope indicates decay
    else:
        decay_slope = 0.0

    # Early vs Late comparisons (first 20% vs last 20%)
    k_seg = max(1, len(events) // 5)
    early_centroid = float(np.mean(centroids[:k_seg]))
    late_centroid = float(np.mean(centroids[-k_seg:]))
    centroid_shift = late_centroid - early_centroid

    early_energy = float(np.sum(rms[:k_seg] ** 2))
    late_energy = float(np.sum(rms[-k_seg:] ** 2))
    energy_ratio = float(early_energy / (late_energy + 1e-12))

    # Peak dominance ratio (energy of top 3 peaks vs sum of all peaks)
    sorted_rms = np.sort(rms)[::-1]
    top3_energy = float(np.sum(sorted_rms[:3] ** 2))
    total_impact_energy = float(np.sum(rms ** 2)) + 1e-12
    peak_dom = float(top3_energy / total_impact_energy)

    return {
        "n_impacts": len(events),
        "impact_rate_hz": round(float(len(events) / max(seq_dur, 0.01)), 2),
        "sequence_duration_s": round(seq_dur, 4),
        "mean_delta_t_ms": round(mean_dt, 3),
        "std_delta_t_ms": round(std_dt, 3),
        "cv_delta_t": round(cv_dt, 4),
        "interval_trend_slope": round(slope_dt, 4),
        "mean_spec_sim": round(float(np.mean(spec_sims)), 4),
        "std_spec_sim": round(float(np.std(spec_sims)), 4),
        "mean_wave_corr": round(float(np.mean(wave_corrs)), 4),
        "std_wave_corr": round(float(np.std(wave_corrs)), 4),
        "seq_decay_slope": round(decay_slope, 4),
        "early_late_centroid_shift_hz": round(centroid_shift, 2),
        "early_late_energy_ratio": round(energy_ratio, 3),
        "mean_event_centroid_hz": round(float(np.mean(centroids)), 1),
        "mean_event_hfsr": round(float(np.mean(hfsrs)), 4),
        "peak_dominance_ratio": round(peak_dom, 4),
    }


def plot_representative_sequence(y: np.ndarray, sr: int, env: np.ndarray, t_env: np.ndarray, peaks: np.ndarray, fn: str, grp: str):
    """Generates detection and envelope plot with detected impact markers using Plotly."""
    t_audio = np.arange(len(y)) / sr
    step = max(1, len(y) // 4000)
    t_down = t_audio[::step]
    y_down = y[::step]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=(
            f"2612 Repeated Impacts: {fn} (Group {grp}) — {len(peaks)} impacts detected",
            "Detailed Contact Chatter Segment (100 ms Zoom)"
        )
    )

    fig.add_trace(
        go.Scatter(x=t_down, y=y_down, mode="lines", line=dict(color="#7f8c8d", width=1), name="Raw Audio"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_env, y=env, mode="lines", line=dict(color="#e74c3c", width=1.5), name="Envelope"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_env[peaks], y=env[peaks], mode="markers", marker=dict(symbol="x", size=8, color="#2c3e50"), name="Impacts"),
        row=1, col=1
    )

    if len(peaks) > 5:
        t_start = max(0.0, float(t_env[peaks[2]]) - 0.01)
        t_end = min(float(t_audio[-1]), t_start + 0.10)
    else:
        t_start = 0.0
        t_end = min(float(t_audio[-1]), 0.10)

    mask = (t_audio >= t_start) & (t_audio <= t_end)
    mask_env = (t_env >= t_start) & (t_env <= t_end)
    peaks_in_zoom = [p for p in peaks if t_start <= t_env[p] <= t_end]

    fig.add_trace(
        go.Scatter(x=t_audio[mask], y=y[mask], mode="lines", line=dict(color="#3498db", width=1.2), name="Zoomed Wave"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_env[mask_env], y=env[mask_env], mode="lines", line=dict(color="#e74c3c", width=2, dash="dash"), name="Zoomed Env"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_env[peaks_in_zoom], y=env[peaks_in_zoom], mode="markers", marker=dict(symbol="circle", size=9, color="#27ae60"), name="Zoomed Markers"),
        row=2, col=1
    )

    fig.update_layout(
        title={
            "text": f"Impact Transient Detection — {fn} (Group {grp})",
            "font": {"size": 16, "family": "Inter, sans-serif"}
        },
        template="plotly_white",
        height=650,
        showlegend=True,
    )
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    out_html = PLOTS_DIR / f"detection_{fn.replace('.wav', '')}_{grp}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")



def run_pipeline():
    print("=== STAGE 2 & 3: REPEATED-IMPACT FEATURE EXTRACTION (2612) ===")
    labels = load_verified_labels()
    files = sorted(list(DATA_DIR.glob("*.wav")), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)

    all_events = []
    file_summaries = []

    # Representative files for diagnostic plots
    plot_targets = ["8.wav", "10.wav", "11.wav", "1.wav", "2.wav", "3.wav"]

    for f in files:
        fn = f.name.strip()
        grp = labels.get(fn, "UNKNOWN")
        y, sr = dsp_inverse.decode_wav(str(f))
        dur = len(y) / sr

        events, env, t_env, peaks = detect_and_extract_events(y, sr)

        for ev in events:
            all_events.append({
                "file": fn,
                "group": grp,
                "event_idx": ev["event_idx"],
                "time_s": round(ev["time_s"], 5),
                "peak_amp": round(ev["peak_amp"], 6),
                "rms": round(ev["rms"], 6),
                "centroid_hz": round(ev["centroid_hz"], 1),
                "bandwidth_hz": round(ev["bandwidth_hz"], 1),
                "dominant_freq_hz": round(ev["dominant_freq_hz"], 1),
                "hfsr": round(ev["hfsr"], 4),
                "duration_ms": ev["duration_ms"],
                "delta_t_ms": round(ev["delta_t_ms"], 3) if not np.isnan(ev["delta_t_ms"]) else "",
                "succ_spec_sim": round(ev["succ_spec_sim"], 4) if not np.isnan(ev["succ_spec_sim"]) else "",
                "succ_wave_corr": round(ev["succ_wave_corr"], 4) if not np.isnan(ev["succ_wave_corr"]) else "",
            })

        summary = aggregate_file_descriptors(events, dur)
        summary["file"] = fn
        summary["group"] = grp
        summary["eligible_a_c"] = "YES" if grp in ("A", "C") else "NO"
        file_summaries.append(summary)

        if fn in plot_targets:
            plot_representative_sequence(y, sr, env, t_env, peaks, fn, grp)

    # 1. Save per-event CSV
    df_events = pd.DataFrame(all_events)
    p_events_csv = OUT_DIR / "per_event_features.csv"
    df_events.to_csv(p_events_csv, index=False)
    print(f"Per-event features saved: {p_events_csv} ({len(df_events)} total detected impacts across 37 files)")

    # 2. Save per-file summary CSV
    df_summary = pd.DataFrame(file_summaries)
    p_summary_csv = OUT_DIR / "per_file_impact_summary.csv"
    df_summary.to_csv(p_summary_csv, index=False)
    print(f"Per-file summary saved: {p_summary_csv} ({len(df_summary)} files)")

    # 3. Sensitivity Check: Vary detection threshold (height_ratio 0.05, 0.08, 0.12)
    print("\n--- Running Sensitivity Check across Detection Thresholds ---")
    sens_results = []
    for h_ratio in [0.05, 0.08, 0.12]:
        counts_a, counts_c = [], []
        sims_a, sims_c = [], []
        for f in files:
            fn = f.name.strip()
            grp = labels.get(fn, "UNKNOWN")
            if grp not in ("A", "C"):
                continue
            y, sr = dsp_inverse.decode_wav(str(f))
            evs, _, _, _ = detect_and_extract_events(y, sr, height_ratio=h_ratio)
            s_dict = aggregate_file_descriptors(evs, len(y) / sr)
            if grp == "A":
                counts_a.append(s_dict["n_impacts"])
                sims_a.append(s_dict["mean_spec_sim"])
            else:
                counts_c.append(s_dict["n_impacts"])
                sims_c.append(s_dict["mean_spec_sim"])

        sens_results.append({
            "threshold_height_ratio": h_ratio,
            "mean_impacts_A": round(float(np.mean(counts_a)), 1),
            "mean_impacts_C": round(float(np.mean(counts_c)), 1),
            "mean_spec_sim_A": round(float(np.mean(sims_a)), 3),
            "mean_spec_sim_C": round(float(np.mean(sims_c)), 3),
        })
    df_sens = pd.DataFrame(sens_results)
    print(df_sens.to_string(index=False))

    # 4. Comparative Evaluation on Verified Group A vs Group C (N=18)
    print("\n=== STAGE 5: COMPARATIVE EVALUATION (GROUP A VS GROUP C, N=18) ===")
    df_ac = df_summary[df_summary["eligible_a_c"] == "YES"].copy().reset_index(drop=True)

    # Load baseline DSP and Chroma features from session_2612_relative.csv
    baseline_csv = ROOT / "dsp_out" / "chroma_relative_validation" / "session_2612_relative.csv"
    df_base = pd.read_csv(baseline_csv)
    df_merged = pd.merge(df_ac, df_base, on=["file", "group"])

    feature_sets = {
        "1_Baseline_DSP": ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"],
        "2_Absolute_Chroma": [f"chroma_{n.replace('#', 'Sharp')}" for n in dsp_inverse.CHROMA_NOTE_NAMES],
        "3_Peak_Circular_Chroma": [f"chroma_circ_bin_{i}" for i in range(12)],
        "4A_Repeated_Impact_Full": [
            "n_impacts", "mean_delta_t_ms", "cv_delta_t", "interval_trend_slope",
            "mean_spec_sim", "mean_wave_corr", "seq_decay_slope",
            "early_late_centroid_shift_hz", "early_late_energy_ratio", "peak_dominance_ratio"
        ],
        "4B_Repeated_Impact_Compact": [
            "mean_event_centroid_hz", "std_spec_sim"
        ],
        "5_Combined_DSP_plus_Impact": [
            "impact", "duration", "mel_mean_db", "impact_freq_hz",
            "mean_event_centroid_hz", "std_spec_sim"
        ]
    }

    comparison_results = []
    y_true = df_merged["group"].to_numpy()
    classes = np.unique(y_true)

    loo = LeaveOneOut()
    rng = np.random.default_rng(42)

    for name, cols in feature_sets.items():
        X = df_merged[cols].to_numpy(dtype=np.float64)

        # Nearest Centroid
        y_pred_nc = np.zeros_like(y_true)
        for train_idx, test_idx in loo.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = NearestCentroid()
            clf.fit(X_tr, y_true[train_idx])
            y_pred_nc[test_idx] = clf.predict(X_te)

        raw_acc_nc = float(accuracy_score(y_true, y_pred_nc))
        bal_acc_nc = float(balanced_accuracy_score(y_true, y_pred_nc))
        cm_nc = confusion_matrix(y_true, y_pred_nc, labels=classes).tolist()
        rec_nc = recall_score(y_true, y_pred_nc, labels=classes, average=None, zero_division=0)

        # Logistic Regression
        y_pred_lr = np.zeros_like(y_true)
        for train_idx, test_idx in loo.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
            clf.fit(X_tr, y_true[train_idx])
            y_pred_lr[test_idx] = clf.predict(X_te)

        bal_acc_lr = float(balanced_accuracy_score(y_true, y_pred_lr))

        # Permutation test (B=1000) for Nearest Centroid
        perm_accs = []
        for _ in range(1000):
            y_shuf = rng.permutation(y_true)
            preds_p = np.zeros_like(y_true)
            for train_idx, test_idx in loo.split(X):
                scaler_p = StandardScaler()
                X_tr_p = scaler_p.fit_transform(X[train_idx])
                X_te_p = scaler_p.transform(X[test_idx])
                clf_p = NearestCentroid()
                clf_p.fit(X_tr_p, y_shuf[train_idx])
                preds_p[test_idx] = clf_p.predict(X_te_p)
            perm_accs.append(balanced_accuracy_score(y_shuf, preds_p))

        p_val_nc = float(np.mean([acc >= bal_acc_nc for acc in perm_accs]))

        comparison_results.append({
            "feature_set": name,
            "n_features": len(cols),
            "nc_raw_accuracy": round(raw_acc_nc, 4),
            "nc_balanced_accuracy": round(bal_acc_nc, 4),
            "nc_permutation_p": round(p_val_nc, 4),
            "lr_balanced_accuracy": round(bal_acc_lr, 4),
            "recall_A": round(float(rec_nc[0]), 4),
            "recall_C": round(float(rec_nc[1]), 4),
            "confusion_matrix": cm_nc,
        })

    df_comp = pd.DataFrame(comparison_results)
    out_comp_json = OUT_DIR / "baseline_comparison_results.json"
    with open(out_comp_json, "w") as f:
        json.dump(comparison_results, f, indent=2)

    print(df_comp[["feature_set", "n_features", "nc_balanced_accuracy", "nc_permutation_p", "lr_balanced_accuracy", "recall_A", "recall_C"]].to_string(index=False))



if __name__ == "__main__":
    run_pipeline()
