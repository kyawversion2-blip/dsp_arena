#!/usr/bin/env python3
"""analyze_recording_variability.py — Repeatability & Acoustic Variability Analysis.

Investigates how DSP features (2x2 Ruler, damping sigma, HFSR, centroid, peak amplitude)
change across repeated recordings, microphone types (built-in vs Bluetooth),
distances, and horizontal positioning (Left vs Right chassis shadowing).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

ROOT = Path("/Users/kyawnandar/Documents/interactive-v2/gamestudio-subagents/projects/dsp_arena")
sys.path.insert(0, str(ROOT))

import dsp_inverse
from recording_metadata import create_recording_metadata

OUT_DIR = ROOT / "dsp_out" / "recording_variability_audit"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_recording_dsp_features(y: np.ndarray, sr: int, filename: str) -> dict:
    """Extract standard DSP Arena features, 2x2 ruler features, and signal metrics."""
    # 1. Base DSP features
    fe = dsp_inverse.measure_features(y, sr, nfft=4096, hop=1024, top_db=35.0, win=0.5, fmax=None)
    if fe is None:
        # Fallback to direct time-domain and FFT metrics if transient trim rejected
        N = min(len(y), 4096)
        X = np.fft.rfft(y[:N] * np.hanning(N))
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(N, 1.0 / sr)
        peak_idx = int(np.argmax(mag))
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
        fe = {
            "impact": float(np.max(np.abs(y))),
            "duration": float(len(y) / sr),
            "peak_time": float(np.argmax(np.abs(y)) / sr),
            "mel_mean_db": float(10 * np.log10(np.mean(mag) + 1e-12)),
            "impact_freq_hz": float(freqs[peak_idx]),
            "spectral_centroid_hz": centroid,
            "spectral_bandwidth_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_flatness": 0.0,
            "rms_energy": float(np.sqrt(np.mean(y**2))),
            "zcr": float(np.mean(np.signbit(y[1:]) != np.signbit(y[:-1]))),
            "attack_slope_db_ms": 0.0,
            "hf_transient_ratio": 0.0,
            "crest_factor": float(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-12)),
        }

    # 2. Clipping indicator (samples within 99.5% of full scale)
    peak_amp = float(np.max(np.abs(y)))
    clip_threshold = 0.995 if peak_amp >= 0.99 else 0.995 * peak_amp
    clipped_samples = int(np.sum(np.abs(y) >= 0.999))
    clipping_ratio = float(clipped_samples / max(len(y), 1))

    # 3. Candidate Impact count and event interval statistics
    hop = int(sr * 0.0005)  # 0.5 ms blocks
    n_blocks = len(y) // hop
    env = np.array([np.max(np.abs(y[i * hop : (i + 1) * hop])) for i in range(n_blocks)])
    max_env = float(np.max(env)) if len(env) > 0 else 1e-9
    peaks, _ = find_peaks(env, height=0.08 * max_env, distance=int(0.003 / (hop / sr)), prominence=0.04 * max_env)

    impact_count = len(peaks)
    if impact_count > 1:
        t_peaks = peaks * (hop / sr) * 1000.0  # in ms
        dt = np.diff(t_peaks)
        mean_dt_ms = float(np.mean(dt))
        std_dt_ms = float(np.std(dt))
        cv_dt = float(std_dt_ms / (mean_dt_ms + 1e-9))
    else:
        mean_dt_ms = 0.0
        std_dt_ms = 0.0
        cv_dt = 0.0

    # 4. Approximate Laplace damping sigma from decay
    dur = float(fe.get("duration", 0.0))
    sigma_approx = round(1.0 / max(dur, 0.005), 2)

    return {
        "filename": filename,
        "sample_rate_hz": sr,
        "duration_s": round(float(len(y) / sr), 5),
        "peak_amplitude": round(peak_amp, 6),
        "rms_energy": round(float(fe.get("rms_energy", 0.0)), 6),
        "crest_factor": round(float(fe.get("crest_factor", 0.0)), 4),
        "clipped_samples": clipped_samples,
        "clipping_ratio": round(clipping_ratio, 6),
        "impact_freq_hz": round(float(fe.get("impact_freq_hz", 0.0)), 1),
        "spectral_centroid_hz": round(float(fe.get("spectral_centroid_hz", 0.0)), 1),
        "spectral_bandwidth_hz": round(float(fe.get("spectral_bandwidth_hz", 0.0)), 1),
        "spectral_rolloff_hz": round(float(fe.get("spectral_rolloff_hz", 0.0)), 1),
        "spectral_flatness": round(float(fe.get("spectral_flatness", 0.0)), 6),
        "hf_transient_ratio": round(float(fe.get("hf_transient_ratio", 0.0)), 6),
        "mel_mean_db": round(float(fe.get("mel_mean_db", 0.0)), 3),
        "peak_time_ms": round(float(fe.get("peak_time", 0.0)) * 1000.0, 3),
        "laplace_sigma": sigma_approx,
        "impact_event_count": impact_count,
        "mean_impact_interval_ms": round(mean_dt_ms, 3),
        "cv_impact_interval": round(cv_dt, 4),
    }


def calculate_safe_cv(vals: np.ndarray, eps: float = 1e-6) -> float:
    """Safe coefficient of variation (%) preventing zero-division inflation."""
    if len(vals) < 2:
        return 0.0
    m = float(np.mean(vals))
    s = float(np.std(vals))
    if abs(m) < eps:
        return 0.0 if s < eps else float(s / eps * 100.0)
    return float((s / abs(m)) * 100.0)


def calculate_pairwise_relative_diffs(vals: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
    """Calculate mean and max pairwise relative difference (%) across repeated measurements."""
    n = len(vals)
    if n < 2:
        return 0.0, 0.0
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(vals[i] - vals[j])
            denom = 0.5 * (abs(vals[i]) + abs(vals[j])) + eps
            diffs.append((d / denom) * 100.0)
    return float(np.mean(diffs)), float(np.max(diffs))


def run_recording_variability_audit():
    print("=== DSP ARENA: RECORDING & MICROPHONE VARIABILITY INVESTIGATION ===")

    session_2612_p = Path("/Users/kyawnandar/Downloads/2612")
    session_audio_p = Path("/Users/kyawnandar/Downloads/audio")

    # Load labels
    labels_csv = ROOT / "labels.csv"
    lbl_map = {}
    if labels_csv.exists():
        df_lbl = pd.read_csv(labels_csv)
        for _, r in df_lbl.iterrows():
            lbl_map[str(r["File_Name"]).strip()] = str(r["Result"]).strip()

    datasets = [
        ("Session_2612", session_2612_p),
        ("Session_Audio", session_audio_p),
    ]

    records = []
    metadata_list = []

    for s_name, s_dir in datasets:
        if not s_dir.exists():
            print(f"Skipping {s_name}: directory not found")
            continue
        files = sorted(list(s_dir.glob("*.wav")), key=lambda p: (0, int(p.stem.strip())) if p.stem.strip().isdigit() else (1, p.name))
        print(f"Analyzing {s_name}: {len(files)} WAV files...")

        for f in files:
            fn = f.name.strip()
            y, sr = dsp_inverse.decode_wav(str(f))
            feats = compute_recording_dsp_features(y, sr, fn)
            feats["session"] = s_name
            feats["ground_truth_group"] = lbl_map.get(fn, "UNKNOWN")
            records.append(feats)

            # Metadata provenance tracking
            meta = create_recording_metadata(
                source_type="imported_wav",
                requested_sr=sr,
                actual_sr=sr,
                channels=1,
                duration_s=len(y) / sr,
                position="unknown",
                distance_cm=None,
                device_name="MacBook Built-in / External Capture",
                preprocessing=["native_decode"],
                file_name=fn,
                session_id=s_name,
            )
            metadata_list.append(meta)

    df_feats = pd.DataFrame(records)
    out_feats_csv = OUT_DIR / "recording_variability_features.csv"
    df_feats.to_csv(out_feats_csv, index=False)
    print(f"Features saved to: {out_feats_csv} ({len(df_feats)} rows)")

    out_meta_csv = OUT_DIR / "recording_metadata_inventory.csv"
    pd.DataFrame(metadata_list).to_csv(out_meta_csv, index=False)
    print(f"Metadata inventory saved to: {out_meta_csv}")

    # Compute repeatability statistics per session and per group
    numeric_cols = [
        "peak_amplitude", "rms_energy", "crest_factor", "clipping_ratio",
        "impact_freq_hz", "spectral_centroid_hz", "spectral_bandwidth_hz",
        "spectral_rolloff_hz", "hf_transient_ratio", "mel_mean_db",
        "laplace_sigma", "impact_event_count", "mean_impact_interval_ms"
    ]

    stats_rows = []
    pairwise_rows = []

    for s_name in df_feats["session"].unique():
        df_s = df_feats[df_feats["session"] == s_name]

        for grp in sorted(df_s["ground_truth_group"].unique()):
            df_grp = df_s[df_s["ground_truth_group"] == grp]
            n_samples = len(df_grp)
            if n_samples < 2:
                continue

            for col in numeric_cols:
                vals = df_grp[col].to_numpy(dtype=np.float64)
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                cv_pct = calculate_safe_cv(vals)
                mean_pw_diff, max_pw_diff = calculate_pairwise_relative_diffs(vals)

                stats_rows.append({
                    "session": s_name,
                    "group": grp,
                    "n_samples": n_samples,
                    "feature": col,
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "cv_percent": round(cv_pct, 2),
                    "mean_pairwise_diff_pct": round(mean_pw_diff, 2),
                    "max_pairwise_diff_pct": round(max_pw_diff, 2),
                })

    df_stats = pd.DataFrame(stats_rows)
    out_stats_csv = OUT_DIR / "repeatability_statistics.csv"
    df_stats.to_csv(out_stats_csv, index=False)
    print(f"Repeatability statistics saved to: {out_stats_csv}")

    # Summary table of high-variability vs stable features across repeated runs
    print("\n--- Feature Repeatability Summary (Mean CV % Across Groups) ---")
    summary_cv = df_stats.groupby("feature")["cv_percent"].mean().reset_index().sort_values(by="cv_percent")
    print(summary_cv.to_string(index=False))

    # Generate Comparative Distribution Plots (Plotly HTML)
    key_features = ["impact_freq_hz", "spectral_centroid_hz", "hf_transient_ratio", "rms_energy"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Feature: {f}" for f in key_features],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    colors = {"Session_2612": "#3498db", "Session_Audio": "#e67e22"}

    for idx, feat in enumerate(key_features):
        r = (idx // 2) + 1
        c = (idx % 2) + 1

        for s_name in df_feats["session"].unique():
            sub = df_feats[df_feats["session"] == s_name]
            fig.add_trace(
                go.Box(
                    y=sub[feat],
                    name=s_name,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(color=colors.get(s_name, "#9b59b6"), size=5),
                    showlegend=(idx == 0),
                ),
                row=r, col=c
            )

    fig.update_layout(
        title={
            "text": "Cross-Session & Recording Variability Analysis (Session 2612 vs Audio)",
            "font": {"size": 16, "family": "Inter, sans-serif"}
        },
        template="plotly_white",
        height=700,
    )
    out_plot_html = PLOTS_DIR / "cross_session_variability_comparison.html"
    fig.write_html(str(out_plot_html), include_plotlyjs="cdn")
    print(f"Interactive plot saved to: {out_plot_html}")


if __name__ == "__main__":
    run_recording_variability_audit()
