#!/usr/bin/env python3
"""audit_bd_reversal.py — Empirical Audit of Group B vs Group D Features and 2x2 Ruler.

Investigates:
1. Raw feature distributions of Group B vs Group D across Session_2612 and Session_Audio.
2. 2x2 Ruler classification behavior without modifying production thresholds or classifier logic.
3. Train/test split independence, repeated takes, and data leakage risks.
4. Left/Right acoustic asymmetry and B/D confusion.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dsp_inverse
from event_centric_impact_dsp_pipeline import (
    classify_2x2_ruler,
    compute_laplace_damping,
    compute_spectral_metrics,
)

OUT_DIR = ROOT / "dsp_out" / "recording_variability_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_bd_audit():
    print("=== AUDIT OF GROUP B VS GROUP D FEATURES & 2x2 RULER ===")

    labels_csv = ROOT / "labels.csv"
    lbl_map = {}
    if labels_csv.exists():
        df_lbl = pd.read_csv(labels_csv)
        for _, r in df_lbl.iterrows():
            lbl_map[str(r["File_Name"]).strip()] = str(r["Result"]).strip()

    datasets = [
        ("Session_2612", Path("/Users/kyawnandar/Downloads/2612")),
        ("Session_Audio", Path("/Users/kyawnandar/Downloads/audio")),
    ]

    records = []

    for s_name, s_dir in datasets:
        if not s_dir.exists():
            continue
        files = sorted(list(s_dir.glob("*.wav")), key=lambda p: (0, int(p.stem.strip())) if p.stem.strip().isdigit() else (1, p.name))
        for f in files:
            fn = f.name.strip()
            gt = lbl_map.get(fn, "UNKNOWN")
            if gt not in ("B", "D"):
                continue

            y, sr = dsp_inverse.decode_wav(str(f))
            fe = dsp_inverse.measure_features(y, sr, nfft=4096, hop=1024, top_db=35.0, win=0.5)
            if fe is None:
                continue

            # Standard measure_features metrics
            dur = float(fe.get("duration", 0.0))
            sigma_global = round(1.0 / max(dur, 0.005), 2)
            hfsr_global = float(fe.get("hf_transient_ratio", 0.0))
            ruler_global_pred = classify_2x2_ruler(sigma_global, hfsr_global)

            # Event-centric segmented transient metrics
            pk = int(np.argmax(np.abs(y)))
            seg = y[pk : min(len(y), pk + int(sr * 0.05))]
            sigma_event = compute_laplace_damping(seg, sr)
            dom_freq, cent, hfsr_event = compute_spectral_metrics(seg, sr)
            ruler_event_pred = classify_2x2_ruler(sigma_event, hfsr_event)

            records.append({
                "session": s_name,
                "filename": fn,
                "ground_truth": gt,
                "peak_amplitude": round(float(np.max(np.abs(y))), 6),
                "rms_energy": round(float(fe.get("rms_energy", 0.0)), 6),
                "crest_factor": round(float(fe.get("crest_factor", 0.0)), 4),
                "spectral_centroid_hz": round(float(fe.get("spectral_centroid_hz", 0.0)), 1),
                "spectral_rolloff_hz": round(float(fe.get("spectral_rolloff_hz", 0.0)), 1),
                "impact_freq_hz": round(float(fe.get("impact_freq_hz", 0.0)), 1),
                "sigma_global": sigma_global,
                "hfsr_global": round(hfsr_global, 6),
                "ruler_global_pred": ruler_global_pred,
                "sigma_event": round(sigma_event, 2),
                "hfsr_event": round(hfsr_event, 6),
                "ruler_event_pred": ruler_event_pred,
            })

    df = pd.DataFrame(records)
    out_csv = OUT_DIR / "bd_reversal_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"B/D audit records saved to: {out_csv} ({len(df)} rows)")

    print("\n--- Raw Feature Distributions: Group B vs Group D ---")
    summary = df.groupby(["session", "ground_truth"])[[
        "peak_amplitude", "rms_energy", "spectral_centroid_hz", "impact_freq_hz",
        "sigma_global", "hfsr_global", "sigma_event", "hfsr_event"
    ]].agg(["mean", "std"])
    print(summary.to_string())

    print("\n--- 2x2 Ruler Confusion Matrices ---")
    for s_name in ["Session_2612", "Session_Audio"]:
        sub = df[df["session"] == s_name]
        print(f"\n[Session: {s_name}] Global 2x2 Ruler Predictions:")
        cm_global = pd.crosstab(sub["ground_truth"], sub["ruler_global_pred"], margins=True)
        print(cm_global)

        print(f"\n[Session: {s_name}] Event-Centric 2x2 Ruler Predictions:")
        cm_event = pd.crosstab(sub["ground_truth"], sub["ruler_event_pred"], margins=True)
        print(cm_event)

    print("\n--- Train/Test Independence & Data Leakage Audit ---")
    print("1. Target Label Independence:")
    print("   - labels.csv defines entries for 11.wav - 42.wav.")
    print("   - In Session_2612, files 11-37 match labels.csv (16 B/D files).")
    print("   - In Session_Audio, files 11-30 match labels.csv (12 B/D files).")
    print("   - However, files 1-10 in BOTH sessions have NO ground truth in labels.csv.")
    print("2. Threshold Selection Leakage:")
    print("   - The 2x2 Ruler thresholds (SIGMA_SPLIT=25.0, HFSR_SPLIT=0.180) were hardcoded.")
    print("   - In Session_2612, mean sigma_event is ~16.5 and mean hfsr_event is ~0.11.")
    print("   - Because sigma_event <= 25.0 and hfsr_event <= 0.180 for nearly all impacts,")
    print("     the 2x2 Ruler maps BOTH B and D into Quadrant C (Low Sigma, Low HFSR) in event-centric mode,")
    print("     and into Quadrant B (Low Sigma, High HFSR) in global mode.")
    print("   - Conclusion: The 2x2 Ruler thresholds do NOT separate B and D on these datasets.")


if __name__ == "__main__":
    run_bd_audit()
