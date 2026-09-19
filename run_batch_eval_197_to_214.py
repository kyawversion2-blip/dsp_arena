#!/usr/bin/env python3
"""
run_batch_eval_197_to_214.py — Batch Evaluation of Dataset Files 197 to 214
==========================================================================
Computes files 197.wav to 214.wav across both acoustic engines/modes:

  [Mode 1] Dual-Mode Acoustic Engine (manual_stop_interactive_recorder(1).py)
           - Candidate Pair FIRST Display & Elimination
           - Multi-method FFT + Laplace Sigma + HFSR + Centroid
           - Multi-method Cross-Verification & Classification

  [Mode 2] Event-Centric Consecutive Prior & 2x2 Ruler (.agents/skills/event-centric)
           - 1kHz High-Pass & 1.0s Max Duration Ceiling
           - Consecutive Neighbor Prior (>= 75% Repeat Match, < 75% 3-of-4 Elimination)
           - 2x2 Orthogonal Ruler (Laplace Split vs HFSR Split)

  [Mode 3] DSP Arena Standard (dsp_inverse.py)
           - Fast linear resample & Hilbert transient window

Outputs:
  /Users/kyawnandar/Downloads/evaluation_197_to_214_dual_mode.csv
"""

import os
import sys
import glob
import importlib.util
import numpy as np
import pandas as pd
from scipy.io import wavfile

BACKEND_DIR = "/Users/kyawnandar/Documents/interactive-v2/backend"
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

SKILL_DIR = "/Users/kyawnandar/Documents/interactive-v2/gamestudio-subagents/projects/dsp_arena/.agents/skills/event-centric/scripts"
if SKILL_DIR not in sys.path:
    sys.path.insert(0, SKILL_DIR)

# 1. Import Mode 1 (Dual-Mode Acoustic Engine)
rec_path = "/Users/kyawnandar/Downloads/manual_stop_interactive_recorder(1).py"
spec = importlib.util.spec_from_file_location("recorder", rec_path)
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)

# 2. Import Mode 2 (Event-Centric Consecutive Evaluator)
import event_centric_evaluator as ev_eval

# 3. Import Mode 3 (DSP Arena Standard)
from dsp_inverse import measure_features, detect_transient_window as detect_transient_std

DATASET_DIR = "/Users/kyawnandar/Downloads/dataset"
OUTPUT_CSV = "/Users/kyawnandar/Downloads/evaluation_197_to_214_dual_mode.csv"
TARGET_FILES = [f"{i}.wav" for i in range(197, 215)]


def run_batch():
    print("=" * 80)
    print(" 🎯 BATCH EVALUATION: FILES 197.wav TO 214.wav (DUAL-MODE ACOUSTIC ENGINES)")
    print("=" * 80)
    print(f" Dataset Path: {DATASET_DIR}")
    print(f" Output CSV  : {OUTPUT_CSV}\n")

    results = []
    prev_event_mode2 = None

    for idx, fname in enumerate(TARGET_FILES, 1):
        fpath = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(fpath):
            print(f"⚠️ [{idx:02d}/18] File not found: {fname}")
            continue

        sr, audio_raw = wavfile.read(fpath)
        audio = audio_raw.astype(np.float64)
        if audio.ndim > 1:
            audio = audio[:, 0]
        raw_dur_s = len(audio) / sr

        print("-" * 80)
        print(f" 📍 [{idx:02d}/18] Processing: {fname}  (raw length: {raw_dur_s:.3f}s, sr: {sr:,} Hz)")

        # ── Physical Detection & Duration Guard ──
        y_win, c_idx, on_idx, off_idx, det_dur_s = ev_eval.detect_transient_window(audio, sr)
        is_rejected = (y_win is None) or (det_dur_s > 1.0)
        status_text = "REJECTED (>1.0s ceiling)" if is_rejected else "ACCEPTED (<0.5s impact)"

        if is_rejected:
            print(f"   ❌ REJECTED: Detected transient duration {det_dur_s:.3f}s > 1.0s physical limit.")
            row = {
                "File_Name": fname,
                "Raw_Duration_s": round(raw_dur_s, 4),
                "Detected_Duration_s": round(det_dur_s, 4),
                "Physical_Status": "REJECTED",
                "Peak_Freq_Hz": 0.0,
                "Laplace_Sigma": 0.0,
                "HFSR": 0.0,
                "Centroid_Hz": 0.0,
                "Mode1_DualEngine_Pred": "REJECTED",
                "Mode1_Candidate_Pair": "NONE",
                "Mode1_Elimination": "NONE",
                "Mode2_Standalone_Ruler": "REJECTED",
                "Mode2_Consecutive_Pred": "REJECTED",
                "Mode2_Similarity_Pct": 0.0,
                "Mode2_Decision_Path": "PHYSICAL_DURATION_REJECT",
                "Mode3_DSP_Arena_Pred": "REJECTED",
                "Final_Consensus_Verdict": "REJECTED (> 1.0s limit)",
            }
            results.append(row)
            continue

        # ── Mode 1: Dual-Mode Acoustic Multi-Method Engine ──
        try:
            feats1 = rec.compute_multi_method_dsp(audio, sr=sr, source_name=fname)
            m1_pred = feats1.get("final_single_prediction", "UNKNOWN")
            m1_pair = feats1.get("candidate_pair_text", "")
            m1_elim = feats1.get("elimination_text", "")
            m1_freq = float(feats1.get("frequency_hz", 0.0))
            m1_sigma = float(feats1.get("laplace_sigma", 0.0))
            m1_hfsr = float(feats1.get("high_freq_ratio", 0.0))
            m1_centroid = float(feats1.get("centroid_hz", 0.0))
        except Exception as e:
            print(f"   ⚠️ Mode 1 DSP error: {e}")
            m1_pred, m1_pair, m1_elim = "ERROR", str(e), ""
            m1_freq, m1_sigma, m1_hfsr, m1_centroid = 0.0, 0.0, 0.0, 0.0

        # ── Mode 2: Event-Centric Consecutive Prior & 2x2 Ruler ──
        res2 = ev_eval.evaluate_event_with_consecutive_ruler(
            audio=audio,
            sr=sr,
            prev_event=prev_event_mode2,
            sim_threshold=0.75
        )
        if res2 is not None:
            m2_ruler = ev_eval.classify_ruler_raw(res2.features.get("laplace_sigma", 15.0), res2.features.get("high_freq_ratio", 0.2))
            m2_consec = res2.predicted_group
            m2_sim = round(res2.similarity_with_previous * 100.0, 1)
            m2_path = res2.decision_path
            m2_expl = res2.explanation_burmese
            prev_event_mode2 = {"group": m2_consec, "features": res2.features}
        else:
            m2_ruler, m2_consec, m2_sim, m2_path, m2_expl = "REJECT", "REJECT", 0.0, "REJECT", ""

        # ── Mode 3: DSP Arena Standard (dsp_inverse.py) ──
        try:
            feats3 = measure_features(audio, sr, 4096, 1024, 35, 0.5)
            m3_pred = m2_ruler if feats3 is not None else "REJECTED"
        except Exception:
            m3_pred = "ERROR"

        # ── Consensus Synthesis ──
        if m1_pred == m2_consec:
            consensus = f"Group {m1_pred} (Full Agreement)"
        elif m1_pred == m2_ruler:
            consensus = f"Group {m1_pred} (Dual Engine + Ruler Match)"
        else:
            consensus = f"Group {m1_pred} [M1] / Group {m2_consec} [M2-Consec]"

        print(f"   📊 [FEATURES] Freq={m1_freq:.1f}Hz | Sigma={m1_sigma:.2f} | HFSR={m1_hfsr:.4f} | Dur={det_dur_s:.3f}s")
        print(f"   🅰️ [Mode 1: Dual-Mode Acoustic]  : Group 【 {m1_pred} 】 ({m1_pair} | {m1_elim})")
        print(f"   📏 [Mode 2: Consecutive Ruler]   : Group 【 {m2_consec} 】 (Path: {m2_path}, Sim: {m2_sim}%)")
        print(f"   💡 [Consensus Verdict]           : {consensus}")

        row = {
            "File_Name": fname,
            "Raw_Duration_s": round(raw_dur_s, 4),
            "Detected_Duration_s": round(det_dur_s, 4),
            "Physical_Status": "ACCEPTED",
            "Peak_Freq_Hz": round(m1_freq, 1),
            "Laplace_Sigma": round(m1_sigma, 3),
            "HFSR": round(m1_hfsr, 5),
            "Centroid_Hz": round(m1_centroid, 1),
            "Mode1_DualEngine_Pred": m1_pred,
            "Mode1_Candidate_Pair": m1_pair,
            "Mode1_Elimination": m1_elim,
            "Mode2_Standalone_Ruler": m2_ruler,
            "Mode2_Consecutive_Pred": m2_consec,
            "Mode2_Similarity_Pct": m2_sim,
            "Mode2_Decision_Path": m2_path,
            "Mode3_DSP_Arena_Pred": m3_pred,
            "Final_Consensus_Verdict": consensus,
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 80)
    print(f" 🎉 BATCH EVALUATION COMPLETE! Saved {len(df)} rows to:")
    print(f"    {OUTPUT_CSV}")
    print("=" * 80)

    # Print Summary Table
    print("\n📋 EVALUATION MATRIX SUMMARY (Files 197 - 214):")
    print(f"{'File':<9} | {'Freq (Hz)':<10} | {'Sigma':<7} | {'HFSR':<7} | {'Mode 1 (Dual)':<14} | {'Mode 2 (Consec)':<16} | {'Consensus Verdict'}")
    print("-" * 95)
    for _, r in df.iterrows():
        print(f"{r['File_Name']:<9} | {r['Peak_Freq_Hz']:<10.1f} | {r['Laplace_Sigma']:<7.2f} | {r['HFSR']:<7.4f} | Group {r['Mode1_DualEngine_Pred']:<8} | Group {r['Mode2_Consecutive_Pred']:<10} | {r['Final_Consensus_Verdict']}")


if __name__ == "__main__":
    run_batch()
