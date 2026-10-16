#!/usr/bin/env python3
"""evaluate_chroma_relative.py — Phase 2B: Relative Chroma & Spectral Ratio Validation.

Evaluates whether relative representation transforms (circular semitone shift,
spectral peak ratios) improve Group A vs Group C discrimination across sessions:
  1. Session 2612: /Users/kyawnandar/Downloads/2612 (37 files, recorded 2025-12-26)
  2. Session Audio: /Users/kyawnandar/Downloads/audio (30 files, recorded 2026-01-04)

Approaches Evaluated:
  - Approach A: Baseline Absolute 12-bin Chroma
  - Approach B1: Peak-Aligned Circular Chroma (circular roll so max bin = 0)
  - Approach B2: Impact-Frequency-Aligned Circular Chroma (circular roll by pitch class of impact_freq_hz)
  - Approach C: Relative Modal Ratios (f2/f1, f3/f1, spectral centroid ratio)
  - Approach D: Combined Baseline DSP + Peak-Aligned Circular Chroma
  - Approach Baseline: Existing 5 DSP features only

All outputs written strictly to:
  dsp_out/chroma_relative_validation/
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

# Ensure root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dsp_inverse
from dsp_inverse import CHROMA_NOTE_NAMES, compute_chroma_12, measure_features

OUT_DIR = ROOT / "dsp_out" / "chroma_relative_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_2612 = Path("/Users/kyawnandar/Downloads/2612")
PATH_AUDIO = Path("/Users/kyawnandar/Downloads/audio")

BASELINE_FEATS = ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]
CHROMA_FULL_COLS = [f"chroma_{name.replace('#', 'Sharp')}" for name in CHROMA_NOTE_NAMES]
CHROMA_CIRC_COLS = [f"chroma_circ_bin_{i}" for i in range(12)]
CHROMA_REF_CIRC_COLS = [f"chroma_ref_circ_bin_{i}" for i in range(12)]
MODAL_RATIO_COLS = ["ratio_f2_f1", "ratio_f3_f1", "ratio_centroid_f1"]

GROUND_TRUTH = {
    "1.wav": "C", "2.wav": "C", "3.wav": "C", "4.wav": "D", "5.wav": "C",
    "6.wav": "C", "7.wav": "D", "8.wav": "A", "9.wav": "B", "10.wav": "A",
    "11.wav": "D", "12.wav": "C", "13.wav": "C", "14.wav": "D", "15.wav": "C",
    "16.wav": "B", "17.wav": "D", "18.wav": "D", "19.wav": "B", "20.wav": "A",
    "21.wav": "B", "22.wav": "C", "23.wav": "D", "24.wav": "D", "25.wav": "C",
    "26.wav": "D", "27.wav": "A", "28.wav": "B", "29.wav": "A", "30.wav": "B",
    "31.wav": "A", "32.wav": "D", "33.wav": "B", "34.wav": "C", "35.wav": "B",
    "36.wav": "D", "37.wav": "A"
}


def compute_pitch_class(freq_hz: float | None) -> int | None:
    """Safe conversion from frequency in Hz to pitch class 0..11 (0=C, 9=A)."""
    if freq_hz is None:
        return None
    try:
        f = float(freq_hz)
    except Exception:
        return None
    if not np.isfinite(f) or f < 20.0 or f > 20000.0:
        return None
    midi = 12.0 * np.log2(f / 440.0) + 69.0
    return int(np.round(midi)) % 12


def compute_circular_chroma(chroma_12: list[float] | np.ndarray, ref_pitch_idx: int | None = None) -> list[float]:
    """Circularly rolls the 12-bin chroma vector so reference pitch class is at index 0."""
    v = np.asarray(chroma_12, dtype=np.float64)
    if v.ndim != 1 or len(v) != 12 or not np.isfinite(v).all():
        return [0.0] * 12
    if np.max(np.abs(v)) < 1e-9:
        return [0.0] * 12

    if ref_pitch_idx is None or not (0 <= ref_pitch_idx < 12):
        ref_idx = int(np.argmax(v))
    else:
        ref_idx = int(ref_pitch_idx)

    shifted = np.roll(v, -ref_idx)
    return [round(float(x), 6) for x in shifted]


def extract_modal_ratios(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extracts top modal resonance frequencies and computes inharmonic ratios.

    Uses high-resolution windowed FFT to locate top prominent peaks in [200, 8000 Hz].
    """
    N = 8192
    if len(y) < N:
        y_pad = np.pad(y, (0, N - len(y)))
    else:
        y_pad = y[:N]

    w = np.hanning(len(y_pad))
    spec = np.abs(np.fft.rfft(y_pad * w))
    freqs = np.fft.rfftfreq(len(y_pad), 1.0 / sr)

    # Restrict to ceramic plate vibration range [200, 8000 Hz]
    valid = (freqs >= 200.0) & (freqs <= 8000.0)
    spec_band = spec[valid]
    freqs_band = freqs[valid]

    # Find spectral peaks with minimum distance of ~50 Hz
    min_dist_bins = max(1, int(50.0 / (sr / len(y_pad))))
    peak_indices, _ = signal.find_peaks(spec_band, distance=min_dist_bins, prominence=np.max(spec_band) * 0.05)

    if len(peak_indices) == 0:
        # Fallback to absolute max
        peak_indices = [int(np.argmax(spec_band))]

    peak_freqs = freqs_band[peak_indices]
    peak_mags = spec_band[peak_indices]

    # Sort peaks by magnitude descending
    sorted_order = np.argsort(peak_mags)[::-1]
    sorted_peaks = peak_freqs[sorted_order]

    f1 = float(sorted_peaks[0])
    f2 = float(sorted_peaks[1]) if len(sorted_peaks) > 1 else f1 * 1.5
    f3 = float(sorted_peaks[2]) if len(sorted_peaks) > 2 else f1 * 2.0

    # Ensure f1 < f2 < f3 ordering
    modes = sorted([f1, f2, f3])
    m1, m2, m3 = modes[0], modes[1], modes[2]

    # Spectral centroid in band
    centroid = float(np.sum(freqs_band * spec_band) / (np.sum(spec_band) + 1e-12))

    return {
        "modal_f1_hz": round(m1, 2),
        "modal_f2_hz": round(m2, 2),
        "modal_f3_hz": round(m3, 2),
        "ratio_f2_f1": round(m2 / (m1 + 1e-6), 4),
        "ratio_f3_f1": round(m3 / (m1 + 1e-6), 4),
        "ratio_centroid_f1": round(centroid / (m1 + 1e-6), 4),
    }


def scan_and_extract_session(session_dir: Path, session_name: str) -> pd.DataFrame:
    """Scans and extracts baseline, absolute chroma, circular chroma, and modal ratios."""
    records = []
    for f in sorted(session_dir.glob("*.wav")):
        norm_name = f.name.strip()
        grp = GROUND_TRUTH.get(norm_name, None)
        if grp is None:
            continue

        try:
            y, native_sr = dsp_inverse.decode_wav(str(f))
            y = np.asarray(y, dtype=np.float64)
            target_sr = 44100
            if native_sr != target_sr:
                y_proc = dsp_inverse.resample_linear(y, native_sr, target_sr)
            else:
                y_proc = y

            fe = measure_features(y_proc, target_sr, nfft=4096, hop=1024, top_db=35.0, win=0.5, fmax=None)
            if fe is None:
                N = 4096
                X = np.fft.rfft(y_proc[:N])
                mag = np.abs(X)
                peak_idx = int(np.argmax(mag))
                fe = {
                    "impact": float(np.max(np.abs(y_proc))),
                    "duration": float(len(y_proc) / target_sr),
                    "peak_time": float(np.argmax(np.abs(y_proc)) / target_sr),
                    "mel_mean_db": float(10 * np.log10(np.mean(mag) + 1e-12)),
                    "impact_freq_hz": float(peak_idx * target_sr / N),
                }

            ch = compute_chroma_12(y_proc, sr=target_sr, detect_transient=True)
            chroma_abs = ch["chroma_12"]

            # Candidate B1: Peak-aligned circular chroma
            chroma_circ_peak = compute_circular_chroma(chroma_abs, ref_pitch_idx=None)

            # Candidate B2: Reference-frequency-aligned circular chroma
            ref_pitch = compute_pitch_class(fe.get("impact_freq_hz"))
            chroma_circ_ref = compute_circular_chroma(chroma_abs, ref_pitch_idx=ref_pitch)

            # Candidate C: Modal frequency ratios
            modal_ratios = extract_modal_ratios(y_proc, target_sr)

            rec = {
                "session": session_name,
                "file": norm_name,
                "group": grp,
                "native_sr": native_sr,
                "proc_sr": target_sr,
                "impact_freq_hz": fe.get("impact_freq_hz", np.nan),
                "impact_ref_pitch_class": ref_pitch if ref_pitch is not None else -1,
                "chroma_status": ch["status"],
                "chroma_dominant_note": ch["dominant_note"],
                "chroma_purity": ch["purity"],
            }

            for k in BASELINE_FEATS:
                rec[k] = fe.get(k, np.nan)

            for col in CHROMA_FULL_COLS:
                rec[col] = ch.get(col, 0.0)

            for i, col in enumerate(CHROMA_CIRC_COLS):
                rec[col] = chroma_circ_peak[i]

            for i, col in enumerate(CHROMA_REF_CIRC_COLS):
                rec[col] = chroma_circ_ref[i]

            rec.update(modal_ratios)
            records.append(rec)

        except Exception as exc:
            print(f"Error processing {norm_name}: {exc}")

    df = pd.DataFrame(records)
    return df


def evaluate_loocv(X: np.ndarray, y: np.ndarray, clf_type: str = "nearest_centroid") -> dict:
    """Performs Leave-One-Out Cross-Validation with permutation testing."""
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)
    classes = np.unique(y)

    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        if clf_type == "nearest_centroid":
            clf = NearestCentroid()
        else:
            clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)

        clf.fit(X_train, y[train_idx])
        y_pred[test_idx] = clf.predict(X_test)

    acc = float(accuracy_score(y, y_pred))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    cm = confusion_matrix(y, y_pred, labels=classes).tolist()
    prec = precision_score(y, y_pred, labels=classes, average=None, zero_division=0)
    rec = recall_score(y, y_pred, labels=classes, average=None, zero_division=0)
    f1 = f1_score(y, y_pred, labels=classes, average=None, zero_division=0)

    # Permutation test (N=1000)
    rng = np.random.default_rng(42)
    n_perm = 1000
    perm_accs = []
    for _ in range(n_perm):
        y_shuffled = rng.permutation(y)
        perm_preds = np.zeros_like(y)
        for train_idx, test_idx in loo.split(X):
            scaler_p = StandardScaler()
            X_tr_p = scaler_p.fit_transform(X[train_idx])
            X_te_p = scaler_p.transform(X[test_idx])
            if clf_type == "nearest_centroid":
                clf_p = NearestCentroid()
            else:
                clf_p = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
            clf_p.fit(X_tr_p, y_shuffled[train_idx])
            perm_preds[test_idx] = clf_p.predict(X_te_p)
        perm_accs.append(balanced_accuracy_score(y_shuffled, perm_preds))

    p_value = float(np.mean([p >= bal_acc for p in perm_accs]))

    return {
        "classifier": clf_type,
        "n_samples": len(y),
        "raw_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "p_value_permutation": round(p_value, 4),
        "classes": [str(c) for c in classes],
        "confusion_matrix": cm,
        "precision_per_class": [round(float(p), 4) for p in prec],
        "recall_per_class": [round(float(r), 4) for r in rec],
        "f1_per_class": [round(float(f), 4) for f in f1],
        "macro_f1": round(float(np.mean(f1)), 4),
    }


def evaluate_cross(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   clf_type: str = "nearest_centroid") -> dict:
    """Evaluates cross-session generalization."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    if clf_type == "nearest_centroid":
        clf = NearestCentroid()
    else:
        clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)

    clf.fit(X_tr_s, y_train)
    y_pred = clf.predict(X_te_s)

    classes = np.unique(np.concatenate([y_train, y_test]))
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    prec = precision_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)

    return {
        "classifier": clf_type,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "raw_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "confusion_matrix": cm,
        "precision_per_class": [round(float(p), 4) for p in prec],
        "recall_per_class": [round(float(r), 4) for r in rec],
        "f1_per_class": [round(float(x), 4) for x in f1],
        "macro_f1": round(float(np.mean(f1)), 4),
    }


def main():
    print("=" * 80)
    print(" 🎯 PHASE 2B: RELATIVE CHROMA & SPECTRAL RATIO VALIDATION")
    print("=" * 80)

    # 1. Extract Sessions
    print("[1/4] Extracting features for Session 2612 (37 files)...")
    df_2612 = scan_and_extract_session(PATH_2612, "2612")
    path_2612_csv = OUT_DIR / "session_2612_relative.csv"
    df_2612.to_csv(path_2612_csv, index=False)
    print(f"  Saved: {path_2612_csv} ({len(df_2612)} rows)")

    print("[2/4] Extracting features for Session Audio (30 files)...")
    df_audio = scan_and_extract_session(PATH_AUDIO, "audio")
    path_audio_csv = OUT_DIR / "session_audio_relative.csv"
    df_audio.to_csv(path_audio_csv, index=False)
    print(f"  Saved: {path_audio_csv} ({len(df_audio)} rows)")

    # 2. Filter for Group A vs Group C
    df_2612_ac = df_2612[df_2612["group"].isin(["A", "C"])].copy()
    df_audio_ac = df_audio[df_audio["group"].isin(["A", "C"])].copy()
    print(f"\nGroup A vs C Subsets:")
    print(f"  Session 2612: Total {len(df_2612_ac)} (A: {sum(df_2612_ac['group']=='A')}, C: {sum(df_2612_ac['group']=='C')})")
    print(f"  Session Audio: Total {len(df_audio_ac)} (A: {sum(df_audio_ac['group']=='A')}, C: {sum(df_audio_ac['group']=='C')})")

    feature_sets = {
        "Baseline_DSP": BASELINE_FEATS,
        "Candidate_A_Absolute_Chroma": CHROMA_FULL_COLS,
        "Candidate_B1_Peak_Circular_Chroma": CHROMA_CIRC_COLS,
        "Candidate_B2_RefPitch_Circular_Chroma": CHROMA_REF_CIRC_COLS,
        "Candidate_C_Modal_Ratios": MODAL_RATIO_COLS,
        "Candidate_D_Baseline_plus_Peak_Circular": BASELINE_FEATS + CHROMA_CIRC_COLS,
    }

    results = {
        "metadata": {
            "session_2612_files": len(df_2612),
            "session_audio_files": len(df_audio),
            "session_2612_ac_count": len(df_2612_ac),
            "session_audio_ac_count": len(df_audio_ac),
        },
        "intra_session_2612": {},
        "intra_session_audio": {},
        "cross_session_2612_to_audio": {},
        "cross_session_audio_to_2612": {},
        "frequency_reference_audit": {},
    }

    # 3. Frequency reference audit statistics
    for sess_name, df_sub in [("session_2612", df_2612_ac), ("session_audio", df_audio_ac)]:
        f_vals = df_sub["impact_freq_hz"].to_numpy()
        p_vals = df_sub["impact_ref_pitch_class"].to_numpy()
        results["frequency_reference_audit"][sess_name] = {
            "freq_mean": round(float(np.mean(f_vals)), 1),
            "freq_std": round(float(np.std(f_vals)), 1),
            "freq_min": round(float(np.min(f_vals)), 1),
            "freq_max": round(float(np.max(f_vals)), 1),
            "pitch_classes_observed": [int(x) for x in sorted(set(p_vals))],
            "pitch_class_counts": {int(k): int(v) for k, v in pd.Series(p_vals).value_counts().items()},
        }

    # 4. Intra-session Evaluations
    print("\n[3/4] Running Intra-session Evaluations...")
    for set_name, cols in feature_sets.items():
        # Session 2612
        X_2612 = df_2612_ac[cols].to_numpy(dtype=np.float64)
        y_2612 = df_2612_ac["group"].to_numpy()
        nc_2612 = evaluate_loocv(X_2612, y_2612, "nearest_centroid")
        lr_2612 = evaluate_loocv(X_2612, y_2612, "logistic_regression")
        results["intra_session_2612"][set_name] = {"nearest_centroid": nc_2612, "logistic_regression": lr_2612}

        # Session Audio
        X_audio = df_audio_ac[cols].to_numpy(dtype=np.float64)
        y_audio = df_audio_ac["group"].to_numpy()
        nc_audio = evaluate_loocv(X_audio, y_audio, "nearest_centroid")
        lr_audio = evaluate_loocv(X_audio, y_audio, "logistic_regression")
        results["intra_session_audio"][set_name] = {"nearest_centroid": nc_audio, "logistic_regression": lr_audio}

        print(f"  {set_name:40s} | 2612 NC BalAcc: {nc_2612['balanced_accuracy']*100:5.1f}% (p={nc_2612['p_value_permutation']:.3f}) | Audio NC BalAcc: {nc_audio['balanced_accuracy']*100:5.1f}% (p={nc_audio['p_value_permutation']:.3f})")

    # 5. Cross-session Evaluations
    print("\n[4/4] Running Cross-session Generalization Evaluations...")
    for set_name, cols in feature_sets.items():
        X_2612 = df_2612_ac[cols].to_numpy(dtype=np.float64)
        y_2612 = df_2612_ac["group"].to_numpy()
        X_audio = df_audio_ac[cols].to_numpy(dtype=np.float64)
        y_audio = df_audio_ac["group"].to_numpy()

        # Train 2612 -> Test Audio
        fwd_nc = evaluate_cross(X_2612, y_2612, X_audio, y_audio, "nearest_centroid")
        fwd_lr = evaluate_cross(X_2612, y_2612, X_audio, y_audio, "logistic_regression")
        results["cross_session_2612_to_audio"][set_name] = {"nearest_centroid": fwd_nc, "logistic_regression": fwd_lr}

        # Train Audio -> Test 2612
        rev_nc = evaluate_cross(X_audio, y_audio, X_2612, y_2612, "nearest_centroid")
        rev_lr = evaluate_cross(X_audio, y_audio, X_2612, y_2612, "logistic_regression")
        results["cross_session_audio_to_2612"][set_name] = {"nearest_centroid": rev_nc, "logistic_regression": rev_lr}

        mean_bal = (fwd_nc["balanced_accuracy"] + rev_nc["balanced_accuracy"]) / 2.0
        print(f"  {set_name:40s} | 2612->Audio: {fwd_nc['balanced_accuracy']*100:5.1f}% | Audio->2612: {rev_nc['balanced_accuracy']*100:5.1f}% | Mean NC BalAcc: {mean_bal*100:5.1f}%")

    out_json = OUT_DIR / "metrics_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ All Phase 2B evaluation metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
