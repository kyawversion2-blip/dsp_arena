#!/usr/bin/env python3
"""evaluate_chroma_empirical.py — Phase 2: Empirical Chroma Feature Validation.

Evaluates whether 12-D Chroma features discriminate between Group A and Group C
using the two verified cohesive sessions:
  1. Session 2612: /Users/kyawnandar/Downloads/2612 (37 files, recorded 2025-12-26)
  2. Session Audio: /Users/kyawnandar/Downloads/audio (30 files, recorded 2026-01-04)

Configurations evaluated:
  A. Baseline DSP features only (impact, duration, peak_time, mel_mean_db, impact_freq_hz)
  B. 12 Full-window Chroma features only
  C. Baseline DSP + 12 Full-window Chroma features
  D. Baseline DSP + 12 Attack Chroma + 12 Ringdown Chroma features

Analyses:
  • Intra-session LOOCV (Session 2612 and Session Audio separately)
  • Cross-session generalization (Train on Session 2612 -> Test on Session Audio, and vice-versa)
  • ANOVA F-statistic, p-value, Cohen's d for all 12 Chroma bins
  • Within-group vs between-group distances & cosine similarities
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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

OUT_DIR = ROOT / "dsp_out" / "chroma_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATH_2612 = Path("/Users/kyawnandar/Downloads/2612")
PATH_AUDIO = Path("/Users/kyawnandar/Downloads/audio")

BASELINE_FEATS = ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]
CHROMA_FULL_COLS = [f"chroma_{name.replace('#', 'Sharp')}" for name in CHROMA_NOTE_NAMES]
CHROMA_ATTACK_COLS = [f"chroma_attack_{name.replace('#', 'Sharp')}" for name in CHROMA_NOTE_NAMES]
CHROMA_RINGDOWN_COLS = [f"chroma_ringdown_{name.replace('#', 'Sharp')}" for name in CHROMA_NOTE_NAMES]

# Canonical ground-truth mapping
# 1-10 verified from dsp_out/run3/features.csv and dsp_out/test_run/features.csv
# 11-37 verified from labels.csv and dsp_out/run1/features.csv
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


def scan_session_files(session_dir: Path, session_name: str) -> pd.DataFrame:
    """Scans WAV files in session dir and pairs them with authoritative ground truth."""
    records = []
    for f in session_dir.glob("*.wav"):
        norm_name = f.name.strip()
        grp = GROUND_TRUTH.get(norm_name, None)
        records.append({
            "session": session_name,
            "raw_filename": f.name,
            "filename": norm_name,
            "path": str(f),
            "group": grp,
            "has_label": grp is not None,
        })
    df = pd.DataFrame(records)
    return df.sort_values(by="filename").reset_index(drop=True)


def extract_features_from_session(df_files: pd.DataFrame) -> pd.DataFrame:
    """Extracts baseline DSP and Chroma features under consistent resampling (44.1 kHz)."""
    rows = []
    for _, row in df_files.iterrows():
        fn = row["filename"]
        grp = row["group"]
        p_str = row["path"]
        sess = row["session"]

        if not os.path.exists(p_str) or grp is None:
            continue

        p = Path(p_str)
        try:
            y, native_sr = dsp_inverse.decode_wav(str(p))
            y = np.asarray(y, dtype=np.float64)
            target_sr = 44100
            if native_sr != target_sr:
                y_proc = dsp_inverse.resample_linear(y, native_sr, target_sr)
            else:
                y_proc = y

            fe = measure_features(y_proc, target_sr, nfft=4096, hop=1024, top_db=35.0, win=0.5, fmax=None)
            if fe is None:
                # Direct fallback on the slice if window detection ceiling triggers
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

            record = {
                "session": sess,
                "file": fn,
                "group": grp,
                "native_sr": native_sr,
                "proc_sr": target_sr,
                "native_duration_s": round(len(y) / native_sr, 4),
                "chroma_status": ch["status"],
                "chroma_analysis_window_ms": ch["analysis_window_ms"],
                "chroma_attack_window_ms": ch["attack_window_ms"],
                "chroma_ringdown_window_ms": ch["ringdown_window_ms"],
                "chroma_dominant_note": ch["dominant_note"],
                "chroma_purity": ch["purity"],
                "chroma_dominant_note_attack": ch["dominant_note_attack"],
                "chroma_purity_attack": ch["purity_attack"],
                "chroma_dominant_note_ringdown": ch["dominant_note_ringdown"],
                "chroma_purity_ringdown": ch["purity_ringdown"],
            }

            for k in BASELINE_FEATS:
                record[k] = fe.get(k, np.nan)

            for col in CHROMA_FULL_COLS + CHROMA_ATTACK_COLS + CHROMA_RINGDOWN_COLS:
                record[col] = ch.get(col, 0.0)

            rows.append(record)

        except Exception as exc:
            print(f"Error processing {fn}: {exc}")
            continue

    return pd.DataFrame(rows)


def evaluate_classifier_loocv(X: np.ndarray, y: np.ndarray, clf_type: str = "nearest_centroid") -> dict:
    """Leakage-free Leave-One-Out Cross-Validation."""
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)
    classes = np.unique(y)

    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        if clf_type == "nearest_centroid":
            clf = NearestCentroid()
        elif clf_type == "logistic_regression":
            clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
        else:
            raise ValueError(f"Unknown clf_type: {clf_type}")

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


def evaluate_cross_session(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           clf_type: str = "nearest_centroid") -> dict:
    """Evaluates cross-session generalization (Train on Session 1 -> Test on Session 2)."""
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
    f1 = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)

    return {
        "classifier": clf_type,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "raw_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "confusion_matrix": cm,
        "f1_per_class": [round(float(x), 4) for x in f1],
        "macro_f1": round(float(np.mean(f1)), 4),
    }


def compute_group_distances(X: np.ndarray, y: np.ndarray) -> dict:
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    within_dists = []
    between_dists = []

    norm_X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos_sim_matrix = norm_X @ norm_X.T

    within_cos = []
    between_cos = []

    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X_z[i] - X_z[j])
            c = cos_sim_matrix[i, j]
            if y[i] == y[j]:
                within_dists.append(d)
                within_cos.append(c)
            else:
                between_dists.append(d)
                between_cos.append(c)

    within_mean = float(np.mean(within_dists)) if within_dists else 0.0
    between_mean = float(np.mean(between_dists)) if between_dists else 0.0
    ratio = between_mean / (within_mean + 1e-12)

    return {
        "within_group_mean_distance": round(within_mean, 4),
        "between_group_mean_distance": round(between_mean, 4),
        "distance_ratio_inter_intra": round(ratio, 4),
        "within_group_mean_cosine": round(float(np.mean(within_cos)), 4) if within_cos else 0.0,
        "between_group_mean_cosine": round(float(np.mean(between_cos)), 4) if between_cos else 0.0,
    }


def analyze_chroma_bins_ac(df_ac: pd.DataFrame) -> list[dict]:
    bin_stats = []
    group_a = df_ac[df_ac["group"] == "A"]
    group_c = df_ac[df_ac["group"] == "C"]

    for i, name in enumerate(CHROMA_NOTE_NAMES):
        clean_name = name.replace("#", "Sharp")
        col = f"chroma_{clean_name}"
        vals_a = group_a[col].to_numpy()
        vals_c = group_c[col].to_numpy()

        mean_a = float(np.mean(vals_a))
        std_a = float(np.std(vals_a, ddof=1)) if len(vals_a) > 1 else 0.0
        mean_c = float(np.mean(vals_c))
        std_c = float(np.std(vals_c, ddof=1)) if len(vals_c) > 1 else 0.0

        n_a, n_c = len(vals_a), len(vals_c)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_c - 1) * std_c**2) / (n_a + n_c - 2 + 1e-12))
        cohens_d = (mean_a - mean_c) / (pooled_std + 1e-12)

        try:
            f_stat, p_val = stats.f_oneway(vals_a, vals_c)
        except Exception:
            f_stat, p_val = 0.0, 1.0

        bin_stats.append({
            "note": name,
            "column": col,
            "mean_A": round(mean_a, 4),
            "std_A": round(std_a, 4),
            "mean_C": round(mean_c, 4),
            "std_C": round(std_c, 4),
            "delta_A_minus_C": round(mean_a - mean_c, 4),
            "cohens_d": round(float(cohens_d), 4),
            "anova_F": round(float(f_stat), 4),
            "p_value": round(float(p_val), 4),
            "significant_p05": bool(p_val < 0.05),
        })

    return bin_stats


def main():
    print("=" * 80)
    print(" 🎯 PHASE 2: EMPIRICAL CHROMA FEATURE VALIDATION (2 SESSIONS: 37 & 30 FILES)")
    print("=" * 80)

    # 1. Audit both sessions
    df_files_2612 = scan_session_files(PATH_2612, "Session_2612")
    df_files_audio = scan_session_files(PATH_AUDIO, "Session_Audio")

    print(f"\n[Audit] Session 2612: {len(df_files_2612)} files found. All labeled? {df_files_2612['has_label'].all()}")
    print(f"[Audit] Session Audio: {len(df_files_audio)} files found. All labeled? {df_files_audio['has_label'].all()}")

    # 2. Extract features
    print("\n[Extraction] Extracting features for Session 2612 (37 files)...")
    df_feats_2612 = extract_features_from_session(df_files_2612)
    df_feats_2612.to_csv(OUT_DIR / "session_2612_features.csv", index=False)

    print("[Extraction] Extracting features for Session Audio (30 files)...")
    df_feats_audio = extract_features_from_session(df_files_audio)
    df_feats_audio.to_csv(OUT_DIR / "session_audio_features.csv", index=False)

    df_all = pd.concat([df_feats_2612, df_feats_audio], ignore_index=True)
    df_all.to_csv(OUT_DIR / "all_sessions_features.csv", index=False)

    print(f"[Extraction] Done. Total valid records: {len(df_all)}")
    print(f"  Session 2612 counts: {df_feats_2612['group'].value_counts().to_dict()}")
    print(f"  Session Audio counts: {df_feats_audio['group'].value_counts().to_dict()}")

    config_features = {
        "Config_A_Baseline_DSP": BASELINE_FEATS,
        "Config_B_Chroma_Only": CHROMA_FULL_COLS,
        "Config_C_Baseline_Plus_Chroma": BASELINE_FEATS + CHROMA_FULL_COLS,
        "Config_D_Baseline_Plus_DualChroma": BASELINE_FEATS + CHROMA_ATTACK_COLS + CHROMA_RINGDOWN_COLS,
    }

    report = {
        "sessions": {
            "session_2612_count": len(df_feats_2612),
            "session_audio_count": len(df_feats_audio),
            "total_count": len(df_all),
        },
        "session_2612_ac": {},
        "session_audio_ac": {},
        "cross_session_ac": {},
        "chroma_bins_ac_2612": analyze_chroma_bins_ac(df_feats_2612[df_feats_2612["group"].isin(["A", "C"])]),
        "chroma_bins_ac_audio": analyze_chroma_bins_ac(df_feats_audio[df_feats_audio["group"].isin(["A", "C"])]),
    }

    # =========================================================================
    # Part 1: Intra-Session Evaluation (Group A vs Group C)
    # =========================================================================
    for sess_name, df_s, key in [("Session 2612 (2025-12-26)", df_feats_2612, "session_2612_ac"),
                                 ("Session Audio (2026-01-04)", df_feats_audio, "session_audio_ac")]:
        df_ac = df_s[df_s["group"].isin(["A", "C"])].copy()
        y_ac = (df_ac["group"] == "C").to_numpy().astype(int)
        nA = sum(df_ac["group"] == "A")
        nC = sum(df_ac["group"] == "C")

        print("\n" + "=" * 65)
        print(f" 📊 INTRA-SESSION EVALUATION: {sess_name} — A vs C (A={nA}, C={nC}, Total={len(df_ac)})")
        print("=" * 65)

        for cfg_name, feat_cols in config_features.items():
            X = df_ac[feat_cols].to_numpy()
            d_info = compute_group_distances(X, y_ac)
            nc_eval = evaluate_classifier_loocv(X, y_ac, clf_type="nearest_centroid")
            lr_eval = evaluate_classifier_loocv(X, y_ac, clf_type="logistic_regression")

            report[key][cfg_name] = {
                "n_features": len(feat_cols),
                "distances": d_info,
                "nearest_centroid_loocv": nc_eval,
                "logistic_regression_loocv": lr_eval,
            }

            print(f"\n--- {cfg_name} (D = {len(feat_cols)}) ---")
            print(f"  Distance Inter/Intra: {d_info['distance_ratio_inter_intra']:.4f} | Within Cosine: {d_info['within_group_mean_cosine']:.4f} | Between Cosine: {d_info['between_group_mean_cosine']:.4f}")
            print(f"  NearestCentroid Bal Acc: {nc_eval['balanced_accuracy']*100:.1f}% (p={nc_eval['p_value_permutation']:.3f}) | CM [A, C]: {nc_eval['confusion_matrix']}")
            print(f"  LogisticReg     Bal Acc: {lr_eval['balanced_accuracy']*100:.1f}% (p={lr_eval['p_value_permutation']:.3f}) | CM [A, C]: {lr_eval['confusion_matrix']}")

    # =========================================================================
    # Part 2: Cross-Session Evaluation (Train on 2612 -> Test on Audio, and vice-versa)
    # =========================================================================
    print("\n" + "=" * 65)
    print(" 🔄 CROSS-SESSION GENERALIZATION EVALUATION: GROUP A vs GROUP C")
    print("=" * 65)

    ac_2612 = df_feats_2612[df_feats_2612["group"].isin(["A", "C"])].copy()
    ac_audio = df_feats_audio[df_feats_audio["group"].isin(["A", "C"])].copy()

    y_tr = (ac_2612["group"] == "C").to_numpy().astype(int)
    y_te = (ac_audio["group"] == "C").to_numpy().astype(int)

    for cfg_name, feat_cols in config_features.items():
        X_tr = ac_2612[feat_cols].to_numpy()
        X_te = ac_audio[feat_cols].to_numpy()

        res_forward = evaluate_cross_session(X_tr, y_tr, X_te, y_te, clf_type="nearest_centroid")
        res_reverse = evaluate_cross_session(X_te, y_te, X_tr, y_tr, clf_type="nearest_centroid")

        report["cross_session_ac"][cfg_name] = {
            "train_2612_test_audio": res_forward,
            "train_audio_test_2612": res_reverse,
        }

        print(f"\n--- {cfg_name} (D = {len(feat_cols)}) ---")
        print(f"  Train 2612 -> Test Audio: Bal Acc = {res_forward['balanced_accuracy']*100:.1f}% | CM [A, C]: {res_forward['confusion_matrix']}")
        print(f"  Train Audio -> Test 2612: Bal Acc = {res_reverse['balanced_accuracy']*100:.1f}% | CM [A, C]: {res_reverse['confusion_matrix']}")

    # Save JSON report
    report_json = OUT_DIR / "evaluation_report.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] Saved complete evaluation report JSON to {report_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
