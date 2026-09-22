#!/usr/bin/env python3
"""
partition_scout.py — 4-Class to 2-Group Binary Partition Scout
=============================================================================
Statistical & Architectural Engine for Breaking the ~45% Multi-Class Plateau:

  1. 7 Partitions Evaluation:
     - 3 Balanced:   AB|CD, AC|BD, AD|BC
     - 4 One-vs-Rest: A|BCD, B|ACD, C|ABD, D|ABC

  2. Multiple-Testing Correction via Within-Session Permutation (FWER):
     - Shuffles class labels strictly WITHIN each recording session/block (no session leakage).
     - Computes max |d| across all 7 partitions x K features over 2,000 iterations.
     - Family-wise empirical p-value for the global null distribution.

  3. Scale & Metric Invariance:
     - Log10 transforms for ratios and frequency features.
     - Univariate AUC & Cohen's d.

  4. 2-Feature LDA Pair Search with Fold-back Coefficients:
     - Searches best 2-feature pairs for top partitions.
     - Folds z-score scaler constants into raw coefficients: w1*x1 + w2*x2 + b = 0.
     - Structural Gating: If Leave-One-Session-Out (LOSO) CV < 60%, flagged "NOT REPORTABLE".
=============================================================================
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


PARTITIONS = {
    "AB|CD": (["A", "B"], ["C", "D"]),
    "AC|BD": (["A", "C"], ["B", "D"]),
    "AD|BC": (["A", "D"], ["B", "C"]),
    "A|BCD": (["A"], ["B", "C", "D"]),
    "B|ACD": (["B"], ["A", "C", "D"]),
    "C|ABD": (["C"], ["A", "B", "D"]),
    "D|ABC": (["D"], ["A", "B", "C"]),
}


def compute_cohens_d(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Vectorized Cohen's d across columns."""
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return np.zeros(x0.shape[1])
    m0 = np.mean(x0, axis=0)
    m1 = np.mean(x1, axis=0)
    v0 = np.var(x0, axis=0, ddof=1)
    v1 = np.var(x1, axis=0, ddof=1)
    s_pool = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
    s_pool = np.where(s_pool < 1e-12, 1e-12, s_pool)
    return (m1 - m0) / s_pool


def compute_auc_vectorized(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Compute AUC for each feature column via Mann-Whitney U."""
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return np.full(x0.shape[1], 0.5)
    aucs = np.zeros(x0.shape[1])
    for j in range(x0.shape[1]):
        try:
            stat, _ = stats.mannwhitneyu(x1[:, j], x0[:, j], alternative='two-sided')
            u_auc = stat / (n0 * n1)
            aucs[j] = max(u_auc, 1.0 - u_auc)  # Two-sided discriminatory capacity
        except Exception:
            aucs[j] = 0.5
    return aucs


def detect_sessions(df: pd.DataFrame, session_col: str = None) -> np.ndarray:
    """Identify or synthesize session clusters."""
    if session_col and session_col in df.columns:
        return pd.factorize(df[session_col])[0]
    
    # Check common session candidates
    for c in ["session", "Session", "batch", "Batch", "File_Name", "file"]:
        if c in df.columns:
            # If multiple rows per file, file is the session
            val_counts = df[c].value_counts()
            if (val_counts > 1).sum() >= 2:
                return pd.factorize(df[c])[0]

    # Fallback: Partition contiguous recordings into synthetic session blocks of size ~15
    n = len(df)
    block_size = max(8, min(20, n // 8 if n >= 40 else 10))
    sessions = np.arange(n) // block_size
    return sessions


def run_within_session_permutation(
    X: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42
) -> tuple[np.ndarray, dict]:
    """
    Permutation test on the MAXIMUM |d| across all 7 partitions x K features.
    Shuffles labels strictly WITHIN each session block to prevent session leakage.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    unique_sessions = np.unique(sessions)
    session_indices = [np.where(sessions == s)[0] for s in unique_sessions]

    # Calculate observed |d| for all partitions
    obs_effects = {}
    for p_name, (g0_classes, g1_classes) in PARTITIONS.items():
        idx0 = np.isin(labels, g0_classes)
        idx1 = np.isin(labels, g1_classes)
        if idx0.sum() < 3 or idx1.sum() < 3:
            continue
        d_vals = compute_cohens_d(X[idx0], X[idx1])
        obs_effects[p_name] = np.abs(d_vals)

    # Detect if session blocks are homogeneous (single-class per session file)
    session_is_homogeneous = all(len(np.unique(labels[s_idx])) <= 1 for s_idx in session_indices)
    if session_is_homogeneous:
        print("  [Auto-Detect] Sessions are homogeneous (single-class files). Applying Cluster-Level Permutation.")
        session_classes = [labels[s_idx[0]] for s_idx in session_indices]

    # 2000 Permutations of Max |d|
    max_d_null = np.zeros(n_permutations)
    perm_labels = labels.copy()

    for b in range(n_permutations):
        if session_is_homogeneous:
            # Cluster-level permutation: permute the assigned class of each entire file/session
            shuffled_sess_classes = rng.permutation(session_classes)
            for s_i, s_idx in enumerate(session_indices):
                perm_labels[s_idx] = shuffled_sess_classes[s_i]
        else:
            # Within-session shuffle: permute labels within interleaved sessions
            for s_idx in session_indices:
                perm_labels[s_idx] = rng.permutation(labels[s_idx])

        max_stat_b = 0.0
        for p_name, (g0_classes, g1_classes) in PARTITIONS.items():
            idx0 = np.isin(perm_labels, g0_classes)
            idx1 = np.isin(perm_labels, g1_classes)
            if idx0.sum() < 3 or idx1.sum() < 3:
                continue
            d_perm = compute_cohens_d(X[idx0], X[idx1])
            max_p = np.max(np.abs(d_perm))
            if max_p > max_stat_b:
                max_stat_b = max_p

        max_d_null[b] = max_stat_b

    return max_d_null, obs_effects


def evaluate_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray
) -> float:
    """Leave-One-Session-Out Cross Validation Balanced Accuracy using LDA."""
    unique_sessions = np.unique(sessions)
    if len(unique_sessions) < 2:
        return 0.50

    y_true_all = []
    y_pred_all = []

    for test_s in unique_sessions:
        train_idx = (sessions != test_s)
        test_idx = (sessions == test_s)

        y_train = y[train_idx]
        y_test = y[test_idx]

        # Check binary presence in both sets
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        X_train = X[train_idx]
        X_test = X[test_idx]

        # Standardize using train statistics
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std = np.where(std < 1e-12, 1.0, std)

        X_train_std = (X_train - mu) / std
        X_test_std = (X_test - mu) / std

        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_std, y_train)
        preds = clf.predict(X_test_std)

        y_true_all.extend(y_test)
        y_pred_all.extend(preds)

    if len(y_true_all) == 0:
        return 0.50

    return float(balanced_accuracy_score(y_true_all, y_pred_all))


def find_best_lda_pair(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    sessions: np.ndarray
) -> dict:
    """
    Exhaustive search for the best 2-feature LDA pair.
    Fold scaler constants back into raw coefficients: w1*x1 + w2*x2 + b = 0.
    """
    n_features = X.shape[1]
    best_res = None
    best_score = -1.0

    for i in range(n_features):
        for j in range(i + 1, n_features):
            X_sub = X[:, [i, j]]
            cv_score = evaluate_loso_cv(X_sub, y, sessions)

            if cv_score > best_score:
                best_score = cv_score

                # Fit on full dataset to produce deployed formula
                mu = np.mean(X_sub, axis=0)
                sigma = np.std(X_sub, axis=0)
                sigma = np.where(sigma < 1e-12, 1.0, sigma)

                X_std = (X_sub - mu) / sigma
                lda = LinearDiscriminantAnalysis()
                lda.fit(X_std, y)

                w_std = lda.coef_[0]
                b_std = lda.intercept_[0]

                # Fold-back constants to raw units:
                # w1*(x1 - mu1)/sigma1 + w2*(x2 - mu2)/sigma2 + b_std = 0
                # => (w1/sigma1)*x1 + (w2/sigma2)*x2 + (b_std - w1*mu1/sigma1 - w2*mu2/sigma2) = 0
                w1_raw = float(w_std[0] / sigma[0])
                w2_raw = float(w_std[1] / sigma[1])
                b_raw = float(b_std - (w_std[0] * mu[0] / sigma[0]) - (w_std[1] * mu[1] / sigma[1]))

                feat1 = feature_names[i]
                feat2 = feature_names[j]

                # Gate formula if CV < 60%
                is_reportable = (cv_score >= 0.60)
                formula_str = (
                    f"({w1_raw:+.4e}) * [{feat1}] + ({w2_raw:+.4e}) * [{feat2}] + ({b_raw:+.4e}) > 0"
                    if is_reportable else "NOT REPORTABLE (CV < 60% Gating Rule Triggered)"
                )

                best_res = {
                    "feat1": feat1,
                    "feat2": feat2,
                    "feat1_idx": i,
                    "feat2_idx": j,
                    "loso_cv_score": cv_score,
                    "is_reportable": is_reportable,
                    "w1_raw": w1_raw,
                    "w2_raw": w2_raw,
                    "b_raw": b_raw,
                    "formula": formula_str,
                }

    return best_res


def run_partition_scout(csv_path: str, session_col: str = None, n_perm: int = 2000):
    print("=" * 80)
    print(" 🔬 4-CLASS TO 2-GROUP BINARY PARTITION SCOUT")
    print(f" Target CSV: {csv_path}")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    # Locate class label column
    label_col = None
    for cand in ["Group", "group", "Ground_Truth", "ground_truth", "label", "Result"]:
        if cand in df.columns:
            label_col = cand
            break

    if not label_col:
        raise ValueError(f"Could not locate class label column in {csv_path}. Columns: {df.columns.tolist()}")

    # Filter to only valid A, B, C, D classes
    df = df[df[label_col].isin(["A", "B", "C", "D"])].copy()
    labels = df[label_col].values
    print(f"Total samples: {len(df)} across classes {np.unique(labels, return_counts=True)}")

    # Detect Sessions
    sessions = detect_sessions(df, session_col)
    print(f"Detected {len(np.unique(sessions))} distinct sessions/blocks for within-session permutation & LOSO CV.")

    # Extract numerical features
    exclude_cols = [
        label_col, "File_No", "file", "File_Name", "Impact", "Physical_Status",
        "2x2_Ruler_Baseline", "Decision_Path", "Event_Centric_Prediction"
    ]
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)
    ]
    print(f"Discovered {len(numeric_cols)} candidate numerical features: {numeric_cols}")

    # Prepare feature matrix X with log10 transform for ratios and large frequency spans
    X_raw = df[numeric_cols].values.astype(np.float64)

    # Handle NaNs or Infs
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e6, neginf=-1e6)

    # Apply log10 to ratio and frequency features
    transformed_cols = []
    X = np.zeros_like(X_raw)
    for j, col in enumerate(numeric_cols):
        col_lower = col.lower()
        if "ratio" in col_lower or "hfsr" in col_lower:
            # log10(val + 1e-4) for ratio symmetry
            X[:, j] = np.log10(np.maximum(X_raw[:, j], 1e-6) + 1e-4)
            transformed_cols.append(f"log10({col})")
        elif "freq" in col_lower or "hz" in col_lower or "centroid" in col_lower:
            # log10 for frequency scale
            X[:, j] = np.log10(np.maximum(X_raw[:, j], 1.0))
            transformed_cols.append(f"log10({col})")
        else:
            X[:, j] = X_raw[:, j]
            transformed_cols.append(col)

    # 1. Run Within-Session Permutation Test (2000 iter)
    print(f"\nRunning Within-Session Permutation Test ({n_perm} iterations) on Max |d|...")
    null_dist, obs_effects = run_within_session_permutation(X, labels, sessions, n_permutations=n_perm)

    null_95 = np.percentile(null_dist, 95)
    null_99 = np.percentile(null_dist, 99)
    null_max = np.max(null_dist)

    print(f"  Null Distribution Max |d| Thresholds (Family-Wise across 7 partitions x {len(numeric_cols)} features):")
    print(f"    - 95th Percentile: {null_95:.4f}")
    print(f"    - 99th Percentile: {null_99:.4f}")
    print(f"    - Max Null Effect: {null_max:.4f}")

    # 2. Evaluate each of the 7 partitions
    print("\n" + "=" * 115)
    print(f"{'Partition':<10} | {'Best Feature':<25} | {'Max |d|':<8} | {'AUC':<7} | {'FWER p-val':<11} | {'LOSO CV':<8} | {'Status'}")
    print("=" * 115)

    results_table = []

    for p_name, (g0, g1) in PARTITIONS.items():
        if p_name not in obs_effects:
            continue

        d_array = obs_effects[p_name]
        best_feat_idx = int(np.argmax(d_array))
        best_d = float(d_array[best_feat_idx])
        best_feat = transformed_cols[best_feat_idx]

        # Calculate AUC for best feature
        idx0 = np.isin(labels, g0)
        idx1 = np.isin(labels, g1)
        auc_vals = compute_auc_vectorized(X[idx0], X[idx1])
        best_auc = float(auc_vals[best_feat_idx])

        # Family-wise empirical p-value: fraction of null permutations where null_max >= observed best_d
        fwer_p = float((np.sum(null_dist >= best_d) + 1) / (len(null_dist) + 1))

        # Binary labels for LOSO CV (0 for g0, 1 for g1)
        sub_mask = idx0 | idx1
        y_binary = np.where(idx1[sub_mask], 1, 0)
        X_sub = X[sub_mask]
        sess_sub = sessions[sub_mask]

        # 2-Feature LDA pair search on this partition
        lda_res = find_best_lda_pair(X_sub, y_binary, transformed_cols, sess_sub)
        cv_score = lda_res["loso_cv_score"]

        # Empirical Verdict
        is_significant = (best_d > null_95) and (fwer_p < 0.05)
        is_deployable = is_significant and (cv_score >= 0.60)

        if is_deployable:
            status = "✅ GENUINE SIGNAL & DEPLOYABLE"
        elif is_significant:
            status = "⚠️ STATISTICALLY SIGNIFICANT (CV GATED < 60%)"
        else:
            status = "❌ NULL HYPOTHESIS (NO SIGNAL)"

        results_table.append({
            "partition": p_name,
            "best_feature": best_feat,
            "max_d": best_d,
            "auc": best_auc,
            "fwer_p": fwer_p,
            "loso_cv": cv_score,
            "status": status,
            "lda_pair": lda_res,
        })

        print(f"{p_name:<10} | {best_feat:<25} | {best_d:<8.4f} | {best_auc:<7.4f} | {fwer_p:<11.4f} | {cv_score*100:<7.1f}% | {status}")

    print("=" * 115)

    # 3. Print Top Deployable Formulas
    print("\n📋 FORMULA SYNTHESIS & REAL-TIME GATE CHECK:")
    deployable_found = False
    for r in results_table:
        lda = r["lda_pair"]
        print(f"\n--- Partition: {r['partition']} (Best Pair: {lda['feat1']} + {lda['feat2']}) ---")
        print(f"  LOSO CV Balanced Accuracy: {lda['loso_cv_score']*100:.1f}%")
        print(f"  Formula: {lda['formula']}")
        if lda["is_reportable"]:
            deployable_found = True

    if not deployable_found:
        print("\n🚨 RIGOROUS VERDICT: 0 of 7 partitions crossed the 60% LOSO CV threshold after family-wise null control.")
        print("  Recommendation: Feature families exhausted on current microphone setup. Transition to controlled impact rig.")

    return results_table, null_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-Class to 2-Group Binary Partition Scout")
    parser.add_argument("csv_path", help="Path to features CSV file")
    parser.add_argument("--session_col", default=None, help="Name of session/block column")
    parser.add_argument("--perms", type=int, default=2000, help="Number of within-session permutations")
    args = parser.parse_args()

    run_partition_scout(args.csv_path, session_col=args.session_col, n_perm=args.perms)
