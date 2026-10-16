#!/usr/bin/env python3
"""
pca_evaluation.py — Stage 2: Quantitative Evaluation Script
=============================================================
Computes all metrics required by the H2/H3 falsification protocol from the
architecture review. Reads Stage 1 outputs and produces evaluation_report.json.

Metrics computed:
  • Silhouette score (5D z-scored + PC1-2)
  • Nearest-centroid LOOCV accuracy (balanced)
  • Within-group / between-group distances
  • Separation index
  • Permutation tests (N=1000) for silhouette and separation index
  • Fisher discriminability per feature
  • Effect sizes (Cohen's d for all pairwise group comparisons)
  • Feature correlation matrix
  • Duration-only vs full-PCA comparison (review Recommendation 6)

Usage:
  .venv/bin/python pca_evaluation.py --pca-dir dsp_out/pca
  .venv/bin/python pca_evaluation.py --pca-dir dsp_out/pca_run1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import LeaveOneOut

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
CANONICAL_FEATURES = ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]
N_PERM = 1000  # permutation test iterations
RNG_SEED = 42


def load_pca_coordinates(pca_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load PCA coordinates CSV from Stage 1 output."""
    csv_path = pca_dir / "pca_coordinates.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"PCA coordinates not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    files = [r["file"] for r in rows]
    groups = [r["group"] for r in rows]

    # Extract raw, z-scored, and PC columns
    raw_cols = [f"raw_{feat}" for feat in CANONICAL_FEATURES]
    z_cols = [f"z_{feat}" for feat in CANONICAL_FEATURES]

    # Count PC columns
    pc_cols = sorted([k for k in rows[0].keys() if k.startswith("PC")],
                     key=lambda x: int(x[2:]))

    X_raw = np.array([[float(r[c]) for c in raw_cols] for r in rows], dtype=np.float64)
    Z = np.array([[float(r[c]) for c in z_cols] for r in rows], dtype=np.float64)
    PC = np.array([[float(r[c]) for c in pc_cols] for r in rows], dtype=np.float64)

    return X_raw, Z, PC, files, groups


def encode_labels(groups: list[str]) -> tuple[np.ndarray, list[str]]:
    """Encode string labels to integer labels."""
    unique = sorted(set(groups))
    label_map = {g: i for i, g in enumerate(unique)}
    labels = np.array([label_map[g] for g in groups], dtype=np.int32)
    return labels, unique


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score. Returns 0 if fewer than 2 clusters have ≥2 members."""
    unique_labels = np.unique(labels)
    counts = np.bincount(labels, minlength=unique_labels.max() + 1)
    if np.sum(counts >= 2) < 2:
        return 0.0
    return float(silhouette_score(X, labels))


def compute_distances(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute within-group and between-group mean Euclidean distances."""
    unique_labels = np.unique(labels)
    centroids = {}
    for lab in unique_labels:
        mask = labels == lab
        centroids[lab] = X[mask].mean(axis=0)

    # Within-group: mean distance of each point to its centroid
    within_dists = []
    for i in range(len(X)):
        d = np.linalg.norm(X[i] - centroids[labels[i]])
        within_dists.append(d)
    within_mean = float(np.mean(within_dists))

    # Between-group: mean pairwise centroid distance
    between_dists = []
    for a, b in combinations(unique_labels, 2):
        d = np.linalg.norm(centroids[a] - centroids[b])
        between_dists.append(d)
    between_mean = float(np.mean(between_dists)) if between_dists else 0.0

    return {
        "within_group_mean": within_mean,
        "between_group_mean": between_mean,
        "ratio": between_mean / (within_mean + 1e-12),
    }


def separation_index(X: np.ndarray, labels: np.ndarray) -> float:
    """S = d̄_inter / (d̄_inter + 2·d̄_intra)"""
    d = compute_distances(X, labels)
    inter = d["between_group_mean"]
    intra = d["within_group_mean"]
    return inter / (inter + 2 * intra + 1e-12)


def nearest_centroid_loocv(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Leave-one-out cross-validation with nearest-centroid classifier.
    Returns balanced accuracy (mean per-class recall).
    """
    loo = LeaveOneOut()
    predictions = np.zeros_like(labels)

    for train_idx, test_idx in loo.split(X):
        clf = NearestCentroid()
        clf.fit(X[train_idx], labels[train_idx])
        predictions[test_idx] = clf.predict(X[test_idx])

    # Balanced accuracy: mean per-class recall
    unique_labels = np.unique(labels)
    per_class_recall = []
    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        recall = (predictions[mask] == lab).mean()
        per_class_recall.append(float(recall))

    balanced_acc = float(np.mean(per_class_recall))
    raw_acc = float((predictions == labels).mean())

    return {
        "balanced_accuracy": balanced_acc,
        "raw_accuracy": raw_acc,
        "per_class_recall": {
            str(lab): float(r)
            for lab, r in zip(unique_labels, per_class_recall)
        },
        "n_correct": int((predictions == labels).sum()),
        "n_total": len(labels),
    }


def fisher_discriminability(X: np.ndarray, labels: np.ndarray,
                            feature_names: list[str]) -> dict:
    """
    Fisher discriminant ratio per feature:
    F_j = var(class_means_j) / mean(within_class_var_j)
    """
    unique_labels = np.unique(labels)
    n_features = X.shape[1]
    results = {}

    for j in range(n_features):
        class_means = []
        class_vars = []
        for lab in unique_labels:
            mask = labels == lab
            class_means.append(X[mask, j].mean())
            class_vars.append(X[mask, j].var())

        between_var = np.var(class_means)
        within_var = np.mean(class_vars)
        fisher = between_var / (within_var + 1e-12)
        results[feature_names[j]] = float(fisher)

    return results


def pairwise_cohen_d(X: np.ndarray, labels: np.ndarray,
                     feature_names: list[str], label_names: list[str]) -> dict:
    """Cohen's d for all pairwise group comparisons, all features."""
    unique_labels = np.unique(labels)
    results = {}

    for a, b in combinations(range(len(unique_labels)), 2):
        mask_a = labels == unique_labels[a]
        mask_b = labels == unique_labels[b]
        pair_key = f"{label_names[a]}_vs_{label_names[b]}"
        pair_results = {}

        for j, feat in enumerate(feature_names):
            xa = X[mask_a, j]
            xb = X[mask_b, j]
            pooled_std = np.sqrt(
                ((len(xa) - 1) * xa.var(ddof=1) + (len(xb) - 1) * xb.var(ddof=1))
                / (len(xa) + len(xb) - 2)
            )
            d = (xa.mean() - xb.mean()) / (pooled_std + 1e-12)
            pair_results[feat] = float(d)

        results[pair_key] = pair_results

    return results


def permutation_test(
    X: np.ndarray, labels: np.ndarray, metric_fn, n_perm: int = N_PERM,
    rng_seed: int = RNG_SEED
) -> dict:
    """
    Permutation test: shuffle labels, recompute metric.
    p = (# perm >= observed + 1) / (n_perm + 1)
    """
    rng = np.random.RandomState(rng_seed)
    observed = metric_fn(X, labels)

    null_dist = np.zeros(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        null_dist[i] = metric_fn(X, perm_labels)

    p_value = (np.sum(null_dist >= observed) + 1) / (n_perm + 1)
    null_mean = float(null_dist.mean())
    null_std = float(null_dist.std())
    null_p95 = float(np.percentile(null_dist, 95))

    return {
        "observed": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "null_p95": null_p95,
        "p_value": float(p_value),
        "n_permutations": n_perm,
        "significant_005": float(p_value) < 0.05,
        "significant_bonferroni": float(p_value) < 0.05 / 4,  # 4 metrics
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Quantitative evaluation of PCA representation"
    )
    parser.add_argument(
        "--pca-dir",
        default="dsp_out/pca",
        help="Directory containing Stage 1 PCA outputs (default: dsp_out/pca)",
    )
    args = parser.parse_args()

    pca_dir = Path(args.pca_dir)
    print(f"[Stage 2] Loading from {pca_dir}/")
    print()

    X_raw, Z, PC, files, groups = load_pca_coordinates(pca_dir)
    labels, label_names = encode_labels(groups)
    N = len(labels)
    n_groups = len(label_names)
    group_counts = {g: groups.count(g) for g in label_names}

    print(f"  N={N}, groups={group_counts}")
    print(f"  Chance accuracy (4-class): {1/n_groups:.4f}")
    print()

    report = {
        "metadata": {
            "pca_dir": str(pca_dir),
            "n_events": N,
            "n_groups": n_groups,
            "group_counts": group_counts,
            "label_names": label_names,
            "chance_accuracy": 1 / n_groups,
            "n_permutations": N_PERM,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    # ── 1. Feature-space metrics (5D z-scored) ────────────────────────
    print("═" * 60)
    print("  FEATURE SPACE: 5D Z-SCORED")
    print("═" * 60)

    sil_5d = compute_silhouette(Z, labels)
    dist_5d = compute_distances(Z, labels)
    sep_5d = separation_index(Z, labels)
    loocv_5d = nearest_centroid_loocv(Z, labels)
    fisher_5d = fisher_discriminability(Z, labels, CANONICAL_FEATURES)

    print(f"  Silhouette:        {sil_5d:.4f}")
    print(f"  Within-group dist: {dist_5d['within_group_mean']:.4f}")
    print(f"  Between-group:     {dist_5d['between_group_mean']:.4f}")
    print(f"  Distance ratio:    {dist_5d['ratio']:.4f}")
    print(f"  Separation index:  {sep_5d:.4f}")
    print(f"  LOOCV balanced:    {loocv_5d['balanced_accuracy']:.4f}")
    print(f"  LOOCV raw:         {loocv_5d['raw_accuracy']:.4f}")
    print(f"  Fisher scores:     {fisher_5d}")
    print()

    report["feature_space_5d"] = {
        "silhouette": sil_5d,
        "distances": dist_5d,
        "separation_index": sep_5d,
        "loocv": loocv_5d,
        "fisher_discriminability": fisher_5d,
    }

    # ── 2. PCA space metrics (PC1+PC2) ────────────────────────────────
    print("═" * 60)
    print("  PCA SPACE: PC1 × PC2")
    print("═" * 60)

    PC2d = PC[:, :2]
    sil_pc2 = compute_silhouette(PC2d, labels)
    dist_pc2 = compute_distances(PC2d, labels)
    sep_pc2 = separation_index(PC2d, labels)
    loocv_pc2 = nearest_centroid_loocv(PC2d, labels)

    print(f"  Silhouette:        {sil_pc2:.4f}")
    print(f"  Within-group dist: {dist_pc2['within_group_mean']:.4f}")
    print(f"  Between-group:     {dist_pc2['between_group_mean']:.4f}")
    print(f"  Distance ratio:    {dist_pc2['ratio']:.4f}")
    print(f"  Separation index:  {sep_pc2:.4f}")
    print(f"  LOOCV balanced:    {loocv_pc2['balanced_accuracy']:.4f}")
    print(f"  LOOCV raw:         {loocv_pc2['raw_accuracy']:.4f}")
    print()

    report["pca_space_pc12"] = {
        "silhouette": sil_pc2,
        "distances": dist_pc2,
        "separation_index": sep_pc2,
        "loocv": loocv_pc2,
    }

    # ── 3. Duration+Peak_time only (Recommendation 6 test) ───────────
    print("═" * 60)
    print("  FOCUSED 2D: duration × peak_time (z-scored)")
    print("═" * 60)

    dur_idx = CANONICAL_FEATURES.index("duration")
    pt_idx = CANONICAL_FEATURES.index("peak_time")
    Z_2d = Z[:, [dur_idx, pt_idx]]

    sil_2d = compute_silhouette(Z_2d, labels)
    dist_2d = compute_distances(Z_2d, labels)
    sep_2d = separation_index(Z_2d, labels)
    loocv_2d = nearest_centroid_loocv(Z_2d, labels)

    print(f"  Silhouette:        {sil_2d:.4f}")
    print(f"  Within-group dist: {dist_2d['within_group_mean']:.4f}")
    print(f"  Between-group:     {dist_2d['between_group_mean']:.4f}")
    print(f"  Distance ratio:    {dist_2d['ratio']:.4f}")
    print(f"  Separation index:  {sep_2d:.4f}")
    print(f"  LOOCV balanced:    {loocv_2d['balanced_accuracy']:.4f}")
    print(f"  LOOCV raw:         {loocv_2d['raw_accuracy']:.4f}")
    print()

    report["focused_2d_duration_peaktime"] = {
        "features_used": ["duration", "peak_time"],
        "silhouette": sil_2d,
        "distances": dist_2d,
        "separation_index": sep_2d,
        "loocv": loocv_2d,
    }

    # ── 4. Permutation tests ──────────────────────────────────────────
    print("═" * 60)
    print(f"  PERMUTATION TESTS (N={N_PERM})")
    print("═" * 60)

    print("  [1/4] Silhouette (5D z-scored)...")
    perm_sil_5d = permutation_test(Z, labels, compute_silhouette)
    print(f"         observed={perm_sil_5d['observed']:.4f}, null_mean={perm_sil_5d['null_mean']:.4f}, p={perm_sil_5d['p_value']:.4f}")

    print("  [2/4] Separation index (5D z-scored)...")
    perm_sep_5d = permutation_test(Z, labels, separation_index)
    print(f"         observed={perm_sep_5d['observed']:.4f}, null_mean={perm_sep_5d['null_mean']:.4f}, p={perm_sep_5d['p_value']:.4f}")

    print("  [3/4] Silhouette (PC1×PC2)...")
    perm_sil_pc2 = permutation_test(PC2d, labels, compute_silhouette)
    print(f"         observed={perm_sil_pc2['observed']:.4f}, null_mean={perm_sil_pc2['null_mean']:.4f}, p={perm_sil_pc2['p_value']:.4f}")

    print("  [4/4] Silhouette (duration×peak_time)...")
    perm_sil_2d = permutation_test(Z_2d, labels, compute_silhouette)
    print(f"         observed={perm_sil_2d['observed']:.4f}, null_mean={perm_sil_2d['null_mean']:.4f}, p={perm_sil_2d['p_value']:.4f}")
    print()

    report["permutation_tests"] = {
        "silhouette_5d": perm_sil_5d,
        "separation_index_5d": perm_sep_5d,
        "silhouette_pc12": perm_sil_pc2,
        "silhouette_duration_peaktime": perm_sil_2d,
    }

    # ── 5. Effect sizes ───────────────────────────────────────────────
    print("═" * 60)
    print("  EFFECT SIZES (Cohen's d)")
    print("═" * 60)

    cohen_d = pairwise_cohen_d(Z, labels, CANONICAL_FEATURES, label_names)
    for pair, feats in cohen_d.items():
        max_feat = max(feats, key=lambda k: abs(feats[k]))
        max_d = feats[max_feat]
        print(f"  {pair}: max |d|={abs(max_d):.3f} ({max_feat})")
    print()

    report["effect_sizes"] = cohen_d

    # ── 6. Correlation matrix ─────────────────────────────────────────
    corr = np.corrcoef(Z, rowvar=False)
    report["correlation_matrix"] = {
        f"{CANONICAL_FEATURES[i]}_vs_{CANONICAL_FEATURES[j]}": float(corr[i, j])
        for i in range(len(CANONICAL_FEATURES))
        for j in range(i + 1, len(CANONICAL_FEATURES))
    }

    # ── 7. Go/No-Go assessment ────────────────────────────────────────
    print("═" * 60)
    print("  GO / NO-GO ASSESSMENT")
    print("═" * 60)

    criteria = {}

    # N adequacy
    n_adequate = N >= 120
    min_group_n = min(group_counts.values())
    criteria["n_adequate"] = {
        "value": N, "min_per_group": min_group_n,
        "threshold": 120, "met": n_adequate,
        "verdict": "GO" if n_adequate else "INCONCLUSIVE (underpowered)"
    }

    # Silhouette
    sil_met = sil_5d > 0.10 and perm_sil_5d["p_value"] < 0.0125
    criteria["silhouette_5d"] = {
        "value": sil_5d, "p_value": perm_sil_5d["p_value"],
        "threshold_value": 0.10, "threshold_p": 0.0125,
        "met": sil_met,
        "verdict": "GO" if sil_met else ("NO-GO" if sil_5d < 0.02 else "INCONCLUSIVE")
    }

    # LOOCV
    loocv_met = loocv_5d["balanced_accuracy"] > 0.45
    criteria["loocv_balanced"] = {
        "value": loocv_5d["balanced_accuracy"], "chance": 1 / n_groups,
        "threshold": 0.45, "met": loocv_met,
        "verdict": "GO" if loocv_met else ("NO-GO" if loocv_5d["balanced_accuracy"] < 0.30 else "INCONCLUSIVE")
    }

    # PC variance concentration
    pc12_ratio = float(np.sum(PC[:, :2].var(axis=0)) / np.sum(PC.var(axis=0)))
    # Load from pca_params.json for exact values
    pca_params_path = pca_dir / "pca_params.json"
    if pca_params_path.exists():
        with open(pca_params_path) as f:
            pca_params = json.load(f)
        evr = pca_params["explained_variance_ratio"]
        pc12_evr = sum(evr[:2])
    else:
        pc12_evr = pc12_ratio

    pc_met = pc12_evr > 0.65
    criteria["pc_variance_concentration"] = {
        "pc12_explained": pc12_evr,
        "threshold": 0.65, "met": pc_met,
        "verdict": "GO" if pc_met else ("NO-GO" if pc12_evr < 0.40 else "INCONCLUSIVE")
    }

    # Duration comparison
    dur_better = loocv_2d["balanced_accuracy"] >= loocv_pc2["balanced_accuracy"]
    criteria["duration_vs_pca"] = {
        "duration_peaktime_acc": loocv_2d["balanced_accuracy"],
        "pc12_acc": loocv_pc2["balanced_accuracy"],
        "full_5d_acc": loocv_5d["balanced_accuracy"],
        "duration_wins": dur_better,
        "interpretation": "Simple 2D may suffice" if dur_better else "PCA adds value over 2D subset"
    }

    # Composite
    met_count = sum(1 for c in ["silhouette_5d", "loocv_balanced", "pc_variance_concentration"]
                    if criteria[c]["met"])
    composite = "GO" if met_count >= 3 else ("NO-GO" if met_count <= 1 else "INCONCLUSIVE")
    if not n_adequate:
        composite = f"INCONCLUSIVE (N={N} < 120)"

    criteria["composite"] = {
        "criteria_met": met_count,
        "total_criteria": 3,
        "verdict": composite,
    }

    for key, val in criteria.items():
        verdict = val.get("verdict", "—")
        print(f"  {key}: {verdict}")

    report["go_no_go"] = criteria

    # ── Write report ──────────────────────────────────────────────────
    report_path = pca_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(f"  → {report_path}")

    # ── Final summary ─────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  STAGE 2 COMPLETE — Quantitative Evaluation")
    print("═" * 60)
    print(f"  N={N}, Groups={group_counts}")
    print(f"  Silhouette (5D):          {sil_5d:.4f}  (p={perm_sil_5d['p_value']:.4f})")
    print(f"  Separation (5D):          {sep_5d:.4f}  (p={perm_sep_5d['p_value']:.4f})")
    print(f"  LOOCV balanced (5D):      {loocv_5d['balanced_accuracy']:.4f}")
    print(f"  LOOCV balanced (PC1×2):   {loocv_pc2['balanced_accuracy']:.4f}")
    print(f"  LOOCV balanced (dur×pt):  {loocv_2d['balanced_accuracy']:.4f}")
    print(f"  PC1+PC2 explained:        {pc12_evr:.4f}")
    print(f"  ────────────────────────────────────────")
    print(f"  COMPOSITE VERDICT:        {composite}")
    print()


if __name__ == "__main__":
    main()
