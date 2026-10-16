#!/usr/bin/env python3
"""
pca_experiment.py — Stage 1: Offline PCA Computation
=====================================================
Reads the canonical 5-feature CSV, computes z-score normalization and PCA,
and writes frozen parameter files for reproducible chart rendering.

Outputs (all into --outdir, default dsp_out/pca/):
  norm_params.json      — frozen μ, σ per feature + data hash
  pca_params.json       — eigenvectors, explained variance, sign convention
  pca_coordinates.csv   — event-level PC coordinates + z-scored features
  experiment_config.json — full provenance record

Determinism guarantees:
  • Sign convention: largest-magnitude element of each eigenvector is positive
  • Feature ordering is locked: impact, duration, peak_time, mel_mean_db, impact_freq_hz
  • All outputs are hash-pinned to input data

Usage:
  .venv/bin/python pca_experiment.py --csv dsp_out/run3/features.csv
  .venv/bin/python pca_experiment.py --csv dsp_out/run1/features.csv --outdir dsp_out/pca_run1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════
CANONICAL_FEATURES = ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]
N_FEATURES = len(CANONICAL_FEATURES)
ROUND_TRIP_TOL = 1e-10  # reconstruction error tolerance


def sha256_file(path: str) -> str:
    """SHA-256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 hash of a numpy array's bytes (for provenance)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_canonical_csv(csv_path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load canonical 5-feature CSV.

    Returns:
        X: (N, 5) float64 array in canonical feature order
        files: list of filenames
        groups: list of group labels
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Verify columns
    for col in ["file", "group"] + CANONICAL_FEATURES:
        if col not in rows[0]:
            raise ValueError(f"Missing column '{col}' in CSV. Found: {list(rows[0].keys())}")

    files = [r["file"] for r in rows]
    groups = [r["group"] for r in rows]
    X = np.array(
        [[float(r[feat]) for feat in CANONICAL_FEATURES] for r in rows],
        dtype=np.float64,
    )
    return X, files, groups


def fix_sign_convention(components: np.ndarray) -> np.ndarray:
    """
    Enforce deterministic sign convention: for each eigenvector (row),
    the element with the largest absolute value is positive.

    This prevents sign flips between different SVD implementations
    (numpy, sklearn, JavaScript).
    """
    for i in range(components.shape[0]):
        max_idx = np.argmax(np.abs(components[i]))
        if components[i, max_idx] < 0:
            components[i] *= -1
    return components


def compute_normalization(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute z-score normalization: z_i = (x_i - μ_i) / σ_i

    Returns:
        Z: (N, D) normalized array
        mu: (D,) means
        sigma: (D,) standard deviations
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)  # population std (entire dataset is training set for now)

    # Guard against zero variance
    zero_var = sigma < 1e-15
    if zero_var.any():
        bad_feats = [CANONICAL_FEATURES[i] for i in np.where(zero_var)[0]]
        raise ValueError(f"Zero-variance features: {bad_feats}")

    Z = (X - mu) / sigma
    return Z, mu, sigma


def compute_pca(Z: np.ndarray, n_components: int = N_FEATURES) -> PCA:
    """
    Compute deterministic PCA on z-scored data.

    Uses full SVD (not randomized) for reproducibility.
    """
    pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
    pca.fit(Z)

    # Enforce sign convention
    pca.components_ = fix_sign_convention(pca.components_.copy())

    return pca


def verify_round_trip(Z: np.ndarray, pca: PCA) -> float:
    """
    Verify PCA round-trip: Z → PC → reconstruct → compare.
    Returns max absolute reconstruction error.
    """
    pc_coords = Z @ pca.components_.T
    Z_reconstructed = pc_coords @ pca.components_ + pca.mean_
    max_error = np.max(np.abs(Z - Z_reconstructed))
    return float(max_error)


def build_norm_params(
    mu: np.ndarray, sigma: np.ndarray, n_train: int, data_hash: str
) -> dict:
    """Build the frozen normalization parameters record."""
    return {
        "feature_order": CANONICAL_FEATURES,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "train_n": n_train,
        "train_hash": data_hash,
        "frozen": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "z-score (population std, ddof=0)",
    }


def build_pca_params(
    pca: PCA, norm_hash: str, data_hash: str
) -> dict:
    """Build the frozen PCA parameters record."""
    return {
        "n_components": int(pca.n_components_),
        "components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "singular_values": pca.singular_values_.tolist(),
        "total_explained": float(pca.explained_variance_ratio_.sum()),
        "mean": pca.mean_.tolist(),
        "sign_convention": "largest_magnitude_positive",
        "svd_solver": "full",
        "feature_order": CANONICAL_FEATURES,
        "norm_params_hash": norm_hash,
        "data_hash": data_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def build_experiment_config(
    csv_path: str,
    data_hash: str,
    norm_hash: str,
    pca_hash: str,
    n_events: int,
    n_groups: int,
    group_counts: dict,
    round_trip_error: float,
    correlation_matrix: np.ndarray,
) -> dict:
    """Build the full provenance record."""
    config_data = {
        "experiment_id": hashlib.sha256(
            f"{data_hash}:{norm_hash}:{pca_hash}".encode()
        ).hexdigest()[:16],
        "representation": "pca_on_zscore_5d",
        "data_source": str(csv_path),
        "data_hash": data_hash,
        "norm_params_hash": norm_hash,
        "pca_params_hash": pca_hash,
        "n_events": n_events,
        "n_groups": n_groups,
        "group_counts": group_counts,
        "feature_order": CANONICAL_FEATURES,
        "round_trip_max_error": round_trip_error,
        "round_trip_passed": round_trip_error < ROUND_TRIP_TOL,
        "feature_correlations": {
            f"{CANONICAL_FEATURES[i]}_vs_{CANONICAL_FEATURES[j]}": float(
                correlation_matrix[i, j]
            )
            for i in range(N_FEATURES)
            for j in range(i + 1, N_FEATURES)
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    config_data["config_hash"] = hashlib.sha256(
        json.dumps(config_data, sort_keys=True).encode()
    ).hexdigest()[:16]
    return config_data


def write_pca_coordinates(
    outpath: str,
    files: list[str],
    groups: list[str],
    X: np.ndarray,
    Z: np.ndarray,
    pc_coords: np.ndarray,
) -> None:
    """Write event-level CSV with raw features, z-scored, and PC coordinates."""
    n_pcs = pc_coords.shape[1]
    header = (
        ["file", "group"]
        + [f"raw_{f}" for f in CANONICAL_FEATURES]
        + [f"z_{f}" for f in CANONICAL_FEATURES]
        + [f"PC{i+1}" for i in range(n_pcs)]
    )
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for idx in range(len(files)):
            row = (
                [files[idx], groups[idx]]
                + [f"{X[idx, j]:.6f}" for j in range(N_FEATURES)]
                + [f"{Z[idx, j]:.6f}" for j in range(N_FEATURES)]
                + [f"{pc_coords[idx, j]:.6f}" for j in range(n_pcs)]
            )
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Offline PCA computation on canonical 5-feature CSV"
    )
    parser.add_argument(
        "--csv",
        default="dsp_out/run3/features.csv",
        help="Path to canonical 5-feature CSV (default: dsp_out/run3/features.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="dsp_out/pca",
        help="Output directory (default: dsp_out/pca/)",
    )
    args = parser.parse_args()

    csv_path = args.csv
    outdir = Path(args.outdir)

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)
    data_hash = sha256_file(csv_path)

    # ── 1. Load data ──────────────────────────────────────────────────
    print(f"[1/6] Loading {csv_path}")
    X, files, groups = load_canonical_csv(csv_path)
    n_events = X.shape[0]
    group_set = sorted(set(groups))
    group_counts = {g: groups.count(g) for g in group_set}
    print(f"       N={n_events}, groups={group_counts}")
    print(f"       data_hash={data_hash[:16]}...")

    # ── 2. Compute correlation matrix (pre-normalization diagnostic) ──
    print("[2/6] Computing feature correlation matrix")
    corr = np.corrcoef(X, rowvar=False)
    # Find strongest correlation
    max_corr_val = 0.0
    max_corr_pair = ("", "")
    for i in range(N_FEATURES):
        for j in range(i + 1, N_FEATURES):
            if abs(corr[i, j]) > abs(max_corr_val):
                max_corr_val = corr[i, j]
                max_corr_pair = (CANONICAL_FEATURES[i], CANONICAL_FEATURES[j])
    print(f"       Strongest correlation: {max_corr_pair[0]} × {max_corr_pair[1]} = {max_corr_val:.4f}")

    # ── 3. Z-score normalization ──────────────────────────────────────
    print("[3/6] Computing z-score normalization")
    Z, mu, sigma = compute_normalization(X)
    norm_params = build_norm_params(mu, sigma, n_events, data_hash)
    norm_path = outdir / "norm_params.json"
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    norm_hash = sha256_file(str(norm_path))
    print(f"       μ = {mu}")
    print(f"       σ = {sigma}")
    print(f"       → {norm_path}")

    # ── 4. PCA ────────────────────────────────────────────────────────
    print("[4/6] Computing PCA (full SVD, sign-convention enforced)")
    pca = compute_pca(Z)
    pc_coords = Z @ pca.components_.T  # manual transform to use sign-fixed components

    pca_params = build_pca_params(pca, norm_hash[:16], data_hash)
    pca_path = outdir / "pca_params.json"
    with open(pca_path, "w") as f:
        json.dump(pca_params, f, indent=2)
    pca_hash = sha256_file(str(pca_path))

    print("       Explained variance ratios:")
    cumulative = 0.0
    for i, evr in enumerate(pca.explained_variance_ratio_):
        cumulative += evr
        print(f"         PC{i+1}: {evr:.4f}  (cumulative: {cumulative:.4f})")

    print("       Loadings (feature contributions per PC):")
    for i in range(pca.n_components_):
        loads = {
            CANONICAL_FEATURES[j]: f"{pca.components_[i, j]:+.4f}"
            for j in range(N_FEATURES)
        }
        print(f"         PC{i+1}: {loads}")
    print(f"       → {pca_path}")

    # ── 5. Round-trip verification ────────────────────────────────────
    print("[5/6] Round-trip verification")
    rt_error = verify_round_trip(Z, pca)
    status = "PASS" if rt_error < ROUND_TRIP_TOL else "FAIL"
    print(f"       Max reconstruction error: {rt_error:.2e} — {status}")
    if rt_error >= ROUND_TRIP_TOL:
        print(f"       WARNING: Round-trip error {rt_error:.2e} exceeds tolerance {ROUND_TRIP_TOL:.2e}")

    # ── 6. Write outputs ──────────────────────────────────────────────
    print("[6/6] Writing outputs")

    # PCA coordinates CSV
    coords_path = outdir / "pca_coordinates.csv"
    write_pca_coordinates(str(coords_path), files, groups, X, Z, pc_coords)
    print(f"       → {coords_path}")

    # Experiment config (full provenance)
    config = build_experiment_config(
        csv_path, data_hash, norm_hash[:16], pca_hash[:16],
        n_events, len(group_set), group_counts, rt_error, corr,
    )
    config_path = outdir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"       → {config_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  STAGE 1 COMPLETE — Offline PCA Computation")
    print("═" * 60)
    print(f"  Data:           {csv_path} (N={n_events})")
    print(f"  Groups:         {group_counts}")
    print(f"  Collinearity:   {max_corr_pair[0]}×{max_corr_pair[1]} r={max_corr_val:.4f}")
    print(f"  PC1+PC2:        {sum(pca.explained_variance_ratio_[:2]):.4f}")
    print(f"  PC1+PC2+PC3:    {sum(pca.explained_variance_ratio_[:3]):.4f}")
    print(f"  Round-trip:     {status} (err={rt_error:.2e})")
    print(f"  Experiment ID:  {config['experiment_id']}")
    print(f"  Output:         {outdir}/")
    print()


if __name__ == "__main__":
    main()
