"""Baseline-derived relationships for DSP Arena optimizer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

PAIRS = ["AB", "AC", "AD", "BC", "BD", "CD"]
GROUPS = ["A", "B", "C", "D"]


def compute_centroids(
    rows: List[Dict],
    features: List[str],
    pooled_mean: List[float],
    pooled_std: List[float],
) -> Dict[str, List[float]]:
    """Per-group mean of z-scored features."""
    sums = {g: [0.0] * len(features) for g in GROUPS}
    counts = {g: 0 for g in GROUPS}
    for r in rows:
        g = r["group"]
        if g not in sums:
            continue
        for k, f in enumerate(features):
            val = r.get(f)
            if val is None or not math.isfinite(val):
                continue
            sums[g][k] += (val - pooled_mean[k]) / max(pooled_std[k], 1e-12)
        counts[g] += 1
    centroids: Dict[str, List[float]] = {}
    for g in GROUPS:
        if counts[g] == 0:
            continue
        centroids[g] = [s / counts[g] for s in sums[g]]
    return centroids


def compute_distance_matrix(centroids: Dict[str, List[float]]) -> Dict[str, float]:
    """RMS standardized pairwise centroid distance."""
    dm: Dict[str, float] = {}
    for i, g in enumerate(GROUPS):
        if g not in centroids:
            continue
        for h in GROUPS[i + 1 :]:
            if h not in centroids:
                continue
            K = len(centroids[g])
            if K == 0:
                continue  # Group has no features for this centroid
            ss = sum((centroids[g][k] - centroids[h][k]) ** 2 for k in range(K))
            dm[g + h] = math.sqrt(ss / K)
    return dm


def infer_relationships(
    distance_matrix: Dict[str, float],
    tau_similar: float,
    tau_different: float,
    alpha: float,
    tau_abs_max: float,
) -> List[Dict]:
    """Classify pairs into attract / neutral using baseline distances."""
    if tau_similar >= tau_different:
        raise ValueError(
            f"tau_similar ({tau_similar}) must be < tau_different ({tau_different})"
        )
    rels: List[Dict] = []
    for pair in PAIRS:
        d = distance_matrix[pair]
        if d < tau_similar:
            rels.append(
                {
                    "pair": pair,
                    "type": "attract",
                    "d_baseline": round(d, 6),
                    "tau_gh": round(min(tau_abs_max, alpha * d), 6),
                    "weight": 1.0,
                    "source": "auto",
                }
            )
        else:
            rels.append(
                {
                    "pair": pair,
                    "type": "neutral",
                    "d_baseline": round(d, 6),
                    "source": "auto",
                }
            )
    for r in rels:
        if r["source"] == "auto" and r["type"] == "attract":
            assert r["d_baseline"] < tau_similar, (
                f"auto attract {r['pair']} violates rule"
            )
    return rels


def _uf(parent: Dict[str, str], x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: Dict[str, str], a: str, b: str) -> None:
    ra, rb = _uf(parent, a), _uf(parent, b)
    if ra != rb:
        parent[ra] = rb


def collapse_gate(
    relations: List[Dict],
    distance_matrix: Dict[str, float],
    anchor: Optional[str] = None,
    protected: Optional[set] = None,
) -> Tuple[List[Dict], List[str]]:
    """Suggestion-time pruning to prevent full component collapse.

    ``protected`` pairs are never demoted/removed (manual curriculum edges).
    They remain in the graph (two-phase: runtime V_min hinge still applies,
    but collapse_gate will not drop them here). Defaults to empty set so all
    existing callers/tests are unaffected.
    """
    protected = protected or set()
    attract = [r for r in relations if r["type"] == "attract" and r["pair"] not in protected]
    if anchor:
        attract = [r for r in attract if anchor not in r["pair"]]

    def num_components(edges: List[Dict]) -> int:
        p = {g: g for g in GROUPS}
        for e in edges:
            _union(p, e["pair"][0], e["pair"][1])
        return len({_uf(p, g) for g in GROUPS})

    removed: List[str] = []
    current = list(attract)
    while num_components(current) == 1 and current:
        current.sort(key=lambda r: (round(r["d_baseline"] - r["tau_gh"], 9), r["pair"]))
        weakest = current.pop(0)
        removed.append(weakest["pair"])

    removed_set = set(removed)
    anchor_involved = {r["pair"] for r in relations if anchor and anchor in r["pair"]}
    result = []
    for r in relations:
        is_attract = r["type"] == "attract"
        if is_attract and (r["pair"] in removed_set or r["pair"] in anchor_involved):
            result.append(
                {
                    "pair": r["pair"],
                    "type": "neutral",
                    "d_baseline": r["d_baseline"],
                    "source": "auto",
                    "_demoted_by": "anchor" if r["pair"] in anchor_involved else "collapse_gate",
                }
            )
        else:
            result.append(r)
    return result, removed


def compute_V(centroids: Dict[str, List[float]]) -> float:
    dm = compute_distance_matrix(centroids)
    vals = list(dm.values())
    if not vals:
        return 0.0
    return sum(v ** 2 for v in vals) / len(vals)


def L_guard_value(V: float, V_min: float, lambda_vmin: float) -> float:
    if V >= V_min:
        return 0.0
    return lambda_vmin * (1.0 - V / V_min) ** 2


def _z_row(row: Dict, features: List[str], scaler: Dict) -> List[float]:
    return [
        (row[f] - scaler["mean"][k]) / max(scaler["std"][k], 1e-12)
        for k, f in enumerate(features)
    ]


def _centroid(rows: List[Dict], features: List[str], scaler: Dict) -> List[float]:
    zs = [_z_row(r, features, scaler) for r in rows]
    K = len(features)
    return [sum(z[k] for z in zs) / len(zs) for k in range(K)]


def L_attract_value(
    rows: List[Dict],
    relationships: List[Dict],
    scaler: Dict,
    features: List[str],
) -> float:
    from collections import defaultdict

    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    centroids = {g: _centroid(rs, features, scaler) for g, rs in by_group.items()}
    total = 0.0
    for rel in relationships:
        if rel["type"] != "attract":
            continue
        g, h = rel["pair"][0], rel["pair"][1]
        if g not in centroids or h not in centroids:
            continue
        cg, ch = centroids[g], centroids[h]
        K = len(features)
        d = math.sqrt(sum((cg[k] - ch[k]) ** 2 for k in range(K)) / K)
        tau = rel.get("tau_gh", 0.0)
        w = rel.get("weight", 1.0)
        total += w * max(0.0, d - tau) ** 2
    return total


def freeze_table(table: Dict, outpath: str) -> str:
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(table, f, indent=2, sort_keys=False)
    return hashlib.sha256(open(outpath, "rb").read()).hexdigest()


def load_frozen(inpath: str) -> Dict:
    with open(inpath) as f:
        return json.load(f)


def select_best_trial(trials: List[Dict], baseline: Dict) -> Dict:
    valid = [t for t in trials if t.get("obj") is not None]
    if not valid:
        raise ValueError("no valid trials")
    best_obj = min(t["obj"] for t in valid)
    candidates = [t for t in valid if abs(t["obj"] - best_obj) <= 1e-12]
    keys = ["sr", "n_fft", "hop_length", "top_db", "window_sec"]

    def dist(t: Dict) -> float:
        return sum(abs(t.get(k, 0) - baseline.get(k, 0)) for k in keys)

    return min(candidates, key=dist)


def build_baseline_table(
    *,
    run_id: str,
    rows: List[Dict],
    features: List[str],
    pooled_mean: List[float],
    pooled_std: List[float],
    theta: Dict,
    anchor: Optional[str],
    replay: Dict,
    provenance: Optional[Dict] = None,
    tau_similar: float = 0.4,
    tau_different: float = 1.0,
    alpha: float = 0.5,
    tau_abs_max: float = 0.5,
    V_min: float = 0.01,
    lambda_vmin: float = 1000.0,
    guard_formula: str = "normalized_hinge_v1",
    protected: Optional[set] = None,
    relationships_override: Optional[List[Dict]] = None,
) -> Dict:
    centroids = compute_centroids(rows, features, pooled_mean, pooled_std)
    dm = compute_distance_matrix(centroids)
    rels = relationships_override or infer_relationships(
        dm, tau_similar, tau_different, alpha, tau_abs_max)
    pruned, removed = collapse_gate(rels, dm, anchor=anchor, protected=protected)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scale_version": "v1",
        "theta": theta,
        "scaler": {
            "features": features,
            "method": "pooled_zscore",
            "mean": pooled_mean,
            "std": pooled_std,
            "ddof": 1,
            "n_samples": len(rows),
        },
        "centroids": centroids,
        "distance_matrix": dm,
        "tau": {
            "tau_similar": tau_similar,
            "tau_different": tau_different,
            "alpha": alpha,
            "tau_abs_max": tau_abs_max,
        },
        "relationships": pruned,
        "collapse_guard": {
            "strategy": "v_min_regularization",
            "V_min": V_min,
            "lambda_vmin": lambda_vmin,
            "guard_formula": guard_formula,
            "suggestion_time_pruning": True,
            "removed_edges": removed,
            "anchor": {"group": anchor, "enabled": anchor is not None},
        },
        "replay": replay,
        "provenance": provenance,
        "frozen": True,
    }
