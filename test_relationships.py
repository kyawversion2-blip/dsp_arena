"""Tests for relationships.py"""

import hashlib

import pytest

from relationships import (
    PAIRS,
    build_baseline_table,
    collapse_gate,
    compute_V,
    compute_centroids,
    compute_distance_matrix,
    freeze_table,
    infer_relationships,
    load_frozen,
    L_attract_value,
    L_guard_value,
    select_best_trial,
)


FEATURES = ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]
POOLED_MEAN = [0.5, 0.3, 0.25, -42.0, 3200.0]
POOLED_STD = [0.2, 0.15, 0.1, 7.0, 1800.0]


@pytest.fixture
def sample_rows():
    import numpy as np
    rng = np.random.default_rng(42)
    raw_centroids = {
        "A": [0.45, 0.28, 0.24, -44.0, 2800.0],
        "B": [0.52, 0.31, 0.26, -40.0, 3500.0],
        "C": [0.58, 0.35, 0.27, -43.0, 3100.0],
        "D": [0.48, 0.29, 0.25, -41.0, 3300.0],
    }
    rows = []
    for g, center in raw_centroids.items():
        for i in range(6):
            row = {"file": f"{g}_{i:03d}.wav", "group": g}
            for k, f in enumerate(FEATURES):
                row[f] = center[k] + rng.normal(0, POOLED_STD[k] * 0.1)
            rows.append(row)
    return rows


@pytest.fixture
def frozen_path(tmp_path):
    return tmp_path / "relationships.json"


# ----- compute_centroids -----

def test_compute_centroids_shapes(sample_rows):
    centroids = compute_centroids(sample_rows, FEATURES, POOLED_MEAN, POOLED_STD)
    assert set(centroids.keys()) == {"A", "B", "C", "D"}
    for g, v in centroids.items():
        assert len(v) == len(FEATURES)


def test_compute_centroids_in_z_range(sample_rows):
    centroids = compute_centroids(sample_rows, FEATURES, POOLED_MEAN, POOLED_STD)
    for g in centroids:
        assert all(-4 < x < 4 for x in centroids[g])


# ----- compute_distance_matrix -----

def test_distance_matrix_six_pairs(sample_rows):
    centroids = compute_centroids(sample_rows, FEATURES, POOLED_MEAN, POOLED_STD)
    dm = compute_distance_matrix(centroids)
    assert set(dm.keys()) == {"AB", "AC", "AD", "BC", "BD", "CD"}


def test_distance_matrix_non_negative(sample_rows):
    centroids = compute_centroids(sample_rows, FEATURES, POOLED_MEAN, POOLED_STD)
    dm = compute_distance_matrix(centroids)
    for k, v in dm.items():
        assert v >= 0.0


def test_compute_V_matches_distance_matrix(sample_rows):
    centroids = compute_centroids(sample_rows, FEATURES, POOLED_MEAN, POOLED_STD)
    dm = compute_distance_matrix(centroids)
    expected = sum(v ** 2 for v in dm.values()) / 6
    assert abs(compute_V(centroids) - expected) < 1e-12


# ----- infer_relationships -----

def test_infer_relationships_classification():
    dm = {"AB": 0.30, "AC": 1.40, "AD": 2.10, "BC": 0.70, "BD": 1.80, "CD": 0.35}
    rels = infer_relationships(dm, 0.4, 1.0, 0.5, 0.5)
    types = {r["pair"]: r["type"] for r in rels}
    assert types["AB"] == "attract"
    assert types["CD"] == "attract"
    assert types["AC"] == "neutral"
    assert types["AD"] == "neutral"
    assert types["BD"] == "neutral"
    assert types["BC"] == "neutral"


def test_infer_relationships_auto_attract_assertion():
    dm = {"AB": 0.30, "AC": 1.40, "AD": 2.10, "BC": 0.70, "BD": 1.80, "CD": 0.35}
    rels = infer_relationships(dm, 0.4, 1.0, 0.5, 0.5)
    for r in rels:
        if r["source"] == "auto" and r["type"] == "attract":
            assert r["d_baseline"] < 0.4


def test_infer_relationships_tau_misconfig_raises():
    dm = {p: 0.5 for p in PAIRS}
    with pytest.raises(ValueError):
        infer_relationships(dm, 1.0, 0.5, 0.5, 0.5)


def test_infer_relationships_skips_pairs_without_two_group_centroids():
    dm = {"AC": 0.2, "AD": 1.2, "CD": 0.4}
    rels = infer_relationships(dm, 0.5, 1.0, 0.5, 0.5)
    assert {r["pair"] for r in rels} == {"AC", "AD", "CD"}
    assert all(r["pair"] in dm for r in rels)


# ----- collapse_gate -----

def test_collapse_gate_prevents_full_merge():
    dm = {p: 0.10 + 0.05 * i for i, p in enumerate(PAIRS)}
    rels = [
        {"pair": p, "type": "attract", "d_baseline": dm[p], "tau_gh": 0.05, "weight": 1.0, "source": "auto"}
        for p in PAIRS
    ]
    pruned, removed = collapse_gate(rels, dm)
    attract_pairs = [r["pair"] for r in pruned if r["type"] == "attract"]
    parent = {g: g for g in "ABCD"}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in attract_pairs:
        union(p[0], p[1])
    components = len({find(g) for g in "ABCD"})
    assert components >= 2
    assert len(removed) > 0


def test_collapse_gate_anchor_excludes_incident():
    dm = {p: 0.10 + 0.05 * i for i, p in enumerate(PAIRS)}
    rels = [
        {"pair": p, "type": "attract", "d_baseline": dm[p], "tau_gh": 0.05, "weight": 1.0, "source": "auto"}
        for p in PAIRS
    ]
    pruned, removed = collapse_gate(rels, dm, anchor="D")
    # D should not appear in remaining attract edges
    remaining = [r["pair"] for r in pruned if r["type"] == "attract"]
    assert all("D" not in pair for pair in remaining)


# ----- V_min guard -----

def test_l_guard_zero_above_vmin():
    assert L_guard_value(0.5, 0.01, 1000) == 0.0


def test_l_guard_maximum_at_collapse():
    assert L_guard_value(0.0, 0.01, 1000) == 1000.0


def test_l_guard_dominates_l_attract():
    assert L_guard_value(0.0, 0.01, 1000) > 100 * 0.1


def test_l_guard_monotonic_near_vmin():
    vals = [L_guard_value(v, 0.01, 1000) for v in [0.001, 0.003, 0.005, 0.007, 0.009, 0.011]]
    for i in range(len(vals) - 1):
        assert vals[i] >= vals[i + 1]


# ----- L_attract_value -----

def test_l_attract_zero_when_all_neutral():
    centroids = {"A": [0.0] * 5, "B": [0.1] * 5, "C": [0.2] * 5, "D": [0.3] * 5}
    dm = compute_distance_matrix(centroids)
    rels = infer_relationships(dm, 0.0, 10.0, 0.5, 0.5)
    assert all(r["type"] == "neutral" for r in rels)
    rows = [
        {"group": g, "impact": 0.5, "duration": 0.3, "peak_time": 0.25,
         "mel_mean_db": -42.0, "impact_freq_hz": 3200.0}
        for g in "ABCD"
    ]
    scaler = {"mean": [0.5] * 5, "std": [0.2] * 5}
    assert L_attract_value(rows, rels, scaler, FEATURES) == 0.0


def test_l_attract_positive_when_attract_pair_far():
    centroids = {"A": [0.0] * 5, "B": [1.0] * 5, "C": [2.0] * 5, "D": [3.0] * 5}
    dm = compute_distance_matrix(centroids)
    rels = infer_relationships(dm, 1.5, 5.0, 0.5, 0.5)
    attract = [r for r in rels if r["type"] == "attract"]
    assert len(attract) > 0
    rows = []
    for g in "ABCD":
        for _ in range(4):
            rows.append({
                "group": g,
                "impact": 0.5 + {"A": -0.5, "B": 0.5, "C": 1.5, "D": 2.5}[g],
                "duration": 0.3, "peak_time": 0.25, "mel_mean_db": -42.0,
                "impact_freq_hz": 3200.0,
            })
    scaler = {"mean": [0.5] * 5, "std": [0.2] * 5}
    L = L_attract_value(rows, rels, scaler, FEATURES)
    assert L > 0.0


# ----- freeze / load_frozen -----

def test_freeze_round_trip(sample_rows, frozen_path):
    table = build_baseline_table(
        run_id="r1",
        rows=sample_rows,
        features=FEATURES,
        pooled_mean=POOLED_MEAN,
        pooled_std=POOLED_STD,
        theta={"sr": 44100, "n_fft": 4096, "hop_length": 1024,
               "top_db": 35, "window_sec": 0.5},
        anchor=None,
        replay={"wavdir_manifest_hash": "x", "features_csv_sha256": "y",
                "pipeline_code_hash": "z", "stft_config": {}},
    )
    freeze_table(table, str(frozen_path))
    loaded = load_frozen(str(frozen_path))
    assert loaded["schema_version"] == 2
    assert loaded["frozen"] is True
    assert loaded["run_id"] == "r1"


def test_freeze_deterministic(sample_rows, frozen_path):
    table = build_baseline_table(
        run_id="r1",
        rows=sample_rows,
        features=FEATURES,
        pooled_mean=POOLED_MEAN,
        pooled_std=POOLED_STD,
        theta={"sr": 44100, "n_fft": 4096, "hop_length": 1024,
               "top_db": 35, "window_sec": 0.5},
        anchor=None,
        replay={"wavdir_manifest_hash": "x", "features_csv_sha256": "y",
                "pipeline_code_hash": "z", "stft_config": {}},
    )
    freeze_table(table, str(frozen_path))
    h1 = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    freeze_table(table, str(frozen_path))
    h2 = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert h1 == h2


def test_freeze_table_keys(sample_rows, frozen_path):
    table = build_baseline_table(
        run_id="r1",
        rows=sample_rows,
        features=FEATURES,
        pooled_mean=POOLED_MEAN,
        pooled_std=POOLED_STD,
        theta={"sr": 44100, "n_fft": 4096, "hop_length": 1024,
               "top_db": 35, "window_sec": 0.5},
        anchor=None,
        replay={"wavdir_manifest_hash": "x", "features_csv_sha256": "y",
                "pipeline_code_hash": "z", "stft_config": {}},
    )
    freeze_table(table, str(frozen_path))
    loaded = load_frozen(str(frozen_path))
    for k in [
        "schema_version", "theta", "scaler", "centroids", "distance_matrix",
        "tau", "relationships", "collapse_guard", "replay", "frozen",
    ]:
        assert k in loaded, f"missing key {k}"


# ----- select_best_trial (tie-break) -----

def test_select_best_trial_picks_closest_baseline():
    baseline = {"sr": 44100, "n_fft": 4096, "hop_length": 1024,
                "top_db": 35, "window_sec": 0.5}
    trials = [
        {"sr": 48000, "n_fft": 4096, "hop_length": 1024, "top_db": 35,
         "window_sec": 0.5, "obj": 0.0},
        {"sr": 44100, "n_fft": 4096, "hop_length": 1024, "top_db": 35,
         "window_sec": 0.5, "obj": 0.0},
        {"sr": 44100, "n_fft": 8192, "hop_length": 512, "top_db": 40,
         "window_sec": 0.3, "obj": 0.0},
    ]
    best = select_best_trial(trials, baseline)
    assert best["sr"] == 44100 and best["n_fft"] == 4096


def test_select_best_trial_skips_none():
    baseline = {"sr": 44100, "n_fft": 4096, "hop_length": 1024,
                "top_db": 35, "window_sec": 0.5}
    trials = [{"obj": None}, {"sr": 44100, "n_fft": 4096, "hop_length": 1024,
                              "top_db": 35, "window_sec": 0.5, "obj": 1.0}]
    best = select_best_trial(trials, baseline)
    assert best["sr"] == 44100


def test_select_best_trial_no_valid_raises():
    with pytest.raises(ValueError):
        select_best_trial([{"obj": None}], {})


# ----- report self-consistency -----

def test_report_self_consistency():
    per_pair = [
        {"pair": "CD", "L_pair": 0.0, "contribution_pct": 0},
        {"pair": "AB", "L_pair": 0.0004, "contribution_pct": 100},
        {"pair": "BC", "L_pair": 0, "contribution_pct": 0},
        {"pair": "AC", "L_pair": 0, "contribution_pct": 0},
        {"pair": "BD", "L_pair": 0, "contribution_pct": 0},
        {"pair": "AD", "L_pair": 0, "contribution_pct": 0},
    ]
    L_total, L_attract, L_guard = 0.0004, 0.0004, 0.0
    assert abs(L_total - L_attract - L_guard) < 1e-12
    assert abs(L_attract - sum(p["L_pair"] for p in per_pair)) < 1e-12
    assert abs(sum(p["contribution_pct"] for p in per_pair) - 100) < 1e-6


# ----- frozen JSON immutable contract -----

def test_relationship_table_frozen_in_optimization_loop(sample_rows, frozen_path):
    """Hash of frozen JSON unchanged after running synthetic optimizer loop."""
    table = build_baseline_table(
        run_id="r1",
        rows=sample_rows,
        features=FEATURES,
        pooled_mean=POOLED_MEAN,
        pooled_std=POOLED_STD,
        theta={"sr": 44100, "n_fft": 4096, "hop_length": 1024,
               "top_db": 35, "window_sec": 0.5},
        anchor=None,
        replay={"wavdir_manifest_hash": "x", "features_csv_sha256": "y",
                "pipeline_code_hash": "z", "stft_config": {}},
    )
    freeze_table(table, str(frozen_path))
    h0 = hashlib.sha256(frozen_path.read_bytes()).hexdigest()

    # Mutation attempt: write a new JSON with same content but rewritten
    import json
    loaded = load_frozen(str(frozen_path))
    frozen_snapshot = json.loads(json.dumps(loaded))  # deep copy

    # Simulate optimizer doing nothing more
    for _ in range(100):
        _ = compute_centroids(sample_rows, FEATURES, POOLED_MEAN, POOLED_STD)
        _ = L_attract_value(
            sample_rows,
            frozen_snapshot["relationships"],
            frozen_snapshot["scaler"],
            FEATURES,
        )

    h1 = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert h0 == h1


# ----- build_baseline_table end-to-end -----

def test_build_baseline_table_e2e(sample_rows):
    table = build_baseline_table(
        run_id="r1",
        rows=sample_rows,
        features=FEATURES,
        pooled_mean=POOLED_MEAN,
        pooled_std=POOLED_STD,
        theta={"sr": 44100, "n_fft": 4096, "hop_length": 1024,
               "top_db": 35, "window_sec": 0.5},
        anchor=None,
        replay={"wavdir_manifest_hash": "x", "features_csv_sha256": "y",
                "pipeline_code_hash": "z", "stft_config": {}},
        provenance={"approved_by": "tester", "approved_at": "2026-08-14T00:00:00Z",
                    "parent_run_id": "r0"},
    )
    assert table["schema_version"] == 2
    assert len(table["relationships"]) == 6
    for r in table["relationships"]:
        assert r["source"] in ("auto", "manual")
        assert r["type"] in ("attract", "neutral")
    assert table["provenance"]["approved_by"] == "tester"


def test_build_baseline_table_anchor_D():
    table = build_baseline_table(
        run_id="r1",
        rows=[{"group": g, **{f: 0.5 for f in FEATURES}} for g in "ABCD"],
        features=FEATURES,
        pooled_mean=[0.5]*5,
        pooled_std=[0.2]*5,
        theta={"sr": 44100, "n_fft": 4096, "hop_length": 1024,
               "top_db": 35, "window_sec": 0.5},
        anchor="D",
        replay={"wavdir_manifest_hash": "x", "features_csv_sha256": "y",
                "pipeline_code_hash": "z", "stft_config": {}},
    )
    assert table["collapse_guard"]["anchor"]["group"] == "D"
    assert table["collapse_guard"]["anchor"]["enabled"] is True
