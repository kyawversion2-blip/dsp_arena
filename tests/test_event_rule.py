"""tests/test_event_rule.py — Event-training rule engine tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import event_rule as er


def test_canonical_features_and_units():
    assert set(er.FEATURES) == {
        "impact",
        "duration_s",
        "peak_time_ms",
        "impact_freq_hz",
        "mel_mean_db",
    }
    for f in er.FEATURES:
        assert f in er.UNITS
    assert set(er.GROUPS) == {"A", "B", "C", "D"}
    assert len(er.GROUP_COLORS) == 4


def test_make_demo_events_is_deterministic():
    e1 = er.make_demo_events(seed=42)
    e2 = er.make_demo_events(seed=42)
    assert len(e1) == 16
    assert all(e1[i].id == e2[i].id for i in range(16))
    assert all(
        abs(e1[i].features[f] - e2[i].features[f]) < 1e-9
        for i in range(16)
        for f in er.FEATURES
    )


def test_parse_events_validates_groups():
    rows = [{"id": "X", "group": "Z", "impact": 0.5}]
    try:
        er.parse_events(rows)
        assert False, "should have raised"
    except ValueError:
        pass


def test_search_rules_returns_best_per_feature():
    ev = er.make_demo_events(seed=123)
    res = er.search_rules(ev, "A", "D")
    assert len(res) == len(er.FEATURES)
    # results ordered by train_accuracy desc
    accs = [r.train_accuracy for r in res]
    assert accs == sorted(accs, reverse=True)
    for r in res:
        assert r.train_total >= 1
        assert 0 <= r.train_accuracy <= 1.0
        assert 0 <= r.loo_accuracy <= 1.0
        assert r.rule.feature in er.FEATURES
        assert r.rule.direction in ("<", ">")


def test_evaluate_single_loo_counts():
    ev = er.make_demo_events(seed=7)
    res = er.evaluate_single(ev, "A", "B", "impact_freq_hz", 2500.0, ">")
    # LOO is scored only over the two selected groups (binary problem)
    pair_n = sum(1 for e in ev if e.group in ("A", "B"))
    assert res.loo_total == pair_n
    assert res.loo_correct <= res.loo_total
    assert isinstance(res.misclassified_ids, list)
    assert isinstance(res.loo_misclassified_ids, list)
    assert "A" in res.group_counts and "B" in res.group_counts
    # misclassified / LOO-misclassified ids must belong to the selected groups
    pair_ids = {e.id for e in ev if e.group in ("A", "B")}
    assert set(res.misclassified_ids) <= pair_ids
    assert set(res.loo_misclassified_ids) <= pair_ids


def test_map_chart_scales_to_01_and_provides_ticks():
    ev = er.make_demo_events(seed=0)
    m = er.map_chart(ev, "impact_freq_hz", "duration_s")
    assert "points" in m and "x" in m and "y" in m
    for p in m["points"]:
        assert 0.0 <= p["x"] <= 1.0
        assert 0.0 <= p["y"] <= 1.0
        assert p["group"] in er.GROUPS
    assert m["x"]["feature"] == "impact_freq_hz"
    assert m["y"]["feature"] == "duration_s"
    assert len(m["x"]["ticks"]) >= 3
    assert len(m["y"]["ticks"]) >= 3


def test_two_feature_search_is_bounded_and_returns_none_or_better():
    ev = er.make_demo_events(seed=5)
    primary = er.search_rules(ev, "A", "D")[0]
    two = er.search_two_feature(ev, "A", "D", primary)
    if two is not None:
        assert two.train_accuracy >= primary.train_accuracy - 1e-9
        assert isinstance(two.misclassified_ids, list)

