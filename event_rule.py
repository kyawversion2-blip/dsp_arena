#!/usr/bin/env python3
"""event_rule.py — Event-training + rule-discovery engine.

Canonical features (mimic dsp_inverse.py measure_features + browser DSP Arena
chart conventions):
    impact          peak RMS, unitless   (fmt: 4 dp)
    duration_s      RMS-decay duration, s (fmt: 4 dp + 's')
    peak_time_ms    10-90 rise time, ms  (fmt: 2 dp + 'ms')
    impact_freq_hz  dominant freq, Hz    (fmt: 1 dp + 'Hz')
    mel_mean_db     mel-power mean, dB   (fmt: 2 dp)

Public API (pure functions, no side effects):
    parse_events(rows)         list[Event]
    evaluate_rule(events, g0, g1, feature, threshold, direction)
                                RuleResult
    search_best_threshold(events, g0, g1, feature)
                                Rule
    search_rules(events, g0, g1)
                                list[RuleResult] (one per feature, plus best)
    search_two_feature(events, g0, g1, primary_result)
                                RuleResult | None
    map_chart(events, x, y)    dict for scatter-chart rendering
    make_demo_events(seed=42)  deterministic synthetic dataset (4 groups × 4)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Canonical schema (authoritative; do NOT invent alternate names)
# ---------------------------------------------------------------------------
FEATURES: List[str] = [
    "impact",
    "duration_s",
    "peak_time_ms",
    "impact_freq_hz",
    "mel_mean_db",
]
UNITS: Dict[str, str] = {
    "impact": "RMS",
    "duration_s": "s",
    "peak_time_ms": "ms",
    "impact_freq_hz": "Hz",
    "mel_mean_db": "dB",
}
GROUPS: Tuple[str, ...] = ("A", "B", "C", "D")
GROUP_COLORS = {
    "A": "#e74c3c",
    "B": "#3498db",
    "C": "#2ecc71",
    "D": "#9b59b6",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Event:
    id: str
    group: str  # one of GROUPS
    features: Dict[str, float]  # must contain all canonical FEATURE_NAMEs


@dataclass
class Rule:
    feature: str
    direction: str  # "<" : predict g0 when feature < threshold
    threshold: float


@dataclass
class RuleResult:
    rule: Rule
    train_correct: int
    train_total: int
    train_accuracy: float
    loo_correct: int
    loo_total: int
    loo_accuracy: float
    # event ids misclassified on training set
    misclassified_ids: List[str] = field(default_factory=list)
    # event ids misclassified on leave-one-out
    loo_misclassified_ids: List[str] = field(default_factory=list)
    group_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(v: object) -> float:
    """Convert a possibly-missing / non-numeric value to a finite float."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _balanced_accuracy(correct_0: int, total_0: int, correct_1: int, total_1: int) -> float:
    """F1-inspired balanced accuracy (mean of recall per class). Handles degenerate counts."""
    r0 = correct_0 / max(total_0, 1)
    r1 = correct_1 / max(total_1, 1)
    return (r0 + r1) / 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_events(rows: List[dict]) -> List[Event]:
    """Normalize raw row dicts (CSV-style) into Event objects.

    Expected columns: 'id' (str), 'group' (A/B/C/D), plus any subset of FEATURES.
    Missing canonical features default to 0.0; missing 'id' gets auto-serial #.
    """
    events = []
    for i, row in enumerate(rows):
        eid = str(row.get("id", f"evt_{i+1:03d}")).strip()
        grp = str(row.get("group", "")).strip().upper()
        if grp not in GROUPS:
            raise ValueError(f"unknown group {grp!r} in row {eid!r}")
        feats = {f: _safe_float(row.get(f, 0.0)) for f in FEATURES}
        events.append(Event(id=eid, group=grp, features=feats))
    return events


def _evaluate_pure(
    events: List[Event],
    g0: str,
    g1: str,
    feature: str,
    threshold: float,
    direction: str,
) -> Tuple[int, int, int, int, List[str], Dict[str, int]]:
    """Cheap training-set-only evaluation. Returns
    (correct_0, total_0, correct_1, total_1, misclassified_ids, group_counts)."""
    correct_0 = correct_1 = total_0 = total_1 = 0
    misclassified: List[str] = []
    group_counts: Dict[str, int] = {}
    for e in events:
        group_counts[e.group] = group_counts.get(e.group, 0) + 1
        pred = g0 if direction == "<" and e.features[feature] < threshold else g1
        if e.group == g0:
            total_0 += 1
            if pred == g0:
                correct_0 += 1
            else:
                misclassified.append(e.id)
        elif e.group == g1:
            total_1 += 1
            if pred == g1:
                correct_1 += 1
            else:
                misclassified.append(e.id)
    return correct_0, total_0, correct_1, total_1, misclassified, group_counts


def evaluate_single(
    events: List[Event],
    g0: str,
    g1: str,
    feature: str,
    threshold: float,
    direction: str,
) -> RuleResult:
    """Apply a fixed rule + compute leave-one-out CV with re-fitted thresholds.

    LOO refits the threshold on each training fold (via search_best_threshold)
    so validation is not circular w.r.t. the training-set threshold.
    """
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}")
    if direction not in ("<", ">"):
        raise ValueError(f"unknown direction {direction!r}")

    c0, t0, c1, t1, misclassified, group_counts = _evaluate_pure(
        events, g0, g1, feature, threshold, direction
    )
    train_acc = _balanced_accuracy(c0, t0, c1, t1)

    # Leave-one-out cross-validation (threshold refit per fold).
    # Only events belonging to the two selected groups take part: the rule is a
    # binary g0-vs-g1 discriminator, so scoring other groups would be dishonest.
    pair = [e for e in events if e.group in (g0, g1)]
    loo_correct = 0
    loo_total = len(pair)
    loo_misclassified: List[str] = []
    for idx, held_out in enumerate(pair):
        train_fold = pair[:idx] + pair[idx + 1:]
        if not train_fold:
            continue
        sub_rule = _search_on_fold(train_fold, g0, g1, feature)
        pred = g0 if sub_rule.direction == "<" and held_out.features[feature] < sub_rule.threshold else g1
        if pred != held_out.group:
            loo_misclassified.append(held_out.id)
        else:
            loo_correct += 1
    loo_acc = loo_correct / max(loo_total, 1)

    return RuleResult(
        rule=Rule(feature=feature, direction=direction, threshold=threshold),
        train_correct=max(c0, c1),
        train_total=max(t0, t1),
        train_accuracy=train_acc,
        loo_correct=loo_correct,
        loo_total=loo_total,
        loo_accuracy=loo_acc,
        misclassified_ids=misclassified,
        loo_misclassified_ids=loo_misclassified,
        group_counts=group_counts,
    )


def search_best_threshold(
    events: List[Event],
    g0: str,
    g1: str,
    feature: str,
) -> Rule:
    """Find threshold + direction maximizing BALANCED training accuracy.

    Uses the cheap _evaluate_pure so it never recurses into LOO (keep the
    grid scan O(K·n), not O(K·n²)).
    Strategy: scan candidate thresholds drawn from inter-group gaps (midpoints
    between sorted values of the two classes), plus a couple of percentile
    anchors. Choose the sweep that maximizes balanced accuracy.
    """
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}")
    v0 = sorted(e.features[feature] for e in events if e.group == g0)
    v1 = sorted(e.features[feature] for e in events if e.group == g1)
    if not v0 or not v1:
        # fallback: midpoint of global range
        vals = [e.features[feature] for e in events]
        return Rule(feature=feature, direction="<", threshold=float(np.mean(vals)))

    candidates: List[float] = []
    # Midpoints between adjacent g0/g1 values (most informative gaps)
    merged = sorted(set(v0 + v1))
    for i in range(len(merged) - 1):
        a, b = merged[i], merged[i + 1]
        if a != b:
            candidates.append((a + b) / 2.0)
    # Also try ±5 % of each group's range around its median
    for group_vals in (v0, v1):
        med = float(np.median(group_vals))
        rng = max(group_vals) - min(group_vals) + 1e-12
        for frac in (0.05, 0.10, 0.15):
            candidates.append(med - frac * rng)
            candidates.append(med + frac * rng)

    best_score = -1.0
    best_rule = None
    for t in candidates:
        for d in ("<", ">"):
            c0, t0, c1, t1, _, _ = _evaluate_pure(events, g0, g1, feature, t, d)
            ba = _balanced_accuracy(c0, t0, c1, t1)
            if ba > best_score:
                best_score = ba
                best_rule = Rule(feature=feature, direction=d, threshold=t)
    if best_rule is None:
        best_rule = Rule(feature=feature, direction="<", threshold=candidates[0] if candidates else 0.0)
    return best_rule


def _search_on_fold(
    fold: List[Event],
    g0: str,
    g1: str,
    feature: str,
) -> Rule:
    """Per-fold threshold search used by LOO (delegates to the shared
    search routine so both training and LOO use the exact same argmax)."""
    return search_best_threshold(fold, g0, g1, feature)


def _evaluate_and_loo(
    events: List[Event],
    g0: str,
    g1: str,
    primary_result: RuleResult,
    f2: str,
    t2: float,
    d2: str,
) -> Tuple[float, int, int, List[str]]:
    """Leave-one-out CV for an AND rule (primary ∧ secondary).

    For each held-out event the thresholds are re-fit on the remaining fold
    (both features), so validation is not circular with the training-set rule.
    Returns (loo_accuracy, loo_correct, loo_total, loo_misclassified_ids).
    """
    pfeat = primary_result.rule.feature
    # Binary g0-vs-g1 problem: score only the two selected groups.
    pair = [e for e in events if e.group in (g0, g1)]
    loo_correct = 0
    loo_total = len(pair)
    loo_missed: List[str] = []

    def _fits(sub_events: List[Event]) -> Tuple[str, float, str, float]:
        """Refit both thresholds on a fold; returns (pd, pt, d2, t2)."""
        p_rule = search_best_threshold(sub_events, g0, g1, pfeat)
        s_rule = search_best_threshold(sub_events, g0, g1, f2)
        return p_rule.direction, p_rule.threshold, s_rule.direction, s_rule.threshold

    for idx, held_out in enumerate(pair):
        train_fold = pair[:idx] + pair[idx + 1:]
        if not train_fold:
            continue
        pd, pt, sd, st = _fits(train_fold)
        pp = (pd == "<" and held_out.features[pfeat] < pt) or (
            pd == ">" and held_out.features[pfeat] > pt
        )
        p2 = (sd == "<" and held_out.features[f2] < st) or (
            sd == ">" and held_out.features[f2] > st
        )
        ok = (pp and p2) == (held_out.group == g0)
        if ok:
            loo_correct += 1
        else:
            loo_missed.append(held_out.id)

    loo_acc = loo_correct / max(loo_total, 1)
    return loo_acc, loo_correct, loo_total, loo_missed


def search_rules(
    events: List[Event],
    g0: str,
    g1: str,
) -> List[RuleResult]:
    """Best rule per feature (direction determined by search). Ordered by
    decreasing balanced training accuracy."""
    results: List[RuleResult] = []
    for feat in FEATURES:
        rule = search_best_threshold(events, g0, g1, feat)
        res = evaluate_single(events, g0, g1, feat, rule.threshold, rule.direction)
        results.append(res)
    results.sort(key=lambda r: r.train_accuracy, reverse=True)
    return results


def search_two_feature(
    events: List[Event],
    g0: str,
    g1: str,
    primary_result: RuleResult,
    max_additions: int = 3,
) -> Optional[RuleResult]:
    """Extend the best single-feature rule with a second feature (AND logic).

    The AND rule predicts g0 iff
        (primary_op primary_thresh) AND (secondary_op secondary_thresh).
    We keep directions fixed to the primary's learned direction and the
    secondary's own optimal direction, and sweep thresholds over
    inter-group-gap midpoints. Return None if no two-feature rule improves
    over the primary by at least 0.05 balanced accuracy.
    """
    pfeat = primary_result.rule.feature
    best: Optional[RuleResult] = None
    improvement = 0.05  # bar for reporting a meaningful improvement
    # Reference bar: the primary rule must be beaten by `improvement`, not tied.
    bar = primary_result.train_accuracy + improvement

    for f2 in FEATURES:
        if f2 == pfeat:
            continue
        v2_0 = sorted(e.features[f2] for e in events if e.group == g0)
        v2_1 = sorted(e.features[f2] for e in events if e.group == g1)
        merged = sorted(set(v2_0 + v2_1))
        candidates: List[float] = []
        for i in range(len(merged) - 1):
            a, b = merged[i], merged[i + 1]
            if a != b:
                candidates.append((a + b) / 2.0)

        for t2 in candidates:
            for d2 in ("<", ">"):
                c0 = c1 = t0 = t1 = 0
                for e in events:
                    if e.group not in (g0, g1):
                        continue
                    pred_g0 = (
                        primary_result.rule.direction == "<"
                        and e.features[pfeat] < primary_result.rule.threshold
                    ) or (
                        primary_result.rule.direction == ">"
                        and e.features[pfeat] > primary_result.rule.threshold
                    )
                    pred_g0_f2 = (d2 == "<" and e.features[f2] < t2) or (
                        d2 == ">" and e.features[f2] > t2
                    )
                    pred = g0 if (pred_g0 and pred_g0_f2) else g1
                    if e.group == g0:
                        t0 += 1
                        if pred == g0:
                            c0 += 1
                    else:
                        t1 += 1
                        if pred == g1:
                            c1 += 1
                ba = _balanced_accuracy(c0, t0, c1, t1)
                if ba > bar:
                    # Recompute misclassified restricted to the two selected groups
                    misclassified = []
                    for e in events:
                        if e.group not in (g0, g1):
                            continue
                        pp = (
                            (primary_result.rule.direction == "<"
                             and e.features[pfeat] < primary_result.rule.threshold)
                            or (primary_result.rule.direction == ">"
                                and e.features[pfeat] > primary_result.rule.threshold)
                        )
                        p2 = (d2 == "<" and e.features[f2] < t2) or (d2 == ">" and e.features[f2] > t2)
                        if (pp and p2) != (e.group == g0):
                            misclassified.append(e.id)
                    loo_acc, loo_corr, loo_tot, loo_missed = _evaluate_and_loo(
                        events, g0, g1, primary_result, f2, t2, d2
                    )
                    # Reuse the dataclass but override the fields that reflect
                    # the AND combination (train + honest AND-rule LOO).
                    res = evaluate_single(events, g0, g1, f2, t2, d2)
                    res.rule = Rule(feature=f2, direction=d2, threshold=t2)
                    res.train_accuracy = ba
                    res.train_correct = c0 + c1
                    res.misclassified_ids = misclassified
                    res.loo_accuracy = loo_acc
                    res.loo_correct = loo_corr
                    res.loo_total = loo_tot
                    res.loo_misclassified_ids = loo_missed
                    res.group_counts = {
                        g: sum(1 for e in events if e.group == g) for g in (g0, g1)
                    }
                    best = res
                    bar = ba  # raise the bar as we improve
            if best and best.train_accuracy >= 1.0:
                break
        if best and best.train_accuracy >= 1.0:
            break
    return best


def map_chart(
    events: List[Event],
    x_feature: str,
    y_feature: str,
) -> dict:
    """Map events to normalised (0..1) canvas coords plus axis metadata.

    Returns:
        points: [{id, group, x, y}] normalised to [0, 1] in both dims
        x: {feature, unit, min, max, ticks: [label, pos]}
        y: {feature, unit, min, max, ticks: [label, pos]}
    """
    if x_feature not in FEATURES or y_feature not in FEATURES:
        raise ValueError(f"unknown feature(s) in map_chart")
    xv = [e.features[x_feature] for e in events]
    yv = [e.features[y_feature] for e in events]
    x_min, x_max = float(min(xv)), float(max(xv))
    y_min, y_max = float(min(yv)), float(max(yv))
    xr = x_max - x_min if x_max > x_min else 1.0
    yr = y_max - y_min if y_max > y_min else 1.0

    def _norm(v: float, lo: float, hi: float, rng: float) -> float:
        return 0.05 + 0.90 * (v - lo) / rng  # 5% margin on each side

    points = [
        {
            "id": e.id,
            "group": e.group,
            "x": _norm(e.features[x_feature], x_min, x_max, xr),
            "y": _norm(e.features[y_feature], y_min, y_max, yr),
        }
        for e in events
    ]

    def _ticks(lo: float, hi: float, rng: float, n=5) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        step = max(rng / (n - 1), rng * 0.01)
        cur = lo
        for _ in range(n):
            out.append((f"{cur:.3g}", _norm(cur, lo, hi, rng)))
            cur += step
        return out

    return {
        "points": points,
        "x": {
            "feature": x_feature,
            "unit": UNITS[x_feature],
            "min": x_min,
            "max": x_max,
            "range": xr,
            "ticks": _ticks(x_min, x_max, xr),
        },
        "y": {
            "feature": y_feature,
            "unit": UNITS[y_feature],
            "min": y_min,
            "max": y_max,
            "range": yr,
            "ticks": _ticks(y_min, y_max, yr),
        },
    }


# ---------------------------------------------------------------------------
# Deterministic synthetic demo dataset (16 events: 4 groups × 4 each)
# Group distributions follow the ivory-on-bamboo physics:
#   A — unscratched round centre  (low freq, long decay)
#   B — heavy scratches           (mid-high freq, short sharp impact)
#   C — round centre with tail    (mid freq, mid decay)
#   D — vertical S-slit           (highest freq, shortest duration)
# ---------------------------------------------------------------------------
_DEMO_SEED = 42


def make_demo_events(seed: int = _DEMO_SEED) -> List[Event]:
    rng = np.random.default_rng(seed)
    group_baselines = {
        "A": (0.40, 0.35, 0.18, 1800, -48.0),
        "B": (0.75, 0.22, 0.28, 3200, -30.0),
        "C": (0.55, 0.30, 0.22, 2400, -38.0),
        "D": (0.85, 0.18, 0.35, 5200, -24.0),
    }
    events = []
    for grp in GROUPS:
        base = group_baselines[grp]
        for i in range(4):
            eid = f"{grp}{i+1:02d}"
            noise = rng.normal(0.0, 0.08, 5)
            feats = {
                FEATURES[j]: round(base[j] + noise[j], 4)
                for j in range(len(FEATURES))
            }
            events.append(Event(id=eid, group=grp, features=feats))
    return events
