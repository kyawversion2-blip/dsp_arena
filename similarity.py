#!/usr/bin/env python3
"""
similarity.py — deterministic recency-weighted similarity ADVISORY
==================================================================
Given the live event's already-computed 5-feature payload and the recent
confirmed operator decisions, suggest which group the operator is most
likely to assign.

Hard boundaries (this module is advisory only):

  * No feature recomputation — it reads only the 5 values already present
    in the event payload / confirmed record.
  * No auto-labeling. It never writes a decision and never mutates a record.
  * Never touches chart coordinates. It has no import of dsp_chart and no
    knowledge of chart geometry at all.
  * Rejected events and test-mode events are excluded by construction: it
    reads confirmed_events.jsonl only, and skips records without a label.

Determinism: identical inputs (confirmed record set + event payload) always
produce identical output. No randomness, no wall-clock in the output value,
no iteration over unordered containers.

Similarity definition (explicit, so it is auditable):
  For each feature f, scale the |difference| between the candidate and a
  confirmed event by the observed spread of f across the confirmed set:
      d_f = |x_f - y_f| / (max_f - min_f)          (0 if spread == 0)
  Weighted L1:
      dist = sum_f w_f * d_f / sum_f w_f
      sim  = 1 - dist                              (clamped to [0, 1])
  Recency weight for the i-th most recent confirmed record (i = 0 newest):
      r_i = 0.5 ** (i / HALFLIFE)
  Group support = sum of sim * r over confirmed events carrying that label.
  Predicted group = argmax support (ties broken by label order A,B,C,D for
  determinism). Confidence = winning support / total support.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import confirmed_store as _store

# Canonical feature order — the same 5 DSP features used everywhere else.
FEATURE_KEYS = (
    "impact",
    "duration_s",
    "peak_time_ms",
    "mel_mean_db",
    "impact_freq_hz",
)

# Weights are a stated design choice, not a fitted model. impact_freq_hz and
# peak_time_ms carry the most class separation in the frozen relationship
# table, so they dominate; mel/duration/impact break ties.
FEATURE_WEIGHTS = {
    "impact": 0.15,
    "duration_s": 0.15,
    "peak_time_ms": 0.25,
    "mel_mean_db": 0.15,
    "impact_freq_hz": 0.30,
}

# Fewer than this many labelled confirmed events -> not enough history.
MIN_HISTORY = 3
DEFAULT_RECENT_N = 20
RECENCY_HALFLIFE = 8.0
LABEL_ORDER = ("A", "B", "C", "D")


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def event_features(payload: Mapping[str, Any] | None) -> dict[str, float]:
    """Extract the 5 advisory features from an event payload.

    Reads only values already present; never recomputes anything.
    """
    if not isinstance(payload, Mapping):
        return {}
    source: Mapping[str, Any] = payload.get("features") if isinstance(
        payload.get("features"), Mapping
    ) else payload
    # A confirmed record nests the original payload under "event".
    if "event" in source and isinstance(source.get("event"), Mapping):
        source = source["event"]

    out: dict[str, float] = {}
    for key in FEATURE_KEYS:
        val = _num(source.get(key))
        if val is None:
            continue
        out[key] = val
    return out


def _labelled_history(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep only confirmed records that carry a real operator label.

    Skips: rejected decisions, test-mode events, records with label_truth
    null/empty or outside A-D, and records with no usable features.
    """
    keep: list[Mapping[str, Any]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        if rec.get("decision") != "YES":
            continue
        if rec.get("_test_mode") or rec.get("test_mode"):
            continue
        source = rec.get("source")
        if isinstance(source, str) and source.startswith("replay"):
            continue
        label = rec.get("label_truth")
        if not isinstance(label, str):
            continue
        label = label.strip().upper()[:1]
        if label not in LABEL_ORDER:
            continue
        feats = event_features(rec)
        if len(feats) < len(FEATURE_KEYS):
            # Require the full 5-feature set so every comparison is on the
            # same axes; a partial record would silently skew the spread.
            continue
        keep.append(rec)
    return keep


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def advise(
    payload: Mapping[str, Any] | None,
    *,
    recent_n: int = DEFAULT_RECENT_N,
    confirmed: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the advisory dict for `payload`.

    Never raises for ordinary inputs: any insufficient/invalid data yields
    `available: False` with a `reason`, so the caller can render a neutral
    badge instead of guessing.
    """
    if confirmed is None:
        confirmed = _store.load_confirmed(limit=max(1, int(recent_n)))

    history = _labelled_history(list(confirmed)[-int(recent_n):])
    basis_count = len(history)

    if basis_count < MIN_HISTORY:
        return {
            "available": False,
            "reason": "insufficient_confirmed_history",
            "basis_count": basis_count,
            "min_history": MIN_HISTORY,
            "predicted_group": None,
            "confidence": None,
            "score": None,
            "basis_event_ids": [],
        }

    target = event_features(payload)
    if len(target) < len(FEATURE_KEYS):
        return {
            "available": False,
            "reason": "incomplete_event_features",
            "basis_count": basis_count,
            "min_history": MIN_HISTORY,
            "predicted_group": None,
            "confidence": None,
            "score": None,
            "basis_event_ids": [],
        }

    # Per-feature observed spread across the labelled confirmed set.
    ranges: dict[str, float] = {}
    for key in FEATURE_KEYS:
        ranges[key] = _spread([float(h[key]) for h in history])
    used_weights = {k: FEATURE_WEIGHTS[k] for k in FEATURE_KEYS}
    total_weight = sum(used_weights.values())

    support: dict[str, float] = {}
    best_per_label: dict[str, float] = {}
    basis_ids: dict[str, list[str]] = {lab: [] for lab in LABEL_ORDER}

    # records are oldest-first; enumerate from the newest for recency decay.
    n = len(history)
    for idx, rec in enumerate(history):
        age = n - 1 - idx
        rec_w = 0.5 ** (age / RECENCY_HALFLIFE)

        dist = 0.0
        for key in FEATURE_KEYS:
            span = ranges[key]
            if span <= 0.0:
                continue  # feature identical across history -> no signal
            dist += used_weights[key] * abs(target[key] - float(rec[key])) / span
        sim = 1.0 - (dist / total_weight if total_weight else 0.0)
        sim = max(0.0, min(1.0, sim))

        label = str(rec["label_truth"]).strip().upper()[:1]
        support[label] = support.get(label, 0.0) + sim * rec_w
        prev_best = best_per_label.get(label)
        if prev_best is None or sim > prev_best:
            best_per_label[label] = sim
            basis_ids[label] = [rec.get("event_id")]

    total_support = sum(support.values())
    if total_support <= 0.0:
        return {
            "available": False,
            "reason": "no_similarity_signal",
            "basis_count": basis_count,
            "min_history": MIN_HISTORY,
            "predicted_group": None,
            "confidence": None,
            "score": None,
            "basis_event_ids": [],
        }

    # Deterministic argmax: highest support, ties broken by LABEL_ORDER.
    predicted = min(
        LABEL_ORDER,
        key=lambda lab: (-support.get(lab, 0.0), LABEL_ORDER.index(lab)),
    )
    confidence = support.get(predicted, 0.0) / total_support

    return {
        "available": True,
        "reason": None,
        "predicted_group": predicted,
        "confidence": round(confidence, 4),
        "score": round(best_per_label.get(predicted, 0.0), 4),
        "basis_count": basis_count,
        "basis_event_ids": basis_ids.get(predicted, []),
        "advisory_only": True,
    }
