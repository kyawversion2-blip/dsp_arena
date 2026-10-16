#!/usr/bin/env python3
"""tests/test_recording_variability.py — Test suite for recording metadata and variability audit.

Verifies:
1. RecordingMetadata dataclass schema, validation, serialization, and factory.
2. Safe CV and pairwise difference mathematical calculations.
3. Repeatability audit outputs and data invariants.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from recording_metadata import RecordingMetadata, create_recording_metadata
from scripts.analyze_recording_variability import (
    calculate_pairwise_relative_diffs,
    calculate_safe_cv,
)

AUDIT_DIR = ROOT / "dsp_out" / "recording_variability_audit"


def test_recording_metadata_defaults():
    meta = RecordingMetadata()
    assert meta.source_type == "unknown"
    assert meta.recording_position == "unknown"
    assert meta.microphone_distance_cm is None
    assert meta.channel_count is None
    assert meta.recording_device_name == "unknown"

    d = meta.to_dict()
    assert isinstance(d, dict)
    assert d["source_type"] == "unknown"

    reconstructed = RecordingMetadata.from_dict(d)
    assert reconstructed == meta


def test_create_recording_metadata_factory():
    data = create_recording_metadata(
        source_type="builtin_mic",
        requested_sr=48000,
        actual_sr=48000,
        channels=1,
        duration_s=1.2345678,
        position="LEFT",
        distance_cm=15.0,
        device_name="MacBook Air Microphone",
        constraints={"voice_isolation": False},
        preprocessing=["transient_trim", "hanning_window"],
        file_name="test_impact_01.wav",
        session_id="session_test",
        notes="Validation run",
    )
    assert data["source_type"] == "builtin_mic"
    assert data["recording_position"] == "left"
    assert data["microphone_distance_cm"] == 15.0
    assert data["duration_s"] == 1.234568
    assert data["actual_sr_hz"] == 48000
    assert data["recording_constraints"] == {"voice_isolation": False}
    assert "transient_trim" in data["preprocessing_steps"]


def test_create_recording_metadata_invalid_fallbacks():
    data = create_recording_metadata(
        source_type="quantum_mic",  # Invalid -> fallback to 'unknown'
        position="diagonal_ceiling",  # Invalid -> fallback to 'unknown'
        distance_cm=None,
    )
    assert data["source_type"] == "unknown"
    assert data["recording_position"] == "unknown"
    assert data["microphone_distance_cm"] is None


def test_safe_cv_calculation():
    # Identical values -> CV must be 0.0
    arr_const = np.array([10.0, 10.0, 10.0, 10.0])
    assert calculate_safe_cv(arr_const) == 0.0

    # Single element -> CV must be 0.0
    assert calculate_safe_cv(np.array([42.0])) == 0.0
    assert calculate_safe_cv(np.array([])) == 0.0

    # Mean = 100, std = 10 -> CV = 10.0%
    arr_var = np.array([90.0, 100.0, 110.0])
    cv = calculate_safe_cv(arr_var)
    assert 8.0 < cv < 10.0  # std of [90, 100, 110] is sqrt(200/3) ≈ 8.165 -> 8.16%

    # Near zero mean handled safely without division by zero
    arr_zero = np.array([0.0, 0.0, 0.0])
    assert calculate_safe_cv(arr_zero) == 0.0


def test_pairwise_relative_diffs():
    # Identical values -> pairwise diffs 0.0
    arr_const = np.array([5.0, 5.0, 5.0])
    mean_diff, max_diff = calculate_pairwise_relative_diffs(arr_const)
    assert mean_diff == 0.0
    assert max_diff == 0.0

    # Single element
    m, mx = calculate_pairwise_relative_diffs(np.array([1.0]))
    assert m == 0.0 and mx == 0.0

    # 10.0 and 20.0 -> diff = 10, denom = 0.5*(10+20) = 15 -> (10/15)*100 = 66.67%
    m, mx = calculate_pairwise_relative_diffs(np.array([10.0, 20.0]))
    assert pytest.approx(m, rel=1e-3) == 66.6667
    assert pytest.approx(mx, rel=1e-3) == 66.6667


def test_audit_artifacts_generated():
    features_csv = AUDIT_DIR / "recording_variability_features.csv"
    stats_csv = AUDIT_DIR / "repeatability_statistics.csv"
    meta_csv = AUDIT_DIR / "recording_metadata_inventory.csv"
    plot_html = AUDIT_DIR / "plots" / "cross_session_variability_comparison.html"

    assert features_csv.exists(), f"Missing {features_csv}"
    assert stats_csv.exists(), f"Missing {stats_csv}"
    assert meta_csv.exists(), f"Missing {meta_csv}"
    assert plot_html.exists(), f"Missing {plot_html}"

    # Verify features CSV content
    with open(features_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) >= 60
        required_cols = {
            "session", "filename", "ground_truth_group", "peak_amplitude",
            "rms_energy", "spectral_centroid_hz", "impact_freq_hz", "mean_impact_interval_ms"
        }
        assert required_cols.issubset(reader.fieldnames)

    # Verify repeatability stats CSV content
    with open(stats_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) > 0
        stat_cols = {"session", "group", "feature", "mean", "std", "cv_percent", "mean_pairwise_diff_pct"}
        assert stat_cols.issubset(reader.fieldnames)


def test_physical_feature_invariants():
    features_csv = AUDIT_DIR / "recording_variability_features.csv"
    with open(features_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 1. Clipping ratio must be zero (no clipping in dataset)
            assert float(r["clipping_ratio"]) == 0.0
            # 2. Peak amplitude must be positive and bounded by 1.0 (unclipped normalized float)
            assert 0.0 < float(r["peak_amplitude"]) <= 1.0
            # 3. Mean impact interval must be within physical chatter range (2ms to 20ms)
            if float(r["mean_impact_interval_ms"]) > 0:
                assert 2.0 <= float(r["mean_impact_interval_ms"]) <= 20.0
            # 4. Spectral centroid must be positive and below Nyquist
            sr = int(r["sample_rate_hz"])
            assert 100.0 < float(r["spectral_centroid_hz"]) < (sr / 2.0)
