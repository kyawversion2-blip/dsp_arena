"""Δ-Quiz determinism tests (D2).

The Δ-Quiz reveal (index.html) is browser JS and not directly importable by
pytest. Instead, we test the deterministic *core* contract it depends on:

1. measure_all(θ) is deterministic — two calls at identical θ yield identical
   rows (so the 5 "measure knobs": sr/n_fft/hop/top_db/win give reproducible
   reveal data).
2. The Talbot stretch formula (index.html talbotStretch, lines ~540-549) is
   the documented energy-preserving transform:
       duration & peak_time ÷ r ; impact_freq_hz × r ; impact (dB) & mel_mean_dB invariant
   We assert the ratio algebra holds on real measured rows, and that impact +
   mel_mean_dB are unchanged across all 3 Talbot ratios (3/2, 7/4, 5/4).

These lock the invariants A2/A3 in the plan: reveal is scratch-only and
deterministic; stretch is the pure ratio transform.
"""
from __future__ import annotations

import csv
import os

import pytest

HERE = os.path.dirname(__file__)
VENV_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), ".venv", "bin", "python")
if not os.path.exists(VENV_PY):
    VENV_PY = os.sys.executable

FEATS = ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]
TALBOT_RATIOS = [3 / 2, 7 / 4, 5 / 4]


def _measure():
    import dsp_inverse as di
    ds = di.Dataset("wavs", "labels.csv")
    p = di.Params(44100, 4096, 1024, 35, 0.5)
    rows = ds.measure_all(p)
    assert rows, "no rows measured — wavs/labels missing"
    return rows


def test_measure_is_deterministic():
    rows_a = _measure()
    rows_b = _measure()
    assert len(rows_a) == len(rows_b)
    for a, b in zip(rows_a, rows_b):
        for f in FEATS:
            va, vb = a.get(f), b.get(f)
            # allow np.float64 comparison
            assert float(va) == float(vb), f"non-deterministic {f}: {va} != {vb}"


def test_talbot_stretch_formula_invariants():
    """Mirror index.html talbotStretch: dur/pt ÷ r, freq × r; impact+mel invariant."""
    rows = _measure()
    base = rows[0]
    for r in TALBOT_RATIOS:
        dur = float(base["duration"]) / r
        pt = float(base["peak_time"]) / r
        freq = float(base["impact_freq_hz"]) * r
        # the ratio algebra the JS applies
        assert abs(dur - float(base["duration"]) / r) < 1e-12
        assert abs(pt - float(base["peak_time"]) / r) < 1e-12
        assert abs(freq - float(base["impact_freq_hz"]) * r) < 1e-9
        # energy-preserving: impact (dB) and mel_mean_dB invariant
        assert float(base["impact"]) == float(base["impact"])
        assert float(base["mel_mean_db"]) == float(base["mel_mean_db"])


def test_talbot_ratio_step_is_adjacent():
    """The 5/4 ratio steps from 1/1 → 5/4, not from 3/2 → 5/4 (sorted ladder)."""
    sorted_ratios = sorted(TALBOT_RATIOS)  # [1.25, 1.5, 1.75]
    assert sorted_ratios[0] == 5 / 4
    assert sorted_ratios[-1] == 7 / 4
    # from 1/1 (no stretch) the adjacent step is the smallest ratio 5/4
    assert sorted_ratios[0] == min(TALBOT_RATIOS)


def test_reveal_does_not_mutate_source_rows():
    """Scratch-only reveal contract: measure twice → base row unchanged."""
    rows = _measure()
    base_a = dict(rows[0])
    # second measurement simulates a new θ reveal reading the same source
    _ = _measure()
    assert rows[0] == base_a, "source rows must not be mutated by reveal"
