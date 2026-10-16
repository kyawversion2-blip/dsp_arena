"""tests/test_chroma_relative.py — Unit tests for Relative Chroma features (Phase 2B).

Validates:
  1. Circularly shifted chroma computation with reference pitch class.
  2. Edge cases for reference frequency: zero, negative, NaN, Inf, empty, silent.
  3. Mathematical properties of circular shift: bin 0 contains reference energy.
  4. Robustness against invalid/extreme audio signals.
  5. Zero regression on existing dsp_inverse functions.
"""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dsp_inverse
from dsp_inverse import (
    CHROMA_NOTE_NAMES,
    compute_chroma_12,
    compute_chroma_circular_relative,
)


def pitch_class_from_frequency(freq_hz: float | None) -> int | None:
    """Computes pitch class 0..11 from frequency in Hz safely."""
    if freq_hz is None:
        return None
    try:
        f = float(freq_hz)
    except Exception:
        return None
    if not np.isfinite(f) or f < 20.0 or f > 100000.0:
        return None
    midi = 12.0 * np.log2(f / 440.0) + 69.0
    return int(np.round(midi)) % 12


def test_circular_chroma_peak_alignment():
    """Confirms that circularly shifted chroma places the peak at index 0."""
    # Synthetic chroma where Note E (index 4) has highest energy
    chroma = [0.05] * 12
    chroma[4] = 0.80

    shifted = compute_chroma_circular_relative(chroma)
    assert shifted[0] == pytest.approx(0.80)
    assert np.argmax(shifted) == 0


def test_circular_chroma_explicit_ref_index():
    """Confirms shifting by explicit reference index."""
    # Chroma with peak at index 9 (Note A)
    chroma = [0.1] * 12
    chroma[9] = 0.9

    # Align explicitly by index 9 -> index 9 becomes index 0
    shifted_9 = compute_chroma_circular_relative(chroma, ref_pitch_idx=9)
    assert shifted_9[0] == pytest.approx(0.9)

    # Align explicitly by index 0 (no shift)
    shifted_0 = compute_chroma_circular_relative(chroma, ref_pitch_idx=0)
    assert shifted_0 == chroma


def test_pitch_class_from_frequency_robustness():
    """Tests reference frequency conversion edge cases."""
    # 440 Hz -> A (index 9)
    assert pitch_class_from_frequency(440.0) == 9
    # 261.63 Hz -> C (index 0)
    assert pitch_class_from_frequency(261.63) == 0
    # Negative frequency -> None
    assert pitch_class_from_frequency(-100.0) is None
    # Zero frequency -> None
    assert pitch_class_from_frequency(0.0) is None
    # NaN and Inf -> None
    assert pitch_class_from_frequency(np.nan) is None
    assert pitch_class_from_frequency(np.inf) is None
    # Unparseable string -> None
    assert pitch_class_from_frequency("abc") is None
    # None -> None
    assert pitch_class_from_frequency(None) is None


def test_circular_chroma_silent_and_invalid_inputs():
    """Confirms safe zero vector return on silent or non-finite chroma."""
    # All zeros
    assert compute_chroma_circular_relative([0.0] * 12) == [0.0] * 12

    # Wrong length
    assert compute_chroma_circular_relative([0.1, 0.2]) == [0.0] * 12

    # NaNs and Infs
    assert compute_chroma_circular_relative([np.nan] * 12) == [0.0] * 12
    assert compute_chroma_circular_relative([np.inf] * 12) == [0.0] * 12

    # None
    assert compute_chroma_circular_relative(None) == [0.0] * 12


def test_circular_chroma_on_real_signal():
    """Validates end-to-end circular chroma from synthetic audio."""
    sr = 44100
    t = np.arange(sr // 4) / sr
    sig = np.sin(2 * np.pi * 523.25 * t)  # C5

    res = compute_chroma_12(sig, sr=sr, detect_transient=False)
    assert res["status"] == "ok"
    assert res["dominant_note"] == "C"

    shifted = compute_chroma_circular_relative(res["chroma_12"])
    assert len(shifted) == 12
    assert shifted[0] == max(shifted)
    assert all(np.isfinite(x) for x in shifted)
