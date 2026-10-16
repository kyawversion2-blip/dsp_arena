"""Unit and regression tests for 12-pitch chroma music note features (Phase 1).

Validates:
  1. Standalone compute_chroma_12() in dsp_inverse.py.
  2. Edge-case robustness: silent audio, NaNs, Infs, None, short signals.
  3. Physical tonal sanity: synthesized pure notes (A440, C523, E330).
  4. Dual-window separation: Attack Chroma vs Ring-down Chroma.
  5. Real WAV audio validation on recorded impacts.
  6. Zero regression on existing measure_features() outputs.
"""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dsp_inverse
from dsp_inverse import CHROMA_NOTE_NAMES, compute_chroma_12, measure_features


def test_chroma_error_handling_and_edge_cases():
    """Confirms safe, deterministic behavior on invalid/edge-case inputs."""
    # None input
    r_none = compute_chroma_12(None, 44100)
    assert r_none["status"] == "invalid_input"
    assert r_none["dominant_note"] is None
    assert r_none["chroma_12"] == [0.0] * 12

    # Empty array
    r_empty = compute_chroma_12(np.array([]), 44100)
    assert r_empty["status"] == "too_short"

    # Too short (< 32 samples)
    r_short = compute_chroma_12([0.05] * 10, 44100)
    assert r_short["status"] == "too_short"

    # Non-finite (NaNs, Infs)
    r_nan = compute_chroma_12(np.array([1.0, np.nan, 0.5] * 20), 44100)
    assert r_nan["status"] == "invalid_input"

    r_inf = compute_chroma_12(np.array([1.0, np.inf, 0.5] * 20), 44100)
    assert r_inf["status"] == "invalid_input"

    # Pure silence (zeros)
    r_silent = compute_chroma_12(np.zeros(2048), 44100)
    assert r_silent["status"] == "silent"
    assert r_silent["dominant_note"] is None
    assert r_silent["purity"] == 0.0


def test_chroma_musical_tone_sanity():
    """Confirms mathematical pitch class detection on synthetic sine waves."""
    sr = 44100
    t = np.arange(sr // 2) / sr

    # Note A4: 440.0 Hz
    sig_a = np.sin(2 * np.pi * 440.0 * t)
    res_a = compute_chroma_12(sig_a, sr=sr, detect_transient=False)
    assert res_a["status"] == "ok"
    assert res_a["dominant_note"] == "A"
    assert res_a["dominant_index"] == 9
    assert res_a["purity"] > 0.25

    # Note C5: 523.25 Hz
    sig_c = np.sin(2 * np.pi * 523.25 * t)
    res_c = compute_chroma_12(sig_c, sr=sr, detect_transient=False)
    assert res_c["status"] == "ok"
    assert res_c["dominant_note"] == "C"
    assert res_c["dominant_index"] == 0

    # Note E4: 329.63 Hz
    sig_e = np.sin(2 * np.pi * 329.63 * t)
    res_e = compute_chroma_12(sig_e, sr=sr, detect_transient=False)
    assert res_e["status"] == "ok"
    assert res_e["dominant_note"] == "E"
    assert res_e["dominant_index"] == 4


def test_chroma_attack_vs_ringdown_separation():
    """Confirms Attack and Ring-down windows capture distinct temporal pitch modes."""
    sr = 44100
    # Create an impact-like synthetic transient:
    # 0 to 20ms: Note A (440 Hz) high-energy collision
    # 20ms to 120ms: Note C (523.25 Hz) decaying ring-down
    t = np.arange(int(sr * 0.150)) / sr
    attack_mask = t <= 0.020
    ringdown_mask = t > 0.020

    signal = np.zeros_like(t)
    # Attack: A4 with fast onset
    signal[attack_mask] = np.sin(2 * np.pi * 440.0 * t[attack_mask]) * 0.8
    # Ring-down: C5 decaying
    t_decay = t[ringdown_mask] - 0.020
    signal[ringdown_mask] = np.sin(2 * np.pi * 523.25 * t_decay) * 0.5 * np.exp(-t_decay * 40.0)

    res = compute_chroma_12(signal, sr=sr, detect_transient=True)
    assert res["status"] == "ok"
    assert res["dominant_note_attack"] == "A"
    assert res["dominant_note_ringdown"] == "C"


def test_chroma_real_wav_files():
    """Validates compute_chroma_12 on actual project WAV audio files."""
    test_wavs = [
        Path("wavs/11.wav"),
        Path("wavs/A_EVT-0B16F1_1790999252616.wav"),
        Path("wavs/C_EVT-068817_1790999231739.wav"),
    ]

    for p in test_wavs:
        if not p.exists():
            continue
        y, sr = dsp_inverse.decode_wav(str(p))
        y = np.asarray(y, dtype=np.float64)
        if sr > 48000:
            y = dsp_inverse.resample_linear(y, sr, 44100)
            sr = 44100

        res = compute_chroma_12(y, sr=sr, detect_transient=True)
        assert res["status"] == "ok"
        assert len(res["chroma_12"]) == 12
        assert len(res["chroma_attack_12"]) == 12
        assert len(res["chroma_ringdown_12"]) == 12
        assert all(np.isfinite(x) for x in res["chroma_12"])
        assert res["dominant_note"] in CHROMA_NOTE_NAMES
        assert 0 <= res["dominant_index"] < 12
        assert res["purity"] >= 0.0


def test_measure_features_regression_unchanged():
    """Guarantees that compute_chroma_12 does NOT alter measure_features() outputs."""
    sr = 44100
    t = np.arange(sr // 2) / sr
    y = np.sin(2 * np.pi * 2000.0 * t) * np.exp(-t * 20.0)
    y = y.astype(np.float64)

    fe = measure_features(y, sr, nfft=2048, hop=512, top_db=35.0, win=0.5, fmax=None)
    assert fe is not None

    # Exact expected keys of measure_features contract
    expected_keys = {
        "impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz",
        "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff_hz",
        "spectral_flatness", "rms_energy", "zcr", "attack_slope_db_ms",
        "hf_transient_ratio", "crest_factor"
    }
    assert set(fe.keys()) == expected_keys, "measure_features output contract altered!"


def test_chroma_csv_schema_completeness():
    """Validates that all 12 note keys are present and conform to CSV schema."""
    sr = 44100
    t = np.arange(sr // 4) / sr
    y = np.sin(2 * np.pi * 440.0 * t)
    res = compute_chroma_12(y, sr)

    expected_cols = [
        "chroma_C", "chroma_CSharp", "chroma_D", "chroma_DSharp",
        "chroma_E", "chroma_F", "chroma_FSharp", "chroma_G",
        "chroma_GSharp", "chroma_A", "chroma_ASharp", "chroma_B"
    ]
    for col in expected_cols:
        assert col in res
        assert isinstance(res[col], float)
        assert np.isfinite(res[col])
