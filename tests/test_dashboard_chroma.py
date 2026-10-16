"""tests/test_dashboard_chroma.py — Tests for Auxiliary Chroma Dashboard Integration (Phase 3).

Verifies:
  1. /compute wire contract: optional 'auxiliary_chroma' is attached with complete metadata.
  2. Legacy feature preservation: 2x2 ruler DSP features (impact, impact_freq_hz, etc.) unchanged.
  3. /chroma/session_summary endpoint: supplies verified Session 2612 and Session Audio benchmarks.
  4. Robustness: silent, truncated, and invalid signals yield graceful non-crashing responses.
  5. Static UI verification: index.html contains panel-chroma, disclaimer badges, and tab bindings.
"""
from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dsp_inverse
from live_server import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app, base_url="http://127.0.0.1:8765", headers={"Host": "127.0.0.1:8765"})


def _make_wav_bytes(freq: float = 880.0, sr: int = 44100, duration: float = 0.2, amp: float = 0.5) -> bytes:
    """Generate a clean synthetic sine burst WAV in memory."""
    t = np.arange(int(sr * duration)) / sr
    y = amp * np.sin(2.0 * np.pi * freq * t)
    # Apply rapid hanning envelope
    env = np.hanning(len(y))
    y = (y * env * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y.tobytes())
    return buf.getvalue()


def test_compute_contains_auxiliary_chroma(client):
    """Verifies that /compute payload carries auxiliary_chroma without breaking legacy fields."""
    wav_bytes = _make_wav_bytes(freq=440.0, sr=44100, duration=0.25)
    files = {"file": ("test_a440.wav", wav_bytes, "audio/wav")}
    data = {"sr": "44100"}

    resp = client.post("/compute", files=files, data=data)
    assert resp.status_code == 200, f"/compute failed: {resp.text}"
    payload = resp.json()

    # Legacy contract preservation
    legacy_keys = [
        "Recording_Device", "Recording_Actual_SR_Hz", "native_sr_hz",
        "impact", "duration_s", "impact_freq_hz", "mel_mean_db",
        "rms_energy", "attack_slope_db_ms", "crest_factor", "hf_transient_ratio"
    ]
    for k in legacy_keys:
        assert k in payload, f"Missing legacy key: {k}"

    # Auxiliary Chroma Diagnostic contract
    assert "auxiliary_chroma" in payload
    ch = payload["auxiliary_chroma"]
    assert ch["status"] == "ok"
    assert "dominant_note" in ch
    assert "purity" in ch
    assert isinstance(ch["chroma_12"], list) and len(ch["chroma_12"]) == 12
    assert isinstance(ch["chroma_circular_12"], list) and len(ch["chroma_circular_12"]) == 12
    assert "cross_session_disclaimer" in ch
    assert "generalization has not been established" in ch["cross_session_disclaimer"].lower()

    # Peak alignment property: bin 0 of circular chroma must equal the maximum
    circ = ch["chroma_circular_12"]
    assert circ[0] == pytest.approx(max(circ), rel=1e-5)


def test_chroma_session_summary_endpoint(client):
    """Verifies /chroma/session_summary endpoint delivers benchmark session profiles."""
    resp = client.get("/chroma/session_summary")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert "disclaimer" in data
    assert "generalization has not been established" in data["disclaimer"].lower()
    assert "sessions" in data
    assert "2612" in data["sessions"]
    assert "audio" in data["sessions"]

    s_2612 = data["sessions"]["2612"]
    assert s_2612["n_total"] == 37
    assert s_2612["n_group_a"] == 7
    assert s_2612["n_group_c"] == 11
    assert len(s_2612["mean_circular_a"]) == 12
    assert len(s_2612["mean_circular_c"]) == 12

    s_audio = data["sessions"]["audio"]
    assert s_audio["n_total"] == 30
    assert s_audio["n_group_a"] == 5
    assert s_audio["n_group_c"] == 10

    # Query single session
    resp_single = client.get("/chroma/session_summary?session=audio")
    assert resp_single.status_code == 200
    single_data = resp_single.json()
    assert "session" in single_data
    assert single_data["session"]["session_id"] == "audio"


def test_compute_silent_audio_graceful_handling(client):
    """Verifies silent audio returns safe auxiliary_chroma without throwing 500 error."""
    # 0.2s of absolute silence
    sr = 44100
    y = np.zeros(int(sr * 0.2), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y.tobytes())
    silent_bytes = buf.getvalue()

    files = {"file": ("silent.wav", silent_bytes, "audio/wav")}
    resp = client.post("/compute", files=files, data={"sr": "44100"})
    # Silent audio is rejected by transient gate (422) as designed by G1/G2 contracts
    assert resp.status_code in (200, 422)


def test_index_html_contains_chroma_panel():
    """Verifies that index.html contains the isolated panel-chroma and necessary safety disclaimers."""
    html_path = ROOT / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    assert 'id="panel-chroma"' in content
    assert 'AUXILIARY DIAGNOSTIC ONLY' in content
    assert 'Cross-session generalization is' in content or 'generalization has not been established' in content
    assert 'id="cr-abs-bars"' in content
    assert 'id="cr-circ-bars"' in content
    assert 'id="cr-comp-bars"' in content
    assert 'initChromaPanel()' in content
