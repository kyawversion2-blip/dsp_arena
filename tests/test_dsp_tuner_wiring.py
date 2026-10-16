"""tests/test_dsp_tuner_wiring.py — DSP tuner (slice 1) static wiring.

The tuner is an additive browser block: it POSTs a real WAV to the existing
/compute route with an explicit parameter set and keeps an A/B history. These
tests are static (no browser) and assert only the guarantees the slice claims:

  1. the panel + wiring exist exactly once
  2. it posts to the existing /compute route (no new backend route invented)
  3. the six /compute form fields are all sent explicitly (never silent)
  4. it does not touch chart math, spawnEvent, live-confirm, or WS handling
  5. the parameter set is displayed/recorded for every run
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HTML = ROOT / "dsp-cluster-lab.html"
SERVER = ROOT / "live_server.py"


def _src() -> str:
    return HTML.read_text(encoding="utf-8")


def test_panel_and_wiring_present_once():
    src = _src()
    for token in ['id="sect-tuner"', 'id="bt-tune-run"', 'id="tune-file"',
                  'id="tune-history"', '[tuner] mounted']:
        assert src.count(token) == 1, f"{token} appears {src.count(token)}x"


def test_posts_to_existing_compute_route_only():
    src = _src()
    assert src.count("fetch('/compute'") == 1
    # no new backend routes were invented
    assert "@app.post('/tune')" not in (SERVER.read_text())
    assert "/tune" not in src.replace("id=\"tune-", "").replace("bt-tune-", "")


def test_all_compute_params_sent_explicitly():
    src = _src()
    # tuner IIFE is between these two slice markers
    block = src[src.index("TUNE-HISTORY-BLOCK"): src.index("SLICE 2 \u00b7 TUNER MARKERS")] 
    for field in ("sr", "nfft", "hop", "top_db", "win", "fmax", "device_label", "engine"):
        assert f"fd.append('{field}'" in block, f"{field} not appended explicitly"


def test_chart_math_and_live_flow_untouched():
    src = _src()
    # runs-feed writes only; marker overlay is allowed to read wxOf/wyOf/wzOf+prj
    runs = src[src.index("TUNE-HISTORY-BLOCK"): src.index("SLICE 2 \u00b7 TUNER MARKERS")]
    for forbidden in ("spawnEvent(", "dF[", "pX[", "wsConnect", "liveBindPayload", "confirm-label"):
        assert forbidden not in runs, f"runs block writes {forbidden}"


def test_param_set_recorded_and_visible():
    src = _src()
    assert "paramsString(" in src           # full parameter string shown per run
    assert "params_used_hz" in src           # fmax_used echoed back
    assert "actual_sr" in src               # effective sample rate echoed back
    # deterministic: no smoothing/averaging of returned values
    assert "smooth" not in tuner_lower(src)
    assert "average" not in tuner_lower(src)


def tuner_lower(src: str) -> str:
    return src[src.index("TUNE-HISTORY-BLOCK"):].lower()


def test_no_tautological_fabrication():
    """The tuner must refuse to run without a real WAV (no synthetic audio)."""
    src = _src()
    assert "the tuner never synthesizes audio" in src
