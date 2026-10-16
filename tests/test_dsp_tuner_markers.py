"""tests/test_dsp_tuner_markers.py — Slice 2: tuner history -> chart markers.

Static (no browser) guarantees asserted here:

  1. markers are an overlay on their OWN canvas; #stage is never drawn on
  2. marker position uses the EXISTING mapping (wxOf / wyOf / wzOf + prj)
  3. chart engine + live flow are untouched (no calls, no writes)
  4. sigma is never derived from duration; unmeasured sigma -> labelled n/a
  5. stable run id recorded on every run
  6. show/hide toggle + clear-history action exist and are wired
  7. hover/click surface the full required run-information field set
  8. nothing is called "best" / no ranking of parameterisations
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HTML = ROOT / "dsp-cluster-lab.html"


def _src() -> str:
    return HTML.read_text(encoding="utf-8")


def _marker_block() -> str:
    src = _src()
    return src[src.index("SLICE 2 · TUNER MARKERS"): src.index("[tuner] mounted")]


def test_overlay_uses_its_own_canvas_not_stage():
    src = _src()
    assert src.count('id="tuner-overlay"') == 1
    assert 'id="stage"' in src
    blk = _marker_block()
    # overlay drawing targets ovCtx only, never ctx
    assert "ovCtx." in blk
    assert "ctx.drawImage" not in blk and "ctx.fillRect" not in blk


def test_marker_position_uses_existing_mapping_only():
    blk = _marker_block()
    for fn in ("wxOf(m.f)", "wyOf(m.tp)", "wzOf(m.sg)"):
        assert fn in blk, f"missing existing mapping call {fn}"
    assert "prj(X, Y, Z)" in blk
    # no bespoke geometry introduced
    for bad in ("function wxOf", "function wyOf", "function wzOf", "function prj"):
        assert bad not in blk, f"marker block redefines {bad}"


def test_chart_engine_and_live_flow_untouched():
    blk = _marker_block()
    tokens = [line for line in blk.splitlines() if not line.lstrip().startswith("·")]
    code = "\n".join(tokens)
    for forbidden in ("dF[", "dSig[", "dTp[", "dE[", "pX[", "pY[", "pZ[",
                      "wsConnect", "liveBindPayload",
                      "ingestMessage", "lockedId=", "hoverId="):
        assert forbidden not in code, f"marker block writes {forbidden}"


def test_sigma_never_derived_from_duration():
    src = _src()
    blk = _marker_block()
    assert "wzOf(0)" in blk                    # explicit sigma=0 reference plane
    assert "σ n/a" in blk
    # no derivation from duration anywhere in the tuner surface
    tuner = src[src.index("TUNE-HISTORY-BLOCK"):]
    for bad in ("feats.duration_s", "m.dur", "duration_s*", "log10(10)"):
        assert bad not in tuner, f"suspicious sigma derivation: {bad}"


def test_every_run_gets_stable_run_id():
    src = _src()
    assert "runId: 'T' + String(state.runs.length + 1)" in src
    assert "m.runId" in _marker_block()
    assert "registerMarker(run_);" in src


def test_toggle_and_clear_exist_and_are_wired():
    src = _src()
    blk = _marker_block()
    assert src.count('id="tune-show-markers"') == 1
    assert src.count('id="bt-tune-clear"') == 1
    assert "chk.addEventListener('change'" in blk
    assert "clearBtn.addEventListener('click'" in blk
    assert "markerState.enabled" in blk


def test_hover_and_click_expose_required_run_information():
    blk = _marker_block()
    assert "markerAt(mx, my)" in blk or "markerAt(" in blk
    assert "cvs.addEventListener('click'" in blk
    info = blk[blk.index("function runInfoHTML"): blk.index("function showTip")]
    for field in ("runId", "file", "sr", "nfft", "hop", "top_db", "win", "fmax",
                  "actual_sr", "fmax_used", "impact", "dur", "peak_time",
                  "impact_freq_hz", "mel_mean_db"):
        assert field in info, f"run info missing {field}"
    assert "laplace_sigma" in info or ">σ<" in info


def test_no_best_or_ranking_language():
    src = _src()
    tuner = src[src.index("TUNE-HISTORY-BLOCK"): src.index("SLICE 2 · TUNER MARKERS")]
    code = src[src.index("SLICE 2 · TUNER MARKERS"): src.index("[tuner] mounted")]
    # user-facing strings must never say "best"; the code may use `hit` not `best`
    for bad in (" best ", "best=", "Best", "BEST", "optimal", "rank"):
        assert bad not in (tuner + code), f"tuner must not judge settings: {bad}"


def test_no_forbidden_analysis_components():
    src = _src().lower()
    for bad in ("umap", "tsne", "pca", "silhouette_score", "lda"):
        assert bad not in src[src.index("tune-history") - 4000: src.index("[tuner] mounted")], bad
