"""tests/test_confirm_flow.py — LIVE-OPERATIONAL confirm-label flow.

Covers: YES persists + removes pending, NO persists + rejects, unknown id 404,
and similarity is advisory/deterministic. All storage is isolated to a tmp dir
via DSP_CONFIRMED_DIR; no real repo JSONL is written during tests.

Endpoints are called directly as coroutines (the venv has no httpx/pytest-asyncio),
so these are the same code paths FastAPI invokes, minus the transport layer.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import event_store
import confirmed_store
import similarity
import live_server


def _payload(**over) -> dict:
    base = {
        "Recording_Device": "pytest",
        "Recording_Actual_SR_Hz": 44100,
        "peak_time_ms": 120.0,
        "impact": 0.5,
        "duration_s": 0.30,
        "mel_mean_db": -40.0,
        "impact_freq_hz": 3000.0,
    }
    base.update(over)
    return base


class _QuietManager:
    async def broadcast(self, payload):
        return 0


def _emit(payload: dict) -> str:
    return asyncio.run(
        event_store.emit_event(payload, _QuietManager(), source="test_label")
    )


def _body(resp):
    """JSONResponse -> decoded dict."""
    return json.loads(bytes(resp.body).decode("utf-8"))


@pytest.fixture()
def iso_store(tmp_path, monkeypatch):
    """Isolate confirmed_store + event_store state per test."""
    monkeypatch.setenv("DSP_CONFIRMED_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(confirmed_store, "_detector_git_sha", lambda: None)
    snap_pending = dict(event_store.pending_events)
    snap_history = list(event_store.session_history)
    event_store.pending_events.clear()
    event_store.session_history.clear()
    try:
        yield tmp_path / "store"
    finally:
        event_store.pending_events.clear()
        event_store.pending_events.update(snap_pending)
        event_store.session_history.clear()
        event_store.session_history.extend(snap_history)


def test_yes_persists_and_removes_pending(iso_store):
    eid = _emit(_payload())
    assert eid in event_store.pending_events

    resp = asyncio.run(
        live_server.confirm_label(
            live_server._ConfirmBody(event_id=eid, decision="YES", label="B", operator="op1")
        )
    )
    body = _body(resp)
    record = body["record"]

    assert body["status"] == "confirmed"
    # original event payload preserved verbatim under "event"
    assert record["decision"] == "YES"
    assert record["label_truth"] == "B"
    assert record["label_source"] == "manual"
    assert record["operator"] == "op1"
    assert isinstance(record["confirmed_at"], float)
    assert record["detector"]["git_sha"] is None
    assert record["detector_params"]["Recording_Actual_SR_Hz"] == 44100
    assert record["event"]["impact_freq_hz"] == 3000.0
    assert record["event"]["Recording_Device"] == "pytest"

    # persisted, and pending consumed
    lines = (iso_store / "confirmed_events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_id"] == eid
    assert eid not in event_store.pending_events
    assert not (iso_store / "rejected_events.jsonl").exists()

    # advisory returned but explicitly non-authoritative
    assert "advisory" in body
    assert body["advisory"]["available"] is False  # only 1 confirmed record so far


def test_no_persists_rejection_and_removes_pending(iso_store):
    eid = _emit(_payload())

    resp = asyncio.run(
        live_server.confirm_label(
            live_server._ConfirmBody(event_id=eid, decision="NO", operator="op2")
        )
    )
    body = _body(resp)

    assert body["status"] == "rejected"
    assert body["record"]["decision"] == "NO"
    assert body["record"]["label_truth"] is None
    assert body["record"]["operator"] == "op2"
    assert "advisory" not in body

    recs = confirmed_store.load_rejected()
    assert [r["event_id"] for r in recs] == [eid]
    assert confirmed_store.load_confirmed() == []
    assert eid not in event_store.pending_events


def test_unknown_id_404(iso_store):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            live_server.confirm_label(
                live_server._ConfirmBody(event_id="does-not-exist", decision="YES", label="A")
            )
        )
    assert exc.value.status_code == 404


def test_confirmed_listing_and_consume_once(iso_store):
    eid = _emit(_payload())
    asyncio.run(
        live_server.confirm_label(
            live_server._ConfirmBody(event_id=eid, decision="YES", label="C", operator="op3")
        )
    )

    # consume-once: same id cannot be confirmed again
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            live_server.confirm_label(
                live_server._ConfirmBody(event_id=eid, decision="YES")
            )
        )
    assert exc.value.status_code == 404

    listing = _body(asyncio.run(live_server.get_confirmed(limit=5)))
    assert listing["count"] == 1
    assert listing["confirmed"][0]["event_id"] == eid
    assert listing["confirmed"][0]["label_truth"] == "C"


def test_similarity_deterministic_and_advisory_only(iso_store):
    A = {"impact": 0.52, "duration_s": 0.31, "peak_time_ms": 118.0, "mel_mean_db": -40.2, "impact_freq_hz": 3050.0}
    B = {"impact": 0.90, "duration_s": 0.12, "peak_time_ms": 410.0, "mel_mean_db": -22.0, "impact_freq_hz": 12000.0}

    history = []
    for i in range(4):
        history.append({**A, "label_truth": "A", "decision": "YES", "event_id": f"a{i}"})
    for i in range(2):
        history.append({**B, "label_truth": "D", "decision": "YES", "event_id": f"d{i}"})
    # decoys that MUST be ignored: unlabelled, rejected, test-mode, partial-feature
    history.append({**A, "label_truth": None, "decision": "YES", "event_id": "unlabeled"})
    history.append({**A, "label_truth": "A", "decision": "NO", "event_id": "rejected"})
    history.append({**A, "label_truth": "A", "decision": "YES", "event_id": "tm", "_test_mode": True})
    history.append({**A, "label_truth": "A", "decision": "YES", "event_id": "replay", "source": "replay_test"})
    history.append({"impact": 0.5, "label_truth": "A", "decision": "YES", "event_id": "partial"})

    target = {"impact": 0.53, "duration_s": 0.30, "peak_time_ms": 119.0, "mel_mean_db": -40.0, "impact_freq_hz": 3060.0}

    first = similarity.advise(target, confirmed=history)
    second = similarity.advise(target, confirmed=history)
    assert first == second  # deterministic, no randomness / wall clock
    assert first["available"] is True
    assert first["predicted_group"] == "A"
    assert first["advisory_only"] is True
    assert first["basis_count"] == 6  # only the 6 labelled, non-test confirmed records
    assert 0.0 <= first["confidence"] <= 1.0
    assert first["basis_event_ids"] and first["basis_event_ids"][0].startswith("a")

    # the advisory carries no chart/coordinate payload — labels + scores only
    assert set(first) <= {
        "available", "reason", "predicted_group", "confidence", "score",
        "basis_count", "basis_event_ids", "advisory_only",
    }


def test_similarity_reads_only_confirmed_store(iso_store):
    """Rejections must never enter the advisory basis."""
    for i in range(5):
        confirmed_store.append_rejected(
            {**_payload(), "event_id": f"r{i}", "source": "test_label"}
        )
    out = similarity.advise(_payload())
    assert out["available"] is False
    assert out["reason"] == "insufficient_confirmed_history"
    assert out["basis_count"] == 0


def test_similarity_insufficient_history(iso_store):
    out = similarity.advise(_payload(), confirmed=[])
    assert out["available"] is False
    assert out["reason"] == "insufficient_confirmed_history"


def test_malformed_jsonl_lines_are_skipped(iso_store):
    bad = iso_store / "confirmed_events.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text(
        '{"event_id": "good1", "decision": "YES", "label_truth": "A"}\n'
        "THIS IS NOT JSON {{{ truncated\n"
        "\n"
        '{"event_id": "good2", "decision": "YES", "label_truth": "B"}\n'
        "[1,2,3]\n",
        encoding="utf-8",
    )
    recs = confirmed_store.load_confirmed(limit=10)
    assert [r["event_id"] for r in recs] == ["good1", "good2"]


def test_invalid_label_does_not_consume_pending(iso_store):
    """400 for bad label must NOT consume the pending event."""
    eid = _emit(_payload())
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            live_server.confirm_label(
                live_server._ConfirmBody(event_id=eid, decision="YES", label="Z")
            )
        )
    assert exc.value.status_code == 400
    assert eid in event_store.pending_events
    # no record written
    assert confirmed_store.load_confirmed() == []
    # retry with valid label now succeeds
    resp = asyncio.run(
        live_server.confirm_label(
            live_server._ConfirmBody(event_id=eid, decision="YES", label="A")
        )
    )
    assert resp.status_code == 200
