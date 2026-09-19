#!/usr/bin/env python3
"""
event_store.py — Phase 3: unified event pipeline
=================================================
Single source of truth for all DSP events (mic loop + /compute).

Responsibilities:
  1. Assign a UUID event_id to every event
  2. Store in pending_events (bounded dict, max PENDING_MAX)
     → Phase 4 /confirm-label reads + removes from here
  3. Append to session_history (bounded deque, max HISTORY_MAX)
     → Phase 5 replay + RAG-style matching reads from here
  4. Broadcast the full event (with event_id) to all WS clients via manager
  5. Thread-safe — called from asyncio but also importable by test harness

Checklist compliance:
  §6a  pending_events has explicit TTL eviction + missing-ID detection
  §6b  Same schema (Recording_Device / Recording_Actual_SR_Hz) on all paths
  §7b  session_history is cleared at replay-test start to prevent contamination
  §3 (G3/G4) Provenance + ms timing enforced at the emit boundary
"""
from __future__ import annotations

import asyncio
import collections
import time
import uuid
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from live_server import _ConnectionManager

log = logging.getLogger("event_store")

# ---------------------------------------------------------------------------
# Tunable limits
# ---------------------------------------------------------------------------
PENDING_MAX: int  = 200    # max events awaiting confirm-label (§6a)
HISTORY_MAX: int  = 500    # rolling session history (§7b)
PENDING_TTL_S: float = 300.0  # pending events expire after 5 min (§6a)

# ---------------------------------------------------------------------------
# Stores (module-level singletons — one per server process)
# ---------------------------------------------------------------------------
# event_id -> {event dict + "_stored_at": timestamp}
pending_events: dict[str, dict] = {}

# rolling FIFO of all emitted events (confirmed or not)
session_history: collections.deque[dict] = collections.deque(maxlen=HISTORY_MAX)

# asyncio lock for mutation of pending_events (reads from /confirm-label are async)
_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _evict_expired() -> int:
    """Remove pending entries older than PENDING_TTL_S. Returns count removed."""
    cutoff = time.time() - PENDING_TTL_S
    expired = [eid for eid, ev in pending_events.items() if ev.get("_stored_at", 0) < cutoff]
    for eid in expired:
        del pending_events[eid]
    if expired:
        log.debug("evicted %d expired pending events", len(expired))
    return len(expired)


def _trim_pending() -> None:
    """If over PENDING_MAX, remove oldest entries (by _stored_at)."""
    if len(pending_events) <= PENDING_MAX:
        return
    overflow = len(pending_events) - PENDING_MAX
    oldest = sorted(pending_events.items(), key=lambda x: x[1].get("_stored_at", 0))
    for eid, _ in oldest[:overflow]:
        del pending_events[eid]
    log.warning("pending_events overflow: evicted %d oldest entries", overflow)


# ---------------------------------------------------------------------------
# emit_event — the ONE function all paths must call to publish an event
# ---------------------------------------------------------------------------
async def emit_event(
    payload: dict[str, Any],
    manager: "_ConnectionManager",
    *,
    source: str = "unknown",
    test_mode: bool = False,
) -> str:
    """
    Assign event_id, store, broadcast, and return the event_id.

    Args:
        payload:    Feature dict from _build_response() or mic_loop._dsp_worker().
                    Must already contain Recording_Device, Recording_Actual_SR_Hz,
                    peak_time_ms (all enforced by _build_response / mic_loop).
        manager:    WebSocket connection manager for broadcast.
        source:     "mic_auto_trigger" | "compute_endpoint" | "replay_test"
        test_mode:  If True, writes to session_history but NOT to pending_events,
                    so replayed events cannot pollute the confirm-label queue (§7b).

    Returns:
        event_id (str UUID4)

    Raises:
        ValueError if required provenance fields are missing (§6b enforcement).
    """
    # §6b / G3 — enforce provenance schema on every path
    for required in ("Recording_Device", "Recording_Actual_SR_Hz", "peak_time_ms"):
        if required not in payload:
            raise ValueError(
                f"emit_event: payload missing required field {required!r}. "
                "All paths must go through _build_response() or mic_loop payload builder."
            )

    # Physical Rule: Acoustic impact transient is < 0.5s and MUST NEVER exceed 1.0s.
    # Continuous noise, speech, or runaway ringing > 1.0s MUST BE REJECTED.
    dur_val = float(payload.get("duration_s") or payload.get("duration") or payload.get("detect_duration_s") or 0.0)
    if dur_val > 1.0:
        log.warning("emit_event: REJECTED event with duration %.3fs > 1.0s physical limit (source=%s)", dur_val, source)
        raise ValueError(f"emit_event rejected: transient duration {dur_val:.3f}s exceeds 1.0s physical limit (impacts are < 0.5s)")

    event_id = str(uuid.uuid4())
    now = time.time()

    event = {
        **payload,
        "event_id":  event_id,
        "source":    source,
        "emitted_at": now,
        "_stored_at": now,   # internal; stripped before confirm-label response
    }

    async with _lock:
        # Evict expired entries before adding new one
        _evict_expired()

        if not test_mode:
            # §7b — replay events go to history only, never to pending
            pending_events[event_id] = event
            _trim_pending()

        # session_history always gets the event (for replay + RAG), but
        # test_mode events are tagged so real matching can filter them out
        hist_entry = {**event, "_test_mode": test_mode}
        session_history.append(hist_entry)

    log.info(
        "emit  id=%s  freq=%.1f Hz  peak_ms=%.3f  source=%s  pending=%d  history=%d",
        event_id[:8],
        float(payload.get("impact_freq_hz", 0)),
        float(payload.get("peak_time_ms", 0)),
        source,
        len(pending_events),
        len(session_history),
    )

    # Broadcast to all connected WS clients
    broadcast_payload = {k: v for k, v in event.items() if not k.startswith("_")}
    await manager.broadcast(broadcast_payload)

    return event_id


# ---------------------------------------------------------------------------
# consume_pending — used by /confirm-label (Phase 4)
# ---------------------------------------------------------------------------
async def consume_pending(event_id: str) -> dict | None:
    """
    Remove and return the pending event, or None if not found / expired.

    §6a: Returns None (not raises) for missing/expired IDs so the endpoint
    can return a clear 404, not a 500. The caller is responsible for
    distinguishing "never existed" vs "expired" (both return None here —
    add a separate expired-ids set if finer-grained reporting is needed).
    """
    async with _lock:
        _evict_expired()
        event = pending_events.pop(event_id, None)
    if event is None:
        log.warning("consume_pending: event_id=%s not found (expired or never existed)", event_id)
    return event


# ---------------------------------------------------------------------------
# clear_for_replay — used by Phase 5 test harness (§7b)
# ---------------------------------------------------------------------------
async def clear_for_replay() -> dict:
    """
    Clear session_history and return stats. Does NOT clear pending_events
    (confirmed real events stay pending until the operator confirms them).
    Call this at the START of a replay test run.
    """
    async with _lock:
        n = len(session_history)
        session_history.clear()
    log.info("clear_for_replay: cleared %d history entries", n)
    return {"cleared_history": n, "pending_events_kept": len(pending_events)}


# ---------------------------------------------------------------------------
# get_history_snapshot — safe read for Phase 5 / debug endpoint
# ---------------------------------------------------------------------------
async def get_history_snapshot(
    limit: int = 50,
    exclude_test: bool = True,
) -> list[dict]:
    """Return the N most-recent history entries as a list, newest first."""
    async with _lock:
        items = list(session_history)
    if exclude_test:
        items = [e for e in items if not e.get("_test_mode", False)]
    # Strip internal fields before returning
    clean = [{k: v for k, v in e.items() if not k.startswith("_")} for e in items]
    return list(reversed(clean))[-limit:]
