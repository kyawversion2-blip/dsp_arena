#!/usr/bin/env python3
"""
confirmed_store.py — append-only persistence for operator label decisions
=========================================================================
Stores the live-operational confirm-label flow:

  confirmed_events.jsonl   one record per YES decision (operator ground truth)
  rejected_events.jsonl    one record per NO decision

Design rules (deliberately narrow):

  1. Append-only. Existing lines are never rewritten, reordered, or deleted.
  2. Never touches features.csv or any other measurement artifact. This module
     only writes the two JSONL files below.
  3. Robust to a missing file, an empty file, and malformed (partial) lines —
     a corrupt tail must not break the live UI or the advisory.
  4. The original event payload is preserved verbatim under "event"; the
     decision metadata is added alongside it, never merged into it.

Storage location:
  $DSP_CONFIRMED_DIR (if set), else <repo>/dsp_out/confirmed_store/.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("confirmed_store")

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "dsp_out" / "confirmed_store"
CONFIRMED_FILENAME = "confirmed_events.jsonl"
REJECTED_FILENAME = "rejected_events.jsonl"

SCHEMA_VERSION = 1

# Payload keys that describe detector identity / params. Copied by reference
# into the record when present; absent keys are simply omitted.
DETECTOR_PARAM_KEYS = (
    "engine",
    "filename",
    "Recording_Device",
    "Recording_Actual_SR_Hz",
    "native_sr_hz",
    "fmax_used_hz",
    "sr",
    "nfft",
    "hop",
    "top_db",
    "win",
)


def store_dir() -> Path:
    """Resolve the storage directory (env override wins, for tests/isolation)."""
    override = os.environ.get("DSP_CONFIRMED_DIR")
    return Path(override) if override else DEFAULT_DIR


def _detector_git_sha() -> str | None:
    """Best-effort git SHA of the running code. Never raises."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception as exc:  # pragma: no cover - environment dependent
        log.debug("git SHA unavailable: %s", exc)
        return None


def _detector_identity(event: dict) -> dict:
    """Detector identity + version + git SHA for the run that produced `event`."""
    return {
        "engine": event.get("engine"),
        "module": "dsp_inverse.measure_features",
        "server_version": "3.0.0",
        "git_sha": _detector_git_sha(),
    }


def _detector_params(event: dict) -> dict:
    return {k: event[k] for k in DETECTOR_PARAM_KEYS if k in event}


def _strip_internal(event: dict) -> dict:
    return {k: v for k, v in event.items() if not str(k).startswith("_")}


def build_confirmed_record(
    event: dict,
    *,
    label_truth: str | None = None,
    operator: str | None = None,
    decision: str = "YES",
) -> dict:
    """Build the persisted record. The original payload is preserved verbatim."""
    clean_event = _strip_internal(event)
    params = _detector_params(clean_event)
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "label_truth": label_truth,
        "label_source": "manual",
        "operator": operator,
        "confirmed_at": time.time(),
        "event_id": clean_event.get("event_id"),
        "source": clean_event.get("source"),
        "emitted_at": clean_event.get("emitted_at"),
        "detector": _detector_identity(clean_event),
        "detector_params": params,
        # Placeholders: filled only if a later phase freezes a params/stats
        # manifest. Kept explicit so the schema does not shift later.
        "params_hash": None,
        "stats_hash": None,
        "features": {
            "impact": clean_event.get("impact"),
            "duration_s": clean_event.get("duration_s"),
            "peak_time_ms": clean_event.get("peak_time_ms"),
            "mel_mean_db": clean_event.get("mel_mean_db"),
            "impact_freq_hz": clean_event.get("impact_freq_hz"),
        },
        "event": clean_event,
    }


def _append(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception as exc:
        # A persistence failure must be visible but must not crash the flow:
        # the caller still returns the decision to the UI.
        log.error("append to %s failed: %s", path, exc)
        raise


def append_confirmed(
    event: dict,
    *,
    label_truth: str | None = None,
    operator: str | None = None,
) -> dict:
    record = build_confirmed_record(
        event, label_truth=label_truth, operator=operator, decision="YES"
    )
    _append(store_dir() / CONFIRMED_FILENAME, record)
    log.info(
        "confirmed id=%s label=%s operator=%s",
        record.get("event_id"),
        label_truth,
        operator,
    )
    return record


def append_rejected(event: dict, *, operator: str | None = None) -> dict:
    record = build_confirmed_record(event, label_truth=None, operator=operator, decision="NO")
    _append(store_dir() / REJECTED_FILENAME, record)
    log.info("rejected id=%s operator=%s", record.get("event_id"), operator)
    return record


def read_records(path: Path, *, limit: int | None = None) -> list[dict]:
    """Read JSONL records, skipping malformed lines. Missing file -> [].

    Returned oldest-first. `limit` keeps the LAST `limit` records (most recent).
    """
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    # Truncated/partial write from an interrupted process.
                    log.warning(
                        "skipping malformed line %d in %s", lineno, path.name
                    )
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
                else:
                    log.warning("skipping non-object line %d in %s", lineno, path.name)
    except OSError as exc:
        log.error("read %s failed: %s", path, exc)
        return []
    if limit is not None and limit >= 0:
        out = out[-limit:]
    return out


def load_confirmed(limit: int = 20) -> list[dict]:
    """Most recent confirmed records, oldest-first (caller sees newest last)."""
    return read_records(store_dir() / CONFIRMED_FILENAME, limit=limit)


def load_rejected(limit: int = 20) -> list[dict]:
    return read_records(store_dir() / REJECTED_FILENAME, limit=limit)
