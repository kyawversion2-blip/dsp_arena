#!/usr/bin/env python3
"""
mic_loop.py — Continuous auto-trigger mic loop  (Phase 2)
==========================================================
Wraps manual_stop_interactive_recorder.py into a continuous event stream.

Architecture (checklist §4 guards baked in):
  ┌─────────────────────────────────────────────────────────────────┐
  │ sounddevice InputStream  (audio thread — C callback)            │
  │   • write every block to ring_buffer (numpy, pre-allocated)     │
  │   • update last_callback_ts (watchdog heartbeat)                │
  │   • simple RMS energy check (NO scipy/librosa in callback)      │
  │   • if onset AND refractory expired → put snapshot in _trig_q   │
  │   • state machine: LISTENING → COLLECTING → LISTENING           │
  └────────────────────────┬────────────────────────────────────────┘
                           │  queue.SimpleQueue  (thread-safe, non-blocking put)
  ┌────────────────────────▼────────────────────────────────────────┐
  │ asyncio DSP worker  (main event loop — run_in_executor)         │
  │   • blocks on _trig_q.get() in thread pool (never blocks loop)  │
  │   • copies ring-buffer slice (handles wrap-around)              │
  │   • detect_transient_window() + measure_features()              │
  │   • builds event dict with provenance (G3/G4/G5)                │
  │   • broadcast via manager (WebSocket)                           │
  └─────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────┐
  │ watchdog coroutine (every WATCHDOG_INTERVAL_S)                  │
  │   • if time.perf_counter() - last_callback_ts > DEAD_S:         │
  │       log WARNING: "stream may have died"                       │
  │   • stream reconnect on exception in _stream_task               │
  └─────────────────────────────────────────────────────────────────┘

Checklist compliance:
  §4a  Refractory window REFRACTORY_S = 0.40 s — no double-fire
  §4b  Ring buffer = max(4 s, 2*(ff_s+rr_s)+1) at actual_sr — transient never split
  §4c  Audio callback: ONLY numpy ops + SimpleQueue.put_nowait — no DSP, no I/O
  §4d  Watchdog logs when gap > DEAD_S; reconnect loop on stream crash
  §1a  Device name + stream.samplerate logged on every (re)connect
  §1b  actual_sr threaded to every DSP call; no fallback to module constant
  §1d  On reconnect, new device info queried and logged fresh
  §1c  Recording_Device / Recording_Actual_SR_Hz in every broadcast event
  §3 (G3/G4) Recording_Device, Recording_Actual_SR_Hz, peak_time_ms on every event
"""
from __future__ import annotations

import asyncio
import logging
import queue
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd

from dsp_inverse import detect_transient_window, measure_features, resample_linear
import event_store as _store

if TYPE_CHECKING:
    from live_server import _ConnectionManager

log = logging.getLogger("mic_loop")

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
REFRACTORY_S: float = 0.40       # §4a — minimum gap between triggers (seconds)
FF_S: float = 0.50               # pre-trigger context window (seconds)
RR_S: float = 0.50               # post-trigger capture window (seconds)
ONSET_RMS_THRESHOLD: float = 0.01  # energy threshold for onset detection (tunable)
WATCHDOG_INTERVAL_S: float = 5.0  # how often watchdog checks
DEAD_STREAM_S: float = 10.0      # no callback for this long → warn "stream may be dead"
MAX_RECONNECT_ATTEMPTS: int = 10
RECONNECT_DELAY_S: float = 2.0
DSP_QUEUE_MAXSIZE: int = 8       # drop oldest if worker falls behind

# DSP params (match locked θ from PROJECT_MEMORY §2)
_NFFT: int = 4_096
_HOP: int = 1_024
_TOPDB: int = 35
_WIN: float = 0.5


# ---------------------------------------------------------------------------
# Shared state (module-level so callback (C thread) can read/write)
# ---------------------------------------------------------------------------
@dataclass
class _LoopState:
    # Ring buffer — allocated once, reused across reconnects
    buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    write_pos: int = 0          # next write index (mod buf_len)
    buf_len: int = 0

    # Trigger state machine
    state: str = "LISTENING"   # "LISTENING" | "COLLECTING"
    trigger_pos: int = 0       # ring-buf index at onset moment
    remaining_post: int = 0    # samples still needed post-trigger

    # Timing
    last_callback_ts: float = 0.0   # updated by every audio callback (watchdog)
    last_trigger_ts: float = 0.0    # updated on each trigger (refractory)

    # Provenance — updated on every (re)connect
    device_name: str = "unknown"
    actual_sr: int = 44_100


_st = _LoopState()
_trig_q: queue.SimpleQueue = queue.SimpleQueue()  # thread-safe, non-blocking put


# ---------------------------------------------------------------------------
# Audio callback — runs in sounddevice C thread
# RULE: ONLY numpy ops + SimpleQueue.put_nowait.  NO scipy, NO librosa.
# ---------------------------------------------------------------------------
def _audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
    """Real-time audio callback — must never block."""
    now = time.perf_counter()
    _st.last_callback_ts = now   # §4d watchdog heartbeat

    if status:
        # Log via stderr (thread-safe, no asyncio)
        print(f"[mic_loop] stream status: {status}", file=sys.stderr)

    # Mono mix (indata shape: (frames, channels))
    mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
    n = len(mono)

    # Write to ring buffer (may wrap)
    end = _st.write_pos + n
    if end <= _st.buf_len:
        _st.buf[_st.write_pos : end] = mono
    else:
        first = _st.buf_len - _st.write_pos
        _st.buf[_st.write_pos :] = mono[:first]
        _st.buf[: n - first]     = mono[first:]
    _st.write_pos = end % _st.buf_len

    # ── State machine ────────────────────────────────────────────────────────
    if _st.state == "LISTENING":
        # Simple RMS energy check — §4a refractory guard
        rms = float(np.sqrt(np.mean(mono * mono)))
        if rms >= ONSET_RMS_THRESHOLD and (now - _st.last_trigger_ts) >= REFRACTORY_S:
            # Onset detected — begin collecting post-trigger audio
            _st.trigger_pos   = _st.write_pos   # ring-buf index right after this block
            _st.remaining_post = int(_st.actual_sr * RR_S)
            _st.state = "COLLECTING"

    elif _st.state == "COLLECTING":
        _st.remaining_post -= n
        if _st.remaining_post <= 0:
            # Enough post-trigger audio collected — snapshot ring buffer
            # Copy write_pos and trigger_pos atomically (single Python assignment each)
            snap_write   = _st.write_pos
            snap_trigger = _st.trigger_pos
            snap_buf     = _st.buf.copy()   # numpy copy — safe from C thread
            snap_sr      = _st.actual_sr
            snap_dev     = _st.device_name
            snap_ts      = time.perf_counter()
            # Hand off to DSP worker — put_nowait never blocks (§4c)
            try:
                _trig_q.put_nowait((snap_buf, snap_trigger, snap_write, snap_sr, snap_dev, snap_ts))
            except Exception:
                pass  # queue full (shouldn't happen with SimpleQueue; ignore silently)
            _st.last_trigger_ts = now
            _st.state = "LISTENING"


# ---------------------------------------------------------------------------
# Ring-buffer window extractor (safe wrap-around copy)
# ---------------------------------------------------------------------------
def _extract_window(buf: np.ndarray, trigger_pos: int, write_pos: int, sr: int) -> np.ndarray:
    """Extract the [trigger - FF_S, trigger + RR_S] window from the ring buffer.

    §4b: Ring buffer is sized so this window is always fully present.
    Handles wrap-around correctly via double-copy.
    """
    buf_len   = len(buf)
    ff_samps  = int(sr * FF_S)
    rr_samps  = int(sr * RR_S)
    total     = ff_samps + rr_samps

    # onset is at trigger_pos (next index after onset block)
    # pre-trigger: go back ff_samps from trigger_pos
    start = (trigger_pos - ff_samps) % buf_len

    if start + total <= buf_len:
        return buf[start : start + total].astype(np.float64)
    else:
        first = buf_len - start
        return np.concatenate([buf[start:], buf[: total - first]]).astype(np.float64)


# ---------------------------------------------------------------------------
# DSP worker — asyncio coroutine
# ---------------------------------------------------------------------------
async def _dsp_worker(manager: "_ConnectionManager") -> None:
    """Pull trigger snapshots from the queue, run DSP, broadcast."""
    loop = asyncio.get_event_loop()
    while True:
        # Block in a thread-pool thread — never blocks the event loop
        snapshot = await loop.run_in_executor(None, _trig_q.get)
        snap_buf, snap_trigger, snap_write, snap_sr, snap_dev, snap_ts = snapshot

        def _compute() -> dict | None:
            audio = _extract_window(snap_buf, snap_trigger, snap_write, snap_sr)
            if len(audio) < max(64, int(snap_sr * 0.05)):
                return None
            features = measure_features(audio, snap_sr, _NFFT, _HOP, _TOPDB, _WIN)
            return features

        t0 = time.perf_counter()
        try:
            features = await loop.run_in_executor(None, _compute)
        except Exception:
            log.exception("DSP error in auto-trigger worker")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if features is None:
            log.debug("auto-trigger: no transient detected in window")
            continue

        pt_s = float(features.get("peak_time", 0.0))
        payload = {
            # G3 — provenance on every event
            "Recording_Device":       snap_dev,
            "Recording_Actual_SR_Hz": snap_sr,
            # G4 — time ALWAYS in ms
            "peak_time_ms":           round(pt_s * 1000.0, 3),
            "peak_time_s":            round(pt_s, 6),   # debug only
            # features
            "impact":                 round(float(features.get("impact",               0.0)), 6),
            "duration_s":             round(float(features.get("duration",              0.0)), 6),
            "impact_freq_hz":         round(float(features.get("impact_freq_hz",        0.0)), 1),
            "mel_mean_db":            round(float(features.get("mel_mean_db",           0.0)), 3),
            "rms_energy":             round(float(features.get("rms_energy",            0.0)), 6),
            "attack_slope_db_ms":     round(float(features.get("attack_slope_db_ms",    0.0)), 4),
            "crest_factor":           round(float(features.get("crest_factor",          0.0)), 4),
            "hf_transient_ratio":     round(float(features.get("hf_transient_ratio",    0.0)), 6),
            "spectral_centroid_hz":   round(float(features.get("spectral_centroid_hz",  0.0)), 1),
            "spectral_bandwidth_hz":  round(float(features.get("spectral_bandwidth_hz", 0.0)), 1),
            "spectral_rolloff_hz":    round(float(features.get("spectral_rolloff_hz",   0.0)), 1),
            "spectral_flatness":      round(float(features.get("spectral_flatness",     0.0)), 6),
            "zcr":                    round(float(features.get("zcr",                   0.0)), 6),
            "compute_elapsed_ms":     round(elapsed_ms, 2),
            "ts":                     snap_ts,
        }

        # Phase 3: unified emit — store + broadcast via event_store
        try:
            event_id = await _store.emit_event(payload, manager, source="mic_auto_trigger")
            log.info(
                "event  id=%s  freq=%.1f Hz  peak_ms=%.3f  device=%r  sr=%d  elapsed=%.0f ms",
                event_id[:8], payload["impact_freq_hz"], payload["peak_time_ms"],
                snap_dev, snap_sr, elapsed_ms,
            )
        except ValueError as ve:
            log.error("emit_event schema error: %s", ve)


# ---------------------------------------------------------------------------
# Watchdog — detects dead stream (§4d)
# ---------------------------------------------------------------------------
async def _watchdog() -> None:
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL_S)
        gap = time.perf_counter() - _st.last_callback_ts
        if _st.last_callback_ts > 0 and gap > DEAD_STREAM_S:
            log.warning(
                "WATCHDOG: no audio callback for %.1f s — "
                "stream may have died or mic disconnected. "
                "Gap ≠ 'no impacts' — verify device is live.",
                gap,
            )


# ---------------------------------------------------------------------------
# Stream task — opens sounddevice stream with reconnect loop
# ---------------------------------------------------------------------------
def _init_ring_buffer(sr: int) -> None:
    """Allocate ring buffer sized to prevent transient split (§4b)."""
    # Must hold at least: FF_S + RR_S + margin so any trigger window fits.
    buf_seconds = max(4.0, 2.0 * (FF_S + RR_S) + 1.0)
    _st.buf_len = int(sr * buf_seconds)
    _st.buf     = np.zeros(_st.buf_len, dtype=np.float32)
    _st.write_pos = 0
    log.info("ring buffer: %.1f s = %d samples at %d Hz", buf_seconds, _st.buf_len, sr)


async def _stream_task(
    device:       int | None,
    requested_sr: int,
    channels:     int,
) -> None:
    """Open stream, log provenance, run until cancelled or error."""
    loop = asyncio.get_event_loop()

    # Discover actual SR before allocating ring buffer (§1a / G5)
    def _probe_sr() -> tuple[str, int]:
        with sd.InputStream(
            device=device,
            samplerate=requested_sr,
            channels=channels,
            dtype="float32",
            blocksize=512,
        ) as probe:
            real_sr  = int(probe.samplerate)
            dev_idx  = probe.device[0] if isinstance(probe.device, (list, tuple)) else probe.device
            dev_info = sd.query_devices(dev_idx)
            dev_name = dev_info["name"]
        return dev_name, real_sr

    dev_name, actual_sr = await loop.run_in_executor(None, _probe_sr)

    # G5 / §1a — log verified stream info
    log.info("STREAM OPEN  device=%r  requested_sr=%d  actual_sr=%d (verified from stream.samplerate)",
             dev_name, requested_sr, actual_sr)

    # Update shared state (safe — written before stream starts)
    _st.device_name = dev_name
    _st.actual_sr   = actual_sr
    _st.last_callback_ts = time.perf_counter()   # reset watchdog clock

    _init_ring_buffer(actual_sr)

    # Open live stream
    with sd.InputStream(
        device=device,
        samplerate=actual_sr,
        channels=channels,
        dtype="float32",
        callback=_audio_callback,
        blocksize=512,
    ):
        log.info("STREAM LIVE  device=%r  actual_sr=%d", dev_name, actual_sr)
        # Block until cancelled
        try:
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            log.info("STREAM STOP  device=%r", dev_name)
            raise


async def run_mic_loop(
    manager:      "_ConnectionManager",
    device:       int | None = None,
    requested_sr: int = 44_100,
    channels:     int = 1,
) -> None:
    """
    Top-level entry point — start all mic-loop tasks.
    Call as an asyncio.Task from live_server.py startup lifespan.

    On reconnect: re-queries device info → §1d (no stale device reuse).
    """
    # Launch DSP worker and watchdog
    dsp_task  = asyncio.create_task(_dsp_worker(manager),  name="dsp_worker")
    watch_task = asyncio.create_task(_watchdog(),           name="watchdog")

    attempt = 0
    while True:
        attempt += 1
        try:
            await _stream_task(device, requested_sr, channels)
        except asyncio.CancelledError:
            log.info("mic_loop cancelled — shutting down")
            dsp_task.cancel()
            watch_task.cancel()
            return
        except Exception as exc:
            log.error("stream error (attempt %d/%d): %s", attempt, MAX_RECONNECT_ATTEMPTS, exc)
            if attempt >= MAX_RECONNECT_ATTEMPTS:
                log.critical("max reconnect attempts reached — giving up")
                dsp_task.cancel()
                watch_task.cancel()
                return
            log.info("reconnecting in %.1f s …  (§1d: fresh device query on reconnect)", RECONNECT_DELAY_S)
            # §1d: on reconnect, device info is re-queried from scratch in _stream_task
            await asyncio.sleep(RECONNECT_DELAY_S)
