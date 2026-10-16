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
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd

from audio_device_resolver import AudioDeviceResolutionError, resolve_audio_input_device
from dsp_inverse import detect_transient_window, measure_features, resample_linear
import event_store as _store

if TYPE_CHECKING:
    from live_server import _ConnectionManager

log = logging.getLogger("mic_loop")


class StreamStallError(RuntimeError):
    """Raised when the audio callback stalls (stream died or no callbacks received)."""
    pass

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
REFRACTORY_S: float = 0.40       # §4a — minimum gap between triggers (seconds)
FF_S: float = 0.50               # pre-trigger context window (seconds)
RR_S: float = 0.50               # post-trigger capture window (seconds)
ONSET_RMS_THRESHOLD: float = 0.01  # energy threshold for onset detection (tunable)
SILENCE_RMS_THRESHOLD: float = 0.0001 # threshold below which signal is considered silence
WATCHDOG_INTERVAL_S: float = 5.0  # how often watchdog checks
DEAD_STREAM_S: float = 10.0      # no callback for this long → warn "stream may be dead"
MAX_RECONNECT_ATTEMPTS: int = 10
RECONNECT_DELAY_S: float = 2.0
DSP_QUEUE_MAXSIZE: int = 8       # drop oldest if worker falls behind
MAX_SUPPORTED_SR: int = 96_000   # maximum supported sample rate for bounded buffer pool
DEFAULT_MAX_EVENT_SAMPLES: int = int(MAX_SUPPORTED_SR * (FF_S + RR_S))  # 96,000 samples (1.0s at 96 kHz)

# DSP params (match locked θ from PROJECT_MEMORY §2)
_NFFT: int = 4_096
_HOP: int = 1_024
_TOPDB: int = 35
_WIN: float = 0.5


# ---------------------------------------------------------------------------
# Bounded Event Buffer Pool & Zero-Allocation Handoff
# ---------------------------------------------------------------------------
class EventBufferSlot:
    """Pre-allocated event audio buffer slot.

    Slot Ownership & Lifecycle:
    1. Pool creation: Pre-allocates N slots (DSP_QUEUE_MAXSIZE = 8). Each slot holds
       a float32 array sized for max_sr * (FF_S + RR_S) = 96,000 samples (1.0s at 96 kHz).
    2. Acquisition (Producer): When trigger collection finishes in _audio_callback,
       producer acquires a free slot via pool.acquire_nowait(). Slot is marked acquired.
       If pool is empty (saturation), producer drops the trigger, increments
       _st.dropped_events, and returns to LISTENING without blocking.
    3. Population: Producer copies the [trigger - FF_S, trigger + RR_S] window
       directly into slot.data[:total].
    4. Handoff: Producer puts slot into the session's bounded queue.
    5. Consumption (Consumer): DSP worker pulls slot from queue. Consumer has exclusive
       read access while the ring buffer continues writing safely.
    6. Release: Upon finishing feature extraction (or rejection), consumer calls
       slot.release(), returning it to the pool for reuse. Double release is guarded.
    """

    def __init__(self, slot_id: int, max_samples: int, pool: "EventBufferPool") -> None:
        self.slot_id = slot_id
        self.data: np.ndarray = np.zeros(max_samples, dtype=np.float32)
        self.pool = pool
        self.length: int = 0
        self.sr: int = 44100
        self.device: str = "unknown"
        self.ts: float = 0.0
        self._in_use: bool = False
        self._lock: threading.Lock = threading.Lock()

    def mark_acquired(self) -> None:
        with self._lock:
            self._in_use = True

    def release(self) -> None:
        with self._lock:
            if not self._in_use:
                return  # Guard against double release
            self._in_use = False
        self.pool.release(self)


class EventBufferPool:
    """Thread-safe pool of pre-allocated audio event buffer slots."""

    def __init__(self, count: int = DSP_QUEUE_MAXSIZE, max_samples: int = DEFAULT_MAX_EVENT_SAMPLES) -> None:
        self._slots: list[EventBufferSlot] = [
            EventBufferSlot(slot_id=i, max_samples=max_samples, pool=self)
            for i in range(count)
        ]
        self._free: queue.SimpleQueue[EventBufferSlot] = queue.SimpleQueue()
        for slot in self._slots:
            self._free.put_nowait(slot)

    def acquire_nowait(self) -> EventBufferSlot | None:
        try:
            slot = self._free.get_nowait()
            slot.mark_acquired()
            return slot
        except queue.Empty:
            return None

    def release(self, slot: EventBufferSlot) -> None:
        self._free.put_nowait(slot)

    @property
    def available_count(self) -> int:
        return self._free.qsize()

    @property
    def total_count(self) -> int:
        return len(self._slots)


_SHUTDOWN_SENTINEL = object()


class MicSession:
    """Explicit per-run queue & lifecycle management to eliminate zombie threads."""

    def __init__(self, maxsize: int = DSP_QUEUE_MAXSIZE) -> None:
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.stop_event: threading.Event = threading.Event()
        self.is_active: bool = True

    def stop(self) -> None:
        self.is_active = False
        self.stop_event.set()
        try:
            self.queue.put_nowait(_SHUTDOWN_SENTINEL)
        except Exception:
            pass

    def drain_and_release(self) -> int:
        """Drain remaining queued slots and return them to the pool on shutdown."""
        drained = 0
        while True:
            try:
                item = self.queue.get_nowait()
                if isinstance(item, EventBufferSlot):
                    item.release()
                    drained += 1
            except queue.Empty:
                break
            except Exception:
                break
        return drained


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

    # Stream state: "uninitialized" | "stream_open" | "audio_received" | "signal_detected" | "silence" | "callback_stalled" | "capture_error"
    stream_state: str = "uninitialized"
    error_message: str | None = None
    device_identity: dict[str, Any] = field(default_factory=lambda: {
        "index": -1,
        "name": "unknown",
        "hostapi": -1,
        "hostapi_name": "unknown",
    })

    # Timing
    stream_open_ts: float = 0.0              # timestamp when stream was opened
    last_callback_ts: float = 0.0            # updated by every audio callback (perf_counter)
    last_successful_callback_ts: float = 0.0 # updated when callback processes samples
    last_trigger_ts: float = 0.0             # updated on each trigger (refractory)
    last_signal_ts: float = 0.0              # last time RMS >= ONSET_RMS_THRESHOLD
    last_active_audio_ts: float = 0.0        # last time RMS >= SILENCE_RMS_THRESHOLD

    # Metrics
    latest_rms: float = 0.0
    latest_peak: float = 0.0
    clipped_samples: int = 0
    total_samples: int = 0
    callback_count: int = 0
    dropped_events: int = 0
    reconnect_attempts: int = 0

    # Provenance — updated on every (re)connect
    device_name: str = "unknown"
    requested_sr: int = 44_100
    actual_sr: int = 44_100
    fallback_sr: int | None = None
    channels: int = 1


_pool = EventBufferPool()
_active_session: MicSession | None = None
_st = _LoopState()
_trig_q: queue.SimpleQueue = queue.SimpleQueue()  # fallback / test queue


def reset_mic_state() -> None:
    """Reset shared module state for clean test isolation and lifecycle boundaries."""
    global _st, _trig_q, _pool, _active_session
    _st = _LoopState()
    if _active_session is not None:
        _active_session.stop()
        _active_session.drain_and_release()
        _active_session = None
    while not _trig_q.empty():
        try:
            item = _trig_q.get_nowait()
            if isinstance(item, EventBufferSlot):
                item.release()
        except Exception:
            break
    _pool = EventBufferPool()


def get_mic_diagnostics() -> dict[str, Any]:
    """Return a thread-safe snapshot of microphone stream diagnostics."""
    now = time.perf_counter()
    clipping_ratio = float(_st.clipped_samples / max(1, _st.total_samples))
    if _st.last_active_audio_ts > 0.0:
        silence_dur = max(0.0, round(now - _st.last_active_audio_ts, 3))
    elif _st.last_callback_ts > 0.0:
        silence_dur = max(0.0, round(now - _st.last_callback_ts, 3))
    else:
        silence_dur = 0.0

    diag: dict[str, Any] = {
        "stream_state": _st.stream_state,
        "device_identity": dict(_st.device_identity),
        "device_name": _st.device_name,
        "requested_sr": _st.requested_sr,
        "actual_sr": _st.actual_sr,
        "channels": _st.channels,
        "rms": round(_st.latest_rms, 6),
        "peak_amplitude": round(_st.latest_peak, 6),
        "clipped_samples": _st.clipped_samples,
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_duration_s": silence_dur,
        "last_successful_callback_ts": _st.last_successful_callback_ts,
        "callback_count": _st.callback_count,
        "total_samples": _st.total_samples,
        "dropped_events": _st.dropped_events,
        "reconnect_attempts": _st.reconnect_attempts,
        "error_message": _st.error_message,
    }
    if _st.fallback_sr is not None:
        diag["fallback_sr"] = _st.fallback_sr
    return diag



# ---------------------------------------------------------------------------
# Audio callback — runs in sounddevice C thread
# RULE: ONLY numpy ops + SimpleQueue.put_nowait.  NO scipy, NO librosa.
# ---------------------------------------------------------------------------
def _audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
    """Real-time audio callback — must never block."""
    now = time.perf_counter()
    _st.last_callback_ts = now   # §4d watchdog heartbeat
    _st.last_successful_callback_ts = now
    _st.callback_count += 1
    if _st.reconnect_attempts > 0:
        _st.reconnect_attempts = 0

    if status:
        # Log via stderr (thread-safe, no asyncio)
        print(f"[mic_loop] stream status: {status}", file=sys.stderr)

    # Mono mix (indata shape: (frames, channels))
    mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
    n = len(mono)

    rms = float(np.sqrt(np.mean(mono * mono))) if n > 0 else 0.0
    peak = float(np.max(np.abs(mono))) if n > 0 else 0.0
    clipped = int(np.sum(np.abs(mono) >= 0.999)) if n > 0 else 0

    _st.latest_rms = rms
    _st.latest_peak = peak
    _st.clipped_samples += clipped
    _st.total_samples += n

    # Distinguish stream states
    if rms >= ONSET_RMS_THRESHOLD:
        _st.stream_state = "signal_detected"
        _st.last_signal_ts = now
        _st.last_active_audio_ts = now
    elif rms >= SILENCE_RMS_THRESHOLD:
        _st.stream_state = "audio_received"
        _st.last_active_audio_ts = now
    else:
        # RMS is below SILENCE_RMS_THRESHOLD
        if _st.last_active_audio_ts == 0.0 or (now - _st.last_active_audio_ts) >= 0.5:
            _st.stream_state = "silence"
        else:
            _st.stream_state = "audio_received"

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
        if rms >= ONSET_RMS_THRESHOLD and (now - _st.last_trigger_ts) >= REFRACTORY_S:
            # Onset detected — begin collecting post-trigger audio
            _st.trigger_pos   = _st.write_pos   # ring-buf index right after this block
            _st.remaining_post = int(_st.actual_sr * RR_S)
            _st.state = "COLLECTING"

    elif _st.state == "COLLECTING":
        _st.remaining_post -= n
        if _st.remaining_post <= 0:
            # Enough post-trigger audio collected
            # Handoff via bounded buffer pool
            slot = _pool.acquire_nowait()
            if slot is None:
                # Saturation / backpressure: drop event cleanly without blocking C thread
                _st.dropped_events += 1
                _st.last_trigger_ts = now
                _st.state = "LISTENING"
                return

            enqueued = False
            try:
                ff_samps = int(_st.actual_sr * FF_S)
                rr_samps = int(_st.actual_sr * RR_S)
                total = ff_samps + rr_samps
                buf_len = _st.buf_len

                # P1-1: Verify that required sample count fits acquired slot
                if total > len(slot.data):
                    _st.dropped_events += 1
                    return

                start = (_st.trigger_pos - ff_samps) % buf_len
                if start + total <= buf_len:
                    slot.data[:total] = _st.buf[start : start + total]
                else:
                    first = buf_len - start
                    slot.data[:first] = _st.buf[start:]
                    slot.data[first:total] = _st.buf[: total - first]

                slot.length = total
                slot.sr = _st.actual_sr
                slot.device = _st.device_name
                slot.ts = time.perf_counter()

                # Enqueue strictly into active session bounded queue
                session = _active_session
                if session is not None and session.is_active:
                    session.queue.put_nowait(slot)
                    enqueued = True
                else:
                    # Racing with session stop or uninitialized session: drop event safely
                    _st.dropped_events += 1
                    # enqueued remains False -> finally block will call slot.release()
            except queue.Full:
                _st.dropped_events += 1
            except Exception as cb_exc:
                _st.dropped_events += 1
                print(f"[mic_loop] callback exception: {cb_exc}", file=sys.stderr)
            finally:
                if not enqueued:
                    slot.release()
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
async def _dsp_worker(manager: "_ConnectionManager", session: MicSession | None = None) -> None:
    """Pull trigger snapshots from the queue, run DSP, broadcast."""
    loop = asyncio.get_event_loop()
    target_session = session or _active_session
    q = target_session.queue if target_session else _trig_q
    stop_ev = target_session.stop_event if target_session else None

    def _get_next():
        while stop_ev is None or not stop_ev.is_set():
            try:
                return q.get(timeout=0.2)
            except queue.Empty:
                if stop_ev is not None and stop_ev.is_set():
                    break
                continue
        return _SHUTDOWN_SENTINEL

    while True:
        try:
            snapshot = await loop.run_in_executor(None, _get_next)
        except asyncio.CancelledError:
            break

        if snapshot is _SHUTDOWN_SENTINEL or snapshot is None:
            break

        slot: EventBufferSlot | None = snapshot if isinstance(snapshot, EventBufferSlot) else None
        try:
            if slot is not None:
                audio = slot.data[:slot.length].astype(np.float64)
                snap_sr = slot.sr
                snap_dev = slot.device
                snap_ts = slot.ts
            else:
                snap_buf, snap_trigger, snap_write, snap_sr, snap_dev, snap_ts = snapshot
                audio = _extract_window(snap_buf, snap_trigger, snap_write, snap_sr)

            if len(audio) < max(64, int(snap_sr * 0.05)):
                continue

            def _compute() -> dict | None:
                return measure_features(audio, snap_sr, _NFFT, _HOP, _TOPDB, _WIN)

            t0 = time.perf_counter()
            try:
                features = await loop.run_in_executor(None, _compute)
            except Exception:
                log.exception("DSP error in auto-trigger worker")
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if features is None:
                log.info("auto-trigger: signal rejected (duration > 1.0s limit, or below 1kHz, or no impact detected)")
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
        finally:
            if slot is not None:
                slot.release()


# ---------------------------------------------------------------------------
# Watchdog — detects dead stream (§4d)
# ---------------------------------------------------------------------------
async def _watchdog() -> None:
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL_S)
        now = time.perf_counter()
        if _st.stream_state == "stream_open":
            if _st.stream_open_ts > 0 and (now - _st.stream_open_ts) >= 3.0:
                _st.stream_state = "callback_stalled"
                log.warning(
                    "WATCHDOG: stream open but no callbacks received within 3.0 s — "
                    "input may be dead, blocked, or awaiting permissions."
                )
        elif _st.last_callback_ts > 0:
            gap = now - _st.last_callback_ts
            if gap > DEAD_STREAM_S:
                if _st.stream_state not in ("capture_error", "uninitialized"):
                    _st.stream_state = "callback_stalled"
                log.warning(
                    "WATCHDOG: no audio callback for %.1f s — "
                    "stream may have died or mic disconnected. "
                    "Gap ≠ 'no impacts' — marked callback_stalled.",
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


def _init_buffer_pool(sr: int) -> None:
    """Ensure buffer pool is initialized with capacity for at least actual_sr."""
    global _pool
    req_samples = int(sr * (FF_S + RR_S))
    alloc_samples = max(DEFAULT_MAX_EVENT_SAMPLES, req_samples)
    if not hasattr(_pool, "_slots") or not _pool._slots or len(_pool._slots[0].data) < alloc_samples:
        _pool = EventBufferPool(count=DSP_QUEUE_MAXSIZE, max_samples=alloc_samples)
        log.info("buffer pool initialized: %d slots x %d samples (%d Hz capture)",
                 DSP_QUEUE_MAXSIZE, alloc_samples, sr)


async def _stream_task(
    device:       int | str | None,
    requested_sr: int,
    channels:     int,
) -> None:
    """Open stream, log provenance, run until cancelled or error."""
    loop = asyncio.get_event_loop()

    # Discover and resolve device deterministically avoiding virtual loopbacks
    try:
        resolved_idx, dev_info = resolve_audio_input_device(
            device, channels=channels, requested_sr=requested_sr
        )
    except Exception as exc:
        _st.stream_state = "capture_error"
        _st.error_message = str(exc)
        log.error("Failed to resolve audio input device: %s", exc)
        raise

    target_sr = dev_info.get("actual_supported_sr", requested_sr)

    # Discover actual SR before allocating ring buffer (§1a / G5)
    def _probe_sr() -> tuple[str, int, dict[str, Any]]:
        with sd.InputStream(
            device=resolved_idx,
            samplerate=target_sr,
            channels=channels,
            dtype="float32",
            blocksize=512,
        ) as probe:
            real_sr  = int(probe.samplerate)
            dev_idx  = probe.device[0] if isinstance(probe.device, (list, tuple)) else probe.device
            info = sd.query_devices(dev_idx)
            dev_name = info["name"]
            hapi_idx = info.get("hostapi", -1)
            hapi_name = "unknown"
            if hapi_idx >= 0:
                try:
                    hapi_info = sd.query_hostapis(hapi_idx)
                    hapi_name = hapi_info.get("name", "unknown")
                except Exception:
                    pass
            info["hostapi_name"] = hapi_name
        return dev_name, real_sr, info

    try:
        dev_name, actual_sr, probe_info = await loop.run_in_executor(None, _probe_sr)
    except Exception as exc:
        _st.stream_state = "capture_error"
        _st.error_message = str(exc)
        log.error("Failed to probe stream on device [%s]: %s", resolved_idx, exc)
        raise

    # G5 / §1a — log verified stream info
    log.info("STREAM OPEN  device=%r  requested_sr=%d  actual_sr=%d (verified from stream.samplerate)",
             dev_name, requested_sr, actual_sr)

    # Update shared state (safe — written before stream starts)
    now = time.perf_counter()
    _st.device_name = dev_name
    _st.device_identity = {
        "index": resolved_idx,
        "name": dev_name,
        "hostapi": probe_info.get("hostapi", dev_info.get("hostapi", -1)),
        "hostapi_name": probe_info.get("hostapi_name", dev_info.get("hostapi_name", "unknown")),
    }
    _st.requested_sr = requested_sr
    _st.actual_sr   = actual_sr
    _st.fallback_sr = dev_info.get("fallback_sr")
    _st.channels    = channels
    _st.stream_state = "stream_open"
    _st.stream_open_ts = now
    _st.last_callback_ts = 0.0  # Reset so watchdog tracks first callback arrival
    if actual_sr > MAX_SUPPORTED_SR:
        _st.stream_state = "capture_error"
        _st.error_message = (
            f"Actual stream sample rate {actual_sr} Hz exceeds maximum supported capture capacity "
            f"({MAX_SUPPORTED_SR} Hz)."
        )
        raise AudioDeviceResolutionError(_st.error_message)

    _init_ring_buffer(actual_sr)
    _init_buffer_pool(actual_sr)

    # Open live stream
    try:
        with sd.InputStream(
            device=resolved_idx,
            samplerate=actual_sr,
            channels=channels,
            dtype="float32",
            callback=_audio_callback,
            blocksize=512,
        ):
            log.info("STREAM LIVE  device=%r  actual_sr=%d", dev_name, actual_sr)
            # Block until cancelled or stalled
            while True:
                await asyncio.sleep(0.5)
                if _st.stream_state == "callback_stalled":
                    raise StreamStallError(
                        f"Audio callback stalled on device [{resolved_idx}] '{dev_name}' "
                        f"(no callbacks received for >{DEAD_STREAM_S:.1f}s or within initial window)."
                    )
    except asyncio.CancelledError:
        _st.stream_state = "uninitialized"
        log.info("STREAM STOP  device=%r", dev_name)
        raise
    except StreamStallError as exc:
        _st.error_message = str(exc)
        log.warning("Stream stall detected: %s", exc)
        raise
    except Exception as exc:
        _st.stream_state = "capture_error"
        _st.error_message = str(exc)
        log.error("Live InputStream error on device [%s]: %s", resolved_idx, exc)
        raise


async def stop_mic_loop() -> None:
    """Explicitly stop the mic loop session and wake workers cleanly."""
    global _active_session
    if _active_session is not None:
        _active_session.stop()
        _active_session.drain_and_release()
        _active_session = None
    _st.stream_state = "uninitialized"


async def run_mic_loop(
    manager:      "_ConnectionManager" | None = None,
    device:       int | str | None = None,
    requested_sr: int = 44_100,
    channels:     int = 1,
) -> None:
    """
    Top-level entry point — start all mic-loop tasks with session lifecycle.
    Call as an asyncio.Task from live_server.py startup lifespan.

    On reconnect: re-queries device info → §1d (no stale device reuse).
    """
    global _active_session
    session = MicSession(maxsize=DSP_QUEUE_MAXSIZE)
    _active_session = session

    # Launch DSP worker and watchdog
    dsp_task  = asyncio.create_task(_dsp_worker(manager, session=session),  name="dsp_worker")
    watch_task = asyncio.create_task(_watchdog(),                           name="watchdog")

    attempt = 0
    try:
        while True:
            attempt += 1
            _st.reconnect_attempts = attempt
            try:
                if manager is not None and attempt == 1:
                    try:
                        await manager.broadcast({
                            "type": "mic_state",
                            "active": True,
                            "stream_state": "stream_open",
                        })
                    except Exception:
                        pass
                await _stream_task(device, requested_sr, channels)
                break
            except asyncio.CancelledError:
                log.info("mic_loop cancelled — shutting down")
                raise
            except Exception as exc:
                log.error("stream error (attempt %d/%d): %s", attempt, MAX_RECONNECT_ATTEMPTS, exc)
                _st.error_message = str(exc)
                if manager is not None:
                    try:
                        await manager.broadcast({
                            "type": "mic_state",
                            "active": False,
                            "stream_state": _st.stream_state,
                            "reconnect_attempt": attempt,
                            "max_reconnect_attempts": MAX_RECONNECT_ATTEMPTS,
                            "error": str(exc),
                        })
                    except Exception:
                        pass

                if attempt >= MAX_RECONNECT_ATTEMPTS:
                    log.critical("max reconnect attempts reached — giving up")
                    _st.stream_state = "capture_error"
                    _st.error_message = f"Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached: {exc}"
                    if manager is not None:
                        try:
                            await manager.broadcast({
                                "type": "mic_state",
                                "active": False,
                                "stream_state": "capture_error",
                                "error": _st.error_message,
                            })
                        except Exception:
                            pass
                    return
                log.info("reconnecting in %.1f s …  (§1d: fresh device query on reconnect)", RECONNECT_DELAY_S)
                await asyncio.sleep(RECONNECT_DELAY_S)
    finally:
        session.stop()
        session.drain_and_release()
        _active_session = None
        dsp_task.cancel()
        watch_task.cancel()
        try:
            await dsp_task
        except asyncio.CancelledError:
            pass
        try:
            await watch_task
        except asyncio.CancelledError:
            pass
