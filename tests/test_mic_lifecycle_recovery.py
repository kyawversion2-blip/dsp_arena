#!/usr/bin/env python3
"""tests/test_mic_lifecycle_recovery.py — Unit tests for microphone lifecycle, recovery, and safety.

Step 4 tests (Mocked audio hardware — NO physical device opened):
1. Sample-rate probe success, unsupported sample-rate fallback, and aggregate device allowance.
2. Watchdog transitions: waiting for callbacks -> stalled capture vs digital silence.
3. Reconnect retry behavior, success recovery, and MAX_RECONNECT_ATTEMPTS exhaustion.
4. Repeated mic stop/start cycles with isolated sessions proving no stale worker leaks.
5. Bounded-queue saturation, backpressure overflow, and dropped-event accounting.
"""
from __future__ import annotations

import asyncio
import queue
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import audio_device_resolver
from audio_device_resolver import (
    AudioDeviceResolutionError,
    is_virtual_device,
    resolve_audio_input_device,
)
import mic_loop


@pytest.fixture(autouse=True)
def isolate_mic_state():
    mic_loop.reset_mic_state()
    yield
    mic_loop.reset_mic_state()


# ---------------------------------------------------------------------------
# 1. Sample-Rate Validation, Fallback & Aggregate Devices
# ---------------------------------------------------------------------------
MOCK_CORE_AUDIO = (
    {"name": "Core Audio", "default_input_device": 1, "default_output_device": 2},
)

MOCK_DEVS = [
    {
        "name": "BlackHole 16ch",
        "index": 0,
        "hostapi": 0,
        "max_input_channels": 16,
        "max_output_channels": 16,
        "default_samplerate": 384000.0,
    },
    {
        "name": "MacBook Air Microphone",
        "index": 1,
        "hostapi": 0,
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {
        "name": "Studio Aggregate Device",
        "index": 2,
        "hostapi": 0,
        "max_input_channels": 2,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
]


def test_aggregate_devices_not_rejected_as_virtual():
    """Legitimate macOS CoreAudio Aggregate Devices must not be rejected as virtual loopbacks."""
    assert is_virtual_device("Studio Aggregate Device") is False
    assert is_virtual_device("BlackHole 16ch") is True
    assert is_virtual_device("Loopback Audio") is True


@patch("sounddevice.query_hostapis", return_value=MOCK_CORE_AUDIO)
@patch("sounddevice.query_devices", return_value=MOCK_DEVS)
@patch("sounddevice.check_input_settings")
def test_sample_rate_validation_success(mock_check, mock_query, mock_apis):
    # When requested_sr is supported by check_input_settings
    mock_check.return_value = None
    idx, info = resolve_audio_input_device(1, channels=1, requested_sr=44100)
    assert idx == 1
    assert info["actual_supported_sr"] == 44100
    mock_check.assert_called_with(device=1, samplerate=44100, channels=1)


@patch("sounddevice.query_hostapis", return_value=MOCK_CORE_AUDIO)
@patch("sounddevice.query_devices", return_value=MOCK_DEVS)
@patch("sounddevice.check_input_settings")
def test_sample_rate_fallback_when_requested_rate_rejected(mock_check, mock_query, mock_apis):
    """When device rejects 96000 Hz, resolver must probe and negotiate fallback to 48000 Hz."""
    def side_effect(device, samplerate, channels):
        if samplerate == 96000:
            raise sd.PortAudioError("Invalid sample rate [PaErrorCode -9997]")
        elif samplerate == 48000:
            return None  # 48 kHz supported
        raise sd.PortAudioError("Invalid sample rate")

    mock_check.side_effect = side_effect
    idx, info = resolve_audio_input_device(1, channels=1, requested_sr=96000)
    assert idx == 1
    assert info["actual_supported_sr"] == 48000
    assert info.get("fallback_sr") == 48000


@patch("sounddevice.query_hostapis", return_value=MOCK_CORE_AUDIO)
@patch("sounddevice.query_devices", return_value=MOCK_DEVS)
@patch("sounddevice.check_input_settings", side_effect=sd.PortAudioError("All rates invalid"))
def test_sample_rate_rejection_raises_error(mock_check, mock_query, mock_apis):
    """When device rejects requested rate and all fallback rates, raise AudioDeviceResolutionError."""
    with pytest.raises(AudioDeviceResolutionError, match=r"(does not support requested sample rate|exceeds maximum supported capacity)"):
        resolve_audio_input_device(1, channels=1, requested_sr=192000)


# ---------------------------------------------------------------------------
# 2. Watchdog Transitions: Waiting vs Silence vs Stalled
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_watchdog_detects_stall_when_callbacks_never_arrive():
    """If stream opens but no callbacks arrive within 3.0s, watchdog marks callback_stalled."""
    mic_loop._st.stream_state = "stream_open"
    # Stream opened 4 seconds ago, but last_callback_ts is 0.0
    mic_loop._st.stream_open_ts = time.perf_counter() - 4.0
    mic_loop._st.last_callback_ts = 0.0

    # Run one watchdog inspection cycle
    with patch("mic_loop.WATCHDOG_INTERVAL_S", 0.01):
        watchdog_task = asyncio.create_task(mic_loop._watchdog())
        await asyncio.sleep(0.05)
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    assert mic_loop._st.stream_state == "callback_stalled"


@pytest.mark.asyncio
async def test_watchdog_detects_stall_when_running_stream_dies():
    """If callbacks were running but stop arriving for > DEAD_STREAM_S, watchdog marks callback_stalled."""
    mic_loop._st.stream_state = "audio_received"
    mic_loop._st.stream_open_ts = time.perf_counter() - 20.0
    # Last callback arrived 15s ago (> 10s DEAD_STREAM_S)
    mic_loop._st.last_callback_ts = time.perf_counter() - 15.0

    with patch("mic_loop.WATCHDOG_INTERVAL_S", 0.01):
        watchdog_task = asyncio.create_task(mic_loop._watchdog())
        await asyncio.sleep(0.05)
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    assert mic_loop._st.stream_state == "callback_stalled"


def test_callback_silence_state_distinction():
    """Genuine silence is ONLY set when audio callbacks are running and RMS < threshold."""
    mic_loop._st.buf = np.zeros(4800, dtype=np.float32)
    mic_loop._st.buf_len = 4800
    mic_loop._st.actual_sr = 48000
    mic_loop._st.stream_state = "audio_received"
    mic_loop._st.last_active_audio_ts = time.perf_counter() - 1.0

    # Audio callback with zeros (RMS == 0.0 < SILENCE_RMS_THRESHOLD)
    zero_block = np.zeros((512, 1), dtype=np.float32)
    mic_loop._audio_callback(zero_block, 512, None, None)

    assert mic_loop._st.stream_state == "silence"


# ---------------------------------------------------------------------------
# 3. Reconnect Retry and Exhaustion
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_mic_loop_reconnect_retry_and_exhaustion():
    """Stream failures trigger reconnects up to MAX_RECONNECT_ATTEMPTS, then set capture_error."""
    mock_manager = MagicMock()

    with patch("mic_loop.RECONNECT_DELAY_S", 0.01), \
         patch("mic_loop.MAX_RECONNECT_ATTEMPTS", 3), \
         patch("mic_loop._stream_task", side_effect=RuntimeError("Mock PortAudio crash")):

        await mic_loop.run_mic_loop(mock_manager, device=1)

        diag = mic_loop.get_mic_diagnostics()
        assert diag["stream_state"] == "capture_error"
        assert "Max reconnect attempts (3) reached" in str(diag["error_message"])


# ---------------------------------------------------------------------------
# 4. Repeated Mic Stop/Start Lifecycle (No Zombie Workers)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_repeated_start_stop_cycles_no_stale_workers():
    """Starting, stopping, and restarting mic loop must cleanly retire old workers."""
    mock_manager = MagicMock()
    mock_emit = AsyncMock(return_value="event-uuid")

    # 1. Run 1: Create session 1
    session1 = mic_loop.MicSession()
    worker1_task = asyncio.create_task(mic_loop._dsp_worker(mock_manager, session=session1))

    # Stop session 1
    session1.stop()
    await asyncio.sleep(0.05)
    assert worker1_task.done() or worker1_task.cancelled()

    # 2. Run 2: Create session 2
    session2 = mic_loop.MicSession()
    worker2_task = asyncio.create_task(mic_loop._dsp_worker(mock_manager, session=session2))

    # Prepare event in session 2
    slot = mic_loop._pool.acquire_nowait()
    assert slot is not None
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    synthetic_impact = (0.5 * np.sin(2 * np.pi * 2500 * t) * np.exp(-20 * t)).astype(np.float32)
    slot.data[:sr] = synthetic_impact
    slot.length = sr
    slot.sr = sr
    slot.device = "Mic Run 2"
    slot.ts = time.perf_counter()

    session2.queue.put_nowait(slot)
    session2.queue.put_nowait(mic_loop._SHUTDOWN_SENTINEL)

    with patch("mic_loop._store.emit_event", mock_emit):
        await worker2_task

    # Event was processed by run 2
    assert mock_emit.called
    assert mock_emit.call_args[0][0]["Recording_Device"] == "Mic Run 2"


# ---------------------------------------------------------------------------
# 5. Bounded Queue Saturation & Dropped Event Accounting
# ---------------------------------------------------------------------------
def test_bounded_queue_saturation_increments_dropped_events():
    """When buffer pool is exhausted, callback drops incoming events and records count."""
    sr = 44100
    mic_loop._init_ring_buffer(sr)
    mic_loop._st.actual_sr = sr

    # Exhaust all slots in the pool
    acquired = []
    while True:
        slot = mic_loop._pool.acquire_nowait()
        if slot is None:
            break
        acquired.append(slot)

    assert len(acquired) == mic_loop.DSP_QUEUE_MAXSIZE
    assert mic_loop._pool.acquire_nowait() is None

    # Now simulate a trigger in _audio_callback when pool is empty
    mic_loop._st.state = "COLLECTING"
    mic_loop._st.remaining_post = 0  # Trigger completion
    mic_loop._st.dropped_events = 0

    block = np.zeros((512, 1), dtype=np.float32)
    mic_loop._audio_callback(block, 512, None, None)

    # State returns to LISTENING and dropped_events is incremented
    assert mic_loop._st.state == "LISTENING"
    assert mic_loop._st.dropped_events == 1

    diag = mic_loop.get_mic_diagnostics()
    assert diag["dropped_events"] == 1

    # Release slots back to pool
    for s in acquired:
        s.release()

    # Pool is replenished
    new_slot = mic_loop._pool.acquire_nowait()
    assert new_slot is not None
    new_slot.release()


# ---------------------------------------------------------------------------
# 6. P1-3 Remediation Tests: Callback-Stall Detection & Lifecycle Recovery
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_task_raises_stream_stall_error_on_stalled_state():
    """Verify that _stream_task raises StreamStallError and closes InputStream when stalled."""
    mock_stream = MagicMock()
    mock_stream.__enter__.return_value = mock_stream

    with patch("sounddevice.InputStream", return_value=mock_stream), \
         patch("mic_loop.resolve_audio_input_device", return_value=(1, {"name": "Test Mic", "actual_supported_sr": 44100})), \
         patch("mic_loop.sd.query_devices", return_value={"name": "Test Mic", "hostapi": 0}), \
         patch("mic_loop.sd.query_hostapis", return_value={"name": "Core Audio"}):

        # Trigger stall after 100ms
        async def _trigger_stall():
            await asyncio.sleep(0.05)
            mic_loop._st.stream_state = "callback_stalled"

        stall_task = asyncio.create_task(_trigger_stall())
        with pytest.raises(mic_loop.StreamStallError, match="Audio callback stalled"):
            await mic_loop._stream_task(device=1, requested_sr=44100, channels=1)

        await stall_task
        # Verify InputStream was exited cleanly
        assert mock_stream.__exit__.called


@pytest.mark.asyncio
async def test_callback_stall_followed_by_successful_recovery():
    """Verify callback stall triggers single-owner reconnect and successfully resumes streaming."""
    mock_manager = MagicMock()
    mock_manager.broadcast = AsyncMock()

    call_count = 0

    async def _mock_stream_run(dev, sr, ch):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            mic_loop._st.stream_state = "callback_stalled"
            raise mic_loop.StreamStallError("Simulated first run stall")
        else:
            # Second run succeeds
            mic_loop._st.stream_state = "audio_received"
            mic_loop._st.callback_count += 5
            try:
                while True:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                raise

    with patch("mic_loop.RECONNECT_DELAY_S", 0.01), \
         patch("mic_loop.MAX_RECONNECT_ATTEMPTS", 5), \
         patch("mic_loop._stream_task", side_effect=_mock_stream_run):

        run_task = asyncio.create_task(mic_loop.run_mic_loop(mock_manager, device=1))
        # Allow enough time for stall, reconnect delay (0.01s), and second run
        await asyncio.sleep(0.1)
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

        assert call_count >= 2
        # Verify manager received stall broadcast during recovery
        broadcast_calls = [c[0][0] for c in mock_manager.broadcast.call_args_list]
        stall_broadcast = next((b for b in broadcast_calls if b.get("stream_state") == "callback_stalled"), None)
        assert stall_broadcast is not None
        assert stall_broadcast["reconnect_attempt"] == 1


@pytest.mark.asyncio
async def test_callback_stall_retry_exhaustion_consistent_state():
    """Verify that persistent callback stalls reach retry exhaustion consistently across diagnostics and WS."""
    mock_manager = MagicMock()
    mock_manager.broadcast = AsyncMock()

    with patch("mic_loop.RECONNECT_DELAY_S", 0.01), \
         patch("mic_loop.MAX_RECONNECT_ATTEMPTS", 3), \
         patch("mic_loop._stream_task", side_effect=mic_loop.StreamStallError("Persistent callback stall")):

        await mic_loop.run_mic_loop(mock_manager, device=1)

        diag = mic_loop.get_mic_diagnostics()
        assert diag["stream_state"] == "capture_error"
        assert diag["reconnect_attempts"] == 3
        assert "Max reconnect attempts (3) reached" in str(diag["error_message"])

        # Check WebSocket broadcast received exhaustion message
        broadcast_calls = [c[0][0] for c in mock_manager.broadcast.call_args_list]
        final_broadcast = broadcast_calls[-1]
        assert final_broadcast["type"] == "mic_state"
        assert final_broadcast["active"] is False
        assert final_broadcast["stream_state"] == "capture_error"
        assert "Max reconnect attempts (3) reached" in final_broadcast["error"]


@pytest.mark.asyncio
async def test_stop_during_recovery_cleans_up_without_leak():
    """Verify stopping during reconnect delay cleans up session and leaves pool at full capacity."""
    mock_manager = MagicMock()
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE

    with patch("mic_loop.RECONNECT_DELAY_S", 10.0), \
         patch("mic_loop._stream_task", side_effect=mic_loop.StreamStallError("Stream stall")):

        run_task = asyncio.create_task(mic_loop.run_mic_loop(mock_manager, device=1))
        # Wait until _stream_task fails and enters sleep(RECONNECT_DELAY_S)
        await asyncio.sleep(0.05)
        assert not run_task.done()

        # Stop via stop_mic_loop
        await mic_loop.stop_mic_loop()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

        assert mic_loop._active_session is None
        assert mic_loop._st.stream_state == "uninitialized"
        assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE


# ---------------------------------------------------------------------------
# 7. P2 Fix 1 & Fix 2 Tests: Stop-Window Leak & SR Negotiation Surfacing
# ---------------------------------------------------------------------------
def test_audio_callback_drops_event_and_releases_slot_when_session_inactive():
    """Verify that when no active session exists, callback drops event and releases slot back to pool."""
    sr = 44100
    mic_loop._init_ring_buffer(sr)
    mic_loop._st.actual_sr = sr
    mic_loop._active_session = None

    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE
    assert mic_loop._trig_q.empty()

    # Trigger completion in audio callback
    mic_loop._st.state = "COLLECTING"
    mic_loop._st.remaining_post = 0
    mic_loop._st.dropped_events = 0

    block = np.zeros((512, 1), dtype=np.float32)
    mic_loop._audio_callback(block, 512, None, None)

    # State returns to LISTENING, dropped_events is incremented
    assert mic_loop._st.state == "LISTENING"
    assert mic_loop._st.dropped_events == 1
    # Fallback queue MUST NOT receive any slots
    assert mic_loop._trig_q.empty()
    # Pool MUST retain 100% capacity (8/8) — no slot leaked
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE


def test_callback_racing_with_session_stop_no_leak_and_empty_fallback():
    """Verify that if session stops while callback is collecting, slot is returned to pool without leaking."""
    sr = 44100
    mic_loop._init_ring_buffer(sr)
    mic_loop._st.actual_sr = sr

    session = mic_loop.MicSession(maxsize=mic_loop.DSP_QUEUE_MAXSIZE)
    mic_loop._active_session = session

    mic_loop._st.state = "COLLECTING"
    mic_loop._st.remaining_post = 0
    mic_loop._st.dropped_events = 0

    # Simulate race: session is stopped right before callback attempts to enqueue
    session.stop()
    assert not session.is_active

    block = np.zeros((512, 1), dtype=np.float32)
    mic_loop._audio_callback(block, 512, None, None)

    assert mic_loop._st.state == "LISTENING"
    assert mic_loop._st.dropped_events == 1
    # Slot was released back to pool
    assert mic_loop._trig_q.empty()
    assert session.queue.qsize() == 1  # only contains _SHUTDOWN_SENTINEL
    session.drain_and_release()
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE


@pytest.mark.asyncio
async def test_diagnostics_and_rest_surface_sample_rate_negotiation():
    """Verify requested_sr and fallback_sr are accurately exposed only when negotiation occurred."""
    import json
    import live_server

    # Scenario A: Fallback occurred (requested 96000, negotiated 48000)
    mic_loop._st.requested_sr = 96000
    mic_loop._st.actual_sr = 48000
    mic_loop._st.fallback_sr = 48000
    mic_loop._st.device_name = "Mock Physical Mic"

    diag = mic_loop.get_mic_diagnostics()
    assert diag["requested_sr"] == 96000
    assert diag["actual_sr"] == 48000
    assert diag["fallback_sr"] == 48000

    resp = await live_server.get_mic_status()
    body = json.loads(resp.body)
    assert body["requested_sr"] == 96000
    assert body["actual_sr"] == 48000
    assert body["fallback_sr"] == 48000

    # Scenario B: No fallback occurred (requested 44100, accepted 44100)
    mic_loop._st.requested_sr = 44100
    mic_loop._st.actual_sr = 44100
    mic_loop._st.fallback_sr = None

    diag_no_fb = mic_loop.get_mic_diagnostics()
    assert diag_no_fb["requested_sr"] == 44100
    assert diag_no_fb["actual_sr"] == 44100
    assert "fallback_sr" not in diag_no_fb  # MUST NOT report fallback unless it occurred

    resp_no_fb = await live_server.get_mic_status()
    body_no_fb = json.loads(resp_no_fb.body)
    assert body_no_fb["requested_sr"] == 44100
    assert body_no_fb["actual_sr"] == 44100
    assert "fallback_sr" not in body_no_fb


