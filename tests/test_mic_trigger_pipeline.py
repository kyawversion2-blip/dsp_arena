#!/usr/bin/env python3
"""tests/test_mic_trigger_pipeline.py — Unit tests for the complete mic trigger & DSP pipeline.

Step 3 tests (Mocked audio input — NO hardware opened):
- Onset detection and state transition: LISTENING -> COLLECTING
- Completion of post-trigger window (remaining_post <= 0) and transition back to LISTENING
- Snapshot handoff into trigger queue
- _extract_window wrap-around handling (no wrap, start wrap, end wrap)
- Execution of measure_features and emit_event
- Rejection of malformed/steady-state/sub-threshold input without producing false events
"""
from __future__ import annotations

import asyncio
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
import mic_loop
from mic_loop import (
    FF_S,
    ONSET_RMS_THRESHOLD,
    REFRACTORY_S,
    RR_S,
    _audio_callback,
    _dsp_worker,
    _extract_window,
    _init_ring_buffer,
    reset_mic_state,
)


@pytest.fixture(autouse=True)
def isolate_mic_state():
    reset_mic_state()
    yield
    reset_mic_state()


def test_extract_window_no_wrap():
    sr = 1000
    # Buffer length for sr=1000: max(4.0, 3.0)*1000 = 4000 samples
    buf_len = 4000
    buf = np.arange(buf_len, dtype=np.float32)

    ff_samps = int(sr * FF_S)  # 500
    rr_samps = int(sr * RR_S)  # 500
    total = ff_samps + rr_samps  # 1000

    # Middle trigger: trigger at 2000 -> window [1500, 2500]
    trigger_pos = 2000
    extracted = _extract_window(buf, trigger_pos, write_pos=2500, sr=sr)
    assert len(extracted) == total
    expected = buf[1500:2500].astype(np.float64)
    np.testing.assert_array_equal(extracted, expected)


def test_extract_window_start_wrap():
    """When trigger_pos - ff_samps < 0, the pre-trigger context wraps to buffer end."""
    sr = 1000
    buf_len = 4000
    buf = np.arange(buf_len, dtype=np.float32)

    ff_samps = int(sr * FF_S)  # 500
    rr_samps = int(sr * RR_S)  # 500
    total = ff_samps + rr_samps  # 1000

    # trigger at 200: start = (200 - 500) % 4000 = 3700.
    # Window spans buf[3700:4000] (300 samples) + buf[0:700] (700 samples)
    trigger_pos = 200
    extracted = _extract_window(buf, trigger_pos, write_pos=700, sr=sr)
    assert len(extracted) == total
    expected = np.concatenate([buf[3700:], buf[:700]]).astype(np.float64)
    np.testing.assert_array_equal(extracted, expected)


def test_extract_window_end_wrap():
    """When trigger_pos + rr_samps > buf_len, post-trigger context wraps to buffer start."""
    sr = 1000
    buf_len = 4000
    buf = np.arange(buf_len, dtype=np.float32)

    ff_samps = int(sr * FF_S)  # 500
    rr_samps = int(sr * RR_S)  # 500
    total = ff_samps + rr_samps  # 1000

    # trigger at 3800: start = 3300.
    # Window spans buf[3300:4000] (700 samples) + buf[0:300] (300 samples)
    trigger_pos = 3800
    extracted = _extract_window(buf, trigger_pos, write_pos=300, sr=sr)
    assert len(extracted) == total
    expected = np.concatenate([buf[3300:], buf[:300]]).astype(np.float64)
    np.testing.assert_array_equal(extracted, expected)


@pytest.mark.asyncio
async def test_full_trigger_pipeline_valid_transient_emits_event():
    """Verify end-to-end pipeline: onset -> collecting -> snapshot handoff -> dsp -> event emit."""
    sr = 44100
    _init_ring_buffer(sr)
    mic_loop._st.actual_sr = sr
    mic_loop._st.device_name = "Mock Physical Mic"

    session = mic_loop.MicSession(maxsize=mic_loop.DSP_QUEUE_MAXSIZE)
    mic_loop._active_session = session

    # 1. Generate synthetic damped acoustic impact transient
    t = np.linspace(0, 1.0, sr, endpoint=False)
    synthetic_impact = 0.5 * np.sin(2 * np.pi * 2500 * t) * np.exp(-20 * t)
    synthetic_impact = synthetic_impact.astype(np.float32)

    # 2. Feed silence first: state must be LISTENING
    silence_block = np.zeros((512, 1), dtype=np.float32)
    _audio_callback(silence_block, 512, None, None)
    assert mic_loop._st.state == "LISTENING"
    assert session.queue.empty()
    assert mic_loop._trig_q.empty()

    # 3. Feed the beginning of the transient (high energy onset)
    block_size = 512
    onset_block = synthetic_impact[:block_size].reshape(-1, 1)
    _audio_callback(onset_block, block_size, None, None)

    # Check state transitioned to COLLECTING
    assert mic_loop._st.state == "COLLECTING"
    assert mic_loop._st.remaining_post > 0
    assert session.queue.empty()  # still collecting, not yet in queue
    assert mic_loop._trig_q.empty()

    # 4. Feed subsequent blocks until post-trigger window is satisfied
    cursor = block_size
    while mic_loop._st.state == "COLLECTING":
        if cursor + block_size <= len(synthetic_impact):
            blk = synthetic_impact[cursor : cursor + block_size].reshape(-1, 1)
            cursor += block_size
        else:
            blk = np.zeros((block_size, 1), dtype=np.float32)
        _audio_callback(blk, block_size, None, None)

    # Check state transitioned back to LISTENING and snapshot handed off to session queue
    assert mic_loop._st.state == "LISTENING"
    assert not session.queue.empty()
    assert mic_loop._trig_q.empty()  # Production fallback queue MUST remain untouched

    # 5. Run DSP worker on the session queue and verify event emission
    mock_manager = MagicMock()
    mock_emit = AsyncMock(return_value="event-test-uuid-1234")

    # Append sentinel to queue so worker processes the item and cleanly terminates
    session.queue.put_nowait(mic_loop._SHUTDOWN_SENTINEL)

    with patch("mic_loop._store.emit_event", mock_emit):
        await _dsp_worker(mock_manager, session=session)

    # Assert that emit_event was called exactly once with valid physical features
    assert mock_emit.called
    assert mock_emit.call_count == 1
    call_args, call_kwargs = mock_emit.call_args
    payload = call_args[0]

    assert payload["Recording_Device"] == "Mock Physical Mic"
    assert payload["Recording_Actual_SR_Hz"] == 44100
    assert payload["impact"] > 0
    assert payload["duration_s"] < 1.0
    # Peak frequency should be near 2500 Hz
    assert abs(payload["impact_freq_hz"] - 2500.0) < 50.0
    assert call_kwargs.get("source") == "mic_auto_trigger"


@pytest.mark.asyncio
async def test_trigger_pipeline_malformed_nan_input_no_event_emitted():
    """Verify that malformed/corrupted non-finite input is rejected and does not emit a false event."""
    sr = 44100
    _init_ring_buffer(sr)
    mic_loop._st.actual_sr = sr
    mic_loop._st.device_name = "Mock Physical Mic"

    # Simulate an acquired slot containing non-finite/NaN data
    slot = mic_loop._pool.acquire_nowait()
    assert slot is not None
    slot.data[:1000] = np.nan
    slot.length = 1000
    slot.sr = sr
    slot.device = "Mock Physical Mic"
    slot.ts = time.perf_counter()

    mic_loop._trig_q.put_nowait(slot)
    mic_loop._trig_q.put_nowait(mic_loop._SHUTDOWN_SENTINEL)

    mock_manager = MagicMock()
    mock_emit = AsyncMock()

    with patch("mic_loop._store.emit_event", mock_emit):
        await _dsp_worker(mock_manager)

    # Malformed NaN input must not emit an event
    assert mock_emit.call_count == 0


@pytest.mark.asyncio
async def test_trigger_pipeline_incomplete_short_input_no_event_emitted():
    """Verify that incomplete/truncated input (<50ms) is rejected without emitting a false event."""
    sr = 44100
    _init_ring_buffer(sr)

    # Incomplete slot of only 10 samples (< 50ms)
    slot = mic_loop._pool.acquire_nowait()
    assert slot is not None
    slot.data[:10] = 0.5
    slot.length = 10
    slot.sr = sr
    slot.device = "Mock Physical Mic"
    slot.ts = time.perf_counter()

    mic_loop._trig_q.put_nowait(slot)
    mic_loop._trig_q.put_nowait(mic_loop._SHUTDOWN_SENTINEL)

    mock_manager = MagicMock()
    mock_emit = AsyncMock()

    with patch("mic_loop._store.emit_event", mock_emit):
        await _dsp_worker(mock_manager)

    # Incomplete input must not emit an event
    assert mock_emit.call_count == 0


def test_trigger_pipeline_subthreshold_noise_no_trigger():
    """Verify that low-amplitude background noise never enters COLLECTING or trigger queue."""
    sr = 44100
    _init_ring_buffer(sr)
    mic_loop._st.actual_sr = sr

    # Noise well below ONSET_RMS_THRESHOLD (0.01)
    noise = (np.random.randn(512, 1) * 0.001).astype(np.float32)

    for _ in range(10):
        _audio_callback(noise, 512, None, None)

    assert mic_loop._st.state == "LISTENING"
    assert mic_loop._trig_q.empty()
    diag = mic_loop.get_mic_diagnostics()
    assert diag["stream_state"] in ("audio_received", "silence")


# ---------------------------------------------------------------------------
# P1-1 & P1-2 Remediation Tests: Capacity & Slot Ownership
# ---------------------------------------------------------------------------
def test_event_buffer_capacity_at_96khz_fits_without_allocation():
    """Verify that 96 kHz boundary rate capture fits pre-allocated slot without overflow."""
    sr = 96000
    mic_loop._init_ring_buffer(sr)
    mic_loop._init_buffer_pool(sr)
    mic_loop._st.actual_sr = sr

    assert mic_loop._pool.total_count == mic_loop.DSP_QUEUE_MAXSIZE
    slot = mic_loop._pool.acquire_nowait()
    assert slot is not None
    required_samples = int(sr * (mic_loop.FF_S + mic_loop.RR_S))
    assert len(slot.data) >= required_samples
    assert required_samples == 96000

    # Simulate trigger completion in audio callback
    mic_loop._st.state = "COLLECTING"
    mic_loop._st.remaining_post = 0
    mic_loop._st.trigger_pos = 96000
    block = np.zeros((512, 1), dtype=np.float32)

    # Release test slot before callback acquires
    slot.release()
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE

    session = mic_loop.MicSession(maxsize=mic_loop.DSP_QUEUE_MAXSIZE)
    mic_loop._active_session = session

    mic_loop._audio_callback(block, 512, None, None)

    assert mic_loop._st.state == "LISTENING"
    assert mic_loop._trig_q.empty()
    assert not session.queue.empty()
    queued_slot = session.queue.get_nowait()
    assert isinstance(queued_slot, mic_loop.EventBufferSlot)
    assert queued_slot.length == 96000
    assert queued_slot.sr == 96000

    queued_slot.release()
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE


def test_event_buffer_capacity_rejection_prevents_callback_crash():
    """If required sample count exceeds slot capacity, drop safely and return slot to pool."""
    sr = 44100
    mic_loop._init_ring_buffer(sr)
    mic_loop._st.actual_sr = sr

    # Create an artificially constrained slot that cannot hold 44,100 samples
    constrained_pool = mic_loop.EventBufferPool(count=1, max_samples=1000)
    original_pool = mic_loop._pool
    mic_loop._pool = constrained_pool

    try:
        assert constrained_pool.available_count == 1
        mic_loop._st.state = "COLLECTING"
        mic_loop._st.remaining_post = 0
        mic_loop._st.dropped_events = 0

        block = np.zeros((512, 1), dtype=np.float32)
        # Callback must execute without raising any exception
        mic_loop._audio_callback(block, 512, None, None)

        assert mic_loop._st.state == "LISTENING"
        assert mic_loop._st.dropped_events == 1
        # Crucial P1-2 assertion: slot was NOT leaked despite capacity mismatch
        assert constrained_pool.available_count == 1
    finally:
        mic_loop._pool = original_pool


def test_slot_double_release_protection():
    """Verify that calling slot.release() multiple times does not corrupt pool capacity."""
    pool = mic_loop.EventBufferPool(count=4, max_samples=1000)
    assert pool.available_count == 4

    slot1 = pool.acquire_nowait()
    assert slot1 is not None
    assert pool.available_count == 3

    # First release
    slot1.release()
    assert pool.available_count == 4

    # Second release — must be guarded and no-op
    slot1.release()
    assert pool.available_count == 4

    # Acquire all 4 slots cleanly
    slots = [pool.acquire_nowait() for _ in range(4)]
    assert all(s is not None for s in slots)
    assert pool.acquire_nowait() is None

    for s in slots:
        s.release()
    assert pool.available_count == 4


def test_slot_ownership_returned_on_enqueue_failure():
    """When enqueue fails (e.g. queue full or error), slot is returned to pool without leaking."""
    sr = 44100
    mic_loop._init_ring_buffer(sr)
    mic_loop._st.actual_sr = sr

    session = mic_loop.MicSession(maxsize=1)
    mic_loop._active_session = session

    try:
        # Fill the session queue with a dummy item so put_nowait raises queue.Full
        session.queue.put_nowait("dummy_item")

        assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE
        mic_loop._st.state = "COLLECTING"
        mic_loop._st.remaining_post = 0
        mic_loop._st.dropped_events = 0

        block = np.zeros((512, 1), dtype=np.float32)
        mic_loop._audio_callback(block, 512, None, None)

        assert mic_loop._st.state == "LISTENING"
        assert mic_loop._st.dropped_events == 1
        # The acquired slot must have been released in the finally block
        assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE
    finally:
        session.stop()
        mic_loop._active_session = None


@pytest.mark.asyncio
async def test_worker_exception_releases_slot():
    """When DSP feature extraction throws an unexpected error, worker releases slot in finally block."""
    sr = 44100
    mic_loop._init_ring_buffer(sr)
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE

    slot = mic_loop._pool.acquire_nowait()
    assert slot is not None
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE - 1

    slot.data[:5000] = 0.5
    slot.length = 5000
    slot.sr = sr
    slot.device = "Test Device"
    slot.ts = time.perf_counter()

    mic_loop._trig_q.put_nowait(slot)
    mic_loop._trig_q.put_nowait(mic_loop._SHUTDOWN_SENTINEL)

    mock_manager = MagicMock()

    with patch("mic_loop.measure_features", side_effect=RuntimeError("DSP internal crash")):
        await _dsp_worker(mock_manager)

    # Worker must have released the slot in its finally: block
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE


def test_session_drain_and_release_on_shutdown():
    """Verify that stopping/draining a session returns all unconsumed slots to the pool."""
    session = mic_loop.MicSession(maxsize=mic_loop.DSP_QUEUE_MAXSIZE)
    slots = [mic_loop._pool.acquire_nowait() for _ in range(3)]
    for s in slots:
        session.queue.put_nowait(s)

    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE - 3

    drained = session.drain_and_release()
    assert drained == 3
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE


# ---------------------------------------------------------------------------
# P2 Fix 3 Tests: Bounded-Session End-to-End Coverage
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bounded_session_end_to_end_pipeline_no_fallback_queue():
    """Drive _audio_callback -> active MicSession bounded queue -> _dsp_worker -> emit_event.

    Assert:
    - Exactly 1 emitted event.
    - Correct provenance (Recording_Device, Recording_Actual_SR_Hz).
    - No fallback-queue use (_trig_q remains empty).
    - All 8 pool slots returned after processing.
    """
    sr = 48000
    mic_loop._init_ring_buffer(sr)
    mic_loop._init_buffer_pool(sr)
    mic_loop._st.actual_sr = sr
    mic_loop._st.device_name = "Physical Test Mic"

    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE
    assert mic_loop._trig_q.empty()

    session = mic_loop.MicSession(maxsize=mic_loop.DSP_QUEUE_MAXSIZE)
    mic_loop._active_session = session

    mock_manager = MagicMock()
    mock_emit = AsyncMock(return_value="event-test-bounded-uuid")

    worker_task = asyncio.create_task(_dsp_worker(mock_manager, session=session))

    # Generate synthetic acoustic transient
    t = np.linspace(0, 1.0, sr, endpoint=False)
    transient = (0.5 * np.sin(2 * np.pi * 2500 * t) * np.exp(-20 * t)).astype(np.float32)

    block_size = 512
    # 1. Feed onset block
    _audio_callback(transient[:block_size].reshape(-1, 1), block_size, None, None)
    assert mic_loop._st.state == "COLLECTING"

    # 2. Feed remaining blocks to satisfy post-trigger window
    cursor = block_size
    while mic_loop._st.state == "COLLECTING":
        if cursor + block_size <= len(transient):
            blk = transient[cursor : cursor + block_size].reshape(-1, 1)
            cursor += block_size
        else:
            blk = np.zeros((block_size, 1), dtype=np.float32)
        _audio_callback(blk, block_size, None, None)

    assert mic_loop._st.state == "LISTENING"

    # Fallback queue MUST NOT be touched
    assert mic_loop._trig_q.empty()

    # Shutdown worker cleanly
    session.queue.put_nowait(mic_loop._SHUTDOWN_SENTINEL)
    with patch("mic_loop._store.emit_event", mock_emit):
        await worker_task

    # Exactly 1 event emitted
    assert mock_emit.call_count == 1
    call_payload = mock_emit.call_args[0][0]
    assert call_payload["Recording_Device"] == "Physical Test Mic"
    assert call_payload["Recording_Actual_SR_Hz"] == 48000
    assert call_payload["impact"] > 0

    # Fallback queue STILL completely empty
    assert mic_loop._trig_q.empty()

    # All 8 slots returned after feature extraction
    assert mic_loop._pool.available_count == mic_loop.DSP_QUEUE_MAXSIZE


def test_requested_sr_above_96khz_cannot_reach_callback_with_oversized_buffer():
    """Verify that requesting >96 kHz negotiates a supported rate <=96 kHz or fails safely."""
    mock_apis = [{"name": "Core Audio"}]
    mock_devs = [
        {"name": "Studio Interface", "max_input_channels": 2, "default_samplerate": 48000.0, "hostapi": 0}
    ]

    with patch("sounddevice.query_hostapis", return_value=mock_apis), \
         patch("sounddevice.query_devices", return_value=mock_devs):

        # Scenario A: Device accepts fallback (e.g. 48 kHz default)
        def _check_settings(device, samplerate, channels):
            if samplerate > 96000:
                raise sd.PortAudioError("Rate > 96kHz rejected by policy")
            return True

        with patch("sounddevice.check_input_settings", side_effect=_check_settings):
            idx, dev_info = audio_device_resolver.resolve_audio_input_device(
                0, channels=1, requested_sr=192000
            )
            assert dev_info["actual_supported_sr"] <= mic_loop.MAX_SUPPORTED_SR
            assert dev_info["fallback_sr"] == 48000

        # Scenario B: Device rejects all rates <= 96 kHz -> Must raise before opening stream
        with patch("sounddevice.check_input_settings", side_effect=sd.PortAudioError("Invalid")):
            with pytest.raises(audio_device_resolver.AudioDeviceResolutionError, match="exceeds maximum supported capacity"):
                audio_device_resolver.resolve_audio_input_device(0, channels=1, requested_sr=192000)

