#!/usr/bin/env python3
"""tests/test_mic_device_selection.py — Unit tests for microphone device resolution and diagnostics.

Tests (using mocks — NO physical microphone opened, NO production server started):
1. Virtual device identification.
2. Explicit device index and numeric string resolution.
3. Device name exact and substring resolution.
4. Automatic avoidance of BlackHole / virtual loopback on default.
5. Channel count capability validation.
6. Real-time audio callback diagnostics, silence, onset signal, and clipping.
7. Stream startup failure and error state capture.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio_device_resolver import (
    AudioDeviceResolutionError,
    format_device_list,
    get_available_input_devices,
    is_virtual_device,
    resolve_audio_input_device,
)
import mic_loop


@pytest.fixture(autouse=True)
def isolate_mic_state():
    """Ensure strict test isolation: reset shared mic state before and after each test."""
    mic_loop.reset_mic_state()
    yield
    mic_loop.reset_mic_state()


MOCK_HOSTAPIS = (
    {"name": "Core Audio", "default_input_device": 0, "default_output_device": 2},
)

MOCK_DEVICES = [
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
        "name": "MacBook Air Speakers",
        "index": 2,
        "hostapi": 0,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
]


def test_is_virtual_device():
    assert is_virtual_device("BlackHole 16ch") is True
    assert is_virtual_device("Loopback Audio") is True
    assert is_virtual_device("Soundflower (2ch)") is True
    assert is_virtual_device("MacBook Air Microphone") is False
    assert is_virtual_device("External USB Microphone") is False


@patch("sounddevice.query_hostapis", return_value=MOCK_HOSTAPIS)
@patch("sounddevice.query_devices", return_value=MOCK_DEVICES)
def test_get_available_input_devices(mock_devices, mock_apis):
    inputs = get_available_input_devices()
    assert len(inputs) == 2  # Speakers excluded (0 input channels)
    names = [d["name"] for d in inputs]
    assert "BlackHole 16ch" in names
    assert "MacBook Air Microphone" in names


@patch("sounddevice.query_hostapis", return_value=MOCK_HOSTAPIS)
@patch("sounddevice.query_devices", return_value=MOCK_DEVICES)
def test_resolve_by_explicit_int_and_string(mock_devices, mock_apis):
    # Integer index
    idx, info = resolve_audio_input_device(1)
    assert idx == 1
    assert info["name"] == "MacBook Air Microphone"

    # Numeric string index
    idx2, info2 = resolve_audio_input_device("1")
    assert idx2 == 1
    assert info2["name"] == "MacBook Air Microphone"

    # Out-of-bounds integer
    with pytest.raises(AudioDeviceResolutionError, match=r"\[99\] not found"):
        resolve_audio_input_device(99)

    # Output-only device (Speakers = index 2)
    with pytest.raises(AudioDeviceResolutionError, match=r"\[2\] not found"):
        resolve_audio_input_device(2)


@patch("sounddevice.query_hostapis", return_value=MOCK_HOSTAPIS)
@patch("sounddevice.query_devices", return_value=MOCK_DEVICES)
def test_resolve_by_device_name(mock_devices, mock_apis):
    # Exact name match (case-insensitive)
    idx, info = resolve_audio_input_device("macbook air microphone")
    assert idx == 1

    # Substring match
    idx2, info2 = resolve_audio_input_device("MacBook")
    assert idx2 == 1

    # Unknown device
    with pytest.raises(AudioDeviceResolutionError, match="No input device matching 'Quantum Mic'"):
        resolve_audio_input_device("Quantum Mic")


@patch("sounddevice.default.device", (0, 2))
@patch("sounddevice.query_hostapis", return_value=MOCK_HOSTAPIS)
@patch("sounddevice.query_devices", return_value=MOCK_DEVICES)
def test_default_avoids_blackhole_and_selects_physical_mic(mock_devices, mock_apis):
    # Default is device 0 (BlackHole), but resolve_audio_input_device should auto-select physical mic
    idx, info = resolve_audio_input_device(None)
    assert idx == 1
    assert info["name"] == "MacBook Air Microphone"


@patch("sounddevice.default.device", (0, 2))
@patch("sounddevice.query_hostapis", return_value=MOCK_HOSTAPIS)
@patch("sounddevice.query_devices", return_value=[MOCK_DEVICES[0], MOCK_DEVICES[2]])
def test_default_with_only_virtual_devices_raises_error(mock_devices, mock_apis):
    # Only BlackHole and Speakers available -> must raise error, not silently stream silence
    with pytest.raises(AudioDeviceResolutionError, match="virtual loopback device"):
        resolve_audio_input_device(None)


@patch("sounddevice.query_hostapis", return_value=MOCK_HOSTAPIS)
@patch("sounddevice.query_devices", return_value=MOCK_DEVICES)
def test_channel_count_validation(mock_devices, mock_apis):
    # MacBook Air Mic has 1 input channel
    idx, info = resolve_audio_input_device(1, channels=1)
    assert idx == 1

    # Requesting 2 channels on 1-channel mic must fail with clear error
    with pytest.raises(AudioDeviceResolutionError, match="supports 1 input channels, but 2 channels were requested"):
        resolve_audio_input_device(1, channels=2)


def test_audio_callback_diagnostics_and_states():
    # Reset mic loop state
    mic_loop._st.buf = np.zeros(4800, dtype=np.float32)
    mic_loop._st.buf_len = 4800
    mic_loop._st.write_pos = 0
    mic_loop._st.actual_sr = 48000
    mic_loop._st.channels = 1
    mic_loop._st.device_name = "Mock Mic"
    mic_loop._st.device_identity = {"index": 1, "name": "Mock Mic", "hostapi": 0, "hostapi_name": "Core Audio"}
    mic_loop._st.stream_state = "stream_open"
    mic_loop._st.clipped_samples = 0
    mic_loop._st.total_samples = 0
    mic_loop._st.callback_count = 0
    mic_loop._st.last_active_audio_ts = 0.0

    # 1. Test normal audio block (RMS around 0.005 -> audio_received)
    normal_audio = np.full((512, 1), 0.005, dtype=np.float32)
    mic_loop._audio_callback(normal_audio, 512, None, None)

    diag = mic_loop.get_mic_diagnostics()
    assert diag["stream_state"] == "audio_received"
    assert diag["callback_count"] == 1
    assert diag["total_samples"] == 512
    assert pytest.approx(diag["rms"], rel=1e-3) == 0.005
    assert diag["clipped_samples"] == 0
    assert diag["clipping_ratio"] == 0.0

    # 2. Test high-energy onset block (RMS >= 0.01 -> signal_detected)
    onset_audio = np.full((512, 1), 0.05, dtype=np.float32)
    mic_loop._audio_callback(onset_audio, 512, None, None)

    diag = mic_loop.get_mic_diagnostics()
    assert diag["stream_state"] == "signal_detected"
    assert diag["callback_count"] == 2
    assert pytest.approx(diag["peak_amplitude"], rel=1e-3) == 0.05

    # 3. Test clipped audio block (|x| >= 0.999)
    clipped_audio = np.array([[1.0], [-1.0], [0.5], [0.0]], dtype=np.float32)
    mic_loop._audio_callback(clipped_audio, 4, None, None)

    diag = mic_loop.get_mic_diagnostics()
    assert diag["clipped_samples"] == 2
    assert diag["clipping_ratio"] > 0.0

    # 4. Test silence block (< 0.0001)
    silence_audio = np.zeros((512, 1), dtype=np.float32)
    # Simulate time passing so silence detection triggers
    mic_loop._st.last_active_audio_ts = time.perf_counter() - 1.0
    mic_loop._audio_callback(silence_audio, 512, None, None)

    diag = mic_loop.get_mic_diagnostics()
    assert diag["stream_state"] == "silence"
    assert diag["silence_duration_s"] >= 0.5


@pytest.mark.asyncio
async def test_stream_task_device_resolution_failure():
    # If resolving device raises an error, stream_state must become capture_error
    with patch("mic_loop.resolve_audio_input_device", side_effect=AudioDeviceResolutionError("Mock device failure")):
        with pytest.raises(AudioDeviceResolutionError):
            await mic_loop._stream_task("NonExistentDevice", requested_sr=48000, channels=1)

        diag = mic_loop.get_mic_diagnostics()
        assert diag["stream_state"] == "capture_error"
        assert "Mock device failure" in diag["error_message"]
