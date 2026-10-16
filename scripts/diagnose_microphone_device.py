#!/usr/bin/env python3
"""diagnose_microphone_device.py — Microphone Device & Stream Verification.

Verifies:
1. All audio devices and host APIs available via PortAudio/sounddevice.
2. What default device is selected when MIC_DEVICE is omitted vs set.
3. Captures a 1.0s diagnostic recording from the physical MacBook microphone.
4. Analyzes RMS, peak, dynamic range, clipping, and verifies live acoustic input vs digital silence.
5. Provides the deterministic command and environment configuration.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def diagnose_audio_devices():
    print("=== 1. SOUNDDEVICE / PORTAUDIO HOST APIS & DEVICES ===")
    apis = sd.query_hostapis()
    for idx, api in enumerate(apis):
        print(f"Host API [{idx}]: {api['name']} (default in={api['default_input_device']}, default out={api['default_output_device']})")

    devices = sd.query_devices()
    default_in_idx, default_out_idx = sd.default.device
    print(f"\nSystem Default Input Device Index: {default_in_idx}")
    print(f"System Default Output Device Index: {default_out_idx}")

    print("\n--- Available Input Devices ---")
    physical_mic_indices = []
    blackhole_indices = []

    for idx, d in enumerate(devices):
        max_in = d["max_input_channels"]
        if max_in > 0:
            name = d["name"]
            sr = d["default_samplerate"]
            api_name = apis[d["hostapi"]]["name"]
            is_def = " [SYSTEM DEFAULT INPUT]" if idx == default_in_idx else ""
            print(f"  Device ID [{idx:2d}]: {name:<32} | InCh: {max_in:2d} | DefSR: {sr:7.1f} Hz | API: {api_name}{is_def}")

            name_lower = name.lower()
            if "blackhole" in name_lower:
                blackhole_indices.append(idx)
            if "macbook" in name_lower or "built-in" in name_lower or "internal" in name_lower:
                physical_mic_indices.append(idx)

    # Check environment variable MIC_DEVICE
    env_mic_device = os.environ.get("MIC_DEVICE")
    print(f"\nEnvironment variable MIC_DEVICE: {env_mic_device!r}")

    active_target_idx = None
    if env_mic_device is not None and env_mic_device.strip() != "":
        try:
            active_target_idx = int(env_mic_device)
            print(f"Using MIC_DEVICE index: {active_target_idx} ({devices[active_target_idx]['name']})")
        except (ValueError, IndexError) as exc:
            print(f"Invalid MIC_DEVICE {env_mic_device!r}: {exc}")
    else:
        print(f"MIC_DEVICE not set -> sounddevice default input index is: {default_in_idx}")
        active_target_idx = default_in_idx

    selected_device_info = sd.query_devices(active_target_idx)
    selected_name = selected_device_info["name"]
    is_blackhole = "blackhole" in selected_name.lower()
    print(f"\nCurrently selected device: [{active_target_idx}] {selected_name}")
    if is_blackhole:
        print("  WARNING: Selected device IS BlackHole (virtual loopback)! It will produce SILENCE unless piped externally.")
    else:
        print(f"  OK: Selected device appears to be a physical or non-BlackHole audio input ({selected_name}).")

    # Determine best physical mic index
    best_physical_idx = None
    for idx in physical_mic_indices:
        # Prefer Core Audio API
        if apis[devices[idx]["hostapi"]]["name"] == "Core Audio":
            best_physical_idx = idx
            break
    if best_physical_idx is None and physical_mic_indices:
        best_physical_idx = physical_mic_indices[0]

    print(f"\nIdentified physical MacBook microphone index: {best_physical_idx}")
    if best_physical_idx is not None:
        print(f"Physical Mic Info: {devices[best_physical_idx]}")

    # Capture 1.0 second test from physical microphone
    test_target_idx = best_physical_idx if best_physical_idx is not None else active_target_idx
    print(f"\n=== 2. CAPTURING 1.0s DIAGNOSTIC RECORDING ON DEVICE [{test_target_idx}] ({devices[test_target_idx]['name']}) ===")
    
    target_sr = int(devices[test_target_idx]["default_samplerate"])
    duration_s = 1.0
    total_frames = int(target_sr * duration_s)

    status_flags = []
    def callback(indata, frames, time_info, status):
        if status:
            status_flags.append(str(status))

    print(f"Opening InputStream: device={test_target_idx}, sr={target_sr}, channels=1, dtype='float32'...")
    rec_buf = sd.rec(
        frames=total_frames,
        samplerate=target_sr,
        channels=1,
        dtype="float32",
        device=test_target_idx,
        blocking=True,
    )
    rec_data = rec_buf.ravel()

    # Quantitative signal analysis
    min_val = float(np.min(rec_data))
    max_val = float(np.max(rec_data))
    mean_val = float(np.mean(rec_data))
    peak_amp = float(np.max(np.abs(rec_data)))
    rms_val = float(np.sqrt(np.mean(rec_data ** 2)))
    clipped_samples = int(np.sum(np.abs(rec_data) >= 0.999))
    clipping_ratio = float(clipped_samples / len(rec_data))

    # Frequency domain check
    fft_mag = np.abs(np.fft.rfft(rec_data * np.hanning(len(rec_data))))
    fft_freqs = np.fft.rfftfreq(len(rec_data), 1.0 / target_sr)
    peak_fft_freq = float(fft_freqs[np.argmax(fft_mag)])

    print("\n--- Diagnostic Recording Metrics ---")
    print(f"  Duration:            {duration_s:.3f} s ({len(rec_data)} samples at {target_sr} Hz)")
    print(f"  Stream status:       {'OK (no errors)' if not status_flags else ', '.join(status_flags)}")
    print(f"  Min Sample:          {min_val:+.6f}")
    print(f"  Max Sample:          {max_val:+.6f}")
    print(f"  DC Offset (Mean):    {mean_val:+.6f}")
    print(f"  Peak Amplitude:      {peak_amp:.6f} ({20.0 * np.log10(max(peak_amp, 1e-9)):.1f} dBFS)")
    print(f"  RMS Amplitude:       {rms_val:.6f} ({20.0 * np.log10(max(rms_val, 1e-9)):.1f} dBFS)")
    print(f"  Clipped Samples:     {clipped_samples} ({clipping_ratio * 100:.3f}%)")
    print(f"  Spectral Peak Freq:  {peak_fft_freq:.1f} Hz")

    is_silent = rms_val < 1e-6
    if is_silent:
        print("\n  ALERT: RECORDING IS PURE DIGITAL SILENCE! (RMS < 1e-6)")
        print("  Cause: Device is a disconnected/virtual loopback or microphone access is muted/blocked.")
    elif rms_val < 0.001:
        print("\n  NOTICE: Quiet ambient room acoustic signal (RMS between 1e-6 and 1e-3).")
    else:
        print(f"\n  SUCCESS: Verified live acoustic signal detected (RMS = {rms_val:.6f} > 0.001).")

    print("\n=== 3. RECOMMENDED ENVIRONMENT VARIABLES & LAUNCH COMMAND ===")
    print("To reliably launch live_server with the physical MacBook microphone without index drift:")
    print(f"  export MIC_DEVICE={best_physical_idx}")
    print(f"  export MIC_SR=48000")
    print("  uv run uvicorn live_server:app --port 8765")
    print("\nOr inline:")
    print(f"  MIC_DEVICE={best_physical_idx} MIC_SR=48000 VECLIB_MAXIMUM_THREADS=2 uv run uvicorn live_server:app --port 8765")


if __name__ == "__main__":
    diagnose_audio_devices()
