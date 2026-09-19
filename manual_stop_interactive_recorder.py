#!/usr/bin/env python3
"""
manual_stop_interactive_recorder.py
====================================
Interactive Recorder with Dual Engine Selection:

  [1] DSP Arena Simple Recorder
      - Uses local dsp_inverse.py
      - Strict G4 compliance: peak_time_ms always in milliseconds
      - Driver-verified sample rate (G5) and device logging (G3)

  [2] Full Dual-Mode Acoustic Engine
      - Launches /Users/kyawnandar/Documents/interactive-v2/backend/manual_stop_interactive_recorder(1).py
      - Runs in dedicated backend virtualenv with full librosa/pandas/scipy/RAG stack
      - Training Mode 🏋️ & Testing Mode 🧪 with consecutive-consensus & master logging

Usage:
    python manual_stop_interactive_recorder.py
    python manual_stop_interactive_recorder.py --engine 1
    python manual_stop_interactive_recorder.py --engine 2
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Same DSP recipe as the live server — ONE truth (PROJECT_MEMORY §1)
from dsp_inverse import measure_features, resample_linear

BACKEND_DIR = Path("/Users/kyawnandar/Documents/interactive-v2/backend")
BACKEND_SCRIPT = BACKEND_DIR / "manual_stop_interactive_recorder(1).py"
BACKEND_VENV_PYTHON = BACKEND_DIR / ".venv" / "bin" / "python"


def _list_devices() -> None:
    import sounddevice as sd
    devs = sd.query_devices()
    print("\nInput devices:")
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            marker = " <-- default" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}] {d['name']!r:<38}  ch={d['max_input_channels']}  sr={d['default_samplerate']:.0f}{marker}")
    print()


def record_interactive(
    device: int | None,
    requested_sr: int,
    channels: int,
    out_path: Path,
) -> tuple[np.ndarray, int]:
    """Record until user presses Enter. Returns (audio_float64_mono, actual_sr)."""
    import sounddevice as sd

    buf_q: queue.SimpleQueue = queue.SimpleQueue()

    with sd.InputStream(
        device=device,
        samplerate=requested_sr,
        channels=channels,
        dtype="float32",
    ) as stream:
        actual_sr = int(stream.samplerate)
        dev_info = sd.query_devices(stream.device[0] if isinstance(stream.device, (list, tuple)) else stream.device)
        dev_name = dev_info["name"]

        print(f"\n[recorder] Device    : {dev_name!r}")
        print(f"[recorder] Requested SR: {requested_sr} Hz")
        print(f"[recorder] Actual SR  : {actual_sr} Hz  <-- verified from stream.samplerate")
        print(f"[recorder] Channels   : {channels}\n")

    chunks: list[np.ndarray] = []
    recording = True

    def _cb2(indata, frames, time_info, status):
        if status:
            print(f"[recorder] stream status: {status}", file=sys.stderr)
        if recording:
            chunks.append(indata.copy())

    input("👉 Press Enter to START recording... ")
    t_start = time.perf_counter()

    with sd.InputStream(
        device=device,
        samplerate=actual_sr,
        channels=channels,
        dtype="float32",
        callback=_cb2,
    ) as stream2:
        real_sr = int(stream2.samplerate)
        if real_sr != actual_sr:
            print(f"[recorder] SR adjusted by driver: {actual_sr} -> {real_sr}", file=sys.stderr)
            actual_sr = real_sr

        input("🛑 Press Enter to STOP recording... ")
        recording = False
        elapsed = time.perf_counter() - t_start
        print(f"[recorder] Recorded {elapsed:.2f} s at {actual_sr} Hz")

    if not chunks:
        raise RuntimeError("No audio captured")
    audio = np.concatenate(chunks, axis=0)
    mono = audio.mean(axis=1) if channels > 1 else audio[:, 0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(actual_sr)
        wf.writeframes((mono.astype(np.float32)).tobytes())
    print(f"[recorder] Saved -> {out_path} ({len(mono)} samples)")

    return mono, actual_sr


def run_dsp(mono: np.ndarray, actual_sr: int, dev_name: str) -> None:
    """Run DSP on recorded audio — same recipe as live_server /compute."""
    print("\n[dsp] Running feature extraction…")
    t0 = time.perf_counter()
    features = measure_features(
        mono, actual_sr,
        nfft=4096, hop=1024, top_db=35, win=0.5,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if features is None:
        print("[dsp] No transient detected — signal too short or quiet.")
        return

    pt_s = float(features.get("peak_time", 0.0))
    print(f"\n{'═'*66}")
    print(f"  Recording_Device       : {dev_name!r}")
    print(f"  Recording_Actual_SR_Hz : {actual_sr}")
    print(f"  impact_freq_hz         : {features['impact_freq_hz']:.1f} Hz")
    print(f"  peak_time_ms           : {pt_s * 1000:.3f} ms  (G4 compliant)")
    print(f"  impact                 : {features['impact']:.6f}")
    print(f"  duration_s             : {features['duration']:.4f} s")
    print(f"  mel_mean_db            : {features['mel_mean_db']:.3f} dB")
    print(f"  rms_energy             : {features['rms_energy']:.6f}")
    print(f"  attack_slope_db_ms     : {features['attack_slope_db_ms']:.4f}")
    print(f"  crest_factor           : {features['crest_factor']:.4f}")
    print(f"  hf_transient_ratio     : {features['hf_transient_ratio']:.6f}")
    print(f"  spectral_centroid_hz   : {features['spectral_centroid_hz']:.1f} Hz")
    print(f"  spectral_bandwidth_hz  : {features['spectral_bandwidth_hz']:.1f} Hz")
    print(f"  spectral_rolloff_hz    : {features['spectral_rolloff_hz']:.1f} Hz")
    print(f"  spectral_flatness      : {features['spectral_flatness']:.6f}")
    print(f"  zcr                    : {features['zcr']:.6f}")
    print(f"  DSP elapsed            : {elapsed_ms:.1f} ms")
    print(f"{'═'*66}\n")


def sync_to_live_server(wav_path: Path, actual_sr: int, dev_name: str) -> None:
    """Stream recorded WAV to live_server /compute so WebSocket broadcasts to dashboard."""
    try:
        import requests
        url = "http://127.0.0.1:8765/compute"
        with open(wav_path, "rb") as f:
            files = {"file": (wav_path.name, f, "audio/wav")}
            data = {"sr": actual_sr, "device_label": dev_name}
            resp = requests.post(url, files=files, data=data, timeout=3.0)
            if resp.status_code == 200:
                print("📡 [live_server] Successfully streamed event & waveform to Web Dashboard (port 8765)!")
            else:
                print(f"⚠️ [live_server] Server responded with HTTP {resp.status_code}")
    except Exception as exc:
        print(f"ℹ️ [live_server] Web dashboard sync skipped ({exc})")


def run_engine_dsp_arena(args: argparse.Namespace) -> None:
    """Mode 1: Run local DSP Arena recorder."""
    import sounddevice as sd
    if args.list_devices:
        _list_devices()
        return

    mono, actual_sr = record_interactive(args.device, args.sr, args.channels, args.out)
    dev_info = sd.query_devices(args.device if args.device is not None else sd.default.device[0])
    run_dsp(mono, actual_sr, dev_info["name"])
    sync_to_live_server(args.out, actual_sr, dev_info["name"])


def run_engine_backend_dual_mode() -> None:
    """Mode 2: Launch the full backend dual-mode recorder in its virtualenv."""
    if not BACKEND_SCRIPT.exists():
        print(f"❌ Backend script not found at: {BACKEND_SCRIPT}")
        return

    python_bin = BACKEND_VENV_PYTHON if BACKEND_VENV_PYTHON.exists() else Path(sys.executable)
    print(f"\n🚀 Launching Full Dual-Mode Acoustic Engine...")
    print(f"   Interpreter : {python_bin}")
    print(f"   Script      : {BACKEND_SCRIPT}")
    print(f"   Working Dir : {BACKEND_DIR}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BACKEND_DIR}:{env.get('PYTHONPATH', '')}"

    try:
        subprocess.run(
            [str(python_bin), str(BACKEND_SCRIPT)],
            cwd=str(BACKEND_DIR),
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        print("\n👋 Returning from Dual-Mode Acoustic Engine.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Recorder with Dual-Engine Options")
    ap.add_argument("--engine", type=int, choices=[1, 2], default=None,
                    help="1 = DSP Arena Recorder, 2 = Full Dual-Mode Acoustic Engine")
    ap.add_argument("--device", type=int, default=None, help="Input device index")
    ap.add_argument("--sr", type=int, default=44100, help="Requested sample rate")
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("recordings/latest.wav"))
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        _list_devices()
        return

    if engine_choice is None:
        print("\n" + "═" * 70)
        print(" 🔬  4DS DUAL-MODE ACOUSTIC RECORDER & EVENT ENGINE")
        print("═" * 70)
        print(" [1] 🔬 Full Dual-Mode Acoustic Engine (Default: manual_stop_interactive_recorder(1).py)")
        print("     → Auto Impact Detection (1-20kHz BP + Hilbert + find_peaks)")
        print("     → Multi-Method DSP (384kHz STFT, Laplace Decay, FFT, RAG Memory)")
        print("     → Training Mode 🏋️ & Testing Mode 🧪 with Candidate Pairs")
        print(" [2] ⚡ Legacy Simple Recorder (Single Transient fallback)")
        print("═" * 70)
        choice = input("👉 Select Engine [1 or 2] (Default: 1 - Full Dual-Mode Engine): ").strip()
        engine_choice = 1 if choice in ("", "1") else 2

    if engine_choice == 1:
        run_engine_backend_dual_mode()
    else:
        run_engine_dsp_arena(args)


if __name__ == "__main__":
    main()
