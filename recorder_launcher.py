#!/usr/bin/env python3
"""
recorder_launcher.py — Unified Dual-Engine Acoustic Recorder Launcher
======================================================================
Interactive Launcher providing quick access to both acoustic engines:

  [1] DSP Arena Simple Recorder
      - dsp_inverse.py engine
      - G4-compliant peak_time_ms in milliseconds
      - Lightweight, zero-overhead execution

  [2] Full Dual-Mode Acoustic Engine
      - manual_stop_interactive_recorder(1).py
      - Dedicated virtualenv with librosa, pandas, scipy, and RAG memory
      - Dual Training Mode 🏋️ & Testing Mode 🧪

Usage:
    uv run python recorder_launcher.py
    uv run python recorder_launcher.py --engine 1
    uv run python recorder_launcher.py --engine 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ARENA_DIR = Path(__file__).resolve().parent
BACKEND_DIR = Path("/Users/kyawnandar/Documents/interactive-v2/backend")
BACKEND_SCRIPT = BACKEND_DIR / "manual_stop_interactive_recorder(1).py"
BACKEND_VENV_PYTHON = BACKEND_DIR / ".venv" / "bin" / "python"


def run_engine_1() -> None:
    """Run Option 1 (DSP Arena Recorder)."""
    script = ARENA_DIR / "manual_stop_interactive_recorder.py"
    subprocess.run([sys.executable, str(script), "--engine", "1"], cwd=str(ARENA_DIR))


def run_engine_2() -> None:
    """Run Option 2 (Full Backend Dual-Mode Recorder)."""
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

    subprocess.run(
        [str(python_bin), str(BACKEND_SCRIPT)],
        cwd=str(BACKEND_DIR),
        env=env,
        check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-Engine Acoustic Recorder Launcher")
    ap.add_argument("--engine", type=int, choices=[1, 2], default=None,
                    help="1 = DSP Arena Simple Recorder, 2 = Full Dual-Mode Engine")
    args = ap.parse_args()

    choice = args.engine
    if choice is None:
        print("\n" + "═" * 70)
        print(" 🎙️  DSP CLUSTER LAB — DUAL-ENGINE ACOUSTIC RECORDER")
        print("═" * 70)
        print(" [1] ⚡ DSP Arena Simple Recorder (dsp_inverse.py, G4 ms timing)")
        print(" [2] 🔬 Full Dual-Mode Acoustic Engine (Training 🏋️ + Testing 🧪, RAG, Laplace)")
        print(" [q] 🚪 Quit")
        print("═" * 70)
        ans = input("👉 Select Engine [1, 2, or q]: ").strip().lower()
        if ans == "q":
            print("👋 Goodbye!")
            return
        choice = 2 if ans == "2" else 1

    if choice == 1:
        run_engine_1()
    else:
        run_engine_2()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting. Goodbye!")
