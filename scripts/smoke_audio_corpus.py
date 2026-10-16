#!/usr/bin/env python3
"""Smoke-test a real WAV directory against the running local /compute API.

Example:
  ./.venv/bin/python scripts/smoke_audio_corpus.py --wav-dir ~/Downloads/audio

The script does not copy, rename, or modify source WAVs and uses only the
Python standard library. The API performs its documented native-to-target
resampling; a 384 kHz native WAV is never relabelled as a 48 kHz recording.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CANONICAL_FIELDS = ("impact", "duration_s", "peak_time_ms", "mel_mean_db", "impact_freq_hz")


def multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = "----DSPArena" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
            value.encode(),
            b"\r\n",
        ))
    chunks.extend((
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: audio/wav\r\n\r\n",
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ))
    return b"".join(chunks), boundary


def post_compute(url: str, wav_path: Path, sample_rate: int, fmax: str | None = None) -> tuple[int, dict]:
    content = wav_path.read_bytes()
    fields = {
        "sr": str(sample_rate),
        "nfft": "8192",
        "hop": "2048",
        "top_db": "35",
        "win": "0.5",
        "device_label": "dsp_arena_smoke",
    }
    if fmax is not None:
        fields["fmax"] = fmax
    body, boundary = multipart(fields, "file", wav_path.name, content)
    request = Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw[:500]}
        return error.code, payload
    except URLError as error:
        raise RuntimeError(f"Cannot reach {url}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-dir", type=Path, default=Path.home() / "Downloads" / "audio")
    parser.add_argument("--url", default="http://127.0.0.1:8765/compute")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--skip-guard-probes", action="store_true")
    args = parser.parse_args()

    if args.sample_rate < 1 or args.sample_rate > 48000:
        parser.error("--sample-rate must be between 1 and 48000")
    if not args.wav_dir.is_dir():
        parser.error(f"WAV directory not found: {args.wav_dir}")
    files = sorted((p for p in args.wav_dir.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".wave"}), key=lambda p: p.name.lower())
    if not files:
        parser.error(f"No WAV files found in {args.wav_dir}")

    passed = 0
    failed = 0
    for path in files:
        started = time.monotonic()
        try:
            source = path.read_bytes()
            if len(source) < 28 or source[:4] != b"RIFF" or source[8:12] != b"WAVE":
                raise ValueError("not a supported RIFF/WAVE file")
            native_sr = struct.unpack_from("<I", source, 24)[0]
            status, payload = post_compute(args.url, path, args.sample_rate)
            if status != 200:
                raise ValueError(f"HTTP {status}: {payload.get('detail', payload)}")
            missing = [key for key in CANONICAL_FIELDS if key not in payload]
            if missing:
                raise ValueError(f"missing canonical fields: {missing}")
            if payload.get("native_sr_hz") != native_sr:
                raise ValueError(f"native sample-rate provenance mismatch: header={native_sr}, response={payload.get('native_sr_hz')}")
            if payload.get("Recording_Actual_SR_Hz") != args.sample_rate:
                raise ValueError(f"DSP sample-rate mismatch: expected={args.sample_rate}, response={payload.get('Recording_Actual_SR_Hz')}")
            peak_ms = float(payload["peak_time_ms"])
            peak_s = float(payload["peak_time_s"])
            if abs(peak_ms - peak_s * 1000.0) >= 0.0011:
                raise ValueError("peak_time_ms does not match peak_time_s * 1000")
            elapsed = time.monotonic() - started
            passed += 1
            print(f"PASS {path.name!r}: {native_sr} -> {args.sample_rate} Hz; freq={payload['impact_freq_hz']} Hz; peak={peak_ms} ms; {elapsed:.2f}s")
        except Exception as error:
            failed += 1
            print(f"FAIL {path.name!r}: {error}")

    if not args.skip_guard_probes:
        for label, sample_rate, fmax in (
            ("G1 sample-rate guard", 48001, None),
            ("G2 fmax guard", args.sample_rate, "21501"),
        ):
            status, payload = post_compute(args.url, files[0], sample_rate, fmax)
            ok = status == 422
            print(f"{'PASS' if ok else 'FAIL'} {label}: HTTP {status}")
            if ok:
                passed += 1
            else:
                failed += 1
                print(f"  response: {payload.get('detail', payload)}")

    print(f"SUMMARY: {passed} passed, {failed} failed, {len(files)} WAV files; source files unchanged")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
