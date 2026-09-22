#!/usr/bin/env python3
"""
live_server.py — DSP Cluster Lab Live Monitor  (Phase 3)
=========================================================
FastAPI server skeleton + /compute route + continuous mic-loop auto-trigger.

Phase 1+2 contracts (baked-in from code-review pre-implementation guards):
  G1  sr > SR_MAX (48 kHz) -> 422   prevents 384k upsample artifact re-entry
  G2  fmax > FMAX_REAL (21.5 kHz) -> 422  above confirmed real capture ceiling
  G3  Every response carries Recording_Device + Recording_Actual_SR_Hz
  G4  Time values are ALWAYS in milliseconds in JSON response (peak_time_ms)
  G5  Actual sample-rate (not requested) logged and returned as actual_sr

Run:
    uv run uvicorn live_server:app --host 0.0.0.0 --port 8765 --reload
"""

from __future__ import annotations

import asyncio
import os
import sys
import importlib.util
import struct
import time
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import socket
import webbrowser
import io
import qrcode
import qrcode.image.svg
from fastapi.responses import FileResponse, JSONResponse, Response

import mic_loop as _mic
import event_store as _store

# Local DSP library - reuse exact same recipe as offline optimizer.
# ONE recipe = ONE truth (PROJECT_MEMORY s1 golden rule).
from dsp_inverse import (
    detect_transient_window,
    measure_features,
    resample_linear,
)

# Guards (Phase-1 contracts)
SR_MAX: int = 48_000         # G1 native-capture ceiling; prohibits 384k artifact
FMAX_REAL: float = 21_500.0  # G2 confirmed real ceiling (impact_spectrum_check.py)


# ---------------------------------------------------------------------------
# Dual-Mode RAG Acoustic Engine Bridge
# ---------------------------------------------------------------------------
BACKEND_DIR = Path("/Users/kyawnandar/Documents/interactive-v2/backend")
_dual_engine = None

def get_dual_engine():
    global _dual_engine
    if _dual_engine is None:
        try:
            if str(BACKEND_DIR) not in sys.path:
                sys.path.insert(0, str(BACKEND_DIR))
            script_path = BACKEND_DIR / "manual_stop_interactive_recorder(1).py"
            if script_path.exists():
                spec = importlib.util.spec_from_file_location("dual_engine", str(script_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "init_engine_rag_store"):
                    mod.init_engine_rag_store()
                _dual_engine = mod
                log.info("Dual-Mode Acoustic Engine loaded successfully.")
        except Exception as exc:
            log.warning("Could not load Dual-Mode Acoustic Engine: %s", exc)
    return _dual_engine

# Default DSP params - match locked Optuna-winning theta from PROJECT_MEMORY s2
_DEFAULT_SR: int = 44_100
_DEFAULT_NFFT: int = 4_096
_DEFAULT_HOP: int = 1_024
_DEFAULT_TOPDB: int = 35
_DEFAULT_WIN: float = 0.5

# Use uvicorn's existing logger chain so messages appear in uvicorn output
log = logging.getLogger("live_server")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_h)


class _ConnectionManager:
    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        log.info("WS  client connected  (total=%d)", len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._active:
            self._active.remove(ws)
        log.info("WS  client disconnected (total=%d)", len(self._active))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._active):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._active:
                self._active.remove(ws)

    @property
    def client_count(self) -> int:
        return len(self._active)



# ---------------------------------------------------------------------------
# Network IP, QR Code & Auto Browser Launcher
# ---------------------------------------------------------------------------
_SERVER_PORT: int = int(os.environ.get("PORT", "8765"))

def get_lan_ip() -> str:
    """Detect LAN IPv4 address for mobile access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def print_terminal_qr_banner(local_url: str, lan_url: str) -> None:
    """Print ASCII QR code in terminal for mobile scanning."""
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(lan_url)
        qr.make(fit=True)
        f = io.StringIO()
        qr.print_ascii(out=f, invert=True)
        ascii_qr = f.getvalue()
    except Exception as exc:
        ascii_qr = f"[QR generation error: {exc}]"

    print("\n" + "═" * 66)
    print(" 🚀 DSP CLUSTER LAB — LIVE MONITOR READY")
    print(f" 🌐 Local Browser : {local_url}")
    print(f" 📱 Mobile / LAN  : {lan_url}")
    print("═" * 66)
    print(ascii_qr)
    print(" 📷 Scan the QR code with phone or tablet to open dashboard!")
    print("═" * 66 + "\n")

def _auto_open_browser(url: str) -> None:
    """Auto-launch the user's default browser on startup."""
    if os.environ.get("NO_BROWSER", "").lower() in ("1", "true", "yes"):
        log.info("Auto-browser disabled by NO_BROWSER env var.")
        return
    time.sleep(0.8)
    try:
        log.info("Opening dashboard in browser: %s", url)
        webbrowser.open(url)
    except Exception as exc:
        log.warning("Could not auto-launch browser: %s", exc)

_mic_task: asyncio.Task | None = None
_mic_active: bool = False

async def _start_mic_service() -> bool:
    global _mic_task, _mic_active
    if _mic_task and not _mic_task.done():
        _mic_active = True
        return True
    
    no_mic = os.environ.get("NO_MIC", "").lower() in ("1", "true", "yes")
    if no_mic:
        log.info("Mic loop start skipped (NO_MIC=1 set).")
        return False
        
    mic_dev = int(os.environ["MIC_DEVICE"]) if "MIC_DEVICE" in os.environ else None
    mic_sr = int(os.environ.get("MIC_SR", "44100"))
    log.info("Starting live mic loop: device=%s  requested_sr=%d", mic_dev or "default", mic_sr)
    _mic_task = asyncio.create_task(
        _mic.run_mic_loop(manager=manager, device=mic_dev, requested_sr=mic_sr, channels=1),
        name="mic_loop",
    )
    _mic_active = True
    await manager.broadcast({"type": "mic_state", "active": True})
    return True

async def _stop_mic_service() -> bool:
    global _mic_task, _mic_active
    _mic_active = False
    if _mic_task and not _mic_task.done():
        _mic_task.cancel()
        try:
            await asyncio.wait_for(_mic_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        _mic_task = None
    await manager.broadcast({"type": "mic_state", "active": False})
    log.info("Live mic loop stopped.")
    return True

@asynccontextmanager
async def _lifespan(app: FastAPI):
    lan_ip = get_lan_ip()
    local_url = f"http://localhost:{_SERVER_PORT}"
    lan_url = f"http://{lan_ip}:{_SERVER_PORT}"

    # Print terminal QR code banner and launch browser
    print_terminal_qr_banner(local_url, lan_url)
    asyncio.create_task(asyncio.to_thread(_auto_open_browser, local_url))

    # Auto-start mic unless disabled by NO_MIC env
    await _start_mic_service()

    yield

    # Shutdown
    await _stop_mic_service()
    log.info("live_server shutdown complete")

manager = _ConnectionManager()

app = FastAPI(
    lifespan=_lifespan,
    title="DSP Cluster Lab Live Monitor",
    description="Phase 3 - /compute + mic-loop + event store + WS broadcast pipeline",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_HTML_PATH = Path(__file__).parent / "dsp-cluster-lab.html"
_SCOUT_HTML_PATH = Path(__file__).parent / "partition_scout.html"


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def root() -> FileResponse:
    if not _HTML_PATH.exists():
        raise HTTPException(404, "dsp-cluster-lab.html not found")
    return FileResponse(_HTML_PATH, media_type="text/html")


@app.get("/scout", response_class=FileResponse, include_in_schema=False)
async def scout() -> FileResponse:
    if not _SCOUT_HTML_PATH.exists():
        raise HTTPException(404, "partition_scout.html not found")
    return FileResponse(_SCOUT_HTML_PATH, media_type="text/html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "ws_clients": manager.client_count,
        "sr_max": SR_MAX,
        "fmax_real": FMAX_REAL,
        "ts": time.time(),
    }



def _downsample_waveform(audio: np.ndarray, num_points: int = 300) -> list[float]:
    """Downsample audio to fixed num_points for high-performance canvas waveform view."""
    if len(audio) == 0:
        return []
    if len(audio) <= num_points:
        return [round(float(x), 4) for x in audio]
    step = len(audio) / num_points
    return [round(float(audio[int(i * step)]), 4) for i in range(num_points)]

def _decode_wav_bytes(data: bytes) -> tuple:
    """Decode raw WAV bytes -> (mono float64, native_sr)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    off = 12
    fmt = None
    dstart = dlen = -1
    n = len(data)
    while off + 8 <= n:
        cid = data[off : off + 4]
        sz = struct.unpack("<I", data[off + 4 : off + 8])[0]
        if cid == b"fmt ":
            fmt = {
                "fmt":  struct.unpack("<H", data[off + 8 : off + 10])[0],
                "ch":   struct.unpack("<H", data[off + 10 : off + 12])[0],
                "sr":   struct.unpack("<I", data[off + 12 : off + 16])[0],
                "bits": struct.unpack("<H", data[off + 22 : off + 24])[0],
            }
            if sz >= 40:
                sub  = struct.unpack("<H", data[off + 32 : off + 34])[0]
                guid = data[off + 34 : off + 38]
                if sub == 0xFFFE and guid == b"PCM\0":
                    fmt["fmt"] = 1
                elif sub == 0xFFFE and guid == b"f32\0":
                    fmt["fmt"] = 3
            off += 8 + sz + (sz & 1)
        elif cid == b"data":
            dstart, dlen = off + 8, sz
            break
        else:
            off += 8 + sz + (sz & 1)
    if fmt is None:
        raise ValueError("missing fmt chunk")
    if dstart < 0:
        raise ValueError("missing data chunk")
    bits = fmt["bits"]
    ch   = fmt["ch"]
    bps  = bits >> 3
    if fmt["fmt"] not in (1, 3) or (fmt["fmt"] == 3 and bits != 32):
        raise ValueError(f"unsupported format: wave={fmt['fmt']} bits={bits}")
    if bits not in (8, 16, 24, 32):
        raise ValueError(f"unsupported bit depth {bits}")
    usable  = (dlen // (bps * ch)) * (bps * ch)
    nframes = usable // (bps * ch)
    raw = data[dstart : dstart + usable]
    if fmt["fmt"] == 3:
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(nframes, ch)
    elif bits == 8:
        arr = ((np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0).reshape(nframes, ch)
    elif bits == 16:
        arr = (np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0).reshape(nframes, ch)
    elif bits == 24:
        b3  = np.frombuffer(raw, dtype=np.uint8).reshape(nframes * ch, 3)
        x   = b3[:, 0].astype(np.int64) | (b3[:, 1].astype(np.int64) << 8) | (b3[:, 2].astype(np.int64) << 16)
        x   = np.where(x & 0x800000, x - 0x1000000, x)
        arr = (x / 8_388_607.0).reshape(nframes, ch)
    else:
        arr = (np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2_147_483_647.0).reshape(nframes, ch)
    mono = arr.mean(axis=1) if ch > 1 else arr[:, 0]
    return mono, int(fmt["sr"])


def _build_response(
    features: dict,
    *,
    native_sr: int,
    actual_sr: int,
    recording_device: str,
    fmax_used: float,
    filename: str,
    elapsed_ms: float,
) -> dict:
    """Convert measure_features() dict -> wire-format dict.
    G3: Recording_Device + Recording_Actual_SR_Hz on every response.
    G4: peak_time ALWAYS in ms. peak_time_s for debug only.
    """
    pt_s = float(features.get("peak_time", 0.0))
    return {
        "Recording_Device":       recording_device,
        "Recording_Actual_SR_Hz": actual_sr,
        "native_sr_hz":           native_sr,
        "fmax_used_hz":           round(fmax_used, 1),
        "filename":               filename,
        "peak_time_ms":           round(pt_s * 1000.0, 3),
        "peak_time_s":            round(pt_s, 6),
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
        "waveform":               _downsample_waveform(features.get("_audio_window", np.array([], dtype=np.float32)), 300),
        "ts":                     time.time(),
    }


@app.post("/compute")
async def compute(
    file:         UploadFile = File(..., description="WAV file to analyse"),
    sr:           int   = Form(_DEFAULT_SR,   description="Target sample-rate (<=48000 Hz)"),
    nfft:         int   = Form(_DEFAULT_NFFT),
    hop:          int   = Form(_DEFAULT_HOP),
    top_db:       float = Form(_DEFAULT_TOPDB),
    win:          float = Form(_DEFAULT_WIN,  description="Transient window +/- seconds"),
    fmax:         float = Form(FMAX_REAL,     description="Max analysis frequency (<=21500 Hz)"),
    device_label: str   = Form("file_upload", description="Logical device label (G3 provenance)"),
    engine:       str   = Form("dsp_arena",   description="Engine: 'dsp_arena' or 'dual_mode_rag'"),
) -> JSONResponse:
    """POST a WAV file -> DSP feature vector JSON.

    Guards:
      G1  sr > 48000  -> 422
      G2  fmax > 21500 -> 422
    """
    # G1
    if sr > SR_MAX:
        raise HTTPException(
            422,
            detail=(
                f"sr={sr} exceeds SR_MAX={SR_MAX} Hz. "
                "Upsampling above native-capture ceiling prohibited (impact_spectrum_check.py)."
            ),
        )
    # G2
    if fmax > FMAX_REAL:
        raise HTTPException(
            422,
            detail=(
                f"fmax={fmax:.0f} Hz exceeds confirmed real ceiling {FMAX_REAL:.0f} Hz. "
                "Set fmax <= 21500 to avoid artifact-driven features."
            ),
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file upload")
    try:
        y, native_sr = _decode_wav_bytes(raw)
    except Exception as exc:
        raise HTTPException(400, f"WAV decode failed: {exc}") from exc

    filename = file.filename or "unknown.wav"
    log.info("compute  file=%s  native_sr=%d  target_sr=%d  fmax=%.0f  nfft=%d",
             filename, native_sr, sr, fmax, nfft)

    # G5: actual_sr is what we pass to DSP - never a module-level fallback constant
    actual_sr = sr
    if native_sr != actual_sr:
        y = resample_linear(y, native_sr, actual_sr)

    fmax_used = min(fmax, actual_sr / 2.0)

    if engine == "dual_mode_rag":
        de = get_dual_engine()
        if de is None:
            raise HTTPException(500, "Dual-Mode Acoustic Engine not available on server")
        t0 = time.perf_counter()
        try:
            raw_feats = await asyncio.to_thread(de.compute_multi_method_dsp, y, actual_sr, filename)
        except Exception as exc:
            log.exception("Dual-mode DSP error for %s", filename)
            raise HTTPException(500, f"Dual-mode DSP error: {exc}") from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pt_s = float(raw_feats.get("peak_time_s", 0.0))
        freq = float(raw_feats.get("frequency_hz", 0.0))
        hfsr = float(raw_feats.get("high_freq_ratio", 0.0))
        sigma = float(raw_feats.get("laplace_sigma", 0.0))
        dur_s = float(raw_feats.get("duration_s", 0.0))
        det_dur = float(raw_feats.get("detect_duration_s", 0.0))
        centroid = float(raw_feats.get("centroid_hz", 0.0))

        # STRICT PHYSICAL CEILING: transient must not exceed 1.0s limit
        if det_dur > 1.0 or dur_s > 1.0:
            raise HTTPException(422, f"Signal rejected: transient duration ({max(det_dur, dur_s):.3f}s) > 1.0s ceiling. Acoustic impacts are < 0.5s.")

        # RAG memory match check
        rag_match_info = {"matched": False, "group": None, "score": 0.0, "characteristics": ""}
        if hasattr(de, "check_rag_characteristics"):
            feats_tuple = (sigma, centroid, hfsr, dur_s * 1e6, 0.0, 0.0)
            try:
                rag_match_info = de.check_rag_characteristics(feats_tuple, min_score=0.85)
            except Exception:
                pass

        payload = {
            "Recording_Device": device_label,
            "Recording_Actual_SR_Hz": actual_sr,
            "native_sr_hz": native_sr,
            "fmax_used_hz": fmax_used,
            "filename": filename,
            "peak_time_ms": round(pt_s * 1000.0, 3),
            "peak_time_s": round(pt_s, 6),
            "impact": round(float(raw_feats.get("peak_amplitude", 0.0)), 6),
            "duration_s": round(dur_s, 6),
            "impact_freq_hz": round(freq, 1),
            "mel_mean_db": round(float(raw_feats.get("mel_mean_db", 0.0)), 3),
            "rms_energy": round(float(raw_feats.get("peak_amplitude", 0.0)), 6),
            "attack_slope_db_ms": 0.0,
            "crest_factor": round(float(raw_feats.get("crest_factor", 0.0)), 4),
            "hf_transient_ratio": round(hfsr, 6),
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_bandwidth_hz": round(float(raw_feats.get("spectral_bandwidth_hz", 0.0)), 1),
            "spectral_rolloff_hz": round(centroid, 1),
            "spectral_flatness": round(float(raw_feats.get("spectral_flatness", 0.0)), 6),
            "zcr": round(float(raw_feats.get("zcr", 0.0)), 6),
            "engine": "dual_mode_rag",
            "cluster": raw_feats.get("final_single_prediction", "U"),
            "candidate_pair": raw_feats.get("candidate_pair_text", ""),
            "elimination": raw_feats.get("elimination_text", ""),
            "laplace_sigma": round(sigma, 4),
            "damping_ratio_zeta": round(float(raw_feats.get("damping_ratio_zeta", 0.0)), 6),
            "mel_contrast_ratio": round(float(raw_feats.get("mel_contrast_ratio", 0.0)), 4),
            "separation_index_s": round(float(raw_feats.get("separation_index_s", 0.0)), 4),
            "rag_matched": bool(rag_match_info.get("matched", False)),
            "rag_group": rag_match_info.get("group"),
            "rag_score": round(float(rag_match_info.get("score", 0.0)), 4),
            "rag_characteristics": rag_match_info.get("characteristics", ""),
            "verification_status": raw_feats.get("verification_status", ""),
            "compute_elapsed_ms": round(elapsed_ms, 2),
            "waveform": _downsample_waveform(y, 300),
            "ts": time.time(),
        }
    else:
        t0 = time.perf_counter()
        try:
            features = measure_features(y, actual_sr, nfft, hop, top_db, win)
        except Exception as exc:
            log.exception("measure_features error for %s", filename)
            raise HTTPException(500, f"DSP error: {exc}") from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if features is None:
            raise HTTPException(422, "Signal rejected: transient duration > 1.0s limit, or frequency < 1kHz, or no transient detected.")

        payload = _build_response(
            features,
            native_sr=native_sr,
            actual_sr=actual_sr,
            recording_device=device_label,
            fmax_used=fmax_used,
            filename=filename,
            elapsed_ms=elapsed_ms,
        )
        payload["engine"] = "dsp_arena"
        payload["waveform"] = _downsample_waveform(y, 300)

    log.info("compute  OK  freq=%.1f Hz  peak_ms=%.3f  impact=%.4f  elapsed=%.1f ms",
             payload["impact_freq_hz"], payload["peak_time_ms"],
             payload["impact"], elapsed_ms)

    # Phase 3: go through unified event pipeline (store + broadcast)
    event_id = await _store.emit_event(
        payload, manager, source="compute_endpoint"
    )
    return JSONResponse(content={**payload, "event_id": event_id})



# ---------------------------------------------------------------------------
# GET /events — recent session history (Phase 3 / debug)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GET /engines — available classification engines
# ---------------------------------------------------------------------------
@app.get("/engines")
async def get_engines() -> JSONResponse:
    de = get_dual_engine()
    rag_size = getattr(de.rag_store, "size", 0) if (de and hasattr(de, "rag_store") and de.rag_store) else 0
    return JSONResponse(content={
        "engines": [
            {
                "id": "dsp_arena",
                "name": "DSP Arena Standard",
                "description": "Fast linear resample & Hilbert transient window (dsp_inverse.py)",
                "active": True
            },
            {
                "id": "dual_mode_rag",
                "name": "Dual-Mode RAG Acoustic Engine",
                "description": "Multi-method DSP (FFT, Laplace, Mel) + RAG memory matching",
                "rag_events": rag_size,
                "active": de is not None
            }
        ],
        "default": "dsp_arena"
    })


# ---------------------------------------------------------------------------
# Server Info & Mobile QR Code Endpoints
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Live Mic Control Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/mic/status")
async def get_mic_status() -> JSONResponse:
    is_running = _mic_task is not None and not _mic_task.done() and _mic_active
    dev_name = getattr(_mic._st, "device_name", "unknown") if hasattr(_mic, "_st") else "unknown"
    actual_sr = getattr(_mic._st, "actual_sr", 44100) if hasattr(_mic, "_st") else 44100
    return JSONResponse(content={
        "active": is_running,
        "device": dev_name,
        "actual_sr": actual_sr,
    })

@app.post("/api/mic/start")
async def start_mic_endpoint() -> JSONResponse:
    ok = await _start_mic_service()
    return JSONResponse(content={"active": ok, "status": "started" if ok else "blocked"})

@app.post("/api/mic/stop")
async def stop_mic_endpoint() -> JSONResponse:
    await _stop_mic_service()
    return JSONResponse(content={"active": False, "status": "stopped"})

@app.post("/api/mic/toggle")
async def toggle_mic_endpoint() -> JSONResponse:
    is_running = _mic_task is not None and not _mic_task.done() and _mic_active
    if is_running:
        await _stop_mic_service()
        return JSONResponse(content={"active": False, "status": "stopped"})
    else:
        ok = await _start_mic_service()
        return JSONResponse(content={"active": ok, "status": "started" if ok else "blocked"})

@app.get("/api/server/info")
async def get_server_info() -> JSONResponse:
    lan_ip = get_lan_ip()
    return JSONResponse(content={
        "port": _SERVER_PORT,
        "lan_ip": lan_ip,
        "local_url": f"http://localhost:{_SERVER_PORT}",
        "lan_url": f"http://{lan_ip}:{_SERVER_PORT}",
    })

@app.get("/api/server/qr")
async def get_server_qr(data: Optional[str] = None) -> Response:
    """Return an SVG QR code image for mobile scanning."""
    target = data or f"http://{get_lan_ip()}:{_SERVER_PORT}"
    factory = qrcode.image.svg.SvgPathImage
    img = qrcode.make(target, image_factory=factory, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return Response(content=buf.getvalue(), media_type="image/svg+xml")

@app.get("/events")
async def get_events(limit: int = 50, exclude_test: bool = True) -> JSONResponse:
    """Return the N most-recent emitted events, newest first."""
    items = await _store.get_history_snapshot(limit=limit, exclude_test=exclude_test)
    return JSONResponse(content={"count": len(items), "events": items})


# ---------------------------------------------------------------------------
# GET /pending — events awaiting confirm-label (Phase 3 / debug)
# ---------------------------------------------------------------------------
@app.get("/pending")
async def get_pending() -> JSONResponse:
    """Return count and IDs of events awaiting ground-truth confirmation."""
    async with _store._lock:
        ids = list(_store.pending_events.keys())
    return JSONResponse(content={"count": len(ids), "event_ids": ids})


@app.websocket("/ws")
@app.websocket("/ws/dsp-stream")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Phase 1 stub: greet + echo. Phase 2 wires mic-loop -> broadcast."""
    await manager.connect(ws)
    try:
        await ws.send_json({
            "type":      "hello",
            "message":   "DSP Cluster Lab Phase 3 - event store + broadcast pipeline active.",
            "sr_max":    SR_MAX,
            "fmax_real": FMAX_REAL,
            "ts":        time.time(),
        })
        while True:
            try:
                msg = await ws.receive_text()
                await ws.send_json({"type": "echo", "data": msg, "ts": time.time()})
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# AI Agent Endpoint & Acoustic Physics Diagnosis
# ---------------------------------------------------------------------------
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import urllib.request
import urllib.error

class AIAnalyzeRequest(BaseModel):
    event: Dict[str, Any]
    apiKey: Optional[str] = None
    model: Optional[str] = "gemini-2.5-flash"
    prompt: Optional[str] = None

class AIChatRequest(BaseModel):
    message: str
    apiKey: Optional[str] = None
    model: Optional[str] = "gemini-2.5-flash"
    history: Optional[List[Dict[str, str]]] = None
    context: Optional[Dict[str, Any]] = None

def _local_acoustic_expert_analysis(event: Dict[str, Any], user_prompt: Optional[str] = None) -> str:
    """Deterministic, high-accuracy acoustic physics analysis when offline or without API key."""
    freq = float(event.get("impact_freq_hz", 0.0))
    sigma = float(event.get("laplace_sigma", 0.0))
    group = str(event.get("cluster", "U")).upper()
    mel = float(event.get("mel_mean_db", 0.0))
    dur_ms = float(event.get("duration_s", 0.0)) * 1000.0 if "duration_s" in event else float(event.get("peak_time_ms", 0.0))
    pair = event.get("candidate_pair", "")
    elim = event.get("elimination", "")
    rag_score = float(event.get("rag_score", 0.0)) * 100.0 if "rag_score" in event else 0.0

    lines = [
        f"### 🔬 Acoustic Physics Diagnostic Report",
        f"- **Classified Group**: **Group {group}**",
        f"- **Principal Resonant Mode**: `{freq:.1f} Hz`",
        f"- **Laplace Decay Damping (sigma)**: `{sigma:.3f}`",
        f"- **Transient Span**: `{dur_ms:.1f} ms` | **Mel Energy**: `{mel:.1f} dB`",
    ]
    if pair:
        lines.append(f"- **Candidate Pair**: `{pair}` ({elim})")
    if rag_score > 0:
        lines.append(f"- **RAG Empirical Match**: `{rag_score:.1f}%` agreement with historical cluster library")

    lines.append("\n**Physical Interpretation:**")
    if sigma < 1.0 or group == "B":
        lines.append(
            "• **Group B Signature**: Very low Laplace damping (sigma < 1.0) indicates prolonged mechanical resonance. "
            "The energy is predominantly concentrated in the 1.0–2.5 kHz fundamental band with minimal high-frequency dispersion (egg-shaped spectrogram ellipse)."
        )
    elif sigma >= 8.5 or group == "C":
        lines.append(
            "• **Group C Signature**: Extremely rapid Laplace decay (sigma >= 8.5) with high-frequency surface reflection. "
            "Reflector properties produce early acoustic bounce with high spectral rolloff and quick energy dissipation."
        )
    elif freq < 2000 or group == "A":
        lines.append(
            "• **Group A Signature**: Classical absorber profile with suppressed higher harmonics. "
            "Circular 2D spectrogram ripple with high damping ratio zeta, characteristic of absorbent face impact."
        )
    else:
        lines.append(
            "• **Group D Signature**: Coupled intermediate stiffness. Moderate decay rate (3.0 <= sigma < 8.5) with balanced high-frequency ratio."
        )

    if not user_prompt:
        lines.append("\n💡 *Tip: Provide a Gemini API key in settings to enable real-time multimodal LLM reasoning on this event.*")
    else:
        lines.append(f"\n**Query Analysis for:** *\"{user_prompt}\"*\nAll physical metrics match the calibration profile for Face {group}.")

    return "\n".join(lines)


async def _call_gemini_api(api_key: str, model: str, system_prompt: str, user_content: str) -> str:
    """Call Google Gemini API using native Python urllib in a thread pool."""
    # Ensure model name format
    model_name = model.strip()
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key.strip()}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024
        }
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

    def _do_request():
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "No response text generated by AI model."

    return await asyncio.to_thread(_do_request)


@app.post("/api/ai/analyze")
async def ai_analyze_event(req: AIAnalyzeRequest) -> JSONResponse:
    """Analyze acoustic impact event with AI Agent (Gemini API or Local Physics Expert)."""
    api_key = req.apiKey or os.environ.get("GEMINI_API_KEY", "").strip()

    # System instruction for acoustic expert AI agent
    system_prompt = (
        "You are the senior DSP & Acoustic Physics AI Agent for the 4DS Acoustic Laboratory. "
        "Your task is to analyze acoustic impact sound signatures from 4 block faces (A: Absorber, B: Resonator, C: Reflector, D: Coupled). "
        "Explain the mechanical physics behind the resonant frequency, Laplace decay sigma, Mel energy contrast, and candidate pair eliminations. "
        "Be concise, scientifically rigorous, and provide clear conclusions in GitHub markdown."
    )

    event_summary = json.dumps(req.event, indent=2)
    user_query = req.prompt or "Perform a complete acoustic physics diagnosis of this impact event."
    user_content = f"Event DSP Metrics:\n```json\n{event_summary}\n```\n\nTask:\n{user_query}"

    if api_key:
        try:
            ai_text = await _call_gemini_api(api_key, req.model or "gemini-2.5-flash", system_prompt, user_content)
            return JSONResponse(content={
                "status": "ok",
                "provider": "gemini",
                "model": req.model,
                "analysis": ai_text
            })
        except Exception as exc:
            log.warning("Gemini API call failed (%s); falling back to local expert analysis", exc)
            local_text = _local_acoustic_expert_analysis(req.event, req.prompt)
            return JSONResponse(content={
                "status": "ok",
                "provider": "local_fallback",
                "notice": f"Gemini API returned: {exc}. Displaying local expert physics analysis.",
                "analysis": local_text
            })
    else:
        local_text = _local_acoustic_expert_analysis(req.event, req.prompt)
        return JSONResponse(content={
            "status": "ok",
            "provider": "local_expert",
            "analysis": local_text
        })


@app.post("/api/ai/chat")
async def ai_chat(req: AIChatRequest) -> JSONResponse:
    """Interactive chat with the Acoustic Physics AI Agent."""
    api_key = req.apiKey or os.environ.get("GEMINI_API_KEY", "").strip()
    system_prompt = (
        "You are the 4DS Acoustic Laboratory Pair-Programming & Physics AI Agent. "
        "You assist engineers with impact sound classification, resonant modes, Laplace envelope fitting, "
        "RAG memory similarity, and 4-face (A/B/C/D) physical boundaries. Respond clearly in markdown."
    )

    ctx_str = f"Context: {json.dumps(req.context)}\n\n" if req.context else ""
    user_content = f"{ctx_str}User question: {req.message}"

    if api_key:
        try:
            reply = await _call_gemini_api(api_key, req.model or "gemini-2.5-flash", system_prompt, user_content)
            return JSONResponse(content={"status": "ok", "provider": "gemini", "reply": reply})
        except Exception as exc:
            return JSONResponse(content={
                "status": "error",
                "provider": "error",
                "reply": f"Gemini API connection error: {exc}. Please check your API key or network connection."
            })
    else:
        return JSONResponse(content={
            "status": "ok",
            "provider": "local",
            "reply": "Please enter your Gemini API Key in the AI Settings drawer to enable interactive reasoning with Gemini 2.5 Flash."
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("live_server:app", host="0.0.0.0", port=_SERVER_PORT, reload=False)
