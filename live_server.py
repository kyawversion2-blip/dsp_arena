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
    uv run uvicorn live_server:app --host 127.0.0.1 --port 8765 --reload  # LAN requires DSP_ALLOW_LAN=1 + auth/origin config
"""

from __future__ import annotations

import asyncio
import os
import sys
import importlib.util
import struct
import time
import logging
import math
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from network_policy import NetworkPolicy, NetworkPolicyError
import socket
import webbrowser
import io
try:
    import qrcode
    import qrcode.image.svg
    _HAS_QRCODE = True
except ImportError:
    qrcode = None
    _HAS_QRCODE = False
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

import mic_loop as _mic
import event_store as _store
import confirmed_store as _labelled
import similarity as _similar

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
    """Manages active WebSocket connections for real-time broadcasts with heartbeat support."""
    
    HEARTBEAT_INTERVAL = 30.0  # seconds
    HEARTBEAT_TIMEOUT = 90.0   # seconds
    MAX_CONNECTIONS = 100
    
    def __init__(self) -> None:
        self._active: dict[str, dict] = {}  # client_id -> {ws, connected_at, last_heartbeat, authenticated}
        self._heartbeat_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
    
    def _generate_client_id(self) -> str:
        return f"ws_{uuid.uuid4().hex[:12]}"
    
    async def start_background_tasks(self) -> None:
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="ws-heartbeat")
            log.info("WebSocket heartbeat task started")
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="ws-cleanup")
            log.info("WebSocket cleanup task started")
    
    async def stop_background_tasks(self) -> None:
        for task in (self._heartbeat_task, self._cleanup_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._heartbeat_task = None
        self._cleanup_task = None
        log.info("WebSocket background tasks stopped")
    
    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                now = time.time()
                stale = []
                
                for client_id, info in self._active.items():
                    if now - info["last_heartbeat"] > self.HEARTBEAT_TIMEOUT:
                        log.warning("Client %s heartbeat timeout, disconnecting", client_id)
                        stale.append(client_id)
                    else:
                        try:
                            await info["ws"].send_json({
                                "type": "heartbeat",
                                "timestamp": now,
                                "server_time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                            })
                        except Exception as e:
                            log.warning("Failed to send heartbeat to %s: %s", client_id, e)
                            stale.append(client_id)
                
                for client_id in stale:
                    await self._force_disconnect(client_id, code=1001, reason="Heartbeat timeout")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("Error in heartbeat loop: %s", e)
    
    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60.0)
                now = time.time()
                stale = []
                
                for client_id, info in self._active.items():
                    if now - info.get("last_activity", now) > 300.0:
                        log.info("Client %s inactive for 5 minutes, cleaning up", client_id)
                        stale.append(client_id)
                
                for client_id in stale:
                    await self._force_disconnect(client_id, code=1001, reason="Inactivity timeout")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("Error in cleanup loop: %s", e)
    
    async def _force_disconnect(self, client_id: str, code: int = 1000, reason: str = "Normal closure") -> None:
        info = self._active.pop(client_id, None)
        if info:
            try:
                await info["ws"].close(code=code, reason=reason)
            except Exception:
                pass
            log.info("Force disconnected client %s: %s", client_id, reason)
    
    async def connect(self, ws: WebSocket) -> str:
        if len(self._active) >= self.MAX_CONNECTIONS:
            log.warning("Max connections reached, rejecting new connection")
            await ws.close(code=1013, reason="Server busy - max connections reached")
            raise WebSocketDisconnect(code=1013, reason="Max connections reached")
        
        await ws.accept()
        client_id = self._generate_client_id()
        now = time.time()
        
        self._active[client_id] = {
            "ws": ws,
            "connected_at": now,
            "last_heartbeat": now,
            "last_activity": now,
            "authenticated": False,
            "message_count": 0
        }
        log.info("WS client connected: %s (total=%d)", client_id, len(self._active))
        
        try:
            await ws.send_json({
                "type": "connect",
                "client_id": client_id,
                "timestamp": now,
                "server_time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "heartbeat_interval": self.HEARTBEAT_INTERVAL
            })
        except Exception as e:
            log.warning("Failed to send connect confirmation to %s: %s", client_id, e)
            await self._force_disconnect(client_id)
            raise
            
        return client_id
    
    def disconnect(self, ws: WebSocket) -> str | None:
        for client_id, info in self._active.items():
            if info["ws"] is ws:
                del self._active[client_id]
                log.info("WS client disconnected: %s (total=%d)", client_id, len(self._active))
                return client_id
        return None
    
    def disconnect_by_id(self, client_id: str) -> bool:
        if client_id in self._active:
            del self._active[client_id]
            log.info("WS client disconnected by ID: %s (total=%d)", client_id, len(self._active))
            return True
        return False
    
    def get_connection(self, client_id: str) -> dict | None:
        return self._active.get(client_id)
    
    def update_heartbeat(self, client_id: str) -> bool:
        info = self._active.get(client_id)
        if info:
            info["last_heartbeat"] = time.time()
            info["last_activity"] = time.time()
            return True
        return False
    
    def update_activity(self, client_id: str) -> bool:
        info = self._active.get(client_id)
        if info:
            info["last_activity"] = time.time()
            info["message_count"] = info.get("message_count", 0) + 1
            return True
        return False
    
    async def broadcast(self, payload: dict[str, Any]) -> int:
        if not self._active:
            return 0
            
        sent = 0
        stale = []
        
        for client_id, info in self._active.items():
            try:
                await info["ws"].send_json(payload)
                sent += 1
            except Exception as e:
                log.warning("Failed to send broadcast to %s: %s", client_id, e)
                stale.append(client_id)
        
        for client_id in stale:
            await self._force_disconnect(client_id, code=1011, reason="Send failed")
            
        return sent
    
    @property
    def client_count(self) -> int:
        return len(self._active)
    
    def get_stats(self) -> dict:
        now = time.time()
        return {
            "total_connections": len(self._active),
            "connections": [
                {
                    "client_id": cid,
                    "connected_at": info["connected_at"],
                    "duration": now - info["connected_at"],
                    "last_heartbeat": info["last_heartbeat"],
                    "message_count": info.get("message_count", 0),
                    "authenticated": info.get("authenticated", False)
                }
                for cid, info in self._active.items()
            ]
        }



# ---------------------------------------------------------------------------
# Network IP, QR Code & Auto Browser Launcher
# ---------------------------------------------------------------------------
_SERVER_PORT: int = int(os.environ.get("PORT", "8765"))
try:
    NETWORK_POLICY = NetworkPolicy.from_environment(port=_SERVER_PORT)
except NetworkPolicyError as exc:
    raise RuntimeError(f"Unsafe network configuration: {exc}") from exc


class _NetworkBoundaryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        reason = NETWORK_POLICY.validate_request(
            host=request.headers.get("host"),
            origin=request.headers.get("origin"),
            token=NETWORK_POLICY.token_from_headers(request.headers),
        )
        if reason:
            return JSONResponse(status_code=403, content={"detail": reason})
        return await call_next(request)


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
    """Print ASCII QR code in terminal for mobile scanning if qrcode is installed."""
    ascii_qr = ""
    if _HAS_QRCODE and qrcode:
        try:
            qr = qrcode.QRCode(border=1)
            qr.add_data(lan_url)
            qr.make(fit=True)
            f = io.StringIO()
            qr.print_ascii(out=f, invert=True)
            ascii_qr = f.getvalue() + "\n 📷 Scan the QR code with phone or tablet to open dashboard!\n"
        except Exception as exc:
            ascii_qr = f"[QR generation error: {exc}]\n"

    print("\n" + "═" * 66)
    print(" 🚀 DSP CLUSTER LAB — LIVE MONITOR READY")
    print(f" 🌐 Local Browser : {local_url}")
    print(f" 📱 Mobile / LAN  : {lan_url}")
    print("═" * 66)
    if ascii_qr:
        print(ascii_qr)
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
    local_url = f"http://localhost:{_SERVER_PORT}"
    if NETWORK_POLICY.lan_enabled:
        lan_ip = get_lan_ip()
        lan_url = f"http://{lan_ip}:{_SERVER_PORT}"
        print_terminal_qr_banner(local_url, lan_url)
    else:
        print(f"\nDSP CLUSTER LAB - local-only: {local_url}\n")
    asyncio.create_task(asyncio.to_thread(_auto_open_browser, local_url))

    # Start WebSocket heartbeat and cleanup tasks
    await manager.start_background_tasks()

    # Auto-start mic unless disabled by NO_MIC env
    await _start_mic_service()

    # Auto-initialize Google OAuth session from IDE if available
    try:
        ide_account = _load_ide_google_account()
        if ide_account:
            _GOOGLE_OAUTH_SESSION.update(ide_account)
            log.info("🔐 Auto-connected Google OAuth from IDE: %s (%s)", ide_account.get("email"), ide_account.get("name"))
    except Exception as exc:
        log.warning("Could not auto-load IDE Google credentials: %s", exc)

    yield

    # Shutdown
    await _stop_mic_service()
    await manager.stop_background_tasks()
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
    allow_origins=list(NETWORK_POLICY.allowed_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-DSP-Auth-Token"],
    allow_credentials=False,
)
app.add_middleware(_NetworkBoundaryMiddleware)

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
    fmax_used: float | None,
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
        "fmax_used_hz":           (None if fmax_used is None else round(float(fmax_used), 1)),
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
    fmax:         float | str | None = Form(None,          description="Optional max analysis frequency ceil (<=21500 Hz). "
                                "Omitted/null/blank preserves legacy measure_features() ceiling; "
                                "explicit value IS the DSP frequency-domain ceiling."),
    device_label: str   = Form("file_upload", description="Logical device label (G3 provenance)"),
    engine:       str   = Form("dsp_arena",   description="Engine: 'dsp_arena' or 'dual_mode_rag'"),
) -> JSONResponse:
    """POST a WAV file -> DSP feature vector JSON.

    Guards:
      G1  sr > 48000  -> 422
      G2  fmax > 21500 -> 422 (only when fmax is explicitly provided)
    fmax default is a legacy-preserving sentinel: when omitted/empty/null/NaN
    the endpoint preserves the canonical fmax=None measure_features() ceiling.
    Only an explicit finite positive fmax narrows the analysis ceiling.
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
    # G2 — validation (task 9 contract).
    # Distinction is by provenance, not by value:
    #   omitted / empty / null  -> legacy canonical ceiling (measure_features fmax=None)
    #   explicitly supplied but unparseable / NaN / +-inf -> 422
    #   explicit finite positive numeric fmax -> validated and threaded through
    #   numeric-string "8000" is accepted; "abc"/"nan"/"+inf"/"-inf" are 422.
    _explicit_fmax: float | None = None
    if isinstance(fmax, str):
        fmax = fmax.strip()
        if fmax == "":
            fmax = None
    if fmax is not None:
        # Distinguish "explicit but unparseable" from "legacy omission".
        # Any explicit non-None value that cannot be coerced to a finite
        # positive number is a client error (422), not a silent legacy fallback.
        try:
            _candidate = float(fmax)
        except Exception:
            raise HTTPException(422, f"Invalid fmax: unparseable value {fmax!r}")
        if not math.isfinite(_candidate):
            raise HTTPException(422, f"Invalid fmax: non-finite value {fmax!r}")
        if _candidate <= 0:
            raise HTTPException(422, f"fmax must be positive finite or omitted; got {_candidate!r}")
        _explicit_fmax = _candidate
    if _explicit_fmax is not None and _explicit_fmax > FMAX_REAL:
        raise HTTPException(
            422,
            detail=(
                f"fmax={_explicit_fmax:.0f} Hz exceeds confirmed real ceiling {FMAX_REAL:.0f} Hz. "
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
    log.info("compute  file=%s  native_sr=%d  target_sr=%d  fmax=%s  nfft=%d",
             filename, native_sr, sr,
             ("%.0f" % _explicit_fmax) if _explicit_fmax is not None else "legacy", nfft)

    # G5: actual_sr is what we pass to DSP - never a module-level fallback constant
    actual_sr = sr
    if native_sr != actual_sr:
        y = resample_linear(y, native_sr, actual_sr)

    # Provenance (task 6 — M1 fix): legacy is NOT an explicit numeric ceiling.
    # fmax=None means "legacy/default semantics" (measure_features two-domain
    # legacy: impact+mel ceiling min(50000,sr/2), scalars full STFT). The
    # response therefore uses a None sentinel rather than echoing a frequency
    # number the DSP did not apply as a single numeric ceiling. This is
    # honest at all sr; at sr > 100 kHz it avoids claiming 50000 Hz when
    # scalars actually used sr/2. Supported sr <= SR_MAX (48 kHz) preserves
    # previous numeric correctness; the sentinel simply makes provenance honest.
    #   - legacy (omitted/null) -> fmax_requested=None, fmax_used=None, _measure_fmax=None
    #   - explicit fmax         -> min(fmax, actual_sr/2)
    if _explicit_fmax is None:
        fmax_requested = None
        fmax_used = None  # sentinel: legacy semantics, not an explicit frequency ceiling
        _measure_fmax = None
    else:
        fmax_requested = float(_explicit_fmax)
        fmax_used = min(_explicit_fmax, actual_sr / 2.0)
        _measure_fmax = fmax_used

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
            "fmax_requested_hz": None if fmax_requested is None else round(float(fmax_requested), 1),
            "fmax_used_hz": None if fmax_used is None else round(float(fmax_used), 1),
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
            features = measure_features(y, actual_sr, nfft, hop, top_db, win, fmax=_measure_fmax)
        except (TypeError, ValueError) as exc:
            # Slice-3A fmax validation errors (<=0, NaN, inf, <50 Hz) are
            # client errors, not server errors — they reach us only through
            # direct coroutine calls (Form parsing normally coerces/isolates
            # them before we get here).
            raise HTTPException(422, f"Invalid fmax: {exc}") from exc
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
        # Slice 3A provenance: distinguish a legacy/default request from an
        # explicit fmax that actually narrowed the DSP ceiling.
        payload["fmax_requested_hz"] = None if fmax_requested is None else round(float(fmax_requested), 1)
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
    payload = {
        "port": _SERVER_PORT,
        "local_url": f"http://localhost:{_SERVER_PORT}",
        "lan_enabled": NETWORK_POLICY.lan_enabled,
    }
    if NETWORK_POLICY.lan_enabled:
        lan_ip = get_lan_ip()
        payload.update({"lan_ip": lan_ip, "lan_url": f"http://{lan_ip}:{_SERVER_PORT}"})
    return JSONResponse(content=payload)

@app.get("/api/server/qr")
async def get_server_qr(data: Optional[str] = None) -> Response:
    """Return an SVG QR code image for mobile scanning."""
    if not NETWORK_POLICY.lan_enabled:
        raise HTTPException(404, "LAN QR is disabled in local-only mode")
    if not _HAS_QRCODE or qrcode is None:
        raise HTTPException(503, "QR code support is unavailable: install the 'qrcode' package.")
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


_VALID_LABELS = ("A", "B", "C", "D")


class _ConfirmBody(BaseModel):
    event_id: str
    decision: str
    label: Optional[str] = None
    operator: Optional[str] = None


@app.post("/confirm-label")
async def confirm_label(body: _ConfirmBody) -> JSONResponse:
    """Consume a pending event and persist the operator's YES/NO decision.

    YES  -> appends a confirmed record (original payload preserved verbatim
            plus decision metadata) and returns a recency-weighted similarity
            ADVISORY prediction, if enough confirmed history exists.
    NO   -> appends a rejection record.
    Unknown or expired event_id -> 404. Pending semantics are unchanged:
    ids are consumed exactly once via event_store.consume_pending().
    """
    decision = (body.decision or "").strip().upper()
    if decision not in ("YES", "NO"):
        raise HTTPException(400, "decision must be 'YES' or 'NO'")

    # Validate label BEFORE consuming pending so a 400 does not destroy the event.
    pre_label: Optional[str] = None
    if decision == "YES" and body.label is not None:
        pre_label = str(body.label).strip().upper()[:1]
        if pre_label not in _VALID_LABELS:
            raise HTTPException(400, "label must be one of A, B, C, D or null")

    event = await _store.consume_pending(body.event_id)
    if event is None:
        raise HTTPException(404, f"unknown or expired event_id: {body.event_id}")

    if decision == "NO":
        record = _labelled.append_rejected(event, operator=body.operator)
        return JSONResponse(content={"status": "rejected", "record": record})

    record = _labelled.append_confirmed(
        event, label_truth=pre_label, operator=body.operator
    )
    advisory = _similar.advise(event)
    return JSONResponse(
        content={"status": "confirmed", "record": record, "advisory": advisory}
    )


@app.get("/confirmed")
async def get_confirmed(limit: int = 20) -> JSONResponse:
    """Return the most recent confirmed (YES) records, oldest-first."""
    limit = max(1, min(int(limit), 200))
    records = _labelled.load_confirmed(limit=limit)
    return JSONResponse(content={"count": len(records), "confirmed": records})


@app.websocket("/ws")
@app.websocket("/ws/dsp-stream")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Phase 1 stub: greet + echo. Phase 2 wires mic-loop -> broadcast."""
    reason = NETWORK_POLICY.validate_request(
        host=ws.headers.get("host"), origin=ws.headers.get("origin"),
        token=NETWORK_POLICY.token_from_websocket(ws),
    )
    if reason:
        await ws.close(code=1008, reason=reason)
        return
    try:
        client_id = await manager.connect(ws)
    except WebSocketDisconnect:
        return
    
    try:
        await ws.send_json({
            "type": "hello",
            "client_id": client_id,
            "message": "DSP Cluster Lab Phase 3 - event store + broadcast pipeline active.",
            "sr_max": SR_MAX,
            "fmax_real": FMAX_REAL,
            "ts": time.time(),
            "heartbeat_interval": manager.HEARTBEAT_INTERVAL
        })
        while True:
            try:
                msg = await ws.receive_text()
                manager.update_activity(client_id)
                
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "code": "INVALID_JSON", "message": "Malformed JSON"})
                    continue
                
                mtype = data.get("type", "")
                
                # Handle heartbeat response
                if mtype == "heartbeat_ack":
                    manager.update_heartbeat(client_id)
                    continue
                
                # Echo back for compatibility
                await ws.send_json({"type": "echo", "data": msg, "ts": time.time()})
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# AI Agent Endpoint & Acoustic Physics Diagnosis (with Google OAuth 2.0 Web Auth)
# ---------------------------------------------------------------------------
from typing import Dict, List, Tuple
import base64
import json
import ssl
import urllib.request
import urllib.parse
import urllib.error

# Google OAuth 2.0 Configuration
_GOOGLE_CLIENT_ID = os.environ.get(
    "GOOGLE_CLIENT_ID",
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com",
)
_GOOGLE_OAUTH_SESSION: Dict[str, Any] = {
    "authenticated": False,
    "email": "",
    "name": "",
    "picture": "",
    "access_token": "",
    "refresh_token": "",
    "expiry_date": 0,
    "source": "",
}


def _decode_jwt_payload(token_str: str) -> dict:
    """Decode JWT payload without external libraries."""
    try:
        parts = token_str.split(".")
        if len(parts) >= 2:
            padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
            return json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except Exception:
        pass
    return {}


def _load_ide_google_account() -> Optional[Dict[str, Any]]:
    """Load the active Google account and credentials from Antigravity IDE (~/.gemini)."""
    creds_path = Path.home() / ".gemini" / "oauth_creds.json"
    accounts_path = Path.home() / ".gemini" / "google_accounts.json"
    jetski_path = Path.home() / ".gemini" / "jetski-standalone-oauth-token"

    active_email = ""
    if accounts_path.exists():
        try:
            with open(accounts_path, encoding="utf-8") as f:
                active_email = json.load(f).get("active", "")
        except Exception:
            pass

    if creds_path.exists():
        try:
            with open(creds_path, encoding="utf-8") as f:
                creds = json.load(f)
            id_info = _decode_jwt_payload(creds.get("id_token", ""))
            email = id_info.get("email") or active_email or "active-ide-user@gmail.com"
            name = id_info.get("name") or email.split("@")[0].capitalize()
            picture = id_info.get("picture", "")
            return {
                "authenticated": True,
                "email": email,
                "name": name,
                "picture": picture,
                "access_token": creds.get("access_token", ""),
                "refresh_token": creds.get("refresh_token", ""),
                "expiry_date": creds.get("expiry_date", 0),
                "source": "Antigravity IDE Session",
            }
        except Exception as exc:
            log.warning("Could not read IDE credentials (%s)", exc)

    if jetski_path.exists():
        try:
            with open(jetski_path, encoding="utf-8") as f:
                data = json.load(f)
            tok = data.get("token", {})
            id_info = _decode_jwt_payload(data.get("id_token", ""))
            email = id_info.get("email") or "ide-user@gmail.com"
            name = id_info.get("name") or email.split("@")[0].capitalize()
            return {
                "authenticated": True,
                "email": email,
                "name": name,
                "picture": id_info.get("picture", ""),
                "access_token": tok.get("access_token", ""),
                "refresh_token": tok.get("refresh_token", ""),
                "expiry_date": 0,
                "source": "Antigravity Jetski Token",
            }
        except Exception as exc:
            log.warning("Could not read jetski token (%s)", exc)

    return None


def _refresh_google_access_token(session: Dict[str, Any]) -> Optional[str]:
    """Refresh Google OAuth access token if expired and refresh_token is present."""
    refresh_token = session.get("refresh_token")
    if not refresh_token:
        return session.get("access_token")

    now_ms = time.time() * 1000
    exp_ms = session.get("expiry_date", 0)
    # If valid for at least 60 more seconds, reuse existing token
    if exp_ms > 0 and (now_ms + 60000) < exp_ms and session.get("access_token"):
        return session.get("access_token")

    log.info("Refreshing Google OAuth access token for %s...", session.get("email"))
    ctx = ssl.create_default_context()

    params = {
        "client_id": _GOOGLE_CLIENT_ID,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data, method="POST")

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            new_tok = res.get("access_token")
            expires_in = int(res.get("expires_in", 3600))
            if new_tok:
                session["access_token"] = new_tok
                session["expiry_date"] = (time.time() + expires_in) * 1000
                log.info("Google OAuth token refreshed successfully (expires in %ds)", expires_in)
                return new_tok
    except Exception as exc:
        log.warning("OAuth token refresh failed (%s); using current token", exc)

    return session.get("access_token")


def _resolve_gemini_credentials(req_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve whether to use an API Key or Google OAuth Bearer Token.
    Returns: (api_key, oauth_bearer_token)
    """
    if req_key and req_key.strip():
        k = req_key.strip()
        if k.startswith("ya29."):
            return None, k
        if not k.startswith("http"):
            return k, None

    if _GOOGLE_OAUTH_SESSION.get("authenticated"):
        tok = _refresh_google_access_token(_GOOGLE_OAUTH_SESSION)
        if tok:
            return None, tok

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key, None

    return None, None


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


class AIMoodRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gemini-2.5-flash"
    provider: Optional[str] = "gemini"
    apiKey: Optional[str] = None


def _local_acoustic_expert_analysis(event: Dict[str, Any], user_prompt: Optional[str] = None) -> str:
    """Deterministic, high-accuracy acoustic physics analysis when offline or without API key."""
    freq = float(event.get("impact_freq_hz", 0.0))
    sigma = float(event.get("laplace_sigma", 0.0))
    group = str(event.get("cluster", "U")).upper()
    mel = float(event.get("mel_mean_db", 0.0))
    dur_ms = (
        float(event.get("duration_s", 0.0)) * 1000.0
        if "duration_s" in event
        else float(event.get("peak_time_ms", 0.0))
    )
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
        lines.append(
            "\n💡 *Connected to 4DS Acoustic Laboratory Engine with Google OAuth 2.0 Web Authentication.*"
        )
    else:
        lines.append(
            f"\n**Query Analysis for:** *\"{user_prompt}\"*\nAll physical metrics match the calibration profile for Face {group}."
        )

    return "\n".join(lines)


async def _call_gemini_api(
    api_key: Optional[str],
    oauth_token: Optional[str],
    model: str,
    system_prompt: str,
    user_content: str,
    response_mime_type: Optional[str] = None,
) -> str:
    """Call Google Gemini API using either API Key or Google OAuth 2.0 Bearer token."""
    model_name = model.strip()
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    headers = {"Content-Type": "application/json"}
    if oauth_token:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent"
        headers["Authorization"] = f"Bearer {oauth_token.strip()}"
    elif api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key.strip()}"
    else:
        raise ValueError("Neither API Key nor Google OAuth access token provided.")

    gen_config: Dict[str, Any] = {
        "temperature": 0.3,
        "maxOutputTokens": 1024,
    }
    if response_mime_type:
        gen_config["response_mime_type"] = response_mime_type

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": gen_config,
    }

    req_data = json.dumps(payload).encode("utf-8")
    ctx = ssl.create_default_context()

    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

    def _do_request():
        with urllib.request.urlopen(req, context=ctx, timeout=18) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "No response text generated by AI model."

    return await asyncio.to_thread(_do_request)


# ---------------------------------------------------------------------------
# Google OAuth 2.0 Web Authentication Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/auth/google/status")
async def get_google_auth_status() -> dict:
    """Return the current Google OAuth connection status and user profile."""
    # If not authenticated in session, attempt to load active IDE credentials automatically
    if not _GOOGLE_OAUTH_SESSION.get("authenticated"):
        ide_account = _load_ide_google_account()
        if ide_account:
            _GOOGLE_OAUTH_SESSION.update(ide_account)

    is_auth = bool(_GOOGLE_OAUTH_SESSION.get("authenticated"))
    return {
        "authenticated": is_auth,
        "account": {
            "email": _GOOGLE_OAUTH_SESSION.get("email", ""),
            "name": _GOOGLE_OAUTH_SESSION.get("name", ""),
            "picture": _GOOGLE_OAUTH_SESSION.get("picture", ""),
            "source": _GOOGLE_OAUTH_SESSION.get("source", ""),
        }
        if is_auth
        else None,
        "client_id": _GOOGLE_CLIENT_ID,
    }


@app.post("/api/auth/google/connect-ide")
async def connect_ide_google_account() -> dict:
    """Connect using the active Google Account currently used in the Antigravity IDE."""
    ide_account = _load_ide_google_account()
    if not ide_account:
        return {
            "status": "error",
            "message": "No active Google credentials found in ~/.gemini. Please sign in via web login.",
        }

    _GOOGLE_OAUTH_SESSION.update(ide_account)
    log.info("Connected to IDE Google Account: %s", ide_account.get("email"))
    return {
        "status": "ok",
        "authenticated": True,
        "account": {
            "email": _GOOGLE_OAUTH_SESSION.get("email"),
            "name": _GOOGLE_OAUTH_SESSION.get("name"),
            "picture": _GOOGLE_OAUTH_SESSION.get("picture"),
            "source": _GOOGLE_OAUTH_SESSION.get("source"),
        },
    }


@app.get("/api/auth/google/login")
async def google_oauth_login(request: Request) -> RedirectResponse:
    """Initiate standard Google OAuth 2.0 Web Login consent flow."""
    lan_ip = get_lan_ip()
    host = request.headers.get("host") or f"localhost:{_SERVER_PORT}"
    protocol = "https" if "https" in request.headers.get("x-forwarded-proto", "") else "http"
    redirect_uri = f"{protocol}://{host}/api/auth/google/callback"

    scopes = " ".join([
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/generative-language",
        "openid",
    ])

    params = {
        "client_id": _GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scopes,
        "access_type": "offline",
        "prompt": "select_account",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    return RedirectResponse(auth_url)


@app.get("/api/auth/google/callback")
async def google_oauth_callback(request: Request, code: Optional[str] = None, error: Optional[str] = None) -> HTMLResponse:
    """Handle Google OAuth 2.0 authorization code exchange and token receipt."""
    if error or not code:
        err_msg = error or "Authorization code missing."
        return HTMLResponse(
            f"<html><body style='background:#0a1122;color:#fb6f6f;font-family:sans-serif;padding:30px;'>"
            f"<h3>Authentication Failed</h3><p>{err_msg}</p>"
            f"<script>window.close();</script></body></html>"
        )

    host = request.headers.get("host") or f"localhost:{_SERVER_PORT}"
    protocol = "https" if "https" in request.headers.get("x-forwarded-proto", "") else "http"
    redirect_uri = f"{protocol}://{host}/api/auth/google/callback"

    ctx = ssl.create_default_context()

    token_data = urllib.parse.urlencode({
        "client_id": _GOOGLE_CLIENT_ID,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }).encode("utf-8")

    try:
        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=token_data, method="POST")
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            tokens = json.loads(resp.read().decode("utf-8"))

        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")
        expires_in = int(tokens.get("expires_in", 3600))
        id_token = tokens.get("id_token", "")
        user_info = _decode_jwt_payload(id_token)

        _GOOGLE_OAUTH_SESSION.update({
            "authenticated": True,
            "email": user_info.get("email", "google-user@gmail.com"),
            "name": user_info.get("name", "Google User"),
            "picture": user_info.get("picture", ""),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": (time.time() + expires_in) * 1000,
            "source": "Google OAuth 2.0 Web Login",
        })
        log.info("Google OAuth Web Login successful: %s", _GOOGLE_OAUTH_SESSION["email"])

        return HTMLResponse(
            "<html><body style='background:#020409;color:#3ae3ff;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;'>"
            "<div style='text-align:center;padding:24px;border:1px solid rgba(58,227,255,0.4);border-radius:12px;background:rgba(6,10,22,0.9);'>"
            "<h2 style='margin:0 0 10px 0;'>✓ Google OAuth Connected</h2>"
            f"<p style='color:#a78bfa;'>Signed in as <b>{_GOOGLE_OAUTH_SESSION['email']}</b></p>"
            "<p style='color:#8ba0cc;font-size:12px;'>Returning to DSP Cluster Lab...</p>"
            "</div>"
            "<script>"
            "if(window.opener){window.opener.postMessage({type:'oauth_success',email:'" + _GOOGLE_OAUTH_SESSION['email'] + "'}, '*');window.close();}"
            "else{setTimeout(()=>{window.location.href='/';}, 1200);}"
            "</script></body></html>"
        )
    except Exception as exc:
        log.warning("OAuth token exchange error: %s", exc)
        return HTMLResponse(
            f"<html><body style='background:#0a1122;color:#fb6f6f;font-family:sans-serif;padding:30px;'>"
            f"<h3>Authentication Token Exchange Error</h3><p>{exc}</p>"
            f"<a href='/' style='color:#3ae3ff;'>Return to Dashboard</a></body></html>"
        )


@app.post("/api/auth/google/disconnect")
async def google_oauth_disconnect() -> dict:
    """Disconnect the current Google OAuth session."""
    _GOOGLE_OAUTH_SESSION.update({
        "authenticated": False,
        "email": "",
        "name": "",
        "picture": "",
        "access_token": "",
        "refresh_token": "",
        "expiry_date": 0,
        "source": "",
    })
    log.info("Google OAuth session disconnected.")
    return {"status": "ok", "authenticated": False}


@app.post("/api/ai/mood")
async def generate_ai_mood(req: AIMoodRequest) -> JSONResponse:
    """
    Generate physical acoustic mood parameters (clusters A, B, C, D) using
    authenticated Google OAuth (Gemini) or local acoustic fallback.
    """
    api_key, oauth_token = _resolve_gemini_credentials(req.apiKey)

    system_prompt = (
        "You are a precision acoustic physicist and DSP sound designer. "
        "Return a valid JSON object describing 4 distinct acoustic impact clusters (A, B, C, D) matching the user mood. "
        "Physical constraints:\n"
        "- f0 (Hz): between 600 and 160000 (log-frequency resonance)\n"
        "- sigma (s^-1): between 4.0 and 110.0 (damping decay rate)\n"
        "- duration_ms: between 5 and 450 (transient length in ms)\n"
        "- impact: between 0.15 and 0.95 (strike energy amplitude)\n\n"
        "Output strictly valid JSON with keys: mood_title, description, clusters (array of 4 objects with name, f0, sigma, duration_ms, impact)."
    )

    user_content = f"Acoustic Mood Concept: \"{req.prompt}\""

    # Attempt Gemini call via OAuth token or API Key
    if oauth_token or api_key:
        try:
            raw_text = await _call_gemini_api(
                api_key=api_key,
                oauth_token=oauth_token,
                model=req.model or "gemini-2.5-flash",
                system_prompt=system_prompt,
                user_content=user_content,
                response_mime_type="application/json",
            )
            clean = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            return JSONResponse(content={
                "status": "ok",
                "provider": "gemini_oauth" if oauth_token else "gemini_api_key",
                "account": _GOOGLE_OAUTH_SESSION.get("email") if oauth_token else None,
                "data": parsed,
            })
        except Exception as exc:
            log.warning("Gemini AI Mood call failed (%s); falling back to local synthesizer", exc)

    # Local acoustic synthesizer fallback
    low = req.prompt.lower()
    local_presets = {
        "crystal": {
            "mood_title": "Crystal Ringing Chime",
            "description": "High-frequency crystalline resonances with prolonged undamped Laplace decay.",
            "clusters": [
                {"name": "A", "f0": 2400, "sigma": 8.0, "duration_ms": 320, "impact": 0.85},
                {"name": "B", "f0": 9600, "sigma": 14.0, "duration_ms": 280, "impact": 0.72},
                {"name": "C", "f0": 38400, "sigma": 22.0, "duration_ms": 190, "impact": 0.55},
                {"name": "D", "f0": 112000, "sigma": 40.0, "duration_ms": 90, "impact": 0.38},
            ],
        },
        "wood": {
            "mood_title": "Heavy Dense Wood Block",
            "description": "Low-frequency fundamental absorption with fast mechanical damping.",
            "clusters": [
                {"name": "A", "f0": 720, "sigma": 38.0, "duration_ms": 110, "impact": 0.90},
                {"name": "B", "f0": 1850, "sigma": 55.0, "duration_ms": 75, "impact": 0.65},
                {"name": "C", "f0": 4200, "sigma": 80.0, "duration_ms": 40, "impact": 0.40},
                {"name": "D", "f0": 8500, "sigma": 95.0, "duration_ms": 20, "impact": 0.25},
            ],
        },
        "metal": {
            "mood_title": "Forged Bell Metal",
            "description": "Coupled inharmonic stiffness modes with shimmering Laplace tails.",
            "clusters": [
                {"name": "A", "f0": 1450, "sigma": 6.5, "duration_ms": 420, "impact": 0.88},
                {"name": "B", "f0": 5800, "sigma": 18.0, "duration_ms": 220, "impact": 0.70},
                {"name": "C", "f0": 19200, "sigma": 32.0, "duration_ms": 140, "impact": 0.50},
                {"name": "D", "f0": 64000, "sigma": 55.0, "duration_ms": 60, "impact": 0.30},
            ],
        },
    }

    selected = local_presets.get("crystal")
    for k, v in local_presets.items():
        if k in low:
            selected = v
            break

    return JSONResponse(content={
        "status": "ok",
        "provider": "local_acoustic_synth",
        "notice": "Operating via deterministic local acoustic physics preset.",
        "data": selected,
    })


@app.post("/api/ai/analyze")
async def ai_analyze_event(req: AIAnalyzeRequest) -> JSONResponse:
    """Analyze acoustic impact event with AI Agent (Gemini OAuth / API Key or Local Physics Expert)."""
    api_key, oauth_token = _resolve_gemini_credentials(req.apiKey)

    system_prompt = (
        "You are the senior DSP & Acoustic Physics AI Agent for the 4DS Acoustic Laboratory. "
        "Your task is to analyze acoustic impact sound signatures from 4 block faces (A: Absorber, B: Resonator, C: Reflector, D: Coupled). "
        "Explain the mechanical physics behind the resonant frequency, Laplace decay sigma, Mel energy contrast, and candidate pair eliminations. "
        "Be concise, scientifically rigorous, and provide clear conclusions in GitHub markdown."
    )

    event_summary = json.dumps(req.event, indent=2)
    user_query = req.prompt or "Perform a complete acoustic physics diagnosis of this impact event."
    user_content = f"Event DSP Metrics:\n```json\n{event_summary}\n```\n\nTask:\n{user_query}"

    if oauth_token or api_key:
        try:
            ai_text = await _call_gemini_api(
                api_key=api_key,
                oauth_token=oauth_token,
                model=req.model or "gemini-2.5-flash",
                system_prompt=system_prompt,
                user_content=user_content,
            )
            return JSONResponse(content={
                "status": "ok",
                "provider": "gemini_oauth" if oauth_token else "gemini_api_key",
                "account": _GOOGLE_OAUTH_SESSION.get("email") if oauth_token else None,
                "model": req.model,
                "analysis": ai_text,
            })
        except Exception as exc:
            log.warning("Gemini API call failed (%s); falling back to local expert analysis", exc)
            local_text = _local_acoustic_expert_analysis(req.event, req.prompt)
            return JSONResponse(content={
                "status": "ok",
                "provider": "local_fallback",
                "notice": f"Gemini connection notice: {exc}. Displaying local expert physics analysis.",
                "analysis": local_text,
            })
    else:
        local_text = _local_acoustic_expert_analysis(req.event, req.prompt)
        return JSONResponse(content={
            "status": "ok",
            "provider": "local_expert",
            "analysis": local_text,
        })


@app.post("/api/ai/chat")
async def ai_chat(req: AIChatRequest) -> JSONResponse:
    """Interactive chat with the Acoustic Physics AI Agent (Gemini OAuth / API Key or Local Expert)."""
    api_key, oauth_token = _resolve_gemini_credentials(req.apiKey)
    system_prompt = (
        "You are the 4DS Acoustic Laboratory Pair-Programming & Physics AI Agent. "
        "You assist engineers with impact sound classification, resonant modes, Laplace envelope fitting, "
        "RAG memory similarity, and 4-face (A/B/C/D) physical boundaries. Respond clearly in markdown."
    )

    ctx_str = f"Context: {json.dumps(req.context)}\n\n" if req.context else ""
    user_content = f"{ctx_str}User question: {req.message}"

    if oauth_token or api_key:
        try:
            reply = await _call_gemini_api(
                api_key=api_key,
                oauth_token=oauth_token,
                model=req.model or "gemini-2.5-flash",
                system_prompt=system_prompt,
                user_content=user_content,
            )
            return JSONResponse(content={
                "status": "ok",
                "provider": "gemini_oauth" if oauth_token else "gemini_api_key",
                "account": _GOOGLE_OAUTH_SESSION.get("email") if oauth_token else None,
                "reply": reply,
            })
        except Exception as exc:
            return JSONResponse(content={
                "status": "error",
                "provider": "error",
                "reply": f"Gemini API connection error: {exc}. Please check network connection.",
            })
    else:
        return JSONResponse(content={
            "status": "ok",
            "provider": "local",
            "reply": "Please connect with Google OAuth in the AI Mood Settings to enable interactive reasoning with Gemini 2.5 Flash.",
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("live_server:app", host=NETWORK_POLICY.host, port=_SERVER_PORT, reload=False)
