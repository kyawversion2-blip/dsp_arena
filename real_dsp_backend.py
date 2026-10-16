#!/usr/bin/env python3
"""
real_dsp_backend.py
Production-Grade Real-Data DSP Backend for RIPPLE MICROSCOPE CLUSTER LAB
Optimized for Apple Silicon M4 / macOS & Real CSV Replay / Live Mic.
"""

import argparse
import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from network_policy import NetworkPolicy, NetworkPolicyError
from fastapi.responses import HTMLResponse
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt, hilbert

try:
    import sounddevice as sd
except Exception:
    sd = None

# ==========================================
# CONFIGURATION
# ==========================================
FS = 48000               # Strict G1: Native capture <= 48kHz
HIGH_PASS_CUTOFF = 1000.0  # Strict G2: >= 1000Hz high-pass
MAX_DURATION = 1.0         # Strict G2: Reject continuous transient > 1.0s
BLOCKSIZE = 1024           # Audio chunk size for mic callback

app = FastAPI(title="RIPPLE DSP Backend")
try:
    NETWORK_POLICY = NetworkPolicy.from_environment(port=int(os.environ.get("PORT", "8765")))
except NetworkPolicyError as exc:
    raise RuntimeError(f"Unsafe network configuration: {exc}") from exc

class _NetworkBoundaryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        reason = NETWORK_POLICY.validate_request(
            host=request.headers.get("host"), origin=request.headers.get("origin"),
            token=NETWORK_POLICY.token_from_headers(request.headers),
        )
        if reason:
            return JSONResponse(status_code=403, content={"detail": reason})
        return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(NETWORK_POLICY.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-DSP-Auth-Token"],
)
app.add_middleware(_NetworkBoundaryMiddleware)

WORKSPACE_DIR = Path(__file__).resolve().parent

# Serve frontends
@app.get("/", response_class=HTMLResponse)
async def serve_main():
    main_html = WORKSPACE_DIR / "dsp-cluster-lab.html"
    if main_html.is_file():
        return HTMLResponse(content=main_html.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})
    return HTMLResponse(content="<h1>dsp-cluster-lab.html not found</h1>", status_code=404)

@app.get("/scout", response_class=HTMLResponse)
async def serve_scout():
    scout_html = WORKSPACE_DIR / "partition_scout.html"
    if scout_html.is_file():
        return HTMLResponse(content=scout_html.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})
    return HTMLResponse(content="<h1>partition_scout.html not found</h1>", status_code=404)

# ==========================================
# WEBSOCKET MANAGER
# ==========================================
class ConnectionManager:
    """Manages active WebSocket connections for real-time broadcasts with heartbeat support."""
    
    HEARTBEAT_INTERVAL = 30.0  # seconds
    HEARTBEAT_TIMEOUT = 90.0   # seconds
    MAX_CONNECTIONS = 100
    
    def __init__(self) -> None:
        self._active: dict[str, dict] = {}  # client_id -> {ws, connected_at, last_heartbeat, last_activity, authenticated, message_count}
        self._heartbeat_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None
    
    def _generate_client_id(self) -> str:
        return f"ws_{uuid.uuid4().hex[:12]}"
    
    async def start_background_tasks(self) -> None:
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(), name="ws-heartbeat")
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="ws-cleanup")
    
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
    
    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                now = time.time()
                stale = []
                
                for client_id, info in self._active.items():
                    if now - info["last_heartbeat"] > self.HEARTBEAT_TIMEOUT:
                        stale.append(client_id)
                    else:
                        try:
                            await info["ws"].send_json({
                                "type": "heartbeat",
                                "timestamp": now,
                                "server_time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                            })
                        except Exception:
                            stale.append(client_id)
                
                for client_id in stale:
                    await self._force_disconnect(client_id, code=1001, reason="Heartbeat timeout")
                    
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60.0)
                now = time.time()
                stale = []
                
                for client_id, info in self._active.items():
                    if now - info.get("last_activity", now) > 300.0:
                        stale.append(client_id)
                
                for client_id in stale:
                    await self._force_disconnect(client_id, code=1001, reason="Inactivity timeout")
                    
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def _force_disconnect(self, client_id: str, code: int = 1000, reason: str = "Normal closure") -> None:
        info = self._active.pop(client_id, None)
        if info:
            try:
                await info["ws"].close(code=code, reason=reason)
            except Exception:
                pass
    
    async def connect(self, ws: WebSocket) -> str:
        if len(self._active) >= self.MAX_CONNECTIONS:
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
        
        try:
            await ws.send_json({
                "type": "connect",
                "client_id": client_id,
                "timestamp": now,
                "server_time": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                "heartbeat_interval": self.HEARTBEAT_INTERVAL
            })
        except Exception:
            await self._force_disconnect(client_id)
            raise
            
        return client_id
    
    def disconnect(self, ws: WebSocket) -> str | None:
        for client_id, info in self._active.items():
            if info["ws"] is ws:
                del self._active[client_id]
                return client_id
        return None
    
    def disconnect_by_id(self, client_id: str) -> bool:
        if client_id in self._active:
            del self._active[client_id]
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
    
    async def broadcast(self, payload: dict) -> int:
        if not self._active:
            return 0
            
        sent = 0
        stale = []
        
        for client_id, info in self._active.items():
            try:
                await info["ws"].send_json(payload)
                sent += 1
            except Exception:
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

manager = ConnectionManager()
event_counter = 0
counter_lock = threading.Lock()

def get_next_id():
    global event_counter
    with counter_lock:
        event_counter += 1
        return event_counter

# Global replay state controller for UI manual triggers
replay_controller = None

# ==========================================
# DSP EXTRACTION ENGINE
# ==========================================
sos_hp = butter(4, HIGH_PASS_CUTOFF, btype='high', fs=FS, output='sos')

def parabolic_interpolation(y, k):
    """3-point parabolic interpolation to avoid FFT bin quantization error."""
    if k <= 0 or k >= len(y) - 1:
        return float(k), float(y[k])
    alpha = float(y[k-1])
    beta = float(y[k])
    gamma = float(y[k+1])
    denom = (alpha - 2*beta + gamma)
    if abs(denom) < 1e-12:
        return float(k), beta
    p = 0.5 * (alpha - gamma) / denom
    return k + p, beta - 0.25 * (alpha - gamma) * p

def extract_features(audio_chunk: np.ndarray, fs: int = FS) -> Dict:
    filtered = sosfiltfilt(sos_hp, audio_chunk)
    N = len(filtered)
    
    # 1. Peak Frequency (FFT + Parabolic Interpolation)
    yf = rfft(filtered)
    magnitudes = np.abs(yf)
    peak_bin = int(np.argmax(magnitudes))
    interp_bin, _ = parabolic_interpolation(magnitudes, peak_bin)
    f0 = max(20.0, interp_bin * (fs / N))
    
    # 2. Laplace Damping (Hilbert Envelope + Log-Linear Regression)
    analytic_signal = hilbert(filtered)
    envelope = np.abs(analytic_signal)
    peak_idx_env = int(np.argmax(envelope))
    sigma = 20.0
    
    if peak_idx_env < len(envelope) - 10:
        decay_env = envelope[peak_idx_env:]
        t_decay = np.arange(len(decay_env)) / fs
        noise_floor = np.max(envelope) * 0.05
        valid_idx = decay_env > noise_floor
        
        if np.sum(valid_idx) > 5:
            t_valid = t_decay[valid_idx]
            y_valid = np.log(np.maximum(decay_env[valid_idx], 1e-12))
            try:
                p = np.polyfit(t_valid, y_valid, 1)
                sigma = -p[0]
                if sigma < 0.1: sigma = 0.1
            except Exception:
                pass
                
    sigma = float(np.clip(sigma, 0.1, 500.0))
    
    # 3. Peak Time
    threshold = np.max(np.abs(filtered)) * 0.05
    above_thresh = np.where(np.abs(filtered) > threshold)[0]
    onset_idx = above_thresh[0] if len(above_thresh) > 0 else 0
    peak_idx = int(np.argmax(np.abs(filtered)))
    peak_time_ms = max(0.0, ((peak_idx - onset_idx) / fs) * 1000.0)
    
    # 4. Duration & Energy
    duration = min(float(N / fs), MAX_DURATION)
    rms = float(np.sqrt(np.mean(np.square(filtered))))
    energy = 0.06 + (rms / 0.3) * 0.94
    energy = float(np.clip(energy, 0.06, 1.0))
    
    # 5. Oscilloscope Waveform (Exactly 300 points)
    indices = np.linspace(0, N-1, 300).astype(int)
    waveform = filtered[indices].astype(float).tolist()
    
    return {
        "f": float(f0),
        "f0": float(f0),
        "frequency": float(f0),
        "sigma": sigma,
        "duration": float(duration),
        "peak_time_ms": float(peak_time_ms),
        "energy": energy,
        "waveform": waveform
    }

# ==========================================
# MICROPHONE CAPTURE ENGINE (LIVE MODE)
# ==========================================
class MicProcessor:
    def __init__(self, loop):
        self.loop = loop
        self.recording = False
        self.transient_buffer = []
        self.threshold = 0.015
        self.silence_chunks = 0
        self.max_silence_chunks = int(0.05 * FS / BLOCKSIZE)
        self.stream = None

    def callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        rms = np.sqrt(np.mean(np.square(mono)))
        
        if not self.recording:
            if rms > self.threshold:
                self.recording = True
                self.transient_buffer = [mono]
                self.silence_chunks = 0
        else:
            self.transient_buffer.append(mono)
            if rms < self.threshold:
                self.silence_chunks += 1
            else:
                self.silence_chunks = 0
                
            duration = len(self.transient_buffer) * BLOCKSIZE / FS
            if duration > MAX_DURATION or self.silence_chunks > self.max_silence_chunks:
                self.recording = False
                full_audio = np.concatenate(self.transient_buffer)
                self.transient_buffer = []
                
                if len(full_audio) > 512:
                    asyncio.run_coroutine_threadsafe(
                        process_event(full_audio, "UNLABELED"), self.loop
                    )

    def start(self):
        if sd is None:
            print("[MIC] sounddevice library not available. Please install sounddevice.")
            return
        try:
            self.stream = sd.InputStream(
                samplerate=FS, channels=1, 
                callback=self.callback, blocksize=BLOCKSIZE
            )
            self.stream.start()
            print(f"[MIC] Listening on default input device @ {FS}Hz...")
        except Exception as e:
            print(f"[MIC] Error starting audio stream: {e}")

# ==========================================
# ROBUST CSV REPLAY ENGINE (REPLAY MODE)
# ==========================================
_COL_ALIASES: Dict[str, tuple] = {
    "f": ("impact_freq_hz", "dominant_frequency_hz", "stft_peak_freq_hz", "peak_freq_hz", 
          "f", "f0", "freq", "frequency", "peak_freq"),
    "sigma": ("laplace_sigma_fit", "laplace_sigma", "zeta_damping", "decay_rate", "damping_rate", 
              "sigma", "damping", "decay"),
    "dur": ("duration", "dur", "length", "event_duration"),
    "energy": ("impact", "rms", "energy", "amp", "amplitude", "impact_energy"),
    "cluster": ("group", "ground_truth", "class", "cluster", "label", "category", "target"),
    "tp": ("peak_time", "peak_time_ms", "t_peak_ms", "tp", "time_to_peak")
}

def resolve_col(df: pd.DataFrame, candidates: tuple, default: float = 0.0) -> pd.Series:
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            real_name = lower_cols[cand.lower()]
            return pd.to_numeric(df[real_name], errors="coerce")
    return pd.Series([default] * len(df))

def resolve_str_col(df: pd.DataFrame, candidates: tuple, default: str = "UNLABELED") -> pd.Series:
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            real_name = lower_cols[cand.lower()]
            return df[real_name].astype(str)
    return pd.Series([default] * len(df))

class ReplayController:
    def __init__(self, csv_path: str, loop: asyncio.AbstractEventLoop, loop_forever: bool = True):
        self.csv_path = csv_path
        self.loop = loop
        self.loop_forever = loop_forever
        self.df = pd.read_csv(csv_path)
        
        self.freq_col = resolve_col(self.df, _COL_ALIASES["f"], default=1200.0)
        self.sigma_col = resolve_col(self.df, _COL_ALIASES["sigma"], default=20.0)
        self.dur_col = resolve_col(self.df, _COL_ALIASES["dur"], default=0.2)
        self.energy_col = resolve_col(self.df, _COL_ALIASES["energy"], default=0.5)
        self.cluster_col = resolve_str_col(self.df, _COL_ALIASES["cluster"], default="UNLABELED")
        self.tp_col = resolve_col(self.df, _COL_ALIASES["tp"], default=0.0)
        
        # Energy normalization
        e_max = self.energy_col.max()
        if pd.notna(e_max) and e_max > 1.5:
            self.energy_col = 0.06 + (self.energy_col / e_max) * 0.94
            
        # Peak time seconds to ms check
        if self.tp_col.max() <= 5.0:
            self.tp_col = self.tp_col * 1000.0
            
        self.current_idx = 0
        self.total_rows = len(self.df)
        print(f"[REPLAY] Loaded {self.total_rows} real events from {Path(csv_path).name}")

    def get_event_payload(self, idx: int) -> dict:
        f = float(self.freq_col.iloc[idx]) if pd.notna(self.freq_col.iloc[idx]) else 1200.0
        sigma = float(self.sigma_col.iloc[idx]) if pd.notna(self.sigma_col.iloc[idx]) else 20.0
        dur = float(self.dur_col.iloc[idx]) if pd.notna(self.dur_col.iloc[idx]) else 0.2
        dur = max(0.01, min(dur, MAX_DURATION))
        energy = float(self.energy_col.iloc[idx]) if pd.notna(self.energy_col.iloc[idx]) else 0.5
        energy = float(np.clip(energy, 0.06, 1.0))
        cluster = str(self.cluster_col.iloc[idx]).strip().upper()
        if cluster not in ["A", "B", "C", "D"]:
            cluster = "UNLABELED"
        tp_ms = float(self.tp_col.iloc[idx]) if pd.notna(self.tp_col.iloc[idx]) else 0.0
        
        # Mathematical damped cosine reconstruction matching dashboard kernel:
        # x(t) = A * e^(-sigma*t) * cos(2*pi*f*t)
        t = np.linspace(0, dur, 300)
        waveform = (energy * np.exp(-sigma * t) * np.cos(2 * np.pi * f * t)).tolist()
        
        return {
            "type": "event",
            "id": get_next_id(),
            "cluster": cluster,
            "f": float(f),
            "f0": float(f),
            "frequency": float(f),
            "sigma": float(sigma),
            "duration": float(dur),
            "peak_time_ms": float(tp_ms),
            "energy": float(energy),
            "waveform": waveform
        }

    async def step(self):
        if self.total_rows == 0:
            return
        payload = self.get_event_payload(self.current_idx)
        await manager.broadcast(payload)
        self.current_idx = (self.current_idx + 1) % self.total_rows

    async def run_loop(self):
        while True:
            await self.step()
            await asyncio.sleep(0.4)
            if not self.loop_forever and self.current_idx == 0:
                print("[REPLAY] Reached end of dataset.")
                break

# ==========================================
# EVENT DISPATCHER
# ==========================================
async def process_event(audio_chunk: np.ndarray, cluster_label: str = "UNLABELED"):
    features = extract_features(audio_chunk, FS)
    payload = {
        "type": "event",
        "id": get_next_id(),
        "cluster": cluster_label,
        **features
    }
    await manager.broadcast(payload)

# ==========================================
# WEBSOCKET ENDPOINTS
# ==========================================
@app.websocket("/ws/dsp-stream")
@app.websocket("/ws/dsp")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    reason = NETWORK_POLICY.validate_request(
        host=websocket.headers.get("host"), origin=websocket.headers.get("origin"),
        token=NETWORK_POLICY.token_from_websocket(websocket),
    )
    if reason:
        await websocket.close(code=1008, reason=reason)
        return
    await manager.connect(websocket)
    try:
        while True:
            raw_msg = await websocket.receive_text()
            # If dashboard sends manual trigger event:
            if "trigger" in raw_msg.lower() and replay_controller is not None:
                await replay_controller.step()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

# ==========================================
# MAIN ENTRY POINT
# ==========================================
def main():
    global replay_controller
    parser = argparse.ArgumentParser(description="Real-DSP Backend for Cluster Lab")
    parser.add_argument("--mode", choices=["mic", "replay"], default="mic", help="Execution mode")
    parser.add_argument("--csv", type=str, help="Path to CSV dataset for replay mode")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop dataset in replay mode")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if args.mode == "mic":
        mic_proc = MicProcessor(loop)
        threading.Thread(target=mic_proc.start, daemon=True).start()
    elif args.mode == "replay":
        if not args.csv or not os.path.exists(args.csv):
            print(f"[ERROR] CSV path '{args.csv}' does not exist.")
            return
        replay_controller = ReplayController(args.csv, loop, loop_forever=args.loop)
        loop.create_task(replay_controller.run_loop())

    config = uvicorn.Config(app, host=NETWORK_POLICY.host, port=args.port, loop="asyncio")
    server = uvicorn.Server(config)
    
    print(f"============================================================")
    print(f"  RIPPLE MICROSCOPE REAL DSP BACKEND RUNNING")
    print(f"  Mode   : {args.mode.upper()}")
    print(f"  UI URL : http://localhost:{args.port}/")
    print(f"  Scout  : http://localhost:{args.port}/scout")
    print(f"============================================================")
    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    main()
