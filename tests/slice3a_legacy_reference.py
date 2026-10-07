#!/usr/bin/env python3
"""DSP Inverse Optimizer - backend for the DSP Arena 4-group system.

Pipeline:  WAV -> decode -> linear resample to sr -> trim(top_db)
        -> RMS peak (2048/512) -> window +/-win -> re-trim
        -> FFT features (librosa, identical recipe to browser chart)
        -> metrics (Mann-Whitney / Fisher / Mahalanobis / silhouette)
        -> Optuna search over DSP params -> report + CSV + PNG.

Commands:
  baseline  --wavdir DIR --labels CSV [--sr --nfft --hop --topdb --win]
            print metrics at given params (default = browser defaults).
  optimize  --mode diff|sim|generic --split AB|CD or --pair AB
            --trials N --out TAG
            Optuna search; writes dsp_out/TAG/{params.json,report.json,
            features.csv,report.md,before_after.png}.
  reproduce --out TAG             re-run the winning params to features.csv
  boot-test --wavdir DIR --labels CSV --out TAG   one-trial optimizer smoke test

Groups: A/B/C/D from labels CSV (File_Name,Result or File,Group).
25 binary splits: 6x1v1, 12x1v2, 4x1v3, 3x2v2.
Metrics identical to the DSP Arena chart (Mann-Whitney U tie-corrected).
"""
import argparse
import json
import math
import os
import re
import struct
import sys
import time

import numpy as np
import librosa

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# 1. WAV decode + DSP recipe - EXACT mirror of ref_dsp.py / browser WAV mode
# ---------------------------------------------------------------------------

def decode_wav(path):
    """PCM 8/16/24/32 int + 32-bit float, WAVE_FORMAT_EXTENSIBLE, stereo mixdown."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    off = 12
    fmt = None
    dstart = dlen = -1
    n = len(data)
    while off + 8 <= n:
        cid = data[off:off + 4]
        sz = struct.unpack("<I", data[off + 4:off + 8])[0]
        if cid == b"fmt ":
            fmt = {
                "fmt": struct.unpack("<H", data[off + 8:off + 10])[0],
                "ch": struct.unpack("<H", data[off + 10:off + 12])[0],
                "sr": struct.unpack("<I", data[off + 12:off + 16])[0],
                "bits": struct.unpack("<H", data[off + 22:off + 24])[0],
            }
            if sz >= 40:
                sub = struct.unpack("<H", data[off + 8 + 24:off + 8 + 26])[0]
                g = data[off + 8 + 26:off + 8 + 30]
                if sub == 0xFFFE and g == b"PCM\0":
                    fmt["fmt"] = 1
                elif sub == 0xFFFE and g == b"f32\0":
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
    if fmt["ch"] < 1 or fmt["ch"] > 8:
        raise ValueError("unsupported channels %d" % fmt["ch"])
    if fmt["fmt"] not in (1, 3) or (fmt["fmt"] == 3 and bits != 32):
        raise ValueError("unsupported format wave=%d bits=%d" % (fmt["fmt"], bits))
    if bits not in (8, 16, 24, 32):
        raise ValueError("unsupported bit depth %d" % bits)
    ch = fmt["ch"]
    bps = bits >> 3
    usable = (dlen // (bps * ch)) * (bps * ch)
    nsamp = usable // (bps * ch)
    if nsamp < 1:
        return np.zeros(0, dtype=np.float64), fmt["sr"]
    arr = np.frombuffer(data, dtype=np.uint8, count=usable, offset=dstart)
    if fmt["fmt"] == 3:
        mono = arr.view("<f4").reshape(-1, ch).astype(np.float64).mean(axis=1)
    elif bits == 8:
        mono = (arr.reshape(-1, ch).astype(np.float64) - 128.0).mean(axis=1) / 128.0
    elif bits == 16:
        mono = arr.view("<i2").reshape(-1, ch).astype(np.float64).mean(axis=1) / 32767.0
    elif bits == 32:
        mono = arr.view("<i4").reshape(-1, ch).astype(np.float64).mean(axis=1) / 2147483647.0
    else:  # 24-bit
        b24 = arr.reshape(-1, 3).astype(np.int64)
        x = b24[:, 0] | (b24[:, 1] << 8) | (b24[:, 2] << 16)
        x = np.where(x & 0x800000, x - 0x1000000, x)
        mono = x.reshape(-1, ch).astype(np.float64).mean(axis=1) / 8388607.0
    return np.clip(mono, -1.0, 1.0), fmt["sr"]


def resample_linear(x, from_sr, to_sr):
    if from_sr == to_sr:
        return x
    n = max(1, int(round(len(x) * to_sr / from_sr)))
    pos = np.arange(n) * (from_sr / to_sr)
    return np.interp(pos, np.arange(len(x)), x)


def _frame_energy(y, fl, fh):
    """Vectorized frame energy via as_strided (zero-copy view, G7 compliant)."""
    nf = (len(y) - fl) // fh + 1
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(nf, fl),
        strides=(y.strides[0] * fh, y.strides[0]),
    )
    return np.mean(frames * frames, axis=1)  # shape (nf,)


def trim_wav(y, top_db):
    """Frame-power trim: frame 2048 hop 512, threshold = max * 10^(-top_db/10)."""
    FL, FH = 2048, 512
    L = len(y)
    if L < FL:
        return None
    E = _frame_energy(y, FL, FH)
    emax = E.max()
    if emax <= 0:
        return None
    thr = emax * 10.0 ** (-top_db / 10.0)
    idx = np.flatnonzero(E > thr)
    if len(idx) == 0:
        return None
    start, end = int(idx[0] * FH), min(L, int(idx[-1] * FH + FL))
    return y[start:end], start


def max_rms_frame(y, fl=2048, fh=512):
    L = len(y)
    if L < fl:
        return float(np.sqrt(np.mean(y * y)))
    E = _frame_energy(y, fl, fh)
    return float(np.sqrt(E.max()))


def detect_transient_window(audio, sr, freq_threshold_hz=1000.0,
                             max_duration_s=1.0, ff_s=0.5, rr_s=0.5,
                             env_threshold_ratio=0.20):
    """Detect → center → FF/RR windowing (mirrors manual_stop_interactive_recorder(1).py).

    1) High-pass filter above freq_threshold_hz (default 1000 Hz).
    2) Hilbert envelope + threshold (20% of peak) → contiguous transient region.
    3) STRICT PHYSICAL CEILING: True acoustic impact transient is < 0.5s and NEVER > 1.0s.
       If detected span > max_duration_s (1.0s) -> REJECT immediately (returns None).
    4) Center = midpoint of detected region.
    5) Cut FF = ff_s (0.5s) BEFORE center + RR = rr_s (0.5s) AFTER center -> 1.0s total window.

    Returns (windowed_audio, center_sample, onset_sample, offset_sample, detected_duration_s).
    """
    import scipy.signal as _sig
    n = len(audio)
    if n == 0:
        return None, 0, 0, 0, 0.0
    # 1) high-pass filter
    nyq = sr / 2.0
    hp_norm = min(max(freq_threshold_hz / nyq, 1e-4), 0.99)
    try:
        b, a = _sig.butter(4, hp_norm, btype='highpass')
        audio_hp = _sig.filtfilt(b, a, audio)
    except Exception:
        audio_hp = audio
    # 2) Hilbert envelope + threshold
    envelope = np.abs(_sig.hilbert(audio_hp))
    smooth_win = max(1, int(sr * 0.001))
    if smooth_win > 1:
        kernel = np.ones(smooth_win) / smooth_win
        envelope = np.convolve(envelope, kernel, mode='same')
    peak_env = float(np.max(envelope)) if n > 0 else 0.0
    if peak_env <= 0:
        peak_idx = int(np.argmax(np.abs(audio))) if n > 0 else 0
        onset_idx = max(0, peak_idx - int(sr * max_duration_s / 2))
        offset_idx = min(n, peak_idx + int(sr * max_duration_s / 2))
    else:
        thresh = env_threshold_ratio * peak_env
        above = np.where(envelope >= thresh)[0]
        if len(above) == 0:
            peak_idx = int(np.argmax(envelope))
            onset_idx = max(0, peak_idx - int(sr * max_duration_s / 2))
            offset_idx = min(n, peak_idx + int(sr * max_duration_s / 2))
        else:
            onset_idx, offset_idx = int(above[0]), int(above[-1])
    detected_duration_s = (offset_idx - onset_idx) / sr
    # 3) STRICT PHYSICAL CEILING: reject signals > 1.0s
    if detected_duration_s > max_duration_s:
        return None, -1, onset_idx, offset_idx, detected_duration_s
    # 4) center
    center_idx = (onset_idx + offset_idx) // 2
    # 5) FF + RR window (exact 1.0s centered window)
    start = max(0, center_idx - int(sr * ff_s))
    end = min(n, center_idx + int(sr * rr_s))
    return audio[start:end], center_idx, onset_idx, offset_idx, detected_duration_s


def measure_features(y, sr, nfft, hop, top_db, win):
    """Return feature dict or None.
    Detects a >1kHz transient (ceiling 1.0s, impact is <0.5s), centers a ±0.5s window,
    then measures:
      - duration: RMS envelope decay (peak → 10%) from the full signal,
                  anchored to the detected transient region.
      - peak_time: 10%→90% rise time of the windowed transient itself.
    """
    # Validate input: must be 1D float array with finite values
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all() or len(y) < max(512, int(round(sr * 0.01))):
        return None
    # --- event detection: high-pass 1kHz, Hilbert envelope, center, FF=0.5s RR=0.5s ---
    y_win, center_idx, detect_onset, detect_offset, detect_dur = \
        detect_transient_window(y, sr, freq_threshold_hz=1000.0,
                                max_duration_s=1.0, ff_s=0.5, rr_s=0.5)
    if y_win is None or detect_dur > 1.0 or len(y_win) < max(64, int(round(sr * 0.01))):
        return None
    fl, fh = 2048, 512
    # --- top_db trim within the windowed segment (for mel/spectral features) ---
    t2 = trim_wav(y_win, top_db)
    if t2 is None or len(t2[0]) < max(64, int(round(sr * 0.01))):
        st = y_win
        t_offset = 0
    else:
        st, t_offset = t2
    # --- impact: max RMS frame within trimmed segment ---
    impact = max_rms_frame(st, fl, fh)
    if impact <= 0:
        return None
    # --- peak_time: 10%→90% rise time of the windowed transient
    #     attack_slope_db_ms: 19.08 dB (10%→90% swing) / rise_time_ms
    #     Both reuse the same as_strided RMS envelope (G7, zero extra alloc) ---
    _ONSET_DB_SWING = 20.0 * math.log10(9.0)  # log10(0.9/0.1) ≈ 19.08 dB
    try:
        fl_a, fh_a = 2048, 256
        nf_a = max(1, (len(y_win) - fl_a) // fh_a + 1)
        # Vectorized: as_strided view — no Python loop over samples (G7)
        rms_a = np.sqrt(_frame_energy(y_win[:nf_a * fh_a + fl_a], fl_a, fh_a)[:nf_a])
        rms_pk_a = int(np.argmax(rms_a))
        rms_val_a = rms_a[rms_pk_a]
        pre_rms = rms_a[:rms_pk_a + 1]
        above10 = np.where(pre_rms >= rms_val_a * 0.10)[0]
        above90 = np.where(pre_rms >= rms_val_a * 0.90)[0]
        if len(above10) > 0 and len(above90) > 0:
            t10 = above10[0] * fh_a / sr
            t90 = above90[0] * fh_a / sr
            peak_time = max(0.0, t90 - t10)
            # clip denominator to 1 µs to avoid /0 on impulse-width signals
            attack_slope_db_ms = _ONSET_DB_SWING / max(peak_time * 1000.0, 1e-3)
        else:
            peak_time = 0.0
            attack_slope_db_ms = 0.0
    except Exception:
        peak_time = 0.0
        attack_slope_db_ms = 0.0
    # --- duration: RMS envelope decay (peak → 10%) from the FULL signal ---
    # Measure on the full recording (anchored to the detected transient
    # region) so a real decay is captured even when it exceeds the ±0.5s
    # window. Fallback to detect_dur only when the signal never decays to
    # 10% of its RMS peak within the entire recording.
    try:
        fl_d, fh_d = 2048, 256
        nf_d = max(1, (len(y) - fl_d) // fh_d + 1)
        # Vectorized: as_strided view — no Python loop over samples (G7)
        rms_d = np.sqrt(_frame_energy(y[:nf_d * fh_d + fl_d], fl_d, fh_d)[:nf_d])
        lo_f = max(0, detect_onset // fh_d)
        hi_f = min(nf_d, detect_offset // fh_d + 1)
        rms_pk = lo_f + int(np.argmax(rms_d[lo_f:hi_f]))
        rms_val = rms_d[rms_pk]
        post_rms = rms_d[rms_pk:]
        below = np.where(post_rms <= rms_val * 0.10)[0]
        if len(below) > 0:
            duration = below[0] * fh_d / sr
        else:
            # 10% never reached — use the Hilbert-detected transient span
            duration = detect_dur if detect_dur > 0 else len(y) / sr
    except Exception:
        duration = detect_dur if detect_dur > 0 else len(y) / sr

    N = nfft
    X = np.fft.rfft(st, n=N)
    M = np.abs(X)
    freq_bins = np.fft.rfftfreq(N, d=1.0 / sr)
    fhi = min(50000.0, sr / 2.0)
    band = (freq_bins >= 200) & (freq_bins <= fhi)
    Mb = M[band]
    if len(Mb) == 0:
        freq = 0.0
    else:
        bfull = int(np.flatnonzero(band)[0]) + int(np.argmax(Mb))
        if 1 <= bfull < len(M) - 1:
            y1, y2, y3 = M[bfull - 1], M[bfull], M[bfull + 1]
            den = y1 - 2 * y2 + y3
            delta = 0.5 * (y1 - y3) / den if den != 0 else 0.0
            freq = (bfull + max(-0.5, min(0.5, delta))) * sr / N
        else:
            freq = float(freq_bins[bfull])
    impact_freq_hz = round(freq * 10) / 10

    # --- STFT power (mel + spectral extras share it)
    # Browser recipe uses max(1, floor((len-nfft)/hop)+1) frames and zero-pads
    # a too-short final/only frame. librosa center=False raises when len<n_fft,
    # so pad here to preserve browser semantics.
    st_for_stft = st if len(st) >= N else np.pad(st, (0, N - len(st)))
    S = np.abs(librosa.stft(st_for_stft, n_fft=N, hop_length=hop, center=False)) ** 2
    mel = librosa.filters.mel(sr=sr, n_fft=N, n_mels=40, fmin=0.0,
                              fmax=fhi, htk=True, norm=None)
    melpow = np.maximum(mel @ S, np.finfo(np.float32).tiny)
    mel_db = 10.0 * np.log10(melpow)
    mel_mean_db = float(mel_db.mean())

    pf = freq_bins[:S.shape[0]]
    spec_sum = S.sum(axis=0)
    cents = (pf[:, None] * S).sum(axis=0) / np.maximum(spec_sum, 1e-300)
    spectral_centroid_hz = float(cents.mean()) if len(cents) else 0.0
    bw = (((pf[:, None] - cents[None, :]) ** 2) * S).sum(axis=0)
    spectral_bandwidth_hz = float(np.sqrt(bw / np.maximum(spec_sum, 1e-300)).mean())
    # Vectorized spectral rolloff — no Python loop over frames (G7)
    _cs = np.cumsum(S, axis=0)                          # (freq, T)
    _tot = _cs[-1, :]                                    # (T,)
    _valid = _tot > 0
    _roll_bin = np.argmax(_cs >= 0.85 * _tot[np.newaxis, :], axis=0)
    spectral_rolloff_hz = float(np.mean(pf[_roll_bin] * _valid))
    flat = np.exp(np.mean(np.log(S + 1e-300), axis=0)) / np.maximum(np.mean(S, axis=0), 1e-300)
    spectral_flatness = float(np.mean(flat))
    rms_energy = float(np.sqrt(np.mean(st * st)))
    signs = np.signbit(st)
    zcr = float(np.mean(signs[1:] != signs[:-1]))
    # --- hf_transient_ratio: HF (>1 kHz) power fraction (vectorized, G7) ---
    hf_mask = freq_bins[:S.shape[0]] > 1000.0
    hf_energy = float(S[hf_mask].sum())
    total_energy = float(S.sum())
    hf_transient_ratio = hf_energy / max(total_energy, 1e-300)
    # --- crest_factor: peak / RMS (impulsiveness measure) ---
    peak_amp = float(np.max(np.abs(st)))
    crest_factor = peak_amp / max(rms_energy, 1e-12)

    return {
        "impact": impact, "duration": duration, "peak_time": peak_time,
        "mel_mean_db": mel_mean_db, "impact_freq_hz": impact_freq_hz,
        "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_bandwidth_hz": spectral_bandwidth_hz,
        "spectral_rolloff_hz": spectral_rolloff_hz,
        "spectral_flatness": spectral_flatness,
        "rms_energy": rms_energy,
        "zcr": zcr,
        # onset dynamics (from onset_result.md analysis)
        "attack_slope_db_ms": attack_slope_db_ms,
        "hf_transient_ratio": hf_transient_ratio,
        "crest_factor": crest_factor,
    }




# --- end of frozen Slice-3A legacy reference (measurement path only); remainder (labels/metrics/optimize) intentionally omitted ---
