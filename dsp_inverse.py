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


# ---------------------------------------------------------------------------
# 2. labels + dataset
# ---------------------------------------------------------------------------

def load_labels(csv_path):
    labels = {}
    if not csv_path or not os.path.exists(csv_path):
        return labels
    import csv as _csv
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rd = _csv.reader(f)
        head = next(rd, None)
        if not head:
            return labels
        hs = [h.strip().lower() for h in head]
        fkey = None
        for k in ("file_name", "file"):
            if k in hs:
                fkey = hs.index(k)
                break
        gkey = None
        for k in ("result", "group"):
            if k in hs:
                gkey = hs.index(k)
                break
        if fkey is None or gkey is None:
            return labels
        for row in rd:
            if len(row) <= max(fkey, gkey):
                continue
            fn = row[fkey].strip()
            g = re.sub(r"(?i)^group\s*", "", row[gkey].strip()).upper()
            if g in ("A", "B", "C", "D"):
                labels[fn] = g
    return labels


class Dataset:
    def __init__(self, wavdir, labels_csv=None):
        self.wavdir = wavdir
        self.labels = load_labels(labels_csv)
        self.rows = []
        self.missing_labels = []
        for f in sorted(os.listdir(wavdir)):
            if not f.lower().endswith(".wav"):
                continue
            base = os.path.splitext(f)[0]
            g = self.labels.get(f) or self.labels.get(base)
            if g is None:
                # Explicit filename-coded labels are acceptable; silent default is not.
                m = re.match(r"^(\d+)_([ABCD])$", base, re.I) or re.match(r"^([ABCD])_(\d+)$", base, re.I)
                if m:
                    g = (m.group(2) if m.group(2) and m.group(2).upper() in _GROUPS else m.group(1)).upper()
            if g is None or g not in _GROUPS:
                self.missing_labels.append(f)
                g = "?"
            self.rows.append({"file": f, "group": g})
        self._cache = {}

    def measure_one(self, r, params):
        key = (r["file"], params.sr, params.nfft, params.hop, params.topdb, params.win)
        if key in self._cache:
            return self._cache[key]
        try:
            y, native = decode_wav(os.path.join(self.wavdir, r["file"]))
            y = resample_linear(y, native, params.sr)
            fe = measure_features(y, params.sr, params.nfft, params.hop,
                                  params.topdb, params.win)
        except Exception:
            fe = None
        self._cache[key] = fe
        return fe

    def measure_all(self, params):
        out = []
        for r in self.rows:
            fe = self.measure_one(r, params)
            if fe is None:
                continue
            row = {"file": r["file"], "group": r["group"]}
            row.update(fe)
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# 3. metrics - Mann-Whitney U (tie-corrected, two-sided, normal approx),
#    Fisher ratio, Mahalanobis, silhouette. All operate on clean finite arrays.
# ---------------------------------------------------------------------------

def mann_whitney_p(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 1.0
    N = n1 + n2
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(N, dtype=float)
    i = 0
    tie_sum = 0.0
    while i < N:
        j = i
        while j + 1 < N and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        t = j - i + 1
        if t > 1:
            tie_sum += t * t * t - t
        ranks[order[i:j + 1]] = r
        i = j + 1
    R1 = float(ranks[:n1].sum())
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U = min(U1, n1 * n2 - U1)  # chart-parity: use the smaller U for two-sided test
    mu = n1 * n2 / 2.0
    varU = n1 * n2 / 12.0 * ((N + 1) - tie_sum / (N * (N - 1)))
    if varU <= 0:
        return 1.0
    z = (mu - U - 0.5) / math.sqrt(varU)
    # Two-sided p-value: p = 2 * Φ(-|z|) where Φ is standard normal CDF
    # Using erf: Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    abs_z = abs(z)
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs_z / math.sqrt(2.0))))
    return min(1.0, p_val)


def fisher_ratio(groups_vals):
    """groups_vals: list of 1-D arrays (one per group). Fisher = between/within."""
    vs = [np.asarray(g, dtype=float)[np.isfinite(np.asarray(g, dtype=float))]
          for g in groups_vals]
    vs = [v for v in vs if len(v) > 1]
    if len(vs) < 2:
        return 0.0
    allv = np.concatenate(vs)
    if len(allv) < 2 or np.ptp(allv) == 0:
        return 0.0
    grand = allv.mean()
    k = len(vs)
    n = float(len(allv))
    between = 0.0
    within = 0.0
    for v in vs:
        m = v.mean()
        between += len(v) * (m - grand) ** 2
        within += ((v - m) ** 2).sum()
    between /= (k - 1)
    within /= (n - k)
    if within <= 0:
        return 0.0
    return float(between / within)


def mahalanobis_centroid(pairs, feat_names):
    """Mahalanobis distance between group centroids (pooled cov, ridge).
    pairs: [(name, (arrA, arrB)) ...] returns per-feature-pair scalar."""
    out = {}
    for name, (a, b) in pairs.items():
        A = np.asarray(a, dtype=float)
        B = np.asarray(b, dtype=float)
        f = np.isfinite(A) & np.isfinite(B)
        A, B = A[f], B[f]
        if len(A) < 2 or len(B) < 2:
            out[name] = 0.0
            continue
        d = A.mean() - B.mean()
        var = (A.var(ddof=1) + B.var(ddof=1)) / 2.0
        var += 1e-9
        out[name] = float(abs(d) / math.sqrt(var))
    return out


# silhouette with a tiny fallback (sklearn unavailable guard)
def silhouette_score_guarded(X, labels):
    try:
        from sklearn.metrics import silhouette_score
        if len(set(labels)) < 2 or len(X) < 3:
            return 0.0
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return 0.0


SPLITS = []
_GROUPS = ["A", "B", "C", "D"]
for i in range(4):
    for j in range(i + 1, 4):
        SPLITS.append((_GROUPS[i] + "|" + _GROUPS[j], [_GROUPS[i]], [_GROUPS[j]]))
for i in range(4):
    for j in range(4):
        for k in range(j + 1, 4):
            if i == j or i == k:
                continue
            SPLITS.append((_GROUPS[i] + "|" + _GROUPS[j] + _GROUPS[k], [_GROUPS[i]], [_GROUPS[j], _GROUPS[k]]))
for i in range(4):
    rest = [g for g in _GROUPS if g != _GROUPS[i]]
    SPLITS.append((_GROUPS[i] + "|" + "".join(rest), [_GROUPS[i]], rest))
for i in range(4):
    for j in range(i + 1, 4):
        rest = [g for g in _GROUPS if g != _GROUPS[i] and g != _GROUPS[j]]
        if len(rest) == 2:
            SPLITS.append((_GROUPS[i] + _GROUPS[j] + "|" + "".join(rest), [_GROUPS[i], _GROUPS[j]], rest))

CORE_FEATS = ["impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"]
ALL_FEATS = CORE_FEATS + [
    "spectral_centroid_hz", "spectral_bandwidth_hz",
    "spectral_rolloff_hz", "spectral_flatness",
    "rms_energy", "zcr",
    # onset dynamics — added from onset_result.md (2026-08-21)
    "attack_slope_db_ms",   # 19.08 dB / rise_time_ms; B>>D>C>A expected
    "hf_transient_ratio",   # HF(>1kHz) / total STFT power
    "crest_factor",         # peak_amp / RMS
]


def group_arrays(rows, feat):
    by = {g: [] for g in _GROUPS}
    for r in rows:
        val = r.get(feat)
        if val is not None and np.isfinite(val):
            by[r["group"]].append(float(val))
    return {g: np.asarray(by[g]) for g in _GROUPS}


def metrics_for(rows, feats=None, split=None):
    """Compute the full metric bundle. Returns dict:
       split-focused p-values (all 25, best), fisher, mahal, silhouette,
       log10-score (diff objective), merged CV (sim objective).
    """
    feats = feats or CORE_FEATS
    split_focus = split or ("AB|CD", ["A", "B"], ["C", "D"])
    label, left, right = split_focus
    # 1) split p-values over all 25 splits (chart parity)
    ptable = {}
    for slabel, lft, rgt in SPLITS:
        for f in feats:
            lv = [r[f] for r in rows if r["group"] in lft and np.isfinite(r.get(f))]
            rv = [r[f] for r in rows if r["group"] in rgt and np.isfinite(r.get(f))]
            ptable.setdefault(slabel, {})[f] = mann_whitney_p(lv, rv)
    best_split, best_p = label, 1.0
    best_feat = feats[0]
    for slabel, cells in ptable.items():
        for f, p in cells.items():
            if p < best_p:
                best_p = p
                best_split = slabel
                best_feat = f
    # 2) fisher per feature (4 groups)
    fisher = {}
    for f in feats:
        gv = group_arrays(rows, f)
        fisher[f] = fisher_ratio([gv[g] for g in _GROUPS])
    # 3) mahalanobis for the focus split + all 6 pairs (impact_freq_hz only,
    #    the analytic separating axis; plus pooled over CORE for distance)
    mahal = {}
    for (pname, ga, gb) in [("AB|CD", ["A", "B"], ["C", "D"]),
                            ("A|D", ["A"], ["D"]),
                            ("B|C", ["B"], ["C"]),
                            ("A|B", ["A"], ["B"]),
                            ("C|D", ["C"], ["D"])]:
        d = {}
        for f in feats:
            lv = [r[f] for r in rows if r["group"] in ga and np.isfinite(r.get(f))]
            rv = [r[f] for r in rows if r["group"] in gb and np.isfinite(r.get(f))]
            d[f] = abs((np.mean(lv) - np.mean(rv)) / math.sqrt(np.var(lv, ddof=1) + np.var(rv, ddof=1)) / 2 + 1e-12)
        mahal[pname] = d
    # 4) silhouette on z-scored CORE
    X = []
    labs = []
    for r in rows:
        vals = [r.get(f) for f in CORE_FEATS]
        if any(v is None or not np.isfinite(v) for v in vals):
            continue
        X.append(vals)
        labs.append(r["group"])
    sil = 0.0
    if len(X) >= 3 and len(set(labs)) >= 2:
        XX = np.asarray(X, dtype=float)
        mu, sd = XX.mean(axis=0), XX.std(axis=0) + 1e-12
        ZZ = (XX - mu) / sd
        sil = silhouette_score_guarded(ZZ, labs)
    # 5) objectives
    # focus-split log10 score (diff objective for the requested split)
    focus_score = 0.0
    focus_p = {}
    for slabel, lft, rgt in SPLITS:
        if slabel == label:
            for f in feats:
                p = ptable[slabel][f]
                focus_p[f] = p
                focus_score += -math.log10(max(p, 1e-12))
    # merged CV for the focus split (sim objective)
    merged_cv = {}
    for f in feats:
        lv = [r[f] for r in rows if r["group"] in left and np.isfinite(r.get(f))]
        rv = [r[f] for r in rows if r["group"] in right and np.isfinite(r.get(f))]
        pool = np.concatenate([lv, rv])
        if len(pool) < 3 or abs(pool.mean()) < 1e-12:
            merged_cv[f] = 1.0
        else:
            merged_cv[f] = float(pool.std() / abs(pool.mean()))
    return {
        "ptable": ptable, "best_split": best_split, "best_p": best_p,
        "best_feat": best_feat, "fisher": fisher, "mahal": mahal,
        "silhouette": sil, "focus_score": focus_score, "focus_p": focus_p,
        "merged_cv": merged_cv, "label": label,
    }


def bestsplit_for(rows, feats):
    best = "AB|CD"
    bestp = 1.0
    for slabel, lft, rgt in SPLITS:
        for f in feats:
            lv = [r[f] for r in rows if r["group"] in lft and np.isfinite(r.get(f))]
            rv = [r[f] for r in rows if r["group"] in rgt and np.isfinite(r.get(f))]
            p = mann_whitney_p(lv, rv)
            if p < bestp:
                bestp = p
                best = slabel
    return best


# ---------------------------------------------------------------------------
# 3b. TV Game Verdict Screen — statistical verdict reference (Python mirror
#     of index.html JS functions). Pure, deterministic, vectorized (G7).
# ---------------------------------------------------------------------------

# All 6 unordered pairwise group keys in canonical alphabetical order.
_PAIRS = [("A", "B"), ("A", "C"), ("A", "D"),
          ("B", "C"), ("B", "D"), ("C", "D")]


def bonferroni_alpha(n_tests, alpha=0.05):
    """Bonferroni-corrected alpha: alpha / max(n_tests, 1)."""
    return alpha / max(n_tests, 1)


def _resolve_std(scaler_std, feat):
    """Extract the std for *feat* from either a flat {feat: std} dict or a
    scaler dict that carries 'std' (array) + 'features' (list), mirroring
    index.html ``frozenStd()``.  Falls back to 1.0 when absent."""
    if scaler_std is None:
        return 1.0
    if isinstance(scaler_std, dict):
        if feat in scaler_std:
            return float(scaler_std[feat])
        if "std" in scaler_std and "features" in scaler_std:
            feats = scaler_std["features"]
            stds = scaler_std["std"]
            if feat in feats:
                return float(stds[feats.index(feat)])
    return 1.0


def _group_rows(rows, group):
    """Return list of rows belonging to *group* (fast filter)."""
    return [r for r in rows if r.get("group") == group]


def _feat_array(rows, feat):
    """Extract finite values for *feat* from *rows* as a 1-D ndarray."""
    vals = [r.get(feat) for r in rows
            if r.get(feat) is not None and np.isfinite(r.get(feat))]
    return np.asarray(vals, dtype=float)


def _merged_cv(rows_a, rows_b, feats):
    """Per-feature merged coefficient of variation (std/|mean|) for the pooled
    set of *rows_a* + *rows_b*.  Returns {feat: cv} mirroring ``metrics_for``
    merged_cv."""
    out = {}
    pooled = rows_a + rows_b
    for f in feats:
        v = _feat_array(pooled, f)
        if len(v) < 3 or abs(v.mean()) < 1e-12:
            out[f] = 1.0
        else:
            out[f] = float(v.std(ddof=1) / abs(v.mean()))
    return out


def loo_cv(rows, feats, groups):
    """LOO nearest-centroid accuracy (0..1).

    For each row *i*, build centroids from **all other rows** (z-scored by
    pooled std across all groups), classify the held-out row to the nearest
    centroid, return fraction correct.

    Groups not present in the remaining rows are silently skipped.
    """
    n = len(rows)
    if n < 2:
        return 0.0

    # Pre-extract values into a matrix for vectorised z-scoring.
    vals = np.array(
        [[r.get(f, np.nan) for f in feats] for r in rows], dtype=float
    )
    grp = np.array([r.get("group") for r in rows])

    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        other_vals = vals[mask]                  # (n-1, n_feats)
        other_grp = grp[mask]

        # Pooled mean / std across *all* other rows (G7: vectorised).
        valid = np.isfinite(other_vals)
        counts = valid.sum(axis=0)
        if np.any(counts < 2):
            continue                               # not enough data for std
        pooled_mean = np.nanmean(other_vals, axis=0)
        pooled_std = np.nanstd(other_vals, axis=0, ddof=1)
        pooled_std = np.where(pooled_std > 1e-12, pooled_std, 1e-12)

        # Z-score other rows and the held-out row.
        other_z = (other_vals - pooled_mean) / pooled_std
        held_z = (vals[i] - pooled_mean) / pooled_std

        # Centroids per group (from z-scored others).
        cents = {}
        for g in groups:
            gm = other_grp == g
            if gm.sum() > 0:
                gc = other_z[gm]
                fv = np.isfinite(gc)
                if np.any(fv.all(axis=1)):
                    cents[g] = np.nanmean(gc, axis=0)

        if grp[i] not in cents or not np.all(np.isfinite(held_z)):
            continue

        # Nearest centroid (squared Euclidean in z-space).
        best_g = min(cents, key=lambda g: float(np.sum((held_z - cents[g]) ** 2)))
        if best_g == grp[i]:
            correct += 1

    return correct / n


def mean_z_gap(rows_a, rows_b, feat, scaler_std):
    """Return {mean_a, mean_b, z_gap, p, n_a, n_b} for a single feature.

    ``z_gap = (mean_a - mean_b) / scaler_std[feat]``
    ``p      = mann_whitney_p(a_vals, b_vals)`` (two-sided, tie-corrected)
    """
    va = _feat_array(rows_a, feat)
    vb = _feat_array(rows_b, feat)
    n_a, n_b = len(va), len(vb)
    mean_a = float(va.mean()) if n_a else 0.0
    mean_b = float(vb.mean()) if n_b else 0.0
    std = _resolve_std(scaler_std, feat)
    z_gap = (mean_a - mean_b) / std if std > 0 else 0.0
    p = mann_whitney_p(va, vb)
    return {"mean_a": mean_a, "mean_b": mean_b,
            "z_gap": float(z_gap), "p": float(p),
            "n_a": n_a, "n_b": n_b}


def verdict_merge(rows, pair, feats):
    """Screen A (တူ / MERGE).

    *pair* is a two-letter group key such as ``'AB'``.  Computes the merged CV
    (pooled std / |pooled mean|) for every feature, picks the best feature,
    and compares against all six pairwise CVs to decide the verdict.

    Returns ``{best_feat, cv, verdict, n_a, n_b, all_cv}`` where ``all_cv``
    maps every pair key (e.g. ``'AB'``) to its best CV across *feats*.

    verdict:
      * ``'good'`` — CV ≤ 0.05 **and** this pair's CV is the global minimum
      * ``'warn'`` — CV ≤ 0.05 but not the global minimum
      * ``'bad'``  — CV > 0.05
    """
    a, b = pair[0], pair[1]
    all_cv = {}
    pair_cvs = {}
    for ga, gb in _PAIRS:
        ra = _group_rows(rows, ga)
        rb = _group_rows(rows, gb)
        cvs = _merged_cv(ra, rb, feats)
        best_cv = min(cvs.values()) if cvs else 1.0
        all_cv[ga + gb] = best_cv
        if (ga, gb) == (a, b):
            pair_cvs = cvs

    best_feat = min(pair_cvs, key=pair_cvs.get) if pair_cvs else feats[0]
    cv = pair_cvs.get(best_feat, 1.0)
    global_min = min(all_cv.values()) if all_cv else 1.0
    pair_label = a + b

    if cv <= 0.05 and abs(all_cv.get(pair_label, 1.0) - global_min) < 1e-15:
        verdict = "good"
    elif cv <= 0.05:
        verdict = "warn"
    else:
        verdict = "bad"

    n_a = len(_group_rows(rows, a))
    n_b = len(_group_rows(rows, b))
    return {"best_feat": best_feat, "cv": cv, "verdict": verdict,
            "n_a": n_a, "n_b": n_b, "all_cv": all_cv}


def verdict_split(rows, key, feats, n_scanned=1):
    """Screen B (ကွဲ / SPLIT).

    *key* is a split label such as ``'AB|CD'`` (left | right).  Computes the
    Mann-Whitney p-value for every feature, picks the best feature, applies
    Bonferroni correction when many splits were scanned, and runs LOO CV.

    Returns ``{best_feat, p, loo, bonf_alpha, verdict, n, pvals}``.

    bonferroni_alpha:
      * 0.05 / max(n_scanned, 1)  when ``n_scanned >= 35``
      * 0.05                     otherwise

    verdict:
      * ``'bad'``  — p < bonf_alpha
      * ``'warn'`` — bonf_alpha ≤ p < 0.1
      * ``'good'`` — p ≥ 0.1
    """
    left_str, right_str = key.split("|")
    left_groups = list(left_str)
    right_groups = list(right_str)
    n = len(rows)

    pvals = {}
    for f in feats:
        lv = [r.get(f) for r in rows if r["group"] in left_groups
              and np.isfinite(r.get(f, np.nan))]
        rv = [r.get(f) for r in rows if r["group"] in right_groups
              and np.isfinite(r.get(f, np.nan))]
        pvals[f] = mann_whitney_p(lv, rv)

    best_feat = min(pvals, key=pvals.get) if pvals else feats[0]
    p = pvals.get(best_feat, 1.0)
    bonf_alpha = bonferroni_alpha(n_scanned) if n_scanned >= 35 else 0.05
    groups = [g for g in _GROUPS if _group_rows(rows, g)]
    loo = loo_cv(rows, feats, groups) if groups else 0.0

    if p < bonf_alpha:
        verdict = "bad"
    elif p < 0.1:
        verdict = "warn"
    else:
        verdict = "good"

    return {"best_feat": best_feat, "p": p, "loo": loo,
            "bonf_alpha": bonf_alpha, "verdict": verdict,
            "n": n, "pvals": pvals}


def verdict_single(rows, feat, key, scaler_std):
    """Screen C (VALUE / SINGLE).

    *key* is a two-letter pair such as ``'AB'``.  Delegates to ``mean_z_gap``
    for the z-gap and Mann-Whitney p, then classifies.

    Returns ``{mean_a, mean_b, z_gap, p, verdict, n_a, n_b}``.

    verdict:
      * ``'bad'`` — p < 0.05
      * ``'good'`` — p ≥ 0.05
    """
    a, b = key[0], key[1]
    r = mean_z_gap(_group_rows(rows, a), _group_rows(rows, b), feat, scaler_std)
    verdict = "bad" if r["p"] < 0.05 else "good"
    return {"mean_a": r["mean_a"], "mean_b": r["mean_b"],
            "z_gap": r["z_gap"], "p": r["p"], "verdict": verdict,
            "n_a": r["n_a"], "n_b": r["n_b"]}


# ---------------------------------------------------------------------------
# 4. parameter space + optuna
# ---------------------------------------------------------------------------

class Params:
    def __init__(self, sr=44100, nfft=4096, hop=1024, topdb=35, win=0.5):
        self.sr = int(sr)
        self.nfft = int(nfft)
        self.hop = int(hop)
        self.topdb = int(topdb)
        self.win = float(win)

    def asdict(self):
        return {"sr": self.sr, "n_fft": self.nfft, "hop_length": self.hop,
                "top_db": self.topdb, "window_sec": self.win}

    @staticmethod
    def fromdict(d):
        return Params(d.get("sr", 44100), d.get("n_fft", 4096),
                      d.get("hop_length", 1024), d.get("top_db", 35),
                      d.get("window_sec", 0.5))

    def __repr__(self):
        return f"Params(sr={self.sr}, nfft={self.nfft}, hop={self.hop}, top_db={self.topdb}, win={self.win})"


SR_CHOICES = [44100, 48000, 96000, 192000, 384000]
NFFT_CHOICES = [512, 1024, 2048, 4096, 8192, 16384]
HOP_CHOICES = [128, 256, 512, 1024, 2048, 4096]
DB_RANGE = range(10, 61)
WIN_CHOICES = [round(0.10 + 0.05 * i, 2) for i in range(19)]


def suggest_params(trial):
    return Params(
        sr=trial.suggest_categorical("sr", SR_CHOICES),
        nfft=trial.suggest_categorical("n_fft", NFFT_CHOICES),
        hop=trial.suggest_categorical("hop_length", HOP_CHOICES),
        topdb=trial.suggest_int("top_db", 10, 60),
        win=trial.suggest_categorical("window_sec", WIN_CHOICES),
    )


class Objective:
    """mode: 'diff' (maximize -log10 p on split/pair), 'sim' (minimize merged
    CV of the pair), 'generic' (maximize separation across ALL 25 splits:
    min p wins, tie-break by silhouette), 'relationships' (minimize L_attract
    plus V_min collapse guard using a frozen relationship table)."""

    def __init__(self, ds, mode, split=None, pair=None, feats=None, trials=50,
                 relationships=None, frozen_scaler=None, collapse_guard=None):
        self.ds = ds
        self.mode = mode
        self.feats = feats or CORE_FEATS
        self.trials = trials
        self.relationships = relationships or []
        self.frozen_scaler = frozen_scaler or {}
        self.collapse_guard = collapse_guard or {"V_min": 0.01, "lambda_vmin": 1000.0}
        if pair:
            self.focus = (pair[0] + pair[1] + "|" + "".join(g for g in _GROUPS if g not in pair),
                          [pair[0]], [pair[1]])
        elif split:
            left = [c for c in split[0]]
            right = [c for c in split[1]]
            self.focus = (split[0] + "|" + split[1], left, right)
        else:
            self.focus = ("AB|CD", ["A", "B"], ["C", "D"])
        self.best = None

    def _score(self, rows):
        if self.mode == "relationships":
            from relationships import L_attract_value, L_guard_value, compute_centroids, compute_V
            L_a = L_attract_value(rows, self.relationships, self.frozen_scaler, self.feats)
            cents = compute_centroids(
                rows, self.feats, self.frozen_scaler["mean"], self.frozen_scaler["std"]
            )
            V = compute_V(cents)
            L_g = L_guard_value(
                V,
                V_min=self.collapse_guard.get("V_min", 0.01),
                lambda_vmin=self.collapse_guard.get("lambda_vmin", 1000.0),
            )
            return -(L_a + L_g)
        m = metrics_for(rows, self.feats, self.focus)
        if self.mode == "sim":
            cvsum = sum(m["merged_cv"].values()) / max(1, len(m["merged_cv"]))
            return -cvsum  # minimize -> maximize negative
        if self.mode == "diff":
            # Composite: focus-split -log10(p) headline, balanced by the
            # fraction of ALL 25 splits that reach p<0.05 (guards against
            # overfitting one split while the rest stay chance).
            focus = m["focus_score"]
            n_strong = 0
            for slabel, cells in m["ptable"].items():
                if min(cells.values()) < 0.05:
                    n_strong += 1
            spread = n_strong / max(1, len(m["ptable"]))
            return focus + 0.5 * spread
        # generic: overall best single p, tie-break by silhouette
        return -math.log10(max(m["best_p"], 1e-12)) + 0.25 * m["silhouette"]

    def __call__(self, trial):
        p = suggest_params(trial)
        rows = self.ds.measure_all(p)
        score = self._score(rows)
        if self.best is None or score > self.best[0]:
            self.best = (score, p)
        return score


def run_optimize(ds, mode, split=None, pair=None, feats=None, trials=50,
                 relationships=None, frozen_scaler=None, collapse_guard=None,
                 baseline_params=None):
    obj = Objective(ds, mode, split=split, pair=pair, feats=feats, trials=trials,
                    relationships=relationships, frozen_scaler=frozen_scaler,
                    collapse_guard=collapse_guard)
    if not HAS_OPTUNA:
        raise RuntimeError("optuna not installed - run: backend/.venv/bin/pip install optuna")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    if baseline_params:
        def _enqueue_baseline(trial):
            return Params.fromdict(baseline_params)
        # Trial #0: always evaluate the frozen baseline params first.
        study.enqueue_trial(baseline_params)
    study.optimize(obj, n_trials=trials, show_progress_bar=False)
    if mode == "relationships" and baseline_params:
        from relationships import select_best_trial
        trials_list = []
        for t in study.trials:
            if t.value is None:
                continue
            rec = dict(t.params)
            rec["obj"] = -t.value
            trials_list.append(rec)
        best = select_best_trial(trials_list, baseline_params)
        bp = Params.fromdict(best)
    else:
        bp = Params.fromdict(study.best_params)
    return obj, study, bp


# ---------------------------------------------------------------------------
# 5. per-parameter sensitivity (OAT) + causal chain
# ---------------------------------------------------------------------------

def oat_sensitivity(ds, base, mode, focus, feats, done_rows=None):
    """One-at-a-time: vary each param +/- one step from base, recompute the
    objective, report delta. Returns per-param importance + causal chain:
    param -> which FEATURES moved -> which splits' p changed."""
    base_rows = done_rows if done_rows is not None else ds.measure_all(base)
    def score_of(rows):
        m = metrics_for(rows, feats, focus)
        return m

    base_m = score_of(base_rows)
    base_val = base_m["focus_score"] if mode == "diff" else -np.mean(list(base_m["merged_cv"].values()))
    if mode == "generic":
        base_val = -math.log10(max(base_m["best_p"], 1e-12)) + 0.25 * base_m["silhouette"]

    steps = []
    sri = SR_CHOICES.index(base.sr) if base.sr in SR_CHOICES else 0
    nffti = NFFT_CHOICES.index(base.nfft) if base.nfft in NFFT_CHOICES else 2
    hopi = HOP_CHOICES.index(base.hop) if base.hop in HOP_CHOICES else 1
    wini = WIN_CHOICES.index(base.win) if base.win in WIN_CHOICES else 0
    cands = []
    for d in (-1, 1):
        if 0 <= sri + d < len(SR_CHOICES):
            cands.append(("sr", SR_CHOICES[sri + d]))
        if 0 <= nffti + d < len(NFFT_CHOICES):
            cands.append(("n_fft", NFFT_CHOICES[nffti + d]))
        if 0 <= hopi + d < len(HOP_CHOICES):
            cands.append(("hop_length", HOP_CHOICES[hopi + d]))
        if 0 <= wini + d < len(WIN_CHOICES):
            cands.append(("window_sec", WIN_CHOICES[wini + d]))
    for db in (base.topdb - 1, base.topdb + 1):
        if 10 <= db <= 60:
            cands.append(("top_db", db))

    import copy
    for name, val in cands:
        p = copy.deepcopy(base)
        setattr(p, name if name != "n_fft" else "nfft", val)
        rows = ds.measure_all(p)
        m = score_of(rows)
        if mode == "diff":
            v = m["focus_score"]
        elif mode == "sim":
            v = -np.mean(list(m["merged_cv"].values()))
        else:
            v = -math.log10(max(m["best_p"], 1e-12)) + 0.25 * m["silhouette"]
        # feature deltas (mean over rows) for the causal chain
        fdeltas = {}
        fstats = {}
        for f in feats:
            bvals = [r[f] for r in base_rows if np.isfinite(r.get(f))]
            vals = [r[f] for r in rows if np.isfinite(r.get(f))]
            if not bvals or not vals:
                continue
            bm, vm = float(np.mean(bvals)), float(np.mean(vals))
            fstats[f] = {"base_mean": round(bm, 6), "new_mean": round(vm, 6),
                         "delta_mean": round(vm - bm, 6)}
            fdeltas[f] = abs(vm - bm)
        top_feats = sorted(fdeltas, key=fdeltas.get, reverse=True)[:3]
        steps.append({
            "param": name, "from": base.asdict().get(name, val),
            "to": val, "objective_before": round(base_val, 4),
            "objective_after": round(v, 4),
            "delta": round(v - base_val, 4),
            "moved_features": top_feats,
            "feature_stats": fstats,
        })
    steps.sort(key=lambda s: -abs(s["delta"]))
    return steps, base_m


# ---------------------------------------------------------------------------
# 6. report writer
# ---------------------------------------------------------------------------

def fmt_p(p):
    if not np.isfinite(p):
        return "1.0000"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def write_report(outdir, ds, mode, focus, best_params, obj, study, feats,
                 steps, base_m, rows, elapsed, relationships=None,
                 frozen_scaler=None, collapse_guard=None):
    os.makedirs(outdir, exist_ok=True)
    # params.json
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump({"mode": mode, "focus": "".join(focus[0]) + "|" + "".join(focus[1]),
                   "params": best_params.asdict(), "trials": study.best_trial.number + 1,
                   "best_value": study.best_value,
                   "seed": 42}, f, indent=2)
    # features.csv (winning params)
    csv_rows = rows
    feats_out = ["file", "group"] + feats
    with open(os.path.join(outdir, "features.csv"), "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(feats_out)
        for r in csv_rows:
            row = [r.get("file", ""), r.get("group", "")]
            for ft in feats:
                v = r.get(ft)
                row.append("" if v is None else (f"{v:.6f}" if isinstance(v, float) else v))
            w.writerow(row)
    # report.json
    m = metrics_for(rows, feats, focus)
    ptable_best = {}
    for slabel, cells in m["ptable"].items():
        ptable_best[slabel] = {f: fmt_p(p) for f, p in cells.items()}
    report = {
        "mode": mode,
        "focus": "".join(focus[0]) + "|" + "".join(focus[1]),
        "best_params": best_params.asdict(),
        "objective": round(float(study.best_value), 5),
        "best_split": m["best_split"],
        "best_p": fmt_p(m["best_p"]),
        "best_feat": m["best_feat"],
        "silhouette": round(m["silhouette"], 4),
        "fisher_per_feature": {f: round(v, 4) for f, v in m["fisher"].items()},
        "mahalanobis_pairs": {p: {f: round(v, 4) for f, v in cells.items()}
                              for p, cells in m["mahal"].items()},
        "split_pvalues": ptable_best,
        "param_importance_oat": [
            {k: (round(v, 6) if isinstance(v, float) else v)
             for k, v in s.items() if k != "feature_stats"}
            for s in steps[:8]
        ],
        "n_files": len(rows),
        "groups": {g: int(sum(1 for r in rows if r["group"] == g)) for g in _GROUPS},
        "elapsed_s": round(elapsed, 1),
    }
    if mode == "relationships" and relationships is not None and frozen_scaler is not None and collapse_guard is not None:
        from relationships import compute_centroids, compute_V, L_guard_value
        cents = compute_centroids(rows, feats, frozen_scaler["mean"], frozen_scaler["std"])
        dm_after = {}
        for i, g in enumerate(_GROUPS):
            if g not in cents:
                continue
            for h in _GROUPS[i + 1:]:
                if h not in cents:
                    continue
                K = len(feats)
                dm_after[g + h] = math.sqrt(sum((cents[g][k] - cents[h][k]) ** 2 for k in range(K)) / K)
        per_pair = []
        L_attract = 0.0
        for rel in relationships:
            pair = rel["pair"]
            d_after = dm_after.get(pair, 0.0)
            tau = rel.get("tau_gh", 0.0)
            w = rel.get("weight", 1.0)
            L_pair = w * max(0.0, d_after - tau) ** 2 if rel["type"] == "attract" else 0.0
            L_attract += L_pair
            per_pair.append({
                "pair": pair,
                "type": rel["type"],
                "d_before": rel.get("d_baseline", 0.0),
                "d_after": round(d_after, 6),
                "tau_gh": rel.get("tau_gh"),
                "L_pair": round(L_pair, 8),
                "contribution_pct": 0.0,
            })
        for p in per_pair:
            p["contribution_pct"] = round(p["L_pair"] / L_attract * 100.0, 3) if L_attract > 0 else 0.0
        V_after = compute_V(cents)
        L_guard = L_guard_value(
            V_after,
            V_min=collapse_guard.get("V_min", 0.01),
            lambda_vmin=collapse_guard.get("lambda_vmin", 1000.0),
        )
        report["objective"] = {
            "L_total": round(L_attract + L_guard, 8),
            "L_attract": round(L_attract, 8),
            "L_guard": round(L_guard, 8),
        }
        report["per_pair"] = per_pair
        report["V_min_guard"] = {
            "V_after": round(V_after, 6),
            "L_guard": round(L_guard, 8),
            "V_min": collapse_guard.get("V_min", 0.01),
        }
        report["collapse_guard"] = collapse_guard
        report["tie_break"] = {"mode": "closest_to_baseline", "used": True}
        report["relationships_frozen"] = True
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    # report.md
    md = []
    md.append("# DSP Inverse Optimizer - Report")
    md.append("")
    md.append(f"- **Mode**: `{mode}` (focus split `{''.join(focus[0])}|{''.join(focus[1])}`)")
    md.append(f"- **Best DSP params**: `sr={best_params.sr}`, `n_fft={best_params.nfft}`, "
              f"`hop_length={best_params.hop}`, `top_db={best_params.topdb}`, `window={best_params.win}s`")
    md.append(f"- **Objective** (trial {study.best_trial.number + 1}/{len(study.trials)}): `{study.best_value:.5f}`")
    md.append(f"- **Best overall split**: `{m['best_split']}` p={fmt_p(m['best_p'])} @ `{m['best_feat']}`")
    md.append(f"- **Silhouette** (z-scored core features): {m['silhouette']:.4f}")
    md.append(f"- **Files used**: {len(rows)}  (A={sum(1 for r in rows if r['group']=='A')} "
              f"B={sum(1 for r in rows if r['group']=='B')} "
              f"C={sum(1 for r in rows if r['group']=='C')} "
              f"D={sum(1 for r in rows if r['group']=='D')})")
    md.append("")
    md.append("## Feature importance (one-at-a-time param sensitivity)")
    md.append("")
    md.append("| param | from -> to | objective delta | most-moved features |")
    md.append("|---|---|---|---|")
    for s in steps[:8]:
        md.append(f"| {s['param']} | {s['from']} -> {s['to']} | {s['delta']:+.4f} "
                  f"| {', '.join(s['moved_features'])} |")
    md.append("")
    md.append("## Mathematical relationship (parameter -> feature -> separation)")
    md.append("")
    for s in steps[:4]:
        md.append(f"### `{s['param']}`: {s['from']} -> {s['to']}")
        for f in s["feature_stats"]:
            fs = s["feature_stats"][f]
            md.append(f"- `{f}`: {fs['base_mean']} -> {fs['new_mean']} "
                      f"(delta {fs['delta_mean']:+.6f})")
        md.append("")
    md.append("## 25-split p-value table (winning params)")
    md.append("")
    md.append("| split | " + " | ".join(feats) + " | best |")
    md.append("|" + "---|" * (len(feats) + 2))
    for slabel, cells in m["ptable"].items():
        row = "| " + slabel + " | " + " | ".join(fmt_p(cells[f]) for f in feats) + " |"
        row += " " + fmt_p(min(cells.values())) + " |"
        md.append(row)
    md.append("")
    md.append("## Fisher ratio per feature")
    md.append("")
    for f, v in m["fisher"].items():
        md.append(f"- `{f}`: {v:.4f}")
    md.append("")
    md.append("## Mahalanobis centroid distance (per feature-pair, focus split)")
    md.append("")
    for pname, cells in m["mahal"].items():
        md.append(f"- `{pname}`: " + ", ".join(f"{f}={v:.3f}" for f, v in cells.items()))
    md.append("")
    md.append("> Reproduce: `python3 dsp_inverse.py reproduce --out " + os.path.basename(outdir) + "`")
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(md))

    # before/after PNG (if matplotlib)
    if HAS_MPL:
        base_rows = ds.measure_all(Params())
        xf, yf = "impact_freq_hz", "impact"
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        GHEX = {"A": "#ff4d6d", "B": "#4d9fff", "C": "#3ddc84", "D": "#ff9f43"}
        for axx, rows, title in ((ax[0], base_rows, "Before (default params)"),
                                 (ax[1], rows, "After (optimized params)")):
            for g in _GROUPS:
                xs = [r[xf] for r in rows if r["group"] == g and np.isfinite(r.get(xf))]
                ys = [r[yf] for r in rows if r["group"] == g and np.isfinite(r.get(yf))]
                axx.scatter(xs, ys, s=22, c=GHEX[g], label=g, alpha=0.85)
            axx.set_xlabel(xf)
            axx.set_ylabel(yf)
            axx.set_title(title)
            axx.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "before_after.png"), dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def cmd_baseline(args):
    ds = Dataset(args.wavdir, args.labels)
    p = Params(args.sr, args.nfft, args.hop, args.topdb, args.win)
    rows = ds.measure_all(p)
    if not rows:
        print("no rows measured - check wavdir/labels")
        return 1
    m = metrics_for(rows, CORE_FEATS, ("AB|CD", ["A", "B"], ["C", "D"]))
    print(f"n={len(rows)}  " + "  ".join(f"{g}={sum(1 for r in rows if r['group']==g)}" for g in _GROUPS))
    print(f"params: sr={p.sr} nfft={p.nfft} hop={p.hop} top_db={p.topdb} win={p.win}")
    print(f"best split: {m['best_split']} p={fmt_p(m['best_p'])} @ {m['best_feat']}")
    print(f"silhouette: {m['silhouette']:.4f}   fisher(impact_freq): {m['fisher'].get('impact_freq_hz', 0):.3f}")
    print("per-feature best split p:")
    for f in CORE_FEATS:
        bp = min(cells[f] for cells in m["ptable"].values())
        sl = [s for s, c in m["ptable"].items() if c[f] == bp][0]
        print(f"  {f:16s} best p={fmt_p(bp)} @ {sl}")
    return 0


def _curriculum_merge(prior, rows, feats, pooled_mean, pooled_std):
    """Merge a prior approved relationships.json into a fresh baseline.

    C1 single code path: infer relationships on fresh centroids/dm, then
    overlay manual entries verbatim (immutable semantics) with d_after set
    to the fresh distance. Auto entries keep their freshly-inferred type.

    A1: V_min/lambda/guard_formula are NOT recomputed here — they are read
    from prior["collapse_guard"] by the caller.

    Returns (merged_rels, provenance).
    """
    from relationships import compute_centroids, compute_distance_matrix, infer_relationships, PAIRS
    cents = compute_centroids(rows, feats, pooled_mean, pooled_std)
    dm = compute_distance_matrix(cents)
    if prior and "tau" in prior:
        ts = prior["tau"].get("tau_similar", 0.4)
        td = prior["tau"].get("tau_different", 1.0)
        alpha = prior["tau"].get("alpha", 0.5)
        tmax = prior["tau"].get("tau_abs_max", 0.5)
    else:
        ts, td, alpha, tmax = 0.4, 1.0, 0.5, 0.5
    # single inference code path over fresh dm [C1]
    inferred = {r["pair"]: r for r in infer_relationships(dm, ts, td, alpha, tmax)}
    merged, n_manual, n_auto = [], 0, 0
    prior_rels = prior["relationships"] if prior else []
    prior_by_pair = {r["pair"]: r for r in prior_rels}
    for pair in PAIRS:
        d = round(float(dm.get(pair, 0.0)), 6)
        if pair in prior_by_pair and prior_by_pair[pair].get("source") == "manual":
            m = prior_by_pair[pair]
            merged.append({
                "pair": pair,
                "type": m["type"],            # immutable manual semantics
                "d_baseline": d,             # fresh distance
                "tau_gh": m.get("tau_gh", 0.0),
                "weight": m.get("weight", 1.0),
                "source": "manual",
                "d_after": d,
            })
            n_manual += 1
        else:
            r = inferred.get(pair, {"pair": pair, "type": "neutral",
                                    "d_baseline": d, "source": "auto"})
            r["d_after"] = d
            merged.append(r)
            n_auto += 1
    provenance = {
        "prior_run_id": prior.get("run_id") if prior else None,
        "prior_sha": freeze_table_sha256(prior) if prior else None,
        "merged_manual": n_manual,
        "re_inferred_auto": n_auto,
    }
    return merged, provenance


def freeze_table_sha256(table_or_path):
    """sha256 of a frozen relationships.json (for provenance tracking)."""
    import hashlib as _hl
    if isinstance(table_or_path, str) and table_or_path.endswith(".json"):
        raw = open(table_or_path, "rb").read()
    else:
        raw = json.dumps(table_or_path, sort_keys=True).encode()
    return _hl.sha256(raw).hexdigest()[:16]


def cmd_relationships_baseline(args):
    """Compute baseline relationships and freeze JSON to dsp_out/<tag>/."""
    from relationships import build_baseline_table, freeze_table, load_frozen, infer_relationships
    ds = Dataset(args.wavdir, args.labels)
    p = Params()
    rows = ds.measure_all(p)
    if not rows:
        print("no rows measured - check wavdir/labels")
        return 1
    feats = CORE_FEATS
    vals = np.array([[r[f] for f in feats
                      if f in r and np.isfinite(r.get(f))] for r in rows
                     if all(f in r and np.isfinite(r.get(f)) for f in feats)])
    if len(vals) == 0:
        print("no finite rows for scaler")
        return 1
    pooled_mean = vals.mean(axis=0).tolist()
    pooled_std = vals.std(axis=0, ddof=1).tolist()
    pooled_std = [max(s, 1e-12) for s in pooled_std]
    manifest_hash = wavdir_manifest_hash(args.wavdir)
    features_csv_sha = wavdir_placeholder_features_hash(rows, feats)
    replay = {
        "wavdir_manifest_hash": manifest_hash,
        "features_csv_sha256": features_csv_sha,
        "pipeline_code_hash": "git:" + _git_head(),
        "stft_config": {"n_fft": p.nfft, "hop_length": p.hop, "window": "hann", "n_mels": 40},
        "env": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    prior = load_frozen(args.prior) if args.prior else None
    merged_rels, provenance = _curriculum_merge(prior, rows, feats, pooled_mean, pooled_std)
    protected = {r["pair"] for r in merged_rels if r.get("source") == "manual"}
    prior_guard = prior["collapse_guard"] if (prior and "collapse_guard" in prior) else {}
    V_min = prior_guard.get("V_min", 0.01)
    lambda_vmin = prior_guard.get("lambda_vmin", 1000.0)
    guard_formula = prior_guard.get("guard_formula", "normalized_hinge_v1")
    table = build_baseline_table(
        run_id=args.out,
        rows=rows,
        features=feats,
        pooled_mean=pooled_mean,
        pooled_std=pooled_std,
        theta=p.asdict(),
        anchor=args.anchor,
        replay=replay,
        V_min=V_min,
        lambda_vmin=lambda_vmin,
        guard_formula=guard_formula,
        protected=protected,
        relationships_override=merged_rels,
        provenance=provenance,
    )
    outpath = os.path.join("dsp_out", args.out, "relationships.json")
    h = freeze_table(table, outpath)
    print(f"frozen relationships -> {outpath}")
    print(f"  sha256: {h[:16]}…")
    print(f"  attract pairs: {[r['pair'] for r in table['relationships'] if r['type']=='attract']}")
    print(f"  removed edges: {table['collapse_guard']['removed_edges']}")
    if table["collapse_guard"]["anchor"]["enabled"]:
        print(f"  anchor: {table['collapse_guard']['anchor']['group']}")
    return 0


def wavdir_placeholder_features_hash(rows, feats):
    import hashlib as _h
    flat = []
    for r in rows:
        for f in feats:
            v = r.get(f)
            flat.append(f"{v:.6f}" if isinstance(v, float) else str(v))
    return _h.sha256("|".join(flat).encode()).hexdigest()


def wavdir_manifest_hash(wavdir):
    import hashlib as _h
    if not os.path.isdir(wavdir):
        return "missing"
    files = sorted(
        (f for f in os.listdir(wavdir) if f.lower().endswith(".wav")),
        key=lambda s: s,
    )
    parts = []
    for fn in files:
        path = os.path.join(wavdir, fn)
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        try:
            with open(path, "rb") as fh:
                file_sha = _h.sha256(fh.read()).hexdigest()
        except OSError:
            file_sha = "?"
        parts.append(f"{fn}:{size}:{file_sha}")
    return _h.sha256("\n".join(parts).encode()).hexdigest()


def _git_head():
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".git", "HEAD"), "r") as f:
            return f.read().strip() or "unknown"
    except Exception:
        return "unknown"


def cmd_optimize(args):
    if not HAS_OPTUNA:
        print("optuna not installed; run: backend/.venv/bin/pip install optuna")
        return 1
    ds = Dataset(args.wavdir, args.labels)
    if args.mode == "sim":
        if not args.pair:
            print("sim mode needs --pair AB|AC|AD|BC|BD|CD")
            return 1
        pair = [args.pair[i] for i in (0, 1)]
    else:
        pair = None
    split = None
    if args.mode == "diff" and args.split:
        parts = args.split.split("|")
        split = (parts[0], parts[1])
    feats = CORE_FEATS if not args.extra else ALL_FEATS
    relationships = None
    frozen_scaler = None
    collapse_guard = None
    baseline_params = None
    if args.mode == "relationships":
        if not args.relationships:
            print("--mode relationships requires --relationships PATH")
            return 1
        from relationships import load_frozen
        table = load_frozen(args.relationships)
        relationships = list(table["relationships"])
        frozen_scaler = dict(table["scaler"])
        collapse_guard = dict(table["collapse_guard"])
        baseline_params = dict(table["theta"])
        # Skip if all neutral (flat objective)
        if not any(r["type"] == "attract" for r in relationships):
            print("no attract relationships survive — skipping optimization (flat objective)")
            outdir = os.path.join("dsp_out", args.out)
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "report.json"), "w") as f:
                json.dump({
                    "status": "skipped",
                    "reason": "no_attract_relationships",
                    "mode": "relationships",
                    "recommendation": "lower tau_similar or approve manual attract pairs",
                    "n_files": len(ds.rows),
                    "elapsed_s": 0.0,
                }, f, indent=2)
            return 0
    t0 = time.time()
    obj, study, bp = run_optimize(ds, args.mode, split=split, pair=pair, feats=feats,
                                  trials=args.trials, relationships=relationships,
                                  frozen_scaler=frozen_scaler,
                                  collapse_guard=collapse_guard,
                                  baseline_params=baseline_params)
    rows = ds.measure_all(bp)
    if args.mode == "relationships":
        steps = []
        base_m = metrics_for(rows, feats, obj.focus)
    else:
        steps, base_m = oat_sensitivity(ds, bp, args.mode, obj.focus, feats, done_rows=rows)
    outdir = os.path.join("dsp_out", args.out)
    write_report(outdir, ds, args.mode, obj.focus, bp, obj, study, feats,
                 steps, base_m, rows, time.time() - t0,
                 relationships=relationships, frozen_scaler=frozen_scaler,
                 collapse_guard=collapse_guard)
    print(f"done -> {outdir}")
    print(f"  best params: {bp}")
    if args.mode == "relationships":
        from relationships import L_attract_value, L_guard_value, compute_centroids, compute_V
        L_a = L_attract_value(rows, relationships, frozen_scaler, feats)
        cents = compute_centroids(rows, feats, frozen_scaler["mean"], frozen_scaler["std"])
        V = compute_V(cents)
        L_g = L_guard_value(V, V_min=collapse_guard.get("V_min", 0.01),
                            lambda_vmin=collapse_guard.get("lambda_vmin", 1000.0))
        print(f"  L_attract={L_a:.6f}  L_guard={L_g:.6f}  V={V:.6f}")
    else:
        print(f"  objective: {study.best_value:.5f} (trials {trials_done(study)})")
    return 0


def trials_done(study):
    return len(study.trials)


def cmd_reproduce(args):
    outdir = os.path.join("dsp_out", args.out)
    with open(os.path.join(outdir, "params.json")) as f:
        meta = json.load(f)
    bp = Params.fromdict(meta["params"])
    import glob
    wavdir = glob.glob(os.path.join(os.path.dirname(outdir), "..", "*")) or ["."]
    # wavdir is not stored in params.json; use --wavdir override
    wavdir = args.wavdir or "."
    labels = args.labels
    ds = Dataset(wavdir, labels)
    rows = ds.measure_all(bp)
    feats = CORE_FEATS
    with open(os.path.join(outdir, "features.csv"), "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["file", "group"] + feats)
        for r in rows:
            w.writerow([r["file"], r["group"]] + [f"{v:.6f}" if isinstance(v, float) else v for v in (r.get(ft) for ft in feats)])
    print(f"reproduced {len(rows)} rows at {bp} -> {outdir}/features.csv")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name, help_ in (("baseline", "measure + metrics at fixed params"),
                         ("optimize", "optuna search"),
                         ("reproduce", "re-run winning params to CSV"),
                         ("relationships-baseline", "freeze baseline relationships to JSON")):
        p = sub.add_parser(name, help=help_)
        if name == "baseline":
            p.add_argument("--wavdir", required=True)
            p.add_argument("--labels", default=None)
            p.add_argument("--sr", type=int, default=44100)
            p.add_argument("--nfft", type=int, default=4096)
            p.add_argument("--hop", type=int, default=1024)
            p.add_argument("--topdb", type=int, default=35)
            p.add_argument("--win", type=float, default=0.5)
        elif name == "optimize":
            p.add_argument("--wavdir", required=True)
            p.add_argument("--labels", default=None)
            p.add_argument("--mode", choices=["diff", "sim", "generic", "relationships"], default="generic")
            p.add_argument("--split", default=None, help="AB|CD etc for diff")
            p.add_argument("--pair", default=None, help="AB|AC|AD|BC|BD|CD for sim")
            p.add_argument("--relationships", default=None, help="path to frozen relationships.json")
            p.add_argument("--anchor", default=None, help="anchor group (A/B/C/D)")
            p.add_argument("--trials", type=int, default=50)
            p.add_argument("--extra", action="store_true", help="include 6 spectral extras")
            p.add_argument("--out", required=True)
        elif name == "reproduce":
            p.add_argument("--out", required=True)
            p.add_argument("--wavdir", default=None)
            p.add_argument("--labels", default=None)
        elif name == "relationships-baseline":
            p.add_argument("--wavdir", required=True)
            p.add_argument("--labels", default=None)
            p.add_argument("--out", required=True)
            p.add_argument("--anchor", default=None, help="anchor group (A/B/C/D)")
            p.add_argument("--prior", default=None, help="approved relationships.json to merge (manual entries preserved, auto re-inferred); V_min/lambda/guard_formula read from prior.collapse_guard")
    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return 1
    if args.cmd == "baseline":
        return cmd_baseline(args)
    if args.cmd == "relationships-baseline":
        return cmd_relationships_baseline(args)
    if args.cmd == "optimize":
        return cmd_optimize(args)
    if args.cmd == "reproduce":
        return cmd_reproduce(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())