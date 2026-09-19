#!/usr/bin/env python3
"""DSP Arena WAV-mode reference — mirrors the in-browser recipe EXACTLY.

Recipe (scripts/dsp_methods.md, as implemented by the chart's WAV mode):
  1. decode WAV (int/float PCM, stereo mixdown) → mono float, native sr
  2. IF target sr != native: LINEAR resample (np.interp on integer positions)
  3. trim: frame power (2048/512), threshold = max * 10^(-top_db/10)
  4. RMS peak (2048/512) → peak sample (frame centre, absolute)
  5. window ±win around peak → re-trim
  6. features:
       Impact          = max frame RMS (2048/512) of re-trimmed signal
       Duration        = len(st) / sr
       Peak_Time       = absolute |st| argmax position / sr
       Impact_Freq_Hz  = argmax |rfft(st, n=nfft)| in [200, min(50000, sr/2)]
                         with parabolic interpolation (same as JS)
       Mel_Mean_dB     = mean over frames of mean(10*log10(mel power + eps)),
                         mel = librosa melspectrogram (HTK, norm=None, n_mels=40,
                         fmax=min(50000, sr/2)), frames as in the JS
                         (nFrames=floor((len-nfft)/hop)+1, zero-padded tail)

Usage: env -u PYTHONPATH backend/.venv/bin/python3 ref_dsp.py WAV --sr N --nfft N --hop N --topdb N --win N
"""
import argparse, math, os, struct, sys, wave
import numpy as np
import librosa

def decode_wav(path):
    """stdlib WAV decode — mirrors JS wavDecode (fmt 1/3/0xFFFE, 8/16/24/32, stereo mixdown)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    off = 12
    fmt = None
    dstart = dlen = -1
    n = len(data)
    while off + 8 <= n:
        cid = data[off:off+4]; sz = struct.unpack("<I", data[off+4:off+8])[0]
        if cid == b"fmt ":
            audio_format = struct.unpack("<H", data[off+8:off+10])[0]
            channels = struct.unpack("<H", data[off+10:off+12])[0]
            sample_rate = struct.unpack("<I", data[off+12:off+16])[0]
            bits = struct.unpack("<H", data[off+22:off+24])[0]
            fmt = dict(audio_format=audio_format, channels=channels,
                       sample_rate=sample_rate, bits=bits)
            if sz >= 40:
                sub = struct.unpack("<H", data[off+32:off+34])[0]
                guid = data[off+34:off+38]
                if sub == 0xFFFE and guid == b"PCM": fmt["audio_format"] = 1
                if sub == 0xFFFE and guid == b"f32": fmt["audio_format"] = 3
            off += 8 + sz + (sz & 1)
        elif cid == b"data":
            dstart = off + 8; dlen = sz; break
        else:
            off += 8 + sz + (sz & 1)
    if fmt is None: raise ValueError("missing fmt chunk")
    if dstart < 0: raise ValueError("missing data chunk")
    ch = fmt["channels"]; bits = fmt["bits"]; bps = bits // 8
    is_float = fmt["audio_format"] == 3
    usable = (dlen // (bps*ch)) * (bps*ch)
    nframes = usable // (bps*ch)
    raw = data[dstart:dstart+usable]
    if is_float:
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(nframes, ch)
    elif bits == 8:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        arr = arr.reshape(nframes, ch)
    elif bits == 16:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0
        arr = arr.reshape(nframes, ch)
    elif bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(nframes*ch, 3)
        x = b[:,0].astype(np.int64) | (b[:,1].astype(np.int64) << 8) | (b[:,2].astype(np.int64) << 16)
        x = np.where(x & 0x800000, x - 0x1000000, x)
        arr = (x / 8388607.0).reshape(nframes, ch)
    else:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483647.0
        arr = arr.reshape(nframes, ch)
    mono = arr.mean(axis=1) if ch > 1 else arr[:, 0]
    return mono, fmt["sample_rate"]

def resample_linear(x, src_sr, dst_sr):
    if src_sr == dst_sr:
        return x
    n_out = max(1, int(round(len(x) * dst_sr / src_sr)))
    pos = np.arange(n_out, dtype=np.float64) * (src_sr / dst_sr)
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    i1 = np.minimum(i0 + 1, len(x) - 1)
    return x[i0] * (1 - frac) + x[i1] * frac

def trim_wav(y, sr, top_db):
    """Mirror of JS trimWav: frame power (2048/512), threshold = max*10^(-top_db/10)."""
    fl, fh = 2048, 512
    L = len(y)
    nf = max(0, int((L - fl) // fh) + 1)
    if nf < 1:
        return None
    E = np.zeros(nf)
    for k in range(nf):
        seg = y[k*fh:k*fh+fl]
        E[k] = np.mean(seg**2)
    emax = E.max()
    if emax <= 0:
        return None
    thr = emax * 10 ** (-top_db / 10)
    idx = np.flatnonzero(E > thr)
    if len(idx) == 0:
        return None
    k0, k1 = idx[0], idx[-1]
    start = int(k0 * fh)
    end = min(L, int(k1 * fh + fl))
    return y[start:end], start

def max_rms_frame(y, fl=2048, fh=512):
    if len(y) < fl:
        return float(np.sqrt(np.mean(y**2)))
    nf = int((len(y) - fl) // fh) + 1
    best = 0.0
    for k in range(nf):
        r = float(np.sqrt(np.mean(y[k*fh:k*fh+fl]**2)))
        if r > best:
            best = r
    return best

def measure(path, sr, nfft, hop, top_db, win):
    y, native = decode_wav(path)
    y = resample_linear(y, native, sr)
    t = trim_wav(y, sr, top_db)
    if t is None or len(t[0]) < max(64, int(sr * 0.01)):
        return None
    yt, t0 = t
    fl, fh = 2048, 512
    # center=False aligns the frame grid with the JS (frame k = y[k*hop : k*hop+fl])
    rms = librosa.feature.rms(y=yt, frame_length=fl, hop_length=fh, center=False)[0]
    best_k = int(np.argmax(rms))
    impact = float(rms[best_k])
    peak_sample = int(min(len(y), t0 + best_k * fh + fl // 2))
    w = int(round(win * sr))
    ws = max(0, peak_sample - w); we = min(len(y), peak_sample + w)
    seg = y[ws:we]
    t2 = trim_wav(seg, sr, top_db)
    if t2 is None or len(t2[0]) < max(64, int(sr * 0.01)):
        return None
    st = t2[0]
    pk = int(np.argmax(np.abs(st)))
    peak_time = (t0 + ws + t2[1] + pk) / sr
    duration = len(st) / sr
    impact2 = max_rms_frame(st, fl, fh)
    # FFT peak
    X = np.fft.rfft(st, n=nfft)
    M = np.abs(X)
    f = np.fft.rfftfreq(nfft, d=1.0 / sr)
    fmax = 50000.0
    fhi = min(fmax, sr / 2.0)
    band = (f >= 200) & (f <= fhi)
    Mb = M[band]
    fb = f[band]
    bi = int(np.argmax(Mb))
    bm = bi
    # parabolic interpolation on the band-local index
    if 0 < bi < len(Mb) - 1:
        y1, y2, y3 = Mb[bi-1], Mb[bi], Mb[bi+1]
        den = y1 - 2*y2 + y3
        delta = 0.5 * (y1 - y3) / den if den != 0 else 0.0
        bm = bi + max(-0.5, min(0.5, delta))
    freq = fb[bi]  # bin centre freq
    if 0 < bi < len(Mb) - 1:
        y1, y2, y3 = Mb[bi-1], Mb[bi], Mb[bi+1]
        den = y1 - 2*y2 + y3
        delta = 0.5 * (y1 - y3) / den if den != 0 else 0.0
        delta = max(-0.5, min(0.5, delta))
        freq = f[bi + (1 if delta > 0.25 else 0)]  # simplified — see below
    # proper parabolic: bin index in full array
    bi_full = int(np.argmax(Mb))
    start_bin = int(np.flatnonzero(band)[0])
    bfull = start_bin + bi_full
    if 1 <= bfull < len(M) - 1:
        y1, y2, y3 = M[bfull-1], M[bfull], M[bfull+1]
        den = y1 - 2*y2 + y3
        delta = 0.5 * (y1 - y3) / den if den != 0 else 0.0
        delta = max(-0.5, min(0.5, delta))
        freq = (bfull + delta) * sr / nfft
    else:
        freq = f[bfull]
    # Mel_Mean_dB: mean of 10*log10(mel power + eps) over frames
    S = np.abs(librosa.stft(st, n_fft=nfft, hop_length=hop, center=False)) ** 2
    mel = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=40, fmin=0.0,
                              fmax=fhi, htk=True, norm=None)
    melpow = mel @ S
    eps = np.finfo(np.float32).tiny
    mel_db = 10.0 * np.log10(np.maximum(melpow, eps))
    mel_mean_db = float(mel_db.mean())
    return dict(impact=impact2, duration=duration, peak_time=peak_time,
                mel_mean_db=mel_mean_db, impact_freq_hz=round(freq*10)/10,
                native=native, sr=sr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--nfft", type=int, default=4096)
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--topdb", type=int, default=35)
    ap.add_argument("--win", type=float, default=0.5)
    a = ap.parse_args()
    r = measure(a.wav, a.sr, a.nfft, a.hop, a.topdb, a.win)
    if r is None:
        print("REJECT (no signal after trim)")
        sys.exit(2)
    for k in ("impact", "duration", "peak_time", "mel_mean_db", "impact_freq_hz"):
        print(f"{k}={r[k]:.6f}")
    print(f"native={r['native']} sr={r['sr']}")
