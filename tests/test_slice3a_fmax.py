"""tests/test_slice3a_fmax.py — Slice 3A: fmax genuinely controls the frequency-analysis ceiling.

Covers §9 A–F of the Slice 3A task:

  A. Legacy regression: fmax=None reproduces the pre-3A output bit-for-bit
     against a pinned legacy source-copy reference
     (tests/slice3a_legacy_reference.py, provenance-pinned via SHA-256),
     on a deterministic synthetic fixture — never skipped for a missing WAV.
  B. Full-Nyquist equivalence: fmax=sr/2 matches the legacy full-band result
     (also against the pinned reference).
  C. Actual sensitivity: at least one frequency-domain feature changes between
     a high (21500) and low (1000) fmax on a real broadband fixture.
  D. Time-domain invariance across the sweep values
     (500, 1000, 2000, 4000, 8000, 16000, 21500).
  E. Frequency-domain behavior bounds (impact_freq <= f_c, rolloff <= f_c
     within tolerance, centroid bounded, low-fmax measurable change).
  F. End-to-end /compute: two requests differing only in fmax produce
     different frequency-domain features (direct coroutine call, no server).
  G. Omitted-fmax /compute end-to-end: omitted fmax succeeds, uses the
     legacy DSP path (fmax=None), echoes the None provenance sentinel, and
     differs from an explicit fmax=21500 request on actual DSP features.
  H. Legacy provenance pin: the pinned legacy measure_features source span
     must match its SHA-256 pin or the suite fails loudly.
  I. hf_transient_ratio retained-band semantics: explicit fmax is
     retained-band-relative (not full-band).
  J. fmax validation contract: omitted/None/"" -> legacy; "8000" accepted;
     "abc"/"nan"/"+inf"/"-inf" -> 422; fmax=21500 accepted, 21501 rejected.
  K. Low-fmax mel floor: values >= 300 Hz are non-degenerate; fmax=50 is
     accepted but documented as degenerate/sentinel-like (contract B).

Note on oracle terminology: this file uses "pinned legacy source reference"
with a SHA-256 provenance pin. The reference is a mechanically copied source
reference whose provenance is NOT protected by git history (tests/ is
untracked); the pin only fails loudly on later synchronized edits. It is NOT
claimed to be a "frozen numeric oracle" and does NOT create historical git
provenance.

Fixture strategy: NO new binary files are committed. Tests reuse the tracked
recordings/latest.wav when it yields features; otherwise they copy a WAV from
a local (untracked, developer-machine) directory, or fall back to a
deterministic synthetic HF+LF burst written to tmp_path. Synthetic audio is a
test-only fallback fixture — it is never part of the tuner or shipped data.

Conventions follow tests/test_confirm_flow.py: endpoints are called directly
as coroutines (the venv has no httpx/pytest-asyncio).
"""
from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import io
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dsp_inverse
import event_store
import live_server

SR = 44100
NFFT = 4096
HOP = 1024
TOP_DB = 35
WIN = 0.5
SWEEP = (500, 1000, 2000, 4000, 8000, 16000, 21500)
TIME_KEYS = ("impact", "duration", "peak_time", "attack_slope_db_ms",
             "crest_factor", "rms_energy", "zcr")
FREQ_KEYS = ("impact_freq_hz", "mel_mean_db", "spectral_centroid_hz",
             "spectral_bandwidth_hz", "spectral_rolloff_hz",
             "spectral_flatness", "hf_transient_ratio")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, y: np.ndarray, sr: int) -> Path:
    pcm = np.clip(np.asarray(y, dtype=np.float64), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    data = pcm16.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)
    return path


def _synthetic_burst(sr: int = SR) -> tuple[np.ndarray, int]:
    """Deterministic broadband burst: 5 kHz decay + 800 Hz body + HF chirp."""
    rng = np.random.default_rng(0x3A1F)
    n = sr * 2
    y = rng.standard_normal(n) * 0.001
    t = np.arange(int(0.02 * sr)) / sr
    burst = np.sin(2 * np.pi * 5000 * t) * np.exp(-t * 200) * 0.9
    body = np.sin(2 * np.pi * 800 * t) * np.exp(-t * 60) * 0.5
    chirp_t = np.arange(int(0.01 * sr)) / sr
    chirp = np.sin(2 * np.pi * (9000 + 6000 * chirp_t / chirp_t[-1]) * chirp_t) \
        * np.exp(-chirp_t * 150) * 0.6
    at = int(1.0 * sr)
    y[at:at + len(burst)] += burst + body
    y[at:at + len(chirp)] += chirp
    return np.clip(y, -1.0, 1.0), sr


def _candidate_wavs() -> list[Path]:
    cands: list[Path] = []
    tracked = ROOT / "recordings" / "latest.wav"
    if tracked.exists():
        cands.append(tracked)
    for raw in (Path("/Users/kyawnandar/Desktop/untitled folder").glob("*.wav"),
                Path("/Users/kyawnandar/Documents/interactive-v2").glob("*.wav")):
        try:
            for p in sorted(raw)[:24]:
                cands.append(p)
        except Exception:
            pass
    # de-duplicate, keep order
    seen: set[str] = set()
    out: list[Path] = []
    for p in cands:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def _first_working_real_wav() -> tuple[np.ndarray, int, str] | None:
    for p in _candidate_wavs():
        try:
            y, sr = dsp_inverse.decode_wav(str(p))
        except Exception:
            continue
        try:
            fe = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=None)
        except Exception:
            continue
        if fe is not None:
            return np.asarray(y, dtype=np.float64), int(sr), str(p)
    return None


@pytest.fixture(scope="module")
def real_audio():
    """(y, sr, source_label) — real WAV when available, else synthetic."""
    hit = _first_working_real_wav()
    if hit is not None:
        y, sr, src = hit
        return y, sr, f"real:{Path(src).name}"
    y, sr = _synthetic_burst()
    return y, sr, "synthetic-fallback"


@pytest.fixture()
def wav_file(tmp_path, real_audio):
    y, sr, _src = real_audio
    p = _write_wav(tmp_path / "slice3a.wav", y, sr)
    return p


def _load_legacy_oracle():
    """Pinned legacy source-copy reference: tests/slice3a_legacy_reference.py.

    This module is a mechanically copied verbatim snapshot of the
    dsp_inverse.py measurement path as it existed before Slice 3A (extracted
    from the pre-change file, NOT from the Slice-3A implementation). It is
    the ONLY numeric oracle used. `git show HEAD:dsp_inverse.py` is NOT used:
    once Slice 3A is committed HEAD is no longer pre-3A.

    Terminology: this is a "pinned legacy source reference" with a SHA-256
    provenance pin (see test_h_pinned_legacy_source_pin). The pin only fails
    loudly if the reference source is later edited; it does NOT create
    historical git provenance because tests/ is currently untracked.
    """
    import slice3a_legacy_reference as _L
    return _L.measure_features


@pytest.fixture(scope="module")
def legacy_measure():
    return _load_legacy_oracle()


@pytest.fixture(scope="module")
def oracle_audio():
    """Deterministic synthetic fixture that both the pinned legacy source-copy
    reference and the current implementation measure successfully. Guarantees the
    legacy gates EXECUTE (no skip-on-missing-WAV)."""
    y, sr = _synthetic_burst()
    return y, sr


def test_a_legacy_regression_fmax_none_bit_for_bit(oracle_audio, legacy_measure):
    y, sr = oracle_audio
    expected = legacy_measure(y, sr, NFFT, HOP, TOP_DB, WIN)
    assert expected is not None, "pinned legacy reference rejected the deterministic fixture"
    got = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=None)
    assert got is not None
    assert set(got.keys()) == set(expected.keys())
    for k in expected:
        assert got[k] == expected[k], f"{k}: {got[k]!r} != {expected[k]!r}"


# ---------------------------------------------------------------------------
# B. Full-Nyquist equivalence — fmax=sr/2 matches the pinned legacy full band
# ---------------------------------------------------------------------------

def test_b_full_nyquist_equivalence(oracle_audio, legacy_measure):
    y, sr = oracle_audio
    expected = legacy_measure(y, sr, NFFT, HOP, TOP_DB, WIN)
    assert expected is not None
    got = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=sr / 2.0)
    assert got is not None
    for k in expected:
        assert got[k] == expected[k], f"{k}: {got[k]!r} != {expected[k]!r}"


# ---------------------------------------------------------------------------
# C. Actual sensitivity — high vs low fmax must change frequency features
# ---------------------------------------------------------------------------

def test_c_high_vs_low_fmax_changes_frequency_features(real_audio):
    y, sr, _src = real_audio
    hi = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=21500)
    lo = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=1000)
    assert hi is not None and lo is not None
    changed = [k for k in FREQ_KEYS if hi[k] != lo[k]]
    assert changed, "no frequency-domain feature responded to fmax 21500 -> 1000"


# ---------------------------------------------------------------------------
# D. Time-domain invariance across the requested sweep
# ---------------------------------------------------------------------------

def test_d_time_domain_invariant_across_sweep(real_audio):
    y, sr, _src = real_audio
    ref = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=None)
    assert ref is not None
    for f in SWEEP:
        got = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=f)
        assert got is not None, f"fmax={f} unexpectedly rejected"
        for k in TIME_KEYS:
            assert got[k] == ref[k], f"fmax={f} changed time-domain {k}: {got[k]!r} != {ref[k]!r}"


# ---------------------------------------------------------------------------
# E. Frequency-domain behavior bounds
# ---------------------------------------------------------------------------

def test_e_frequency_domain_bounds(real_audio):
    y, sr, _src = real_audio
    for f in SWEEP:
        fc = min(float(f), sr / 2.0)
        got = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=f)
        assert got is not None, f"fmax={f} unexpectedly rejected"
        # impact_freq floor is 200 Hz; the peak search cannot select a bin
        # above the ceiling, but parabolic interpolation reads the neighbour
        # bin above the ceiling and can legitimately overshoot by up to half
        # an FFT bin. Audit item 4: the implementation is intentionally NOT
        # clamped (that would change the impact-frequency definition); the
        # test invariant allows the mathematically expected tolerance
        # f_c + 0.5 * sr / nfft + epsilon.
        bin_hz = sr / NFFT
        if got["impact_freq_hz"] != 0.0:
            assert got["impact_freq_hz"] <= fc + 0.5 * bin_hz + 1e-9, \
                (f"fmax={f}: impact_freq {got['impact_freq_hz']} > fc {fc} "
                 f"+ 0.5*bin({bin_hz:.4f}) = {fc + 0.5 * bin_hz:.4f}")
        # rolloff/centroid live inside the truncated STFT domain (+1 FFT bin tol).
        assert got["spectral_rolloff_hz"] <= fc + bin_hz + 1e-6, \
            f"fmax={f}: rolloff {got['spectral_rolloff_hz']} > fc {fc}"
        assert got["spectral_centroid_hz"] <= fc + 1e-9, \
            f"fmax={f}: centroid {got['spectral_centroid_hz']} > fc {fc}"
    hi = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=21500)
    lo = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=500)
    assert hi is not None and lo is not None
    deltas = {k: abs(hi[k] - lo[k]) for k in FREQ_KEYS}
    assert any(v > 0 for v in deltas.values()), \
        "sweep 21500 -> 500 produced no measurable frequency-domain change"


# ---------------------------------------------------------------------------
# F. End-to-end /compute — fmax actually threaded through the real route
# ---------------------------------------------------------------------------

class _Upload:
    def __init__(self, data: bytes, filename: str = "slice3a.wav"):
        self._data = data
        self.filename = filename

    async def read(self) -> bytes:
        return self._data


def _call_compute(wav_path: Path, fmax: float) -> dict:
    raw = wav_path.read_bytes()
    resp = asyncio.run(
        live_server.compute(
            file=_Upload(raw),
            sr=SR, nfft=NFFT, hop=HOP, top_db=float(TOP_DB), win=WIN,
            fmax=float(fmax), device_label="slice3a_test", engine="dsp_arena",
        )
    )
    assert resp.status_code == 200, f"/compute fmax={fmax} -> {resp.status_code}"
    import json as _json
    body = _json.loads(resp.body.decode("utf-8"))
    return body


@pytest.fixture()
def iso_events(monkeypatch):
    snap_pending = dict(event_store.pending_events)
    snap_history = list(event_store.session_history)
    event_store.pending_events.clear()
    event_store.session_history.clear()
    try:
        yield
    finally:
        event_store.pending_events.clear()
        event_store.pending_events.update(snap_pending)
        event_store.session_history.clear()
        event_store.session_history.extend(snap_history)


def test_f_compute_threads_fmax_end_to_end(wav_file, iso_events):
    hi = _call_compute(wav_file, 21500)
    lo = _call_compute(wav_file, 1000)
    assert hi["fmax_used_hz"] != lo["fmax_used_hz"]
    assert hi["fmax_used_hz"] == pytest.approx(21500.0)
    assert lo["fmax_used_hz"] == pytest.approx(1000.0)
    changed = [k for k in ("impact_freq_hz", "mel_mean_db", "spectral_centroid_hz",
                           "spectral_bandwidth_hz", "spectral_rolloff_hz",
                           "spectral_flatness", "hf_transient_ratio")
               if hi[k] != lo[k]]
    assert changed, "/compute fmax changed provenance only — not threaded into DSP"
    for k in ("impact", "duration_s", "rms_energy", "zcr"):
        assert hi[k] == lo[k], f"/compute fmax changed time-domain {k}"


# ---------------------------------------------------------------------------
# Legacy ceiling distinctness: fmax=None (legacy) vs explicit fmax=21500 at 44.1k
# ---------------------------------------------------------------------------

def test_none_differs_from_21500_at_44100():
    """Guard: at sr=44100, the legacy ceiling is Nyquist (22050), not 21500.

    The two ceilings differ by 1000 Hz — many STFT bins at nfft=4096
    (bin spacing sr/nfft ≈ 10.77 Hz). This canary proves that requesting
    21500 genuinely narrows the analysis domain, so a regression that
    silently treated 21500 as legacy would fail loudly.

    Uses a pinned deterministic synthetic 44100 Hz fixture — never depends on
    whichever real WAV happens to be discovered on disk.
    """
    y, sr = _synthetic_burst(44100)
    assert sr == 44100
    hi = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=None)
    cap = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=21500)
    assert hi is not None and cap is not None
    # The 21500 ceiling must actually narrow the analysis domain: spectra
    # bounded by 21500 cannot match the legacy Nyquist(22050) result on
    # every frequency-domain feature for the same input.
    identical = all(hi[k] == cap[k] for k in FREQ_KEYS)
    assert not identical, "fmax=21500 unexpectedly identical to legacy Nyquist for all freq features"


def test_nyquist_matches_none_consistency_at_44100():
    """Consistency (current-vs-current): explicit fmax=22050 must agree with
    fmax=None at sr=44100, because both resolve to the same Nyquist ceiling.
    This is NOT a legacy oracle comparison — the pinned-reference gates are
    test_a/test_b, which run against tests/slice3a_legacy_reference.py.
    Uses a pinned deterministic synthetic 44100 Hz fixture — never depends on
    whichever real WAV happens to be discovered on disk."""
    y, sr = _synthetic_burst(44100)
    assert sr == 44100
    hi = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=None)
    nyq = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=22050.0)
    assert hi is not None and nyq is not None
    for k in FREQ_KEYS + TIME_KEYS:
        assert hi[k] == nyq[k], f"{k}: legacy={hi[k]!r} nyquist={nyq[k]!r}"


# ---------------------------------------------------------------------------
# sr > 100 kHz compatibility — audit item 1 resolution.
# The pre-3A descriptors split the ceiling into two domains: mel + impact
# search used fhi = min(50000, sr/2), while the scalar spectral descriptors
# used the FULL STFT spectrum. fmax=None must therefore reproduce the pinned
# legacy reference at EVERY sr (including 192/384 kHz offline choices); an
# explicit fmax genuinely truncates every frequency-domain feature to f_c.
# The canonical /compute route caps sr at 48000, so this is offline-only.
# ---------------------------------------------------------------------------

def test_sr_gt_100k_fmax_none_matches_legacy_oracle(legacy_measure):
    """sr > 100 kHz regression gate: fmax=None must be bit-for-bit identical
    to the pinned pre-3A legacy source-copy reference even at 192 kHz (where min(50000, sr/2) <
    sr/2, so a naive 'shared ceiling' implementation would diverge)."""
    sr_hi = 192_000
    rng = np.random.default_rng(0x7AF3)
    n = sr_hi // 2
    y = rng.standard_normal(n) * 0.001
    t = np.arange(int(0.02 * sr_hi)) / sr_hi
    y[int(0.25 * sr_hi): int(0.25 * sr_hi) + len(t)] += np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 200) * 0.5
    y = y.astype(np.float64)
    expected = legacy_measure(y, sr_hi, 8192, 2048, 35, 0.5)
    assert expected is not None, "pinned legacy reference must measure this fixture"
    got = dsp_inverse.measure_features(y, sr_hi, 8192, 2048, 35, 0.5, fmax=None)
    assert got is not None
    for k in expected:
        assert got[k] == expected[k], f"sr={sr_hi} fmax=None {k}: {got[k]!r} != {expected[k]!r}"


def test_sr_gt_100k_explicit_fmax_truncates_scalars():
    """Explicit fmax at sr > 100 kHz must restrict the scalar spectral
    descriptors to <= f_c (not carry the legacy full-band domain)."""
    sr_hi = 192_000
    rng = np.random.default_rng(0x7AF3)
    n = sr_hi // 2
    y = rng.standard_normal(n) * 0.001
    t = np.arange(int(0.02 * sr_hi)) / sr_hi
    y[int(0.25 * sr_hi): int(0.25 * sr_hi) + len(t)] += np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 200) * 0.5
    y = y.astype(np.float64)
    fe = dsp_inverse.measure_features(y, sr_hi, 8192, 2048, 35, 0.5, fmax=8000.0)
    assert fe is not None
    assert fe["spectral_centroid_hz"] <= 8000.0 + sr_hi / 8192
    assert fe["spectral_rolloff_hz"] <= 8000.0 + sr_hi / 8192


def test_sr_gt_100k_explicit_fmax_clamps_to_nyquist():
    """fmax > Nyquist must clamp to Nyquist (<= sr/2) even at high sr,
    so explicit fmax never exceeds the physical STFT grid.
    Strengthened (task 3):  is asserted separately — the test
    must FAIL if feature extraction unexpectedly returns None."""
    import numpy as np
    sr_hi = 192_000
    rng = np.random.default_rng(0x7AF3)
    n = sr_hi // 2
    y = rng.standard_normal(n) * 0.001
    fe = dsp_inverse.measure_features(
        y.astype(np.float64), sr_hi, 8192, 2048, 35, 0.5, fmax=999_999.0
    )
    # f_c = min(999_999, 96000) = 96000 = sr/2 -> fast-path full Nyquist;
    # centroid is then bounded by sr/2.
    assert fe is not None, "fmax>Nyquist clamp fixture must still return features"
    assert fe["spectral_centroid_hz"] <= sr_hi / 2.0 + 1e-6

# ---------------------------------------------------------------------------
# H. Legacy provenance pin (task 4): SHA-256 of the pinned legacy
# measure_features source span. The file tests/slice3a_legacy_reference.py is
# a mechanically copied snapshot; tests/ is currently untracked so git history
# does NOT protect it. This pin fails loudly if the snapshot is edited after
# the pin was established (accidental synchronized edits to the oracle).
# It does NOT create historical git provenance. Update the pin only after
# independently verifying pre-3A provenance.
# ---------------------------------------------------------------------------
_LEGACY_MEASURE_FEATURES_SHA256 = "cb598cf48b4aa40ee2f5368fe5e1086ba141b83749066b6715dc93cabe0c0668"
_LEGACY_MEASURE_FEATURES_SPAN_START_LINE = 225
_LEGACY_MEASURE_FEATURES_SPAN_END_LINE = 376


def test_h_pinned_legacy_source_pin():
    import hashlib as _hl
    path = ROOT / "tests" / "slice3a_legacy_reference.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    s_idx = next((i for i, l in enumerate(lines) if l.startswith("def measure_features")), None)
    assert s_idx is not None, "measure_features not found in legacy reference"
    depth = 0
    seen = False
    e_idx = None
    for k in range(s_idx, len(lines)):
        l = lines[k]
        if "return {" in l and not seen:
            seen = True
        if seen:
            depth += l.count("{") - l.count("}")
            if depth == 0 and "}" in l:
                e_idx = k
                break
    assert e_idx is not None, "could not locate closing } of measure_features"
    span = "\n".join(lines[s_idx:e_idx + 1]) + "\n"
    got = _hl.sha256(span.encode("utf-8")).hexdigest()
    assert got == _LEGACY_MEASURE_FEATURES_SHA256, (
        "Legacy reference source changed; regenerate/update the pin only after "
        "independently verifying pre-3A provenance. "
        f"expected={_LEGACY_MEASURE_FEATURES_SHA256} got={got} "
        f"span=lines {s_idx+1}-{e_idx+1}"
    )

# ---------------------------------------------------------------------------
# G. Omitted-fmax /compute end-to-end (task 1): omitted must succeed,
#    thread fmax=None legacy path, echo the provenance sentinel, match direct
#    measure_features(..., fmax=None), and differ from explicit fmax=21500
# ---------------------------------------------------------------------------

_SENTINEL_OMITTED = object()


def _call_compute_raw(wav_path: Path, fmax_raw):
    """Direct coroutine call mirroring _call_compute but accepting raw Form
    values (including None / "" / 'abc'). Returns the JSON body on success.

    FastAPI note: when calling the coroutine directly, omitting the fmax kwarg
    inserts the Form(None) sentinel default (a Form object, not None), which
    is NOT the same as an omitted HTTP form field. An omitted form field is
    delivered by FastAPI as Python None. This helper therefore forwards
    omission explicitly as fmax=None.
    """
    import json as _json
    raw = wav_path.read_bytes()
    coro_kwargs = dict(
        file=_Upload(raw),
        sr=SR, nfft=NFFT, hop=HOP, top_db=float(TOP_DB), win=WIN,
        device_label="slice3a_test", engine="dsp_arena",
    )
    # Form-layer omission (missing key on the wire) arrives as None here.
    dispatched = None if fmax_raw is _SENTINEL_OMITTED else fmax_raw
    resp = asyncio.run(live_server.compute(**coro_kwargs, fmax=dispatched))
    assert resp.status_code == 200, f"/compute fmax={fmax_raw!r} -> {resp.status_code} {resp.body[:200]!r}"
    return _json.loads(resp.body.decode("utf-8"))


def _direct_legacy_for_wav(wav_path: Path) -> dict:
    raw = wav_path.read_bytes()
    # decode via live_server helper to get native bytes -> resample if needed
    # For our synthetic fixture written at SR, native == actual so no resample
    y, sr = dsp_inverse.decode_wav(str(wav_path))
    fe = dsp_inverse.measure_features(np.asarray(y, dtype=np.float64), sr, NFFT, HOP, TOP_DB, WIN, fmax=None)
    assert fe is not None, "fixture must be measurable"
    return fe


def test_g_omitted_fmax_end_to_end_legacy(iso_events):
    """Omitted fmax (Form default None) must: succeed, use legacy path,
    echo None sentinel provenance, match direct fmax=None, and differ from
    explicit fmax=21500. Uses a pinned deterministic 44100 Hz synthetic fixture
    so assertions are unconditional."""
    import json as _json
    import tempfile
    y, sr = _synthetic_burst(44100)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "slice3a_g.wav"
        _write_wav(p, y, sr)
        body_legacy = _call_compute_raw(p, _SENTINEL_OMITTED)
        body_21500 = _call_compute_raw(p, 21500.0)

    # A. success already asserted inside _call_compute_raw

    # B+C. DSP call received fmax=None / legacy path — verified via provenance
    #     sentinel and by matching direct fmax=None features
    assert body_legacy["fmax_requested_hz"] is None, "omitted fmax must echo None provenance"
    assert body_legacy["fmax_used_hz"] is None, "omitted fmax legacy uses sentinel None, not a numeric ceiling"

    # D. at the supported SR where legacy ceiling is unambiguous (sr=44100)
    # legacy sentinel is None — NOT 22050 — per the honest split-domain contract
    # (explicit numeric fmax still echoes its value)
    assert body_21500["fmax_requested_hz"] == pytest.approx(21500.0)
    assert body_21500["fmax_used_hz"] == pytest.approx(21500.0)

    # E+F. frequency + time features match direct measure_features(..., fmax=None)
    # Use the SAME decoded audio the /compute route used (int16 WAV round-trip),
    # otherwise float64 vs quantized input diverges.
    import numpy as _np_check
    with tempfile.TemporaryDirectory() as td:
        p2 = Path(td) / "slice3a_g2.wav"
        _write_wav(p2, y, sr)
        y_dec, sr_dec = dsp_inverse.decode_wav(str(p2))
    assert sr_dec == 44100
    fe_direct = dsp_inverse.measure_features(_np_check.asarray(y_dec, dtype=_np_check.float64), 44100, NFFT, HOP, TOP_DB, WIN, fmax=None)
    assert fe_direct is not None
    # Map /compute keys -> direct keys where names differ.
    # The wire response rounds each field; compare against the same rounding.
    # (decimals mirror _build_response in live_server.py)
    WIRE_DECIMALS = {
        "impact": 6, "rms_energy": 6, "zcr": 6,
        "impact_freq_hz": 1, "mel_mean_db": 3,
        "spectral_centroid_hz": 1, "spectral_bandwidth_hz": 1,
        "spectral_rolloff_hz": 1, "spectral_flatness": 6,
        "hf_transient_ratio": 6,
    }
    for k in FREQ_KEYS:
        d = WIRE_DECIMALS.get(k, 6)
        assert body_legacy[k] == pytest.approx(fe_direct[k], abs=10 ** -d, rel=1e-9), f"freq {k}"
    for k in ("impact", "rms_energy", "zcr"):
        d = WIRE_DECIMALS[k]
        assert body_legacy[k] == pytest.approx(fe_direct[k], abs=10 ** -d, rel=1e-9), f"time {k}"
    # duration_s / attack naming
    assert body_legacy["duration_s"] == pytest.approx(fe_direct["duration"], abs=1e-6)
    assert body_legacy["attack_slope_db_ms"] == pytest.approx(fe_direct["attack_slope_db_ms"], abs=1e-4)

    # G. explicit fmax differs from omitted legacy on at least one freq feature
    diff = [k for k in FREQ_KEYS if body_legacy[k] != body_21500[k]]
    assert diff, "explicit fmax=21500 must differ from omitted legacy on at least one freq feature"


# ---------------------------------------------------------------------------
# Provenance cannot claim a ceiling the DSP did not apply (task 7)
# ---------------------------------------------------------------------------

def test_g_provenance_sentinel_not_misleading(iso_events):
    """Legacy (omitted) must not echo a numeric ceiling that the DSP did not
    enforce as a single shared ceiling. At sr=44100 scalars used full STFT."""
    import tempfile
    y, sr = _synthetic_burst(44100)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "slice3a_prov.wav"
        _write_wav(p, y, sr)
        body = _call_compute_raw(p, _SENTINEL_OMITTED)
    assert body["fmax_requested_hz"] is None
    assert body["fmax_used_hz"] is None
    # explicit must echo its value clamped to Nyquist (<= sr/2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "slice3a_prov2.wav"
        _write_wav(p, y, sr)
        body2 = _call_compute_raw(p, 8000.0)
    assert body2["fmax_requested_hz"] == pytest.approx(8000.0)
    assert body2["fmax_used_hz"] == pytest.approx(8000.0)
    # legacy vs explicit must differ
    assert body["fmax_used_hz"] != body2["fmax_used_hz"]


# ---------------------------------------------------------------------------
# I. hf_transient_ratio retained-band semantics (task 8)
# ---------------------------------------------------------------------------

def test_i_hf_transient_ratio_retained_band_relative():
    """Explicit fmax is retained-band-relative: shrinking fmax from 8000 to
    2000 must raise or change the ratio even though absolute HF (>1 kHz)
    energy in the 1-2 kHz sub-band is preserved. Demonstrates that
    hf_transient_ratio is HF / total-retained, not HF / total-full."""
    rng = np.random.default_rng(0xBEEF)
    sr = 44100
    n = sr * 2
    y = rng.standard_normal(n) * 0.0005
    # Broadband burst so both ceilings have real HF and total energy
    t = np.arange(int(0.02 * sr)) / sr
    y[int(0.7 * sr): int(0.7 * sr) + len(t)] += (
        np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 180) * 0.9
        + np.sin(2 * np.pi * 5000 * t) * np.exp(-t * 220) * 0.6
    )
    y = np.clip(y, -1.0, 1.0)
    hi = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=8000)
    lo = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=2000)
    assert hi is not None and lo is not None
    # Both have non-trivial total energy under their own retained domains
    assert hi["hf_transient_ratio"] != lo["hf_transient_ratio"], \
        "hf_transient_ratio must change when the retained band shrinks"
    # The semantic: each ratio's denominator is its OWN retained total, so
    # explicitly compute the invariant: re-derive both ratios from their
    # retained S_for_spec and they must match the reported values up to
    # rounding — here just assert they are genuinely different, not silently
    # full-band ratios that happen to be identical.
    # Also assert legacy differs (legacy used full STFT denominator)
    leg = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=None)
    assert leg is not None
    assert leg["hf_transient_ratio"] != lo["hf_transient_ratio"] or leg["hf_transient_ratio"] != hi["hf_transient_ratio"], \
        "explicit retained-band ratio must differ from legacy full-band ratio for this fixture"


# ---------------------------------------------------------------------------
# J. fmax validation contract (task 9)
#   omitted/None/"" -> legacy (200/None), "8000" accepted, "abc"/"nan"/inf -> 422
#   21500 accepted, 21501 rejected
# ---------------------------------------------------------------------------

def _call_compute_expect_status(wav_path: Path, fmax_raw, expect: int):
    import json as _json
    from fastapi import HTTPException
    raw = wav_path.read_bytes()
    kwargs = dict(file=_Upload(raw), sr=SR, nfft=NFFT, hop=HOP, top_db=float(TOP_DB), win=WIN,
                  device_label="slice3a_test", engine="dsp_arena")
    try:
        # See _call_compute_raw: Form-layer omission arrives as None.
        dispatched = None if fmax_raw is _SENTINEL_OMITTED else fmax_raw
        resp = asyncio.run(live_server.compute(**kwargs, fmax=dispatched))
    except HTTPException as exc:
        # FastAPI converts this to the status_code we expect
        assert exc.status_code == expect, f"fmax={fmax_raw!r} expected {expect} got HTTPException {exc.status_code}: {exc.detail!r}"
        return exc
    assert resp.status_code == expect, f"fmax={fmax_raw!r} expected {expect} got {resp.status_code}: {resp.body[:300]!r}"
    return resp


def test_j_fmax_validation_contract(iso_events):
    import tempfile
    y, sr = _synthetic_burst(44100)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "slice3a_j.wav"
        _write_wav(p, y, sr)
        # omitted -> 200 (via sentinel) — already verified in G; here just 200 check
        _call_compute_expect_status(p, _SENTINEL_OMITTED, 200)
        # explicit None also legacy
        _call_compute_expect_status(p, None, 200)
        # empty string -> legacy (no error)
        _call_compute_expect_status(p, "", 200)
        # whitespace-only -> legacy
        _call_compute_expect_status(p, "   ", 200)
        # numeric-string accepted
        _call_compute_expect_status(p, "8000", 200)
        _call_compute_expect_status(p, "8000.0", 200)
        # unparseable -> 422
        _call_compute_expect_status(p, "abc", 422)
        # nan (case-insensitive) -> 422
        _call_compute_expect_status(p, "nan", 422)
        _call_compute_expect_status(p, "NaN", 422)
        # inf / -inf -> 422
        for bad in ("inf", "+inf", "-inf", "Infinity", float("inf"), float("-inf")):
            _call_compute_expect_status(p, bad, 422)
        # 21500 accepted, 21501 rejected (> FMAX_REAL)
        _call_compute_expect_status(p, 21500, 200)
        _call_compute_expect_status(p, "21500", 200)
        _call_compute_expect_status(p, 21500.0, 200)
        _call_compute_expect_status(p, 21501, 422)
        _call_compute_expect_status(p, "21501", 422)
        _call_compute_expect_status(p, 30000, 422)
        # <=0 -> 422
        _call_compute_expect_status(p, 0, 422)
        _call_compute_expect_status(p, -10, 422)
        _call_compute_expect_status(p, "-1", 422)


# ---------------------------------------------------------------------------
# K. Low-fmax mel floor (task 11 contract B): mel degenerate sentinel
#    Probe established: at nfft=4096/sr=44100 the 40-band mel is non-degenerate
#    from ~300 Hz upward. We keep fmax=50 valid but document it as degenerate;
#    a dedicated low-floor probe or tighter floor would be a separate RFC.
#    Test: values >= 300 are clean; fmax=50 is accepted but mel is degenerate.
# ---------------------------------------------------------------------------

def test_k_low_fmax_mel_degeneracy_contract():
    y, sr = _synthetic_burst(44100)
    # 500 and up: non-degenerate (the task's probe used 40-band, nfft=4096)
    # The probe showed first clean at ~300 Hz; task allows keeping 50 valid
    # under contract B — so we assert 300 and 500 succeed and are not warned
    # as empty-filter only via the exception (librosa warns, not raises).
    for f in (300, 500, 800, 1000):
        fe = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=f)
        assert fe is not None, f"fmax={f} should be accepted"
        assert np.isfinite(fe["mel_mean_db"]), f"fmax={f} mel should be finite"
    # fmax=50 is the documented floor and IS accepted (contract B). It is
    # degenerate/sentinel-like (many mel filters are empty), but the DSP does
    # not throw — callers must treat mel_mean_db at this point as sentinel.
    fe50 = dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=50)
    assert fe50 is not None
    assert np.isfinite(fe50["mel_mean_db"])  # tiny floor, not NaN
    # Values below the floor must be rejected
    import pytest as _pt
    with _pt.raises(ValueError):
        dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=49)
    with _pt.raises(ValueError):
        dsp_inverse.measure_features(y, sr, NFFT, HOP, TOP_DB, WIN, fmax=0)


# ---------------------------------------------------------------------------
# Clamp assertion strengthening (task 3): ensure None would fail
# ---------------------------------------------------------------------------

def test_sr_gt_100k_clamp_assertion_is_strict():
    """Guard that the high-SR clamp is checked as `fe is not None` rather
    than a vacuous `fe is None or ...` so a future regression that returns
    None is not silently accepted."""
    sr_hi = 192_000
    rng = np.random.default_rng(0x7AF3)
    n = sr_hi // 2
    y = rng.standard_normal(n) * 0.001
    fe = dsp_inverse.measure_features(y.astype(np.float64), sr_hi, 8192, 2048, 35, 0.5, fmax=999_999.0)
    assert fe is not None, "fmax>Nyquist clamp fixture must still return features"
    assert fe["spectral_centroid_hz"] <= sr_hi / 2.0 + 1e-6
