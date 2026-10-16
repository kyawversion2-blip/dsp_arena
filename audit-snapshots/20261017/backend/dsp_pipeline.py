# ─────────────────────────────────────────────────────────────────────────────
#  backend/dsp_pipeline.py  —  Task B5.1: Vectorized STFT + Laplace Pipeline
#                             Phase B6: Pluggable Analytical Methods (Strategy)
#
#  Apple Silicon M4 Optimized Dual-Domain DSP extraction.
#
#  B6 refactor — the monolithic DSPExtractor is decomposed into a Strategy
#  Pattern so the frontend `dsp_strategy_toggle` protocol can rewire the
#  computation graph at runtime without restarting the stream:
#
#      BaseDSPStrategy (B6.1)  ─ execution contract: analyze(samples, sr, view)
#        ├─ NormalizedLogEnergyEnvelopeStrategy   id = "log_energy_envelope"
#        ├─ LaplaceRatioStrategy                  id = "laplace_ratios"
#        └─ Hack196kHzStrategy                    id = "hack_196k"
#      StrategyRegistry (B6.3/B6.4)  ─ mutable {id: strategy} + enabled state
#
#  DSPExtractor now owns one registry and a single pre-allocated spectral
#  workspace; extract() and recompute() both run the *enabled* strategies into
#  a reused view container, so the manual-trigger and toggle-recompute paths
#  share one source of truth (no duplicated math).
#
#  Allocation contract (G1 / G3 / G7):
#    • Every working array is pre-allocated once (strategy / extractor __init__).
#    • Per-event elementwise + reduction math is in-place via `out=` / scalar
#      reductions — no heap churn in the metric kernels.
#    • Transform calls (rfft / hilbert) allocate their output once per *event*
#      only; G3 guarantees DSP runs on a triggered 1.0 s segment, never per
#      chunk, so this is bounded at event rate (not in any inner loop).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
import scipy.signal
import librosa

from backend.audio_buffer import AudioSegment
from backend.config import SAMPLE_RATE, EVENT_SEGMENT_SAMPLES

logger = logging.getLogger("ripple.dsp_pipeline")

_EPS = 1e-12

# Strategy tunable defaults — kept as named constants so params_schema() reports
# the *true* effective values used by the kernels (no schema/code drift).
_ENV_DURATION_FRAC = 0.05   # relative peak-fraction onset/offset threshold (adaptive, G6)
_ENV_TAIL_FLOOR = 1e-4      # log-envelope tail clip floor, as a fraction of peak
_LAPLACE_SUB_BANDS = 4      # structural A/B/C/D equal sub-band split (fixed)

# No dummy function needed

@dataclass(frozen=True, )
class DSPResult:
    """
    Complete extraction result for one triggered acoustic event.
    """
    trigger_sample:       int
    trigger_time_s:       float
    duration_ms:          float

    impact_centroid_hz:   float
    ringing_freq_hz:      float
    freq_shift_hz:        float

    decay_rate:           float
    sharpness_index:      float

    in_band_energy_ratio: float
    band_energies:        tuple[float, float, float, float]  # E_A, E_B, E_C, E_D

    laplace_cross_ratio:  float    # (E_A × E_D) / (E_B × E_C)
    ripple_fft_ratio:     float    # (E_A × E_C) / (E_B × E_D)
    trigger_ratio:        float

    # STFT domain features for DualDomainFingerprint
    stft_centroid_hz:     float
    stft_bandwidth_hz:    float
    stft_flatness:        float


def compute_stft_features(
    samples: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float
) -> tuple[float, float, float]:
    """
    Computes time-averaged STFT spectral features (centroid, bandwidth, flatness)
    within the [f_min, f_max] frequency limits. Fully vectorized.
    """
    # Guard against empty/too small inputs
    if len(samples) < n_fft:
        n_fft = len(samples)

    # Compute STFT using scipy.signal.stft
    f, t, Zxx = scipy.signal.stft(
        samples.astype(np.float64),
        fs=sr,
        nperseg=n_fft,
        noverlap=max(0, n_fft - hop_length)
    )
    S = np.abs(Zxx)  # shape: (F, T)

    # Mask frequencies
    mask = (f >= f_min) & (f <= f_max)
    f_masked = f[mask]
    S_masked = S[mask, :]

    if len(f_masked) == 0:
        return 0.0, 0.0, 0.0

    # Average magnitude spectrum over time axis
    mean_mag = np.mean(S_masked, axis=1)

    # Spectral Centroid
    denom = np.sum(mean_mag) + _EPS
    centroid = np.dot(f_masked, mean_mag) / denom

    # Spectral Bandwidth
    variance = np.dot((f_masked - centroid) ** 2, mean_mag) / denom
    bandwidth = np.sqrt(max(variance, 0.0))

    # Spectral Flatness: Geometric Mean / Arithmetic Mean
    log_mean = np.mean(np.log(np.maximum(mean_mag, 1e-12)))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(mean_mag)
    flatness = geo_mean / (arith_mean + _EPS)

    return float(centroid), float(bandwidth), float(flatness)


def run_dsp_pipeline(
    segment: AudioSegment,
    cfg: Any,  # EngineConfig type
    scratch: Any,  # _DSPScratch type
    trigger_ratio: float,
) -> DSPResult:
    """
    Full dual-domain DSP extraction for one triggered event, extended with STFT features.
    Vectorized NumPy operations designed to link with Apple Accelerate framework on macOS.
    """
    sr      = cfg.sample_rate
    samples = segment.samples   # float32 (EVENT_SEGMENT_SAMPLES,)

    # ── Slice the segment into impact and sustained windows ───────────────────
    imp_end = int(cfg.impact_duration_s * sr)
    sus_beg = int(cfg.sustained_start_s * sr)
    sus_end = int(cfg.sustained_end_s   * sr)

    impact_raw    = samples[:imp_end].astype(np.float64)
    sustained_raw = samples[sus_beg:sus_end].astype(np.float64)

    # Guard: zero-pad if segment is shorter than expected
    imp_n = len(scratch.impact_buf)
    sus_n = len(scratch.sustained_buf)
    if len(impact_raw) < imp_n:
        impact_raw = np.pad(impact_raw, (0, imp_n - len(impact_raw)))
    if len(sustained_raw) < sus_n:
        sustained_raw = np.pad(sustained_raw, (0, sus_n - len(sustained_raw)))

    # ── In-place Hann windowing (reuse scratch buffers — G1) ─────────────────
    scratch.impact_buf[:]    = (impact_raw[:imp_n]    * scratch.hann_impact).astype(np.float32)
    scratch.sustained_buf[:] = (sustained_raw[:sus_n] * scratch.hann_sustained).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.1  Impact Phase Spectral Centroid
    # ─────────────────────────────────────────────────────────────────────────
    impact_fft  = np.fft.rfft(scratch.impact_buf)
    impact_mag  = np.abs(impact_fft)
    band_mask_i = (scratch.impact_freqs >= cfg.f_min) & (scratch.impact_freqs <= cfg.f_max)
    band_mag_i  = impact_mag * band_mask_i
    impact_centroid_hz = float(
        np.dot(scratch.impact_freqs, band_mag_i) / (float(band_mag_i.sum()) + _EPS)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4.2  Per-band Energy (E_A, E_B, E_C, E_D) for cross-ratios
    # ─────────────────────────────────────────────────────────────────────────
    sustained_fft  = np.fft.rfft(scratch.sustained_buf)
    sustained_mag  = np.abs(sustained_fft)

    be = scratch.band_edges
    E = np.array([
        float(np.sqrt(np.mean(sustained_mag[be[k]:be[k + 1]] ** 2)) + _EPS)
        for k in range(4)
    ], dtype=np.float64)
    E_A, E_B, E_C, E_D = E[0], E[1], E[2], E[3]

    # Total in-band energy ratio
    total_mag_sq      = float(np.mean(sustained_mag ** 2)) + _EPS
    band_slice        = sustained_mag[be[0]:be[4]]
    in_band_energy_ratio = float(
        np.clip(np.mean(band_slice ** 2) / total_mag_sq, 0.0, 1.0)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4.3  Laplace Cross-Ratio: (E_A × E_D) / (E_B × E_C)
    # ─────────────────────────────────────────────────────────────────────────
    laplace_cross_ratio = float((E_A * E_D) / max(E_B * E_C, _EPS))

    # ─────────────────────────────────────────────────────────────────────────
    # 4.4  Ripple FFT Ratio: (E_A × E_C) / (E_B × E_D)
    # ─────────────────────────────────────────────────────────────────────────
    ripple_fft_ratio = float((E_A * E_C) / max(E_B * E_D, _EPS))

    # ─────────────────────────────────────────────────────────────────────────
    # 4.5  Sustained Ringing Frequency ω_d
    # ─────────────────────────────────────────────────────────────────────────
    band_mask_s = (scratch.sustained_freqs >= cfg.f_min) & \
                  (scratch.sustained_freqs <= cfg.f_max)
    masked_mag  = sustained_mag * band_mask_s
    peak_bin    = int(np.argmax(masked_mag))
    ringing_freq_hz = float(scratch.sustained_freqs[peak_bin])

    # ─────────────────────────────────────────────────────────────────────────
    # 4.6  Exponential Decay Envelope via Hilbert Transform
    # ─────────────────────────────────────────────────────────────────────────
    analytic     = scipy.signal.hilbert(scratch.sustained_buf.astype(np.float64))
    envelope     = np.abs(analytic)

    log_env      = np.log(np.maximum(envelope, cfg.decay_epsilon))
    coeffs       = np.polyfit(scratch.t_sustained, log_env, 1)
    decay_rate   = float(max(-coeffs[0], 0.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 4.7  Duration_ms
    # ─────────────────────────────────────────────────────────────────────────
    peak_env     = float(envelope.max()) + _EPS
    threshold    = peak_env * cfg.duration_envelope_ratio
    above_thresh = np.where(envelope > threshold)[0]
    if above_thresh.size > 0:
        last_idx     = int(above_thresh[-1])
        duration_s   = float(scratch.t_sustained[last_idx])
    else:
        duration_s   = float(scratch.t_sustained[-1])
    duration_ms  = duration_s * 1_000.0

    # ─────────────────────────────────────────────────────────────────────────
    # 4.8  Sharpness Index Q = ω_d / bandwidth_3dB
    # ─────────────────────────────────────────────────────────────────────────
    if peak_bin > 0 and peak_bin < len(sustained_mag) - 1:
        half_power  = float(masked_mag[peak_bin]) / np.sqrt(2.0)
        above_half  = np.where(masked_mag > half_power)[0]
        if above_half.size >= 2:
            bw_hz       = float(
                scratch.sustained_freqs[above_half[-1]] -
                scratch.sustained_freqs[above_half[0]]
            )
            sharpness_index = ringing_freq_hz / max(bw_hz, 1.0)
        else:
            sharpness_index = 1.0
    else:
        sharpness_index = 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # 4.9  Absolute Frequency Shift Hz = |ω_d − impact_centroid|
    # ─────────────────────────────────────────────────────────────────────────
    freq_shift_hz = abs(ringing_freq_hz - impact_centroid_hz)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.10  STFT feature extraction (New B5.1 requirement)
    # ─────────────────────────────────────────────────────────────────────────
    stft_centroid, stft_bandwidth, stft_flatness = compute_stft_features(
        samples, sr, cfg.n_fft, cfg.hop_length, cfg.f_min, cfg.f_max
    )

    return DSPResult(
        trigger_sample       = segment.trigger_sample,
        trigger_time_s       = segment.trigger_time_s,
        duration_ms          = duration_ms,
        impact_centroid_hz   = impact_centroid_hz,
        ringing_freq_hz      = ringing_freq_hz,
        freq_shift_hz        = freq_shift_hz,
        decay_rate           = decay_rate,
        sharpness_index      = sharpness_index,
        in_band_energy_ratio = in_band_energy_ratio,
        band_energies        = (E_A, E_B, E_C, E_D),
        laplace_cross_ratio  = laplace_cross_ratio,
        ripple_fft_ratio     = ripple_fft_ratio,
        trigger_ratio        = trigger_ratio,
        stft_centroid_hz     = stft_centroid,
        stft_bandwidth_hz    = stft_bandwidth,
        stft_flatness        = stft_flatness,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Rev. D additions — Dual-Domain WS extractor, vectorized Chroma (B2.11),
#  512-bin decimation (C6/WAVEFORM_CAPACITY), and a provisional surface tracker
#  for Confidence-Ellipse / ridge geometry (A2.8 / A2.11).
#
#  Design contract:
#    G1  Every working array is pre-allocated once in DSPExtractor.__init__.
#    G3  extract() runs ONLY on a triggered 1.0 s segment — never per chunk.
#    G7  No Python loops over raw audio: all reductions are NumPy/SciPy vectorized
#        (cumsum band split, np.add.at chroma binning, reshape-decimation).
# ═══════════════════════════════════════════════════════════════════════════════

# Pitch-class → note name with f_ref = 440 Hz (A), matching
#   pitch_class = round(12 * log2(f / f_ref)) mod 12   →   pc 0 = A.
_CHROMA_NAMES: tuple[str, ...] = (
    "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",
)
_F_REF_HZ: float = 440.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase B6 — Pluggable Analytical Methods (Strategy Pattern)
# ═══════════════════════════════════════════════════════════════════════════════

# Reserved (private) keys placed into the shared `view` container by the
# orchestrator before strategies run. They carry *pre-allocated* workspace
# arrays — strategies read them and write only scalar float metrics back, so
# the per-event hot path performs zero numpy heap allocation (G1).
_VIEW_X        = "_x"          # float64 samples,            shape (n,)
_VIEW_MAG      = "_mag"        # |rfft(window·x)|,           shape (n//2+1,)
_VIEW_BAND_MAG = "_band_mag"   # in-band magnitude,          shape (freqs_band,)
_VIEW_ENV      = "_env"        # |hilbert(x)| envelope,      shape (n,)


@dataclass(frozen=True, )
class SpectralWorkspace:
    """
    Immutable, pre-computed analysis plan shared by every strategy in a
    registry. Built once from the active DSP params; holds the read-only axes
    and index maps the strategies index into (never reallocated per event).
    """
    sr:          int
    n:           int
    n_fft:       int
    hop_length:  int
    f_min:       float
    f_max:       float
    rfft_freqs:  np.ndarray   # (n//2+1,)  full rfft frequency axis
    band_mask:   np.ndarray   # (n//2+1,)  bool in-band mask
    band_idx:    np.ndarray   # (freqs_band,) integer in-band bin indices
    freqs_band:  np.ndarray   # (freqs_band,) in-band frequency axis
    band_edges:  np.ndarray   # (5,) bin edges for the 4 equal sub-bands A..D
    t_axis:      np.ndarray   # (n,) time axis (decay fit)

    @classmethod
    def build(
        cls,
        sr: int,
        n: int,
        n_fft: int,
        hop_length: int,
        f_min: float,
        f_max: float,
    ) -> "SpectralWorkspace":
        rfft_freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        band_mask = (rfft_freqs >= f_min) & (rfft_freqs <= f_max)
        band_idx = np.nonzero(band_mask)[0].astype(np.intp)
        freqs_band = rfft_freqs[band_mask]
        band_edges = np.searchsorted(rfft_freqs, np.linspace(f_min, f_max, 5)).astype(np.intp)
        t_axis = np.arange(n, dtype=np.float64) / sr
        return cls(
            sr=int(sr), n=int(n), n_fft=int(n_fft), hop_length=int(hop_length),
            f_min=float(f_min), f_max=float(f_max),
            rfft_freqs=rfft_freqs, band_mask=band_mask, band_idx=band_idx,
            freqs_band=freqs_band, band_edges=band_edges, t_axis=t_axis,
        )


class BaseDSPStrategy(abc.ABC):
    """
    B6.1 — execution contract for a pluggable analytical method.

    A strategy owns its pre-allocated scratch (sized once from a
    `SpectralWorkspace`) and writes scalar float metrics into a shared,
    reused `view` container. It never allocates on the per-event path: this is
    how runtime strategy toggling stays compatible with Guardrail G1.

    Concrete subclasses MUST set ``id`` and ``display_name`` and implement
    :meth:`analyze`.
    """

    __slots__ = ("id", "display_name", "ws")

    id: str
    display_name: str
    ws: SpectralWorkspace

    def __init__(self, strategy_id: str, display_name: str, ws: SpectralWorkspace) -> None:
        self.id = strategy_id
        self.display_name = display_name
        self.ws = ws

    @abc.abstractmethod
    def analyze(
        self,
        samples: np.ndarray,
        sr: int,
        pre_allocated_view: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute this strategy's metrics from the shared pre-allocated arrays in
        ``pre_allocated_view`` (reserved ``_``-prefixed keys) and write the
        resulting scalar floats back into the same dict (G1 — update-in-place).
        Returns the same ``pre_allocated_view`` for chaining.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def params_schema(self) -> dict[str, Any]:
        """
        B6.1 — declare this strategy's tunable / structural parameters so the
        frontend can render controls and the registry can report capability.

        Returns ``{param_name: {type, value, ...}}`` where each entry carries at
        least ``type`` and the current effective ``value``; numeric tunables add
        ``min``/``max``, read-only structural params add ``"readonly": True``.
        Pure metadata — never touches the per-event buffers (G1-neutral).
        """
        raise NotImplementedError


class NormalizedLogEnergyEnvelopeStrategy(BaseDSPStrategy):
    """
    id = "log_energy_envelope" — tracks the robust, amplitude-invariant
    envelope form contour of the event.

    Operates on the pre-computed Hilbert envelope (``_env``):
      • signal duration via a *relative* (peak-fraction) threshold — adaptive,
        never a static dB floor (G6);
      • decay rate σ from a closed-form least-squares slope of the log-envelope
        tail (amplitude-invariant: a global gain is an additive log constant
        that cancels out of the slope);
      • envelope form factor = mean/peak of the envelope (crest-invariant
        contour descriptor).

    All math is in-place over pre-allocated buffers (`out=` + scalar reductions).
    """

    __slots__ = ("_log", "_clip", "_mask")

    def __init__(self, ws: SpectralWorkspace) -> None:
        super().__init__("log_energy_envelope", "Normalized Log-Energy Envelope", ws)
        self._log = np.empty(ws.n, dtype=np.float64)   # log-tail scratch
        self._clip = np.empty(ws.n, dtype=np.float64)  # floored-tail scratch
        self._mask = np.empty(ws.n, dtype=bool)        # threshold mask scratch

    def analyze(self, samples, sr, pre_allocated_view):
        env = pre_allocated_view[_VIEW_ENV]            # pre-allocated (n,)
        n = self.ws.n
        peak_env = float(env.max().item()) + _EPS

        # ── Duration via relative peak-fraction threshold (adaptive, G6) ──────
        np.greater(env, peak_env * _ENV_DURATION_FRAC, out=self._mask)
        if bool(self._mask.any()):
            first = self._mask.argmax().item()                  # first True
            last = n - 1 - self._mask[::-1].argmax().item()     # last True (rev view)
            duration_s = float(last - first) / float(sr)
        else:
            duration_s = 0.0

        # ── Decay σ via closed-form LSQ slope of the log-envelope tail ────────
        peak_idx = env.argmax().item()
        k = n - peak_idx
        if k >= 2:
            tail = env[peak_idx:]                               # view, no alloc
            clip = self._clip[:k]
            log = self._log[:k]
            np.maximum(tail, peak_env * _ENV_TAIL_FLOOR, out=clip)
            np.log(clip, out=log)
            t = self.ws.t_axis[:k]                              # view, no alloc
            kk = float(k)
            st = float(t.sum().item())
            sy = float(log.sum().item())
            stt = float(np.dot(t, t).item())
            sty = float(np.dot(t, log).item())
            denom = kk * stt - st * st
            slope = (kk * sty - st * sy) / denom if abs(denom) > _EPS else 0.0
        else:
            slope = 0.0
        decay_rate = max(-slope, 0.0)

        form_factor = float(env.mean().item()) / peak_env             # amplitude-invariant

        pre_allocated_view["duration_s"] = duration_s
        pre_allocated_view["signalDurationUs"] = duration_s * 1_000_000.0
        pre_allocated_view["decayRate"] = decay_rate
        pre_allocated_view["envelopeFormFactor"] = form_factor
        return pre_allocated_view

    def params_schema(self) -> dict[str, Any]:
        return {
            "duration_threshold_frac": {
                "type": "float", "value": _ENV_DURATION_FRAC,
                "min": 0.01, "max": 0.5,
                "label": "Onset/offset threshold (fraction of peak)",
            },
            "tail_floor_frac": {
                "type": "float", "value": _ENV_TAIL_FLOOR,
                "min": 1e-6, "max": 1e-2,
                "label": "Log-envelope tail clip floor (fraction of peak)",
            },
        }


class LaplaceRatioStrategy(BaseDSPStrategy):
    """
    id = "laplace_ratios" — resolves the s-plane pole attributes from the
    magnitude spectrum and emits the structural cross-ratio vector.

    From the four equal sub-band RMS energies E_A..E_D across [f_min, f_max]:
      • Laplace cross-ratio   AD/BC   (pole-pair separation structure)
      • Ripple FFT ratio      AC/BD   (interleaved-band structure)
      • in-band energy ratio  (∑ band power / ∑ total power)
      • sharpness Q           (dominant peak / −3 dB bandwidth)
      • dominant_hz           (in-band spectral peak)

    Sub-band split is a vectorized cumulative-sum gather (no Python loop, G7);
    power + cumsum land in pre-allocated buffers (`out=`).
    """

    __slots__ = ("_power", "_csum")

    def __init__(self, ws: SpectralWorkspace) -> None:
        super().__init__("laplace_ratios", "Laplace Ratios (σ ± jω)", ws)
        nbins = ws.rfft_freqs.shape[0]
        self._power = np.empty(nbins, dtype=np.float64)
        self._csum = np.empty(nbins + 1, dtype=np.float64)

    def _q_factor(self, band_mag: np.ndarray, peak_bin: int, peak_hz: float) -> float:
        if band_mag.size < 3 or peak_bin <= 0 or peak_bin >= band_mag.size - 1:
            return 1.0
        half_power = float(band_mag[peak_bin].item()) / np.sqrt(2.0)
        above = np.nonzero(band_mag > half_power)[0]
        if above.size < 2:
            return 1.0
        bw_hz = float(self.ws.freqs_band[above[-1].item()] - self.ws.freqs_band[above[0].item()])
        return float(peak_hz / max(bw_hz, 1.0))

    def analyze(self, samples, sr, pre_allocated_view):
        mag = pre_allocated_view[_VIEW_MAG]            # full spectrum (n//2+1,)
        band_mag = pre_allocated_view[_VIEW_BAND_MAG]  # in-band (freqs_band,)
        be = self.ws.band_edges

        # ── 4 sub-band RMS energies via cumulative-sum split (vectorized) ─────
        np.multiply(mag, mag, out=self._power)
        self._csum[0] = 0.0
        np.cumsum(self._power, out=self._csum[1:])
        
        # Avoid advanced indexing slice allocations by extracting scalar bounds directly
        E_A = float(np.sqrt((self._csum[be[1]] - self._csum[be[0]]) / max(be[1] - be[0], 1.0)) + _EPS)
        E_B = float(np.sqrt((self._csum[be[2]] - self._csum[be[1]]) / max(be[2] - be[1], 1.0)) + _EPS)
        E_C = float(np.sqrt((self._csum[be[3]] - self._csum[be[2]]) / max(be[3] - be[2], 1.0)) + _EPS)
        E_D = float(np.sqrt((self._csum[be[4]] - self._csum[be[3]]) / max(be[4] - be[3], 1.0)) + _EPS)

        total_power = float(self._csum[-1].item()) + _EPS
        band_power = float(np.dot(band_mag, band_mag).item())
        in_band = float(np.clip(band_power / total_power, 0.0, 1.0))

        if band_mag.size:
            dom_bin = band_mag.argmax().item()
            dominant_hz = float(self.ws.freqs_band[dom_bin])
        else:
            dom_bin, dominant_hz = 0, 0.0
            
        sharpness = self._q_factor(band_mag, dom_bin, dominant_hz)

        pre_allocated_view["inBandEnergyRatio"] = in_band
        pre_allocated_view["sharpnessIndex"] = sharpness
        pre_allocated_view["dominantHz"] = dominant_hz
        pre_allocated_view["bandEnergyA"] = E_A
        pre_allocated_view["bandEnergyB"] = E_B
        pre_allocated_view["bandEnergyC"] = E_C
        pre_allocated_view["bandEnergyD"] = E_D
        return pre_allocated_view

    def params_schema(self) -> dict[str, Any]:
        return {
            "n_sub_bands": {
                "type": "int", "value": _LAPLACE_SUB_BANDS, "readonly": True,
                "label": "Equal A/B/C/D sub-band split (structural)",
            },
            "half_power_ratio": {
                "type": "float", "value": float(1.0 / np.sqrt(2.0)), "readonly": True,
                "label": "−3 dB half-power ratio for Q-factor bandwidth",
            },
            "band_hz": {
                "type": "range", "value": [self.ws.f_min, self.ws.f_max], "readonly": True,
                "label": "In-band analysis range [f_min, f_max] (set via DSP params)",
            },
        }


class Hack196kHzStrategy(BaseDSPStrategy):
    """
    id = "hack_196k" — decimates the raw feed to observe the simulated
    192 kHz → 196 kHz boundary displacement (aliasing fold) shift.

    The true capture Nyquist is sr/2 (192 kHz @ 384 kHz). Decimating by `q`
    drops the observation Nyquist to sr/(2q); any in-band energy above that
    folds back across the boundary. We measure the displacement between the
    true dominant peak folded into the decimated band and the dominant peak as
    *observed* through the decimated spectrum — the "196 kHz hack" telemetry.

    Decimation is a strided gather into a pre-allocated contiguous buffer; the
    only event-rate allocation is the decimated rfft output (G3-bounded).
    """

    __slots__ = (
        "_q", "_nd", "_hack_boundary_hz", "_nyq_dec",
        "_win_dec", "_dec", "_decwin", "_mag_dec",
        "_freqs_dec", "_dec_band_idx", "_dec_band_mag",
    )

    def __init__(self, ws: SpectralWorkspace, decim_factor: int = 2,
                 hack_boundary_hz: float = 196_000.0) -> None:
        super().__init__("hack_196k", "196 kHz Hack", ws)
        self._q = max(1, int(decim_factor))
        self._nd = ws.n // self._q
        self._hack_boundary_hz = float(hack_boundary_hz)
        self._nyq_dec = ws.sr / (2.0 * self._q)

        self._win_dec = scipy.signal.windows.hann(self._nd).astype(np.float64)
        self._dec = np.empty(self._nd, dtype=np.float64)      # contiguous decimated buf
        self._decwin = np.empty(self._nd, dtype=np.float64)   # windowed scratch
        self._mag_dec = np.empty(self._nd // 2 + 1, dtype=np.float64)
        self._freqs_dec = np.fft.rfftfreq(self._nd, d=float(self._q) / ws.sr)

        dec_mask = (self._freqs_dec >= ws.f_min) & (self._freqs_dec <= min(ws.f_max, self._nyq_dec))
        self._dec_band_idx = np.nonzero(dec_mask)[0].astype(np.intp)
        self._dec_band_mag = np.empty(self._dec_band_idx.shape[0], dtype=np.float64)

    def analyze(self, samples, sr, pre_allocated_view):
        x = pre_allocated_view[_VIEW_X]                 # float64 samples (n,)
        q, nd = self._q, self._nd

        # ── Strided decimation into the pre-allocated contiguous buffer ───────
        np.copyto(self._dec, x[:q * nd:q])
        np.multiply(self._dec, self._win_dec, out=self._decwin)
        np.abs(np.fft.rfft(self._decwin), out=self._mag_dec)  # rfft alloc: event-rate

        if self._dec_band_idx.size:
            np.take(self._mag_dec, self._dec_band_idx, out=self._dec_band_mag)
            dom_dec_hz = float(self._freqs_dec[self._dec_band_idx[int(np.argmax(self._dec_band_mag))]])
        else:
            dom_dec_hz = 0.0

        # ── True full-rate dominant peak (independent of laplace toggle) ──────
        band_mag = pre_allocated_view[_VIEW_BAND_MAG]
        dom_full_hz = float(self.ws.freqs_band[int(np.argmax(band_mag))]) if band_mag.size else 0.0

        # ── Fold the true peak across the decimated Nyquist boundary ──────────
        period = sr / q
        g = dom_full_hz % period
        folded = g if g <= self._nyq_dec else (period - g)
        boundary_shift = abs(folded - dom_dec_hz)

        pre_allocated_view["decimatedNyquistHz"] = self._nyq_dec
        pre_allocated_view["decimatedDominantHz"] = dom_dec_hz
        pre_allocated_view["observedAliasHz"] = float(folded)
        pre_allocated_view["boundaryShiftHz"] = float(boundary_shift)
        pre_allocated_view["hackBoundaryHz"] = self._hack_boundary_hz
        return pre_allocated_view

    def params_schema(self) -> dict[str, Any]:
        return {
            "decim_factor": {
                "type": "int", "value": int(self._q),
                "min": 1, "max": 8,
                "label": "Decimation factor q (rebuild-time; drops Nyquist to sr/2q)",
            },
            "hack_boundary_hz": {
                "type": "float", "value": float(self._hack_boundary_hz),
                "min": 1.0, "max": float(self.ws.sr),
                "label": "Simulated 196 kHz alignment boundary",
            },
            "decimated_nyquist_hz": {
                "type": "float", "value": float(self._nyq_dec), "readonly": True,
                "label": "Observation Nyquist after decimation (sr/2q)",
            },
        }


class StrongWeakRelationshipStrategy(BaseDSPStrategy):
    """
    id = "strong_weak_relationship" — Computes the relationship ratios between 
    strong and weak acoustic elements.
    
    Uses the pre-calculated band energies E_A, E_B, E_C, E_D from the Laplace Ratio Strategy 
    to determine:
      • Laplace cross-ratio (A*D / B*C)
      • Ripple FFT ratio (A*C / B*D)
      
    This function was extracted specifically to allow independent toggling
    via the dashboard.
    """

    __slots__ = ()

    def __init__(self, ws: SpectralWorkspace) -> None:
        super().__init__("strong_weak_relationship", "Strong/Weak Relationship", ws)

    def analyze(self, samples, sr, pre_allocated_view):
        # Retrieve energies from view (depends on LaplaceRatioStrategy running first)
        E_A = pre_allocated_view.get("bandEnergyA", 1.0)
        E_B = pre_allocated_view.get("bandEnergyB", 1.0)
        E_C = pre_allocated_view.get("bandEnergyC", 1.0)
        E_D = pre_allocated_view.get("bandEnergyD", 1.0)

        laplace_cross_ratio = (E_A * E_D) / max(E_B * E_C, _EPS)
        ripple_fft_ratio = (E_A * E_C) / max(E_B * E_D, _EPS)

        pre_allocated_view["laplaceCrossRatio"] = laplace_cross_ratio
        pre_allocated_view["rippleFFTRatio"] = ripple_fft_ratio
        return pre_allocated_view

    def params_schema(self) -> dict[str, Any]:
        return {
            "description": {
                "type": "string", "value": "Computes ratios A/C, B/D, etc.", "readonly": True,
                "label": "Strong/Weak Element Ratio Mode",
            }
        }


class LibrosaHumanFeaturesStrategy(BaseDSPStrategy):
    """
    id = "librosa_human_features" — extracts exactly the 5 features from the 
    user's ground truth Colab script using librosa: 
    Duration, Peak Time, Peak Amplitude (Impact), Peak Frequency, Mel_Mean.
    """
    __slots__ = ()

    def __init__(self, ws: SpectralWorkspace) -> None:
        super().__init__("librosa_human_features", "Human Ground Truth (Librosa)", ws)

    def analyze(self, samples, sr, pre_allocated_view):
        # We perform exact logic from the user's Colab script
        # 1. Trim silence (top_db=30)
        try:
            # We operate on the original 1s segment, converting to float32 for librosa
            y = samples.astype(np.float32)
            y_trim, _ = librosa.effects.trim(y, top_db=30)
            
            duration = len(y_trim) / sr
            
            # RMS Energy
            rms = librosa.feature.rms(y=y_trim, frame_length=4096, hop_length=256)[0]
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)
            
            peak_index = int(np.argmax(rms))
            peak_time = float(times[peak_index])
            peak_amp = float(rms[peak_index])
            
            # STFT -> Peak Frequency
            D = librosa.stft(y_trim, n_fft=8192)
            S = np.abs(D)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
            peak_freq = float(freqs[np.argmax(np.mean(S, axis=1))])
            
            # Mel Spectrogram -> Mean Value
            mel = librosa.feature.melspectrogram(
                y=y_trim,
                sr=sr,
                n_fft=8192,
                n_mels=256,
                fmax=sr // 2
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_mean = float(np.mean(mel_db))
            
            pre_allocated_view["librosa_duration"] = duration
            pre_allocated_view["librosa_peak_time"] = peak_time
            pre_allocated_view["librosa_peak_amp"] = peak_amp
            pre_allocated_view["librosa_peak_freq"] = peak_freq
            pre_allocated_view["librosa_mel_mean"] = mel_mean
        except Exception as e:
            logger.error(f"Librosa strategy failed: {e}")
            
        return pre_allocated_view

    def params_schema(self) -> dict[str, Any]:
        return {
            "description": {
                "type": "string", "value": "Extracts Human CSV features (Duration, Impact, Mel Mean)", "readonly": True,
                "label": "Librosa Exact Feature Matching",
            }
        }


class StrategyRegistry:
    """
    B6.3 / B6.4 — runtime registry mapping {id: BaseDSPStrategy} with a mutable
    per-id enabled flag. The frontend `dsp_strategy_toggle` flips `enabled`
    without rebuilding any strategy, so the computation graph is rewired
    in-place between events (no context blocking).

    Registration order is preserved; :meth:`run` dispatches the *enabled*
    strategies in that order into a shared, reused view container.
    """

    __slots__ = ("_index", "_enabled", "_order")

    def __init__(self) -> None:
        self._index: dict[str, BaseDSPStrategy] = {}
        self._enabled: dict[str, bool] = {}
        self._order: list[str] = []

    def register(self, strategy: BaseDSPStrategy, *, enabled: bool = True) -> "StrategyRegistry":
        if strategy.id not in self._index:
            self._order.append(strategy.id)
        self._index[strategy.id] = strategy
        self._enabled[strategy.id] = bool(enabled)
        return self

    def toggle(self, strategy_id: str, enabled: bool) -> bool:
        """Flip a strategy's enabled flag. Returns False for an unknown id."""
        if strategy_id not in self._enabled:
            return False
        self._enabled[strategy_id] = bool(enabled)
        return True

    def is_enabled(self, strategy_id: str) -> bool:
        return self._enabled.get(strategy_id, False)

    def has(self, strategy_id: str) -> bool:
        return strategy_id in self._index

    def status(self) -> dict[str, bool]:
        """Snapshot of {id: enabled} in registration order."""
        return {sid: self._enabled[sid] for sid in self._order}

    def schemas(self) -> dict[str, dict]:
        """
        B6.1 — capability map {id: {display_name, enabled, params}} aggregating
        every registered strategy's params_schema(), in registration order. Pure
        metadata for the frontend strategy panel; allocates only at query time.
        """
        return {
            sid: {
                "display_name": self._index[sid].display_name,
                "enabled": self._enabled[sid],
                "params": self._index[sid].params_schema(),
            }
            for sid in self._order
        }

    def active(self):
        """Yield the enabled strategies in registration order."""
        for sid in self._order:
            if self._enabled[sid]:
                yield self._index[sid]

    def run(
        self,
        samples: np.ndarray,
        sr: int,
        view: dict[str, float],
    ) -> dict[str, float]:
        """Dispatch every enabled strategy into the shared `view` (in order)."""
        for strategy in self.active():
            strategy.analyze(samples, sr, view)
        return view


def build_default_registry(ws: SpectralWorkspace) -> StrategyRegistry:
    """
    B6.2 — assemble the three concrete strategies against a shared workspace,
    with default enabled state matching the frontend `DEFAULT_STRATEGIES`
    (envelope + laplace on, 196 kHz hack off).
    """
    registry = StrategyRegistry()
    registry.register(NormalizedLogEnergyEnvelopeStrategy(ws), enabled=True)
    registry.register(LaplaceRatioStrategy(ws), enabled=True)
    registry.register(StrongWeakRelationshipStrategy(ws), enabled=True)
    registry.register(LibrosaHumanFeaturesStrategy(ws), enabled=False)
    registry.register(Hack196kHzStrategy(ws), enabled=False)
    return registry


class DSPExtractor:
    """
    Pre-allocated dual-domain feature extractor for the Rev. D ``/ws/dsp`` stream.

    Post-B6 this is a thin orchestrator: it owns one pre-allocated spectral
    workspace + scratch and a :class:`StrategyRegistry`, computes the shared
    transform arrays (windowed magnitude, Hilbert envelope) once per event into
    pre-allocated buffers, then runs the *enabled* strategies into a reused
    ``view`` container and assembles the wire dict.

    Always-on base channel (cheap, independent of toggles):
      • peak, peak_dbfs, dominant_hz   (human display)
      • chroma top-4                   (vectorized 12-semitone binning)
      • 512-bin peak-preserving waveform decimation (visualizer)

    Toggle-gated analytical metrics are supplied by the strategies (decay,
    duration, Laplace/Ripple ratios, Q, in-band ratio, 196 kHz boundary shift).
    """

    __slots__ = (
        "sr", "n", "n_fft", "hop_length", "f_min", "f_max", "n_bins",
        "ws", "registry",
        "_window", "_rfft_freqs", "_band_mask", "_band_idx", "_freqs_band",
        "_pc_idx", "_chroma", "_band_edges", "_bin_ar", "_group", "_t_axis",
        "_x", "_winscratch", "_mag", "_band_mag", "_env", "_view", "_trimmed_abs",
    )

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        segment_samples: int = EVENT_SEGMENT_SAMPLES,
        n_fft: int = 4096,
        hop_length: int = 1024,
        f_min: float = 1000.0,
        f_max: float = 120_000.0,
        n_bins: int = 512,
        registry: StrategyRegistry | None = None,
    ) -> None:
        self.sr = int(sample_rate)
        self.n = int(segment_samples)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.n_bins = int(n_bins)

        # ── Shared analysis plan (axes / masks / edges) — built once ──────────
        self.ws = SpectralWorkspace.build(
            self.sr, self.n, self.n_fft, self.hop_length, self.f_min, self.f_max
        )
        self._rfft_freqs = self.ws.rfft_freqs
        self._band_mask = self.ws.band_mask
        self._band_idx = self.ws.band_idx
        self._freqs_band = self.ws.freqs_band
        self._band_edges = self.ws.band_edges
        self._t_axis = self.ws.t_axis

        # Chroma pitch class for every in-band rfft bin (vectorized binning map).
        pc = np.round(12.0 * np.log2(np.maximum(self._freqs_band, 1e-6) / _F_REF_HZ))
        self._pc_idx = np.mod(pc, 12).astype(np.intp)
        self._chroma = np.zeros(12, dtype=np.float64)         # reused every event

        # Peak-preserving decimation helpers.
        self._group = max(1, self.n // self.n_bins)
        self._bin_ar = np.arange(self.n_bins)

        # Analysis window — allocated once.
        self._window = scipy.signal.windows.hann(self.n).astype(np.float64)

        # ── Pre-allocated per-event transform buffers (G1) ────────────────────
        self._x = np.zeros(self.n, dtype=np.float64)
        self._winscratch = np.empty(self.n, dtype=np.float64)
        self._mag = np.empty(self._rfft_freqs.shape[0], dtype=np.float64)
        self._band_mag = np.empty(self._freqs_band.shape[0], dtype=np.float64)
        self._env = np.empty(self.n, dtype=np.float64)
        self._view: dict[str, float] = {}                     # reused container
        # Pre-allocated abs buffer for peak-preserving decimation in _assemble() (G1).
        self._trimmed_abs = np.empty((self.n_bins, self._group), dtype=np.float64)

        # ── B6 strategy registry (default or injected) ────────────────────────
        self.registry = registry if registry is not None else build_default_registry(self.ws)

        logger.info(
            "DSPExtractor ready: %d samples @ %d Hz | band [%.0f, %.0f] Hz | %d-bin decimation | strategies=%s",
            self.n, self.sr, self.f_min, self.f_max, self.n_bins, self.registry.status(),
        )

    # ── Internal: refresh shared transform arrays for one event (G1) ─────────
    def _ingest(self, samples: np.ndarray) -> dict[str, float]:
        """
        Load `samples` into the pre-allocated buffers, recompute the shared
        windowed magnitude + Hilbert envelope, populate the reused view with
        the reserved array handles, and run the enabled strategies. Returns the
        view (scalar metrics + reserved arrays).
        """
        n = self.n
        m = min(n, samples.shape[0])
        self._x[:m] = samples[:m]          # direct type-cast slice-assignment (no intermediate np.asarray)
        if m < n:
            self._x[m:] = 0.0

        # Windowed magnitude spectrum into pre-allocated _mag.
        np.multiply(self._x, self._window, out=self._winscratch)
        np.abs(np.fft.rfft(self._winscratch), out=self._mag)      # rfft alloc: event-rate
        np.take(self._mag, self._band_idx, out=self._band_mag)    # in-band gather, no temp

        # Hilbert envelope into pre-allocated _env.
        np.abs(scipy.signal.hilbert(self._x), out=self._env)      # hilbert alloc: event-rate

        view = self._view
        view.clear()
        view[_VIEW_X] = self._x
        view[_VIEW_MAG] = self._mag
        view[_VIEW_BAND_MAG] = self._band_mag
        view[_VIEW_ENV] = self._env

        self.registry.run(self._x, self.sr, view)
        return view

    # ── Internal: assemble the Rev. D wire dict from a populated view ────────
    def _assemble(self, view: dict[str, float]) -> dict:
        # Always-on base human channel (cheap, toggle-independent).
        peak = float(max(abs(self._x.max().item()), abs(self._x.min().item())))
        peak_dbfs = float(20.0 * np.log10(peak)) if peak > 1e-9 else -120.0
        if self._band_mag.size:
            dominant_hz = float(self._freqs_band[self._band_mag.argmax().item()])
        else:
            dominant_hz = 0.0

        # Chroma top-4 (vectorized np.add.at into pre-zeroed accumulator).
        self._chroma[:] = 0.0
        np.add.at(self._chroma, self._pc_idx, self._band_mag)
        top = np.argsort(self._chroma)[-4:][::-1]
        chroma_top4 = [_CHROMA_NAMES[int(i)] for i in top]

        # 512-bin peak-preserving decimation (reshape view, vectorized, in-place abs).
        trimmed = self._x[: self.n_bins * self._group].reshape(self.n_bins, self._group)
        np.abs(trimmed, out=self._trimmed_abs)                # ← in-place; no transient heap alloc (G1)
        amax = self._trimmed_abs.argmax(axis=1)
        waveform = trimmed[self._bin_ar, amax].astype(np.float32).tolist()

        # Strategy-supplied metrics (graceful defaults when a strategy is off).
        duration_s = float(view.get("duration_s", 0.0))
        fingerprint = {
            "eventTimestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "timeSlice": {"startMs": 0.0, "endMs": 1000.0},
            "dspParams": {"nFft": self.n_fft, "hopLength": self.hop_length, "fMax": int(self.f_max)},
            "metrics": {
                "signalDurationUs": float(view.get("signalDurationUs", 0.0)),
                "decayRate": float(view.get("decayRate", 0.0)),
                "sharpnessIndex": float(view.get("sharpnessIndex", 1.0)),
                "inBandEnergyRatio": float(view.get("inBandEnergyRatio", 0.0)),
            },
            "laplaceCrossRatio": float(view.get("laplaceCrossRatio", 0.0)),
            "rippleFFTRatio": float(view.get("rippleFFTRatio", 0.0)),
            "chroma_top4": chroma_top4,
            "activeStrategies": self.registry.status(),
        }
        # The 196 kHz hack block is only present when that strategy is enabled.
        if "boundaryShiftHz" in view:
            fingerprint["hack196k"] = {
                "decimatedNyquistHz": float(view.get("decimatedNyquistHz", 0.0)),
                "decimatedDominantHz": float(view.get("decimatedDominantHz", 0.0)),
                "observedAliasHz": float(view.get("observedAliasHz", 0.0)),
                "boundaryShiftHz": float(view.get("boundaryShiftHz", 0.0)),
                "hackBoundaryHz": float(view.get("hackBoundaryHz", 196_000.0)),
            }

        # The human channel relies on the Librosa strategy if present, otherwise fallbacks.
        human_impact = float(view.get("librosa_peak_amp", peak))
        
        return {
            "human": {
                "peak": peak,
                "peak_dbfs": peak_dbfs,
                "duration_s": float(view.get("librosa_duration", duration_s)),
                "peak_time_s": float(view.get("librosa_peak_time", 0.0)),
                "dominant_hz": float(view.get("librosa_peak_freq", dominant_hz)),
                "impact": round(human_impact * 1000.0) / 10.0 if "librosa_peak_amp" not in view else float(view["librosa_peak_amp"]),
                "mel_mean_db": float(view.get("librosa_mel_mean", -120.0)),
            },
            "waveform": waveform,
            "chroma_top4": chroma_top4,
            "fingerprint": fingerprint,
        }

    def extract(self, samples: np.ndarray) -> dict:
        """Vectorized dual-domain extraction for one event. Returns a wire dict."""
        view = self._ingest(samples)
        return self._assemble(view)

    def recompute(self, samples: np.ndarray) -> dict:
        """
        B6.4 — re-run the *currently enabled* strategy set over a cached sample
        buffer (used by the `dsp_strategy_toggle` hook). Pure: it does not
        advance any external state (e.g. the surface tracker), so toggling a
        strategy refreshes the metrics for the last event without double-
        counting. Identical computation path to :meth:`extract`.
        """
        return self.extract(samples)


def feature_point(dominant_hz: float, f_max: float, in_band: float) -> tuple[float, float]:
    """
    Map an event onto a normalized 2-D feature plane (cx, cy) ∈ [0, 1]² used by
    the Confidence-Ellipse / ridge geometry. Provisional — the production mapping
    is mapDSPToPhysics (A1.3) / B5.
    """
    cx = float(np.clip(in_band, 0.0, 1.0))
    cy = float(np.clip((np.log10(max(dominant_hz, 1.0)) - 3.0) / 2.0, 0.0, 1.0))  # 1k→0, 100k→1
    return cx, cy


class SurfaceTracker:
    """
    Running per-surface mean + (EMA) covariance for the 4 material clusters.

    Each event's (cx, cy) is assigned to the nearest of 4 seeded centroids (A–D);
    that surface's mean and covariance are updated with scalar bookkeeping only
    (no loops over audio — G7). Emits the geometry the frontend draws as
    Confidence Ellipses / ridge footprints. Placeholder classifier (real
    assignment is B5/B7).
    """

    __slots__ = ("_letters", "_seed", "_mx", "_my", "_sxx", "_sxy", "_syy", "_n")

    def __init__(self) -> None:
        self._letters = ("A", "B", "C", "D")
        self._seed = np.array([[0.30, 0.70], [0.70, 0.70], [0.30, 0.30], [0.70, 0.30]])
        self._mx = self._seed[:, 0].astype(np.float64).copy()
        self._my = self._seed[:, 1].astype(np.float64).copy()
        self._sxx = np.full(4, 2.5e-3)
        self._sxy = np.zeros(4)
        self._syy = np.full(4, 2.5e-3)
        self._n = np.zeros(4)

    def update(self, cx: float, cy: float) -> str:
        d = (self._seed[:, 0] - cx) ** 2 + (self._seed[:, 1] - cy) ** 2
        i = int(np.argmin(d))
        self._n[i] += 1.0
        k = self._n[i]
        dx = cx - self._mx[i]
        dy = cy - self._my[i]
        self._mx[i] += dx / k
        self._my[i] += dy / k
        dx2 = cx - self._mx[i]
        dy2 = cy - self._my[i]
        # Exponentially-weighted covariance (stable, bounded, no growth).
        self._sxx[i] = 0.9 * self._sxx[i] + 0.1 * (dx * dx2)
        self._sxy[i] = 0.9 * self._sxy[i] + 0.1 * (dx * dy2)
        self._syy[i] = 0.9 * self._syy[i] + 0.1 * (dy * dy2)
        return self._letters[i]

    def surfaces(self) -> list[dict]:
        return [
            {
                "group": self._letters[i],
                "cx": float(self._mx[i]),
                "cy": float(self._my[i]),
                "sxx": float(max(self._sxx[i], 1e-4)),
                "sxy": float(self._sxy[i]),
                "syy": float(max(self._syy[i], 1e-4)),
            }
            for i in range(4)
        ]


def synthesize_impact_segment(
    sample_rate: int = SAMPLE_RATE,
    n: int = EVENT_SEGMENT_SAMPLES,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a realistic damped-sinusoid impact for demo/manual triggers when no
    384 kHz hardware feed is attached. Vectorized; allocates only at event rate.
    Random ring frequency + strike force so ridges move along Z and chroma varies.
    """
    rng = rng or np.random.default_rng()
    t = np.arange(n, dtype=np.float64) / sample_rate
    f0 = float(rng.uniform(2_000.0, 90_000.0))
    sigma = float(rng.uniform(8.0, 40.0))
    onset = np.exp(-sigma * np.maximum(t - 0.5, 0.0))         # impact at t = 0.5 s
    sig = np.sin(2.0 * np.pi * f0 * t) * onset
    sig += 0.02 * rng.standard_normal(n)                      # ambient noise floor
    force = float(rng.uniform(0.2, 0.95))                     # random strike amplitude
    sig *= force / (float(np.max(np.abs(sig))) + 1e-9)
    return sig.astype(np.float32)
