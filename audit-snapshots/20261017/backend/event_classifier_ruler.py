"""
event_classifier_ruler.py — Multi-Mode Acoustic Classifier
===========================================================
Supports 3 interchangeable classification engines:

  MODE 1 — "baseline"   : Original Structural Index (SI) centroid ruler
  MODE 2 — "cascade"    : Physics-Cascade Rulers (Laplace → HFSR → ZCR/Dur)
  MODE 3 — "ddt"        : Dynamic Decision Tree (auto-calibrated, sklearn)
  MODE 4 — "orthogonal_2x2" : 2x2 Feature Matrix (Cross/Laplace vs FFT/HFSR)

Switch between modes via `classification_mode` in classify_event_with_ruler().
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# GROUP METADATA (shared across all modes)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroupRulerBounds:
    group: str
    label: str
    material: str
    color: str
    si_range: Tuple[float, float]
    viewport_coords: Dict[str, float]

GROUP_RULERS: Dict[str, GroupRulerBounds] = {
    "A": GroupRulerBounds(
        group="A",
        label="Center-Round (Deep Impact)",
        material="Unscratched Center",
        color="#00ffe7",
        si_range=(-float("inf"), 693.7),
        viewport_coords={"time_center_sec": 0.32, "amp_center_db": 45.0},
    ),
    "B": GroupRulerBounds(
        group="B",
        label="Messy/Sharp (Rough Scratches)",
        material="Heavy Scratches",
        color="#a78bfa",
        si_range=(693.7, 2406.0),
        viewport_coords={"time_center_sec": 0.65, "amp_center_db": 85.0},
    ),
    "C": GroupRulerBounds(
        group="C",
        label="Round with Tail (Delamination)",
        material="Unscratched Center + Tail",
        color="#00ff41",
        si_range=(2406.0, 3182.5),
        viewport_coords={"time_center_sec": 1.05, "amp_center_db": 30.0},
    ),
    "D": GroupRulerBounds(
        group="D",
        label="S-Shape Vertical (Whistling Slit)",
        material="Vertical S-shape Scratch",
        color="#ffb800",
        si_range=(3182.5, float("inf")),
        viewport_coords={"time_center_sec": 1.34, "amp_center_db": 72.0},
    ),
    "B or A/C": GroupRulerBounds(
        group="B or A/C",
        label="Uncertain (Mid-Low Freq)",
        material="Needs Calibration",
        color="#FFA500",
        si_range=(-float("inf"), float("inf")),
        viewport_coords={"time_center_sec": 0.5, "amp_center_db": 60.0},
    ),
    "A/C or D": GroupRulerBounds(
        group="A/C or D",
        label="Uncertain (Mid-High Freq)",
        material="Needs Calibration",
        color="#FFA500",
        si_range=(-float("inf"), float("inf")),
        viewport_coords={"time_center_sec": 1.2, "amp_center_db": 60.0},
    ),
    "A or C": GroupRulerBounds(
        group="A or C",
        label="Uncertain (A vs C)",
        material="Needs Calibration",
        color="#FFA500",
        si_range=(-float("inf"), float("inf")),
        viewport_coords={"time_center_sec": 0.7, "amp_center_db": 30.0},
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — BASELINE: Structural Index (SI) Formula
# ─────────────────────────────────────────────────────────────────────────────

def calculate_structural_index(freq_peak: float, freq_rolloff: float, freq_centroid: float) -> float:
    """Computes the physical Structural Index (SI) from 3 spectral features."""
    return (0.984333 * freq_peak) - (0.138814 * freq_rolloff) + (0.108715 * freq_centroid)


def _classify_baseline(si: float) -> str:
    """MODE 1: Simple SI threshold ruler — original behaviour."""
    if si < 693.7:
        return "A"
    elif si < 2406.0:
        return "B"
    elif si < 3182.5:
        return "C"
    else:
        return "D"


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — CASCADE: Physics Rulers (Laplace → HFSR → ZCR/Dur → B fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_cascade(
    frequency_hz: float,
    laplace_sigma: float,
    high_freq_ratio: float,
    zcr: float,
    duration_ms: float,
    bounce_count: int,
    bounds: Dict[str, float],
) -> Tuple[str, str]:
    """
    MODE 2: Cascade Physics Rulers.
    Uses the 2D Bounding Box / Calibration rule if dynamic bounds are calibrated,
    otherwise falls back to the original relative cascade rules.
    """
    # ── Path A: Calibrated Dynamic Bounds (Algebraic Dynamics Rules) ──────────────
    if "duration_split_A_C" in bounds and "freq_split_B_D" in bounds and "laplace_split_A_D" in bounds:
        dur_split = bounds["duration_split_A_C"]
        freq_split = bounds["freq_split_B_D"]
        lap_split = bounds["laplace_split_A_D"]
        zcr_split = bounds.get("zcr_a_c", 0.0295)
        hfsr_split = bounds.get("hfsr_split_D", 0.45)

        if high_freq_ratio > hfsr_split:
            return "D", f"Calibrated Cascade: HFSR={high_freq_ratio:.3f} > {hfsr_split:.3f} → D"

        if duration_ms >= dur_split:
            # Long duration -> A or C
            if laplace_sigma > 0:
                if laplace_sigma < lap_split:
                    return "A", f"Algebraic Cascade: Dur={duration_ms:.1f}ms >= {dur_split:.1f}ms & Laplace σ={laplace_sigma:.5f} < {lap_split:.5f} → A"
                else:
                    return "C", f"Algebraic Cascade: Dur={duration_ms:.1f}ms >= {dur_split:.1f}ms & Laplace σ={laplace_sigma:.5f} >= {lap_split:.5f} → C"
            else:
                if zcr < zcr_split:
                    return "A", f"Algebraic Cascade: Dur={duration_ms:.1f}ms >= {dur_split:.1f}ms & ZCR={zcr:.5f} < {zcr_split:.5f} → A"
                else:
                    return "C", f"Algebraic Cascade: Dur={duration_ms:.1f}ms >= {dur_split:.1f}ms & ZCR={zcr:.5f} >= {zcr_split:.5f} → C"
        else:
            # Short duration -> B or D
            if frequency_hz < freq_split:
                return "B", f"Algebraic Cascade: Dur={duration_ms:.1f}ms < {dur_split:.1f}ms & Freq={frequency_hz:.1f}Hz < {freq_split:.1f}Hz → B"
            else:
                return "D", f"Algebraic Cascade: Dur={duration_ms:.1f}ms < {dur_split:.1f}ms & Freq={frequency_hz:.1f}Hz >= {freq_split:.1f}Hz → D"

    # ── Path B: Uncalibrated Fallback (Original Cascade Logic) ────────────────
    laplace_split = bounds.get("laplace_split_A", 0.0031)
    hfsr_split    = bounds.get("hfsr_split_D",    0.45)
    zcr_split     = bounds.get("zcr_split_C",     0.0295)
    dur_split     = bounds.get("dur_split_C",     900.0)

    # Step 1: Group A — Laplace alone
    if laplace_sigma > 0 and laplace_sigma < laplace_split:
        return "A", f"Fallback Cascade: Laplace σ={laplace_sigma:.5f} < {laplace_split:.5f} → A"

    # Step 2: Group D — High-Frequency Spectral Ratio
    if high_freq_ratio > hfsr_split:
        return "D", f"Fallback Cascade: HFSR={high_freq_ratio:.3f} > {hfsr_split:.3f} → D"

    # Step 3: Group C — ZCR + Duration
    if zcr > zcr_split and duration_ms >= dur_split:
        return "C", f"Fallback Cascade: ZCR={zcr:.5f} > {zcr_split:.5f} & Dur={duration_ms:.0f}ms >= {dur_split:.0f} → C"

    # Step 4: Default → Group B
    return "B", "Fallback Cascade: Default fallback → B"


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — DDT: Dynamic Decision Tree (sklearn, auto-calibrated)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cached DDT model (trained once per session on startup)
_ddt_model = None
_ddt_feature_names: List[str] = ["Impact_Freq_Hz", "ZCR", "Bounce_Count", "Duration_Ms", "Laplace_Sigma"]


def train_ddt(training_events: List[Dict[str, Any]]) -> bool:
    """
    Train the Dynamic Decision Tree from a list of training event dicts.
    Each dict must contain: Impact_Freq_Hz, ZCR, Bounce_Count, Duration_Ms,
                            Laplace_Sigma, True_Group.
    Returns True if training succeeded.
    """
    global _ddt_model
    try:
        from sklearn.tree import DecisionTreeClassifier
        if len(training_events) < 4:
            return False
        X = np.array([[
            ev["Impact_Freq_Hz"],
            ev["ZCR"],
            ev["Bounce_Count"],
            ev["Duration_Ms"],
            ev["Laplace_Sigma"],
        ] for ev in training_events], dtype=np.float32)
        y = np.array([ev["True_Group"] for ev in training_events])
        clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
        clf.fit(X, y)
        _ddt_model = clf
        return True
    except Exception:
        return False


def _classify_ddt(
    impact_freq: float,
    zcr: float,
    bounce_count: int,
    duration_ms: float,
    laplace_sigma: float,
) -> Tuple[str, str]:
    """MODE 3: Dynamic Decision Tree prediction. Falls back to 'B' if untrained."""
    if _ddt_model is None:
        return "B", "DDT not trained — fallback → B"
    feats = np.array([[impact_freq, zcr, float(bounce_count), duration_ms, laplace_sigma]], dtype=np.float32)
    pred = str(_ddt_model.predict(feats)[0])
    proba = _ddt_model.predict_proba(feats)[0]
    max_prob = float(np.max(proba))
    return pred, f"DDT → {pred} (confidence {max_prob:.1%})"


# ─────────────────────────────────────────────────────────────────────────────
# MODE 4 — ORTHOGONAL 2X2: Cross (Laplace) vs FFT (HFSR)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_orthogonal(
    laplace_sigma: float,
    high_freq_ratio: float,
    bounds: Dict[str, float]
) -> Tuple[str, str]:
    """
    MODE 4: 2x2 Orthogonal Grid using Cross (Laplace) and FFT (HFSR).
    Cross perfectly isolates {A, D} from {B, C}.
    FFT perfectly isolates {A, C} from {B, D}.
    """
    # Use dynamically calibrated bounds if available, else fall back to physics defaults
    cross_split = bounds.get("laplace_split_A_D", bounds.get("laplace_split_A", 0.0032))
    fft_split = bounds.get("hfsr_split", bounds.get("hfsr_split_D", 0.45))

    is_ad = laplace_sigma < cross_split
    is_ac = high_freq_ratio < fft_split

    if is_ad and is_ac:
        return "A", f"2x2 Grid: Laplace(Cross)={laplace_sigma:.5f} < {cross_split} (A/D) & HFSR(FFT)={high_freq_ratio:.3f} < {fft_split} (A/C) → A"
    elif is_ad and not is_ac:
        return "D", f"2x2 Grid: Laplace(Cross)={laplace_sigma:.5f} < {cross_split} (A/D) & HFSR(FFT)={high_freq_ratio:.3f} >= {fft_split} (B/D) → D"
    elif not is_ad and is_ac:
        return "C", f"2x2 Grid: Laplace(Cross)={laplace_sigma:.5f} >= {cross_split} (B/C) & HFSR(FFT)={high_freq_ratio:.3f} < {fft_split} (A/C) → C"
    else:
        return "B", f"2x2 Grid: Laplace(Cross)={laplace_sigma:.5f} >= {cross_split} (B/C) & HFSR(FFT)={high_freq_ratio:.3f} >= {fft_split} (B/D) → B"


# ─────────────────────────────────────────────────────────────────────────────
# GOLDEN "A" TEMPLATE — physical anchor for boundary tie-breaks.
# Group A (unscratched center) gives the cleanest impact: its attack envelope
# and spectrum act as the absolute reference. C shares A's pure impact center
# (attack envelope matches); C also shares A's resonance structure (spectrum
# matches) while D's slit whistle does not. All comparisons are peak-aligned,
# max-normalized Pearson correlations — dimensionless and amplitude-invariant.
# ─────────────────────────────────────────────────────────────────────────────

_GOLDEN_A: Optional[Dict[str, Any]] = None
_GOLDEN_A_ENV_POINTS: int = 256
_GOLDEN_A_SPEC_POINTS: int = 512


def _resample_attack_envelope(envelope: np.ndarray, peak_idx: int, sample_rate: float,
                              attack_ms: float, n_points: int) -> Optional[np.ndarray]:
    """Peak-aligned attack window resampled to a fixed grid, peak-normalized."""
    n_att = max(int(sample_rate * attack_ms / 1000.0), 8)
    seg = np.asarray(envelope[peak_idx: peak_idx + n_att], dtype=np.float64)
    if seg.size < 8:
        return None
    out = np.interp(np.linspace(0.0, 1.0, n_points),
                    np.linspace(0.0, 1.0, seg.size), seg)
    peak = float(np.max(out))
    return out / peak if peak > 0.0 else None


def _resample_spectrum(mag: np.ndarray, freqs: np.ndarray, fmax: float,
                       n_points: int) -> Optional[np.ndarray]:
    """Magnitude spectrum interpolated onto a fixed 0..fmax grid, max-normalized."""
    out = np.interp(np.linspace(0.0, fmax, n_points),
                    np.asarray(freqs, dtype=np.float64),
                    np.asarray(mag, dtype=np.float64))
    top = float(np.max(out))
    return out / top if top > 0.0 else None


def build_golden_a_template(envelope: np.ndarray, peak_idx: int,
                            spectrum_mag: np.ndarray, spectrum_freqs: np.ndarray,
                            sample_rate: float, attack_ms: float = 20.0,
                            spec_fmax: Optional[float] = None,
                            confidence: float = 1.0) -> Optional[Dict[str, Any]]:
    """Build the Golden A anchor from a >95%-confidence Group A event."""
    if spec_fmax is None:
        spec_fmax = min(8000.0, 0.45 * float(sample_rate))
    env = _resample_attack_envelope(envelope, peak_idx, sample_rate,
                                    attack_ms, _GOLDEN_A_ENV_POINTS)
    spec = _resample_spectrum(spectrum_mag, spectrum_freqs, spec_fmax,
                              _GOLDEN_A_SPEC_POINTS)
    if env is None or spec is None:
        return None
    return {"attack_ms": float(attack_ms), "spec_fmax": float(spec_fmax),
            "sample_rate": float(sample_rate), "confidence": float(confidence),
            "envelope": env, "spectrum": spec}


def set_golden_a_template(template: Optional[Dict[str, Any]]) -> None:
    global _GOLDEN_A
    if template is not None:
        template = dict(template)
        template["envelope"] = np.asarray(template["envelope"], dtype=np.float64)
        template["spectrum"] = np.asarray(template["spectrum"], dtype=np.float64)
    _GOLDEN_A = template


def get_golden_a_template() -> Optional[Dict[str, Any]]:
    return _GOLDEN_A


def save_golden_a_template(path: str) -> bool:
    if _GOLDEN_A is None:
        return False
    payload = dict(_GOLDEN_A)
    payload["envelope"] = np.asarray(payload["envelope"]).tolist()
    payload["spectrum"] = np.asarray(payload["spectrum"]).tolist()
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return True


def load_golden_a_template(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as fh:
            set_golden_a_template(json.load(fh))
        return True
    except Exception:
        return False


def golden_a_envelope_correlation(envelope: np.ndarray, peak_idx: int,
                                  sample_rate: float) -> float:
    """Pearson r of the event's peak-aligned attack envelope vs the Golden A anchor."""
    if _GOLDEN_A is None:
        return float("nan")
    ev = _resample_attack_envelope(envelope, peak_idx, sample_rate,
                                   _GOLDEN_A["attack_ms"], _GOLDEN_A["envelope"].size)
    tpl = _GOLDEN_A["envelope"]
    if ev is None or np.std(ev) <= 0 or np.std(tpl) <= 0:
        return float("nan")
    return float(np.corrcoef(ev, tpl)[0, 1])


def golden_a_spectrum_correlation(spectrum_mag: np.ndarray,
                                  spectrum_freqs: np.ndarray) -> float:
    """Pearson r of the event's normalized spectrum vs the Golden A anchor."""
    if _GOLDEN_A is None:
        return float("nan")
    ev = _resample_spectrum(spectrum_mag, spectrum_freqs,
                            _GOLDEN_A["spec_fmax"], _GOLDEN_A["spectrum"].size)
    tpl = _GOLDEN_A["spectrum"]
    if ev is None or np.std(ev) <= 0 or np.std(tpl) <= 0:
        return float("nan")
    return float(np.corrcoef(ev, tpl)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def classify_event_with_ruler(
    frequency_hz: float,
    rolloff_hz: float,
    centroid_hz: float,
    impact_db: float = 0.0,
    duration_ms: float = 0.0,
    laplace_sigma: float = 0.0,
    zcr: float = 0.0,
    bounce_count: int = 1,
    high_freq_ratio: float = 0.0,
    dynamic_bounds: Optional[Dict[str, float]] = None,
    classification_mode: str = "baseline",  # "baseline" | "cascade" | "ddt"
    attack_envelope: Optional[np.ndarray] = None,
    envelope_peak_idx: int = 0,
    spectrum_mag: Optional[np.ndarray] = None,
    spectrum_freqs: Optional[np.ndarray] = None,
    sample_rate: float = 0.0,
    zcr_spike_ratio: float = 0.0,
) -> Dict[str, Any]:
    """
    Unified classifier entry point.

    classification_mode:
      "baseline"       — Structural Index (SI) threshold ruler  [MODE 1]
      "cascade"        — Physics Cascade (Laplace→HFSR→ZCR→B)   [MODE 2]
      "ddt"            — Dynamic Decision Tree (auto-calibrated) [MODE 3]
      "orthogonal_2x2" — 2x2 Feature Matrix (Cross vs FFT)       [MODE 4]
    """
    si = calculate_structural_index(frequency_hz, rolloff_hz, centroid_hz)
    reasoning = ""

    bounds = dynamic_bounds or {}

    # ── Switch across modes ───────────────────────────────────────────────────
    if classification_mode == "orthogonal_2x2":
        group, reasoning = _classify_orthogonal(
            laplace_sigma=laplace_sigma,
            high_freq_ratio=high_freq_ratio,
            bounds=bounds
        )

    elif classification_mode == "cascade":
        group, reasoning = _classify_cascade(
            frequency_hz=frequency_hz,
            laplace_sigma=laplace_sigma,
            high_freq_ratio=high_freq_ratio,
            zcr=zcr,
            duration_ms=duration_ms,
            bounce_count=bounce_count,
            bounds=bounds,
        )

    elif classification_mode == "ddt":
        group, reasoning = _classify_ddt(
            impact_freq=frequency_hz,
            zcr=zcr,
            bounce_count=bounce_count,
            duration_ms=duration_ms,
            laplace_sigma=laplace_sigma,
        )

    else:
        # MODE 1: baseline (legacy SI) — also handles old dynamic_bounds path
        if dynamic_bounds is not None and "duration_split_A_C" in bounds:
            # Hybrid AI-Physics ruler (backward compatible with original code)
            duration_split = bounds.get("duration_split_A_C", 1500.0)
            freq_b_d       = bounds.get("freq_split_B_D",       3180.0)
            laplace_split  = bounds.get("laplace_split_A_D",  0.0032)

            if duration_ms >= duration_split:
                if laplace_sigma > 0:
                    if laplace_sigma < laplace_split:
                        group = "A"
                    else:
                        group = "C"
                else:
                    zcr_a_c = bounds.get("zcr_a_c", 0.0295)
                    group = "A" if zcr < zcr_a_c else "C"
            else:
                group = "B" if frequency_hz < freq_b_d else "D"
            reasoning = f"Baseline hybrid: dur={duration_ms:.0f}ms, lap={laplace_sigma:.5f}, freq={frequency_hz:.0f}Hz → {group}"
        else:
            group = _classify_baseline(si)
            reasoning = f"SI={si:.1f} → {group}"

    # ── Calculate 2D Confidence Score ──────────────────────────────────────────
    confidence_score = 1.00
    if dynamic_bounds and "group_medians" in bounds:
        medians = bounds["group_medians"]
        if group in medians:
            g_data = medians[group]
            
            # 1. Y-Axis: Laplace (Cross) Distance
            lap_std = max(g_data.get("laplace_std", 0.0001), 0.0001)
            lap_bound = bounds.get("laplace_split_A_D", 0.0032)
            lap_margin = abs(laplace_sigma - lap_bound) / lap_std
            
            # 2. X-Axis: STFT (FFT) Distance
            if "hfsr_split" in bounds:
                fft_std = max(g_data.get("hfsr_std", 0.01), 0.01)
                fft_bound = bounds.get("hfsr_split")
                fft_val = high_freq_ratio
            else:
                fft_std = max(g_data.get("freq_std", 10.0), 10.0)
                fft_bound = bounds.get("freq_split_B_D", 3180.0)
                fft_val = frequency_hz
                
            fft_margin = abs(fft_val - fft_bound) / fft_std
            
            import math
            # 2D Euclidean Distance (Combined Margin)
            combined_margin = math.sqrt(fft_margin**2 + lap_margin**2)
            
            # Scale: Margin of 0 (on border) = 50%
            # Margin of 2 std devs away = 100%
            confidence_score = min(1.0, 0.50 + (combined_margin * 0.25))
            
            reasoning += f" | 2D Conf: {confidence_score:.0%} (Lap-Z:{lap_margin:.1f}, FFT-Z:{fft_margin:.1f})"

    # ── Golden "A" template tie-breaks (boundary/low-confidence events only) ──
    tb_conf = bounds.get("tiebreak_conf_threshold", 0.85)
    tb_split = bounds.get("template_corr_split", 0.60)
    if _GOLDEN_A is not None and confidence_score < tb_conf:
        # Rule 1 — B vs C (shared high-Laplace column): C shares A's pure
        # impact center, so its attack envelope correlates with the anchor;
        # B's scratch-damaged attack does not.
        if group in ("B", "C") and attack_envelope is not None and sample_rate > 0:
            r_t = golden_a_envelope_correlation(attack_envelope, envelope_peak_idx, sample_rate)
            if np.isfinite(r_t):
                group = "C" if r_t >= tb_split else "B"
                reasoning += (f" | GoldenA env-corr {r_t:.2f} "
                              f"{'>=' if r_t >= tb_split else '<'} {tb_split:.2f} → {group}")
        # Rule 2 — C vs D (diagonal): C shares A's low-frequency resonance
        # peak, so its spectrum correlates with the anchor; D's edge whistle
        # does not.
        if group in ("C", "D") and spectrum_mag is not None and spectrum_freqs is not None:
            r_f = golden_a_spectrum_correlation(spectrum_mag, spectrum_freqs)
            if np.isfinite(r_f):
                group = "C" if r_f >= tb_split else "D"
                reasoning += (f" | GoldenA spec-corr {r_f:.2f} "
                              f"{'>=' if r_f >= tb_split else '<'} {tb_split:.2f} → {group}")

    # ── ZCR-spike safety net: positive micro-lamellae evidence for Group B ───
    # B's chaotic scattering shows as short-window ZCR spikes; a B verdict
    # without a spike is likely a fallback ("garbage bin") assignment.
    # Calibrated-only (G6): activates when the calibrator provides a split —
    # measured B-vs-rest separation is soft (AUC ≈ 0.66), so no static default.
    zcr_spike_split = bounds.get("zcr_spike_split")
    if zcr_spike_split is not None and group == "B" and zcr_spike_ratio > 0.0:
        if zcr_spike_ratio >= zcr_spike_split:
            confidence_score = min(1.0, confidence_score + 0.10)
            reasoning += f" | ZCR spike x{zcr_spike_ratio:.1f} confirms B scattering"
        else:
            confidence_score = max(0.05, confidence_score - 0.15)
            reasoning += f" | no ZCR spike (x{zcr_spike_ratio:.1f}) — B unconfirmed"

    safe_group = group if group in GROUP_RULERS else "A"
    bounds_meta = GROUP_RULERS[safe_group]

    return {
        "group": group,
        "in_basin": True,
        "confidence_score": round(confidence_score, 2),
        "structural_index": round(si, 2),
        "status": "CONFIRMED",
        "classification_mode": classification_mode,
        "reasoning": reasoning,
        "ruler_match": {"si_matched": True},
        "bounds": {
            "label": bounds_meta.label,
            "material": bounds_meta.material,
            "color": bounds_meta.color,
            "viewport_coords": bounds_meta.viewport_coords,
        },
    }
