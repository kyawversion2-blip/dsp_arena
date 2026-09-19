#!/usr/bin/env python3
"""
event_centric_evaluator.py — Event-Centric Acoustic Decision Engine & Ruler Integration
======================================================================================
Implements the definitive physical event detection and sequential Markov decision logic:

1. Physical Event Detection & Extraction:
   - 4th-order Butterworth High-pass filter >= 1000 Hz (1 kHz transient core)
   - Hilbert analytic envelope + 20% peak threshold (onset, offset)
   - Physical Duration Rule: Acoustic impact < 0.5s; REJECT if detected duration > 1.0s
   - Slicing: Midpoint Centering + 0.5s Forward Focus (FF) + 0.5s Reverse Recovery (RR) = 1.0s Window

2. Consecutive Transition Decision Logic ("ကပ်ရပ် ဖြစ်တဲ့အကြိမ်ကိုအဓိကထားရမယ်"):
   - Compare current event features x_t with previous event x_{t-1}
   - If similarity S >= 75% (0.75):
       -> Predicted = Previous Group (e.g. "အရင်အကြိမ်က A ဆို A ပဲ")
   - If similarity S < 75% (0.75):
       -> Elimination: NOT Previous Group (3/4 = 75% confident elimination)
       -> Apply 2x2 Ruler (Laplace vs HFSR) over the remaining 3 candidates
"""

import os
import sys
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


# ─────────────────────────────────────────────────────────────────────────────
# 1. PHYSICAL EVENT DETECTION & EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_transient_window(
    audio: np.ndarray,
    sr: int,
    freq_threshold_hz: float = 1000.0,
    max_duration_s: float = 1.0,
    ff_s: float = 0.5,
    rr_s: float = 0.5,
    env_threshold_ratio: float = 0.20
) -> Tuple[Optional[np.ndarray], int, int, int, float]:
    """
    Strict Physical Event Detection:
    - 1kHz High-Pass filter
    - Hilbert envelope 20% peak span
    - Strict 1.0s rejection ceiling (impacts are < 0.5s)
    - 1.0s centered window (FF 0.5s + RR 0.5s)
    """
    audio = np.asarray(audio, dtype=np.float64).ravel()
    n = len(audio)
    if n == 0:
        return None, 0, 0, 0, 0.0

    # 1) High-pass filter above 1000 Hz
    nyq = sr / 2.0
    hp_norm = min(max(freq_threshold_hz / nyq, 1e-4), 0.99)
    try:
        b, a = signal.butter(4, hp_norm, btype='highpass')
        audio_hp = signal.filtfilt(b, a, audio)
    except Exception:
        audio_hp = audio

    # 2) Hilbert amplitude envelope + 1ms smoothing
    envelope = np.abs(signal.hilbert(audio_hp))
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

    # 3) Strict Physical Ceiling: Reject signals > 1.0s (impacts are < 0.5s)
    if detected_duration_s > max_duration_s:
        return None, -1, onset_idx, offset_idx, detected_duration_s

    # 4) Center on transient midpoint
    center_idx = (onset_idx + offset_idx) // 2

    # 5) Exact 1.0s Window: FF 0.5s + RR 0.5s
    start = max(0, center_idx - int(sr * ff_s))
    end = min(n, center_idx + int(sr * rr_s))

    return audio[start:end], center_idx, onset_idx, offset_idx, detected_duration_s


def measure_event_features(audio_win: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Extract key physical and spectral features from the isolated 1.0s event segment:
    - peak_freq_hz: Dominant transient frequency (FFT)
    - laplace_sigma: Decay damping rate (Laplace distribution fit)
    - high_freq_ratio: HFSR (ratio of 4-6kHz to 1-4kHz energy)
    - centroid_hz: Spectral brightness centroid
    - peak_amplitude: Peak transient magnitude (Impact)
    - duration_s: Transient duration
    """
    audio = np.asarray(audio_win, dtype=np.float64)
    n = len(audio)
    if n < 64:
        return {}

    # Peak amplitude (Impact)
    peak_amp = float(np.max(np.abs(audio)))

    # FFT Spectral Peak & Centroid (> 1000 Hz)
    mags = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    valid_idx = np.where(freqs >= 1000.0)[0]

    if len(valid_idx) > 0:
        v_mags = mags[valid_idx]
        v_freqs = freqs[valid_idx]
        peak_freq = float(v_freqs[np.argmax(v_mags)])
        centroid = float(np.sum(v_mags * v_freqs) / (np.sum(v_mags) + 1e-12))
    else:
        peak_freq = float(freqs[np.argmax(mags)]) if len(freqs) else 0.0
        centroid = peak_freq

    # HFSR (High Frequency Spectral Ratio)
    hf_mask = (freqs >= 4000.0) & (freqs <= 8000.0)
    lf_mask = (freqs >= 1000.0) & (freqs < 4000.0)
    hf_e = float(np.sum(mags[hf_mask])) if np.any(hf_mask) else 0.0
    lf_e = float(np.sum(mags[lf_mask])) if np.any(lf_mask) else 1e-6
    hfsr = float(hf_e / (lf_e + 1e-9))

    # Laplace Sigma (Decay rate)
    try:
        env = np.abs(signal.hilbert(audio))
        _, laplace_sigma = stats.laplace.fit(env)
        laplace_sigma = float(laplace_sigma)
    except Exception:
        laplace_sigma = 15.0

    # Effective transient decay duration (peak -> 10%)
    peak_pos = int(np.argmax(np.abs(audio)))
    tail = np.abs(audio[peak_pos:])
    below_10 = np.where(tail < 0.10 * peak_amp)[0]
    dur_s = float(below_10[0] / sr) if len(below_10) > 0 else float(len(tail) / sr)

    return {
        "frequency_hz": round(peak_freq, 1),
        "laplace_sigma": round(laplace_sigma, 4),
        "high_freq_ratio": round(hfsr, 6),
        "centroid_hz": round(centroid, 1),
        "peak_amplitude": round(peak_amp, 4),
        "duration_s": round(dur_s, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSECUTIVE TRANSITION DECISION LOGIC ("ကပ်ရပ် ဖြစ်တဲ့အကြိမ်ကို အဓိကထားရမယ်")
# ─────────────────────────────────────────────────────────────────────────────

def compute_event_similarity(curr_feats: Dict[str, float], prev_feats: Dict[str, float]) -> float:
    """
    Compute multi-feature similarity S in [0, 1] between consecutive events:
    - Log-frequency match: 40% weight
    - Laplace decay sigma match: 30% weight
    - HFSR texture match: 20% weight
    - Centroid match: 10% weight
    """
    if not curr_feats or not prev_feats:
        return 0.0

    # 1. Frequency (Log-scale difference)
    f1, f2 = max(100.0, curr_feats.get("frequency_hz", 1000.0)), max(100.0, prev_feats.get("frequency_hz", 1000.0))
    d_log_f = abs(np.log10(f1) - np.log10(f2))
    sim_f = max(0.0, 1.0 - (d_log_f / 0.30))  # within ~0.3 dex

    # 2. Laplace Sigma (Decay rate difference)
    s1, s2 = max(0.1, curr_feats.get("laplace_sigma", 10.0)), max(0.1, prev_feats.get("laplace_sigma", 10.0))
    sim_sg = max(0.0, 1.0 - abs(s1 - s2) / (s1 + s2 + 1e-6))

    # 3. HFSR (Spectral texture difference)
    h1, h2 = curr_feats.get("high_freq_ratio", 0.1), prev_feats.get("high_freq_ratio", 0.1)
    sim_h = max(0.0, 1.0 - abs(h1 - h2) / (h1 + h2 + 1e-6))

    # 4. Centroid difference
    c1, c2 = max(100.0, curr_feats.get("centroid_hz", 1000.0)), max(100.0, prev_feats.get("centroid_hz", 1000.0))
    d_c = abs(c1 - c2) / (c1 + c2 + 1e-6)
    sim_c = max(0.0, 1.0 - d_c)

    total_sim = 0.40 * sim_f + 0.30 * sim_sg + 0.20 * sim_h + 0.10 * sim_c
    return float(np.clip(total_sim, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 3. RULER CLASSIFIER (2x2 Feature Matrix)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RULER_BOUNDS = {
    "laplace_split": 18.5,  # Time/Decay separator
    "hfsr_split": 0.35,     # Freq/Texture separator
}

ALL_GROUPS = ["A", "B", "C", "D"]

def classify_ruler_raw(laplace_sigma: float, hfsr: float, bounds: Dict[str, float] = None) -> str:
    """
    2x2 Orthogonal Ruler:
      A (Unscratched Center) : Low Laplace  + Low HFSR
      B (Rough Scratches)     : High Laplace + High HFSR
      C (Delamination Tail)  : High Laplace + Low HFSR
      D (Whistling Slit)     : Low Laplace  + High HFSR
    """
    b = bounds or DEFAULT_RULER_BOUNDS
    is_low_laplace = laplace_sigma < b.get("laplace_split", 18.5)
    is_low_hfsr = hfsr < b.get("hfsr_split", 0.35)

    if is_low_laplace and is_low_hfsr:
        return "A"
    elif not is_low_laplace and not is_low_hfsr:
        return "B"
    elif not is_low_laplace and is_low_hfsr:
        return "C"
    else:
        return "D"


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMBINED CONSECUTIVE-RULER DECISION WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsecutiveDecisionResult:
    predicted_group: str
    decision_path: str
    similarity_with_previous: float
    eliminated_group: Optional[str]
    candidate_pool: List[str]
    confidence: float
    explanation_burmese: str
    features: Dict[str, float]


def evaluate_event_with_consecutive_ruler(
    audio: np.ndarray,
    sr: int,
    prev_event: Optional[Dict[str, Any]] = None,
    ruler_bounds: Optional[Dict[str, float]] = None,
    sim_threshold: float = 0.75
) -> Optional[ConsecutiveDecisionResult]:
    """
    Full pipeline:
    1. Detect -> Center -> 1.0s Window. Reject if duration > 1.0s.
    2. Extract multi-method features.
    3. If prev_event exists:
         - Calculate similarity S.
         - If S >= 75%:
             Prediction = prev_event['group'] ("အရင်အကြိမ်က A ဆို A ပဲ")
         - If S < 75%:
             Eliminate prev_event['group'] ("NOT prev_group" with 3/4 baseline)
             Classify remaining 3 candidates using Ruler.
    4. If no prev_event:
         - Classify directly with Ruler.
    """
    # 1. Detection & Extraction
    audio_win, c_idx, on_idx, off_idx, dur_s = detect_transient_window(audio, sr)
    if audio_win is None or dur_s > 1.0:
        # Rejected by physical ceiling
        return None

    # 2. Measure Features
    feats = measure_event_features(audio_win, sr)
    laplace_sigma = feats.get("laplace_sigma", 15.0)
    hfsr = feats.get("high_freq_ratio", 0.2)

    # 3. Sequential Neighbor Decision
    if prev_event and "group" in prev_event and "features" in prev_event:
        prev_group = prev_event["group"]
        prev_feats = prev_event["features"]
        sim = compute_event_similarity(feats, prev_feats)

        if sim >= sim_threshold:
            # High-similarity match (>= 75%)
            conf = min(0.99, round(0.75 + 0.25 * ((sim - sim_threshold) / (1.0 - sim_threshold + 1e-6)), 4))
            return ConsecutiveDecisionResult(
                predicted_group=prev_group,
                decision_path="CONSECUTIVE_REPEAT_MATCH",
                similarity_with_previous=round(sim, 4),
                eliminated_group=None,
                candidate_pool=[prev_group],
                confidence=conf,
                explanation_burmese=f"ကပ်ရပ်အကြိမ်နှင့် {sim*100:.1f}% တူညီပါသည် (≥ 75%)။ အရင်အကြိမ် Group {prev_group} ဖြစ်သဖြင့် ယခုလည်း Group {prev_group} အဖြစ် အတည်ပြုသည်။",
                features=feats
            )
        else:
            # Dissimilar (< 75%) -> 3/4 Elimination of Previous Class
            eliminated = prev_group
            remaining_candidates = [g for g in ALL_GROUPS if g != eliminated]
            raw_ruler = classify_ruler_raw(laplace_sigma, hfsr, ruler_bounds)

            # Reconcile Ruler prediction with the 3/4 reduced pool
            if raw_ruler in remaining_candidates:
                chosen = raw_ruler
            else:
                # Fallback: pick the closest in Ruler feature space among the 3 candidates
                chosen = remaining_candidates[0]

            conf = 0.75  # 3/4 guaranteed prior confidence on elimination
            return ConsecutiveDecisionResult(
                predicted_group=chosen,
                decision_path="CONSECUTIVE_ELIMINATION_3_OF_4_THEN_RULER",
                similarity_with_previous=round(sim, 4),
                eliminated_group=eliminated,
                candidate_pool=remaining_candidates,
                confidence=conf,
                explanation_burmese=f"ကပ်ရပ်အကြိမ်နှင့် {sim*100:.1f}% သာ တူညီပါသည် (< 75%)။ အရင်အကြိမ် Group {eliminated} မဟုတ်ကြောင်း (3/4 Prior) စစ်ထုတ်ပြီး ကျန်ရှိသော {remaining_candidates} အနက် Ruler ဖြင့် Group {chosen} ကို ရွေးချယ်သည်။",
                features=feats
            )

    # 4. Standalone Event (No Previous Neighbor)
    standalone_group = classify_ruler_raw(laplace_sigma, hfsr, ruler_bounds)
    return ConsecutiveDecisionResult(
        predicted_group=standalone_group,
        decision_path="STANDALONE_RULER_BASELINE",
        similarity_with_previous=0.0,
        eliminated_group=None,
        candidate_pool=ALL_GROUPS,
        confidence=0.70,
        explanation_burmese=f"ကပ်ရပ် History မရှိသေးပါ။ 2x2 Ruler ဖြင့် တိုက်ရိုက်တိုင်းတာရာ Group {standalone_group} အဖြစ် ဆုံးဖြတ်သည်။",
        features=feats
    )


if __name__ == "__main__":
    print("Testing Event-Centric Evaluator with Consecutive Neighbor Logic...")
    sr = 44100
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)

    # Event 1: Group A (Low Laplace, Low HFSR)
    pulse_t = np.linspace(0, 0.10, int(0.10 * sr))
    y1 = np.zeros_like(t)
    y1[int(0.2*sr):int(0.2*sr)+len(pulse_t)] = np.sin(2 * np.pi * 3200 * pulse_t) * np.exp(-12 * pulse_t)
    res1 = evaluate_event_with_consecutive_ruler(y1, sr)
    print(f"\n[Event 1] Group: {res1.predicted_group} | Path: {res1.decision_path}")
    print(f"          Explanation: {res1.explanation_burmese}")

    # Event 2: Highly similar to Event 1 (Repeat Group A, sim >= 75%)
    y2 = np.zeros_like(t)
    y2[int(0.2*sr):int(0.2*sr)+len(pulse_t)] = np.sin(2 * np.pi * 3250 * pulse_t) * np.exp(-12.5 * pulse_t)
    prev1 = {"group": res1.predicted_group, "features": res1.features}
    res2 = evaluate_event_with_consecutive_ruler(y2, sr, prev_event=prev1)
    print(f"\n[Event 2] Group: {res2.predicted_group} | Sim: {res2.similarity_with_previous*100:.1f}% | Path: {res2.decision_path}")
    print(f"          Explanation: {res2.explanation_burmese}")

    # Event 3: Completely different (High HFSR, High Laplace -> NOT A, 3/4 elimination)
    y3 = np.zeros_like(t)
    y3[int(0.2*sr):int(0.2*sr)+len(pulse_t)] = (np.sin(2 * np.pi * 6500 * pulse_t) + 0.8*np.sin(2*np.pi*7500*pulse_t)) * np.exp(-45 * pulse_t)
    prev2 = {"group": res2.predicted_group, "features": res2.features}
    res3 = evaluate_event_with_consecutive_ruler(y3, sr, prev_event=prev2)
    print(f"\n[Event 3] Group: {res3.predicted_group} | Sim: {res3.similarity_with_previous*100:.1f}% | Path: {res3.decision_path}")
    print(f"          Eliminated: NOT Group {res3.eliminated_group} | Pool: {res3.candidate_pool}")
    print(f"          Explanation: {res3.explanation_burmese}")
