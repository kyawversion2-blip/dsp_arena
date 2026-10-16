---
name: event-centric
description: Use when detecting, extracting, slicing, or classifying acoustic impact events under the Event-Centric Architecture, enforcing 1kHz high-pass filtering, 1.0s rejection ceiling, 1.0s centered FF/RR windowing, consecutive Markov prior matching (>=75% identical repeat, <75% 3/4 elimination), and Ruler boundary classification.
---

# Event-Centric Acoustic Architecture & Consecutive Ruler Skill

## Overview

The **Event-Centric Acoustic Architecture** is designed for high-precision acoustic impact classification (e.g. 4-surface contact classification $A, B, C, D$). Unlike continuous streaming or blind fixed-hop framing, this engine operates strictly on isolated physical contact events:

1. **Acoustic Physics:** Physical contact impulses above $1000\text{ Hz}$ ($1\text{ kHz}$) ring down within $20 - 150\text{ ms}$ and **never exceed $0.5\text{ s}$**. Any signal with detected high-frequency duration $> 1.0\text{ s}$ is **REJECTED** as continuous non-impact sound (speech, hum, motor rumble).
2. **Deterministic Event Window:** Slices an exact $1.0\text{ s}$ window centered on the detected transient ($0.5\text{ s}$ Forward Focus pre-trigger + $0.5\text{ s}$ Reverse Recovery post-trigger).
3. **Consecutive Neighbor Prior ("ကပ်ရပ် ဖြစ်တဲ့အကြိမ်ကို အဓိကထားရမယ်"):** Leverages sequential physics:
   - If current event features are **$\ge 75\%$ similar** to the immediately preceding event ($S \ge 0.75$), the source surface has repeated: predict **$y_t = y_{t-1}$** (e.g. "အရင်အကြိမ်က A ဆို A ပဲ").
   - If **$< 75\%$ similar** ($S < 0.75$), the source surface has transitioned: eliminate the previous class ($y_t \neq y_{t-1}$, yielding an immediate **$3/4$ (75%) confident elimination baseline**), then apply the **Ruler** across the remaining 3 candidates.

---

## 1. Physical Event Detection & Extraction Guardrails

```
             ┌────────────── 1.0s Total Window (FF + RR) ──────────────┐
             │                                                         │
             │           ┌───── Transient Core (< 0.5s) ─────┐         │
─────────────┼───────────┼───────────────────────────────────┼─────────┼─────────────
  Baseline   │  Pre-0.5s │  Onset (1kHz HP) ──> Ring Down    │Post-0.5s│  Quiet Floor
  (Ambient)  │   (RR)    │  (Mean ~20ms, Max < 150ms)        │  (FF)   │
─────────────┼───────────┼───────────────────────────────────┼─────────┼─────────────
             │           └────────── Midpoint (Center) ──────┘         │
```

### Mandatory Rules:
1. **High-Pass Cutoff ($1000\text{ Hz}$):** 
   Always filter raw PCM using a 4th-order Butterworth high-pass filter at $1000\text{ Hz}$ (`freq_threshold_hz = 1000.0`). Mechanical contact transients resonate above 1 kHz, while table thumps and environmental rumble sit below 1 kHz.
2. **Hilbert Analytic Envelope:**
   Compute analytic envelope $E(t) = |x_{\text{HP}}(t) + j \mathcal{H}\{x_{\text{HP}}(t)\}|$ and smooth with a $1\text{ ms}$ moving average kernel.
3. **$20\%$ Peak Threshold:**
   Identify contiguous span where $E(t) \ge 0.20 \times \max(E(t))$ to determine transient onset ($t_{\text{onset}}$) and offset ($t_{\text{offset}}$).
4. **Strict $1.0\text{ s}$ Physical Rejection Ceiling:**
   $$\Delta t_{\text{detected}} = \frac{\text{offset} - \text{onset}}{\text{sr}}$$
   If $\Delta t_{\text{detected}} > 1.0\text{ s}$, **REJECT immediately (`return None`)**. Genuine acoustic impacts finish in $< 0.5\text{ s}$.
5. **Midpoint Centering & $\pm 0.5\text{ s}$ Slicing:**
   - $\text{Center} = \lfloor (\text{onset} + \text{offset}) / 2 \rfloor$
   - $\text{Start} = \max(0, \text{Center} - 0.5 \times \text{sr})$
   - $\text{End} = \min(N, \text{Center} + 0.5 \times \text{sr})$
   - Total segment duration is exactly $1.0\text{ s}$ ($384,000$ samples @ 384 kHz, $44,100$ samples @ 44.1 kHz).

---

## 2. Consecutive Neighbor Decision Logic

In interactive physical experiments, impacts often occur in repeated bursts on the same surface, or switch between distinct surfaces.

### Multi-Feature Similarity Metric $S(\mathbf{x}_t, \mathbf{x}_{t-1}) \in [0, 1]$

Compare the current event $\mathbf{x}_t$ with the immediate predecessor $\mathbf{x}_{t-1}$:

$$S = 0.40 \cdot S_{\text{freq}} + 0.30 \cdot S_{\sigma} + 0.20 \cdot S_{\text{hfsr}} + 0.10 \cdot S_{\text{centroid}}$$

Where:
- $S_{\text{freq}} = \max\left(0, 1 - \frac{|\log_{10}(f_t) - \log_{10}(f_{t-1})|}{0.30}\right)$
- $S_{\sigma} = \max\left(0, 1 - \frac{|\sigma_t - \sigma_{t-1}|}{\sigma_t + \sigma_{t-1} + \epsilon}\right)$
- $S_{\text{hfsr}} = \max\left(0, 1 - \frac{|\text{HFSR}_t - \text{HFSR}_{t-1}|}{\text{HFSR}_t + \text{HFSR}_{t-1} + \epsilon}\right)$
- $S_{\text{centroid}} = \max\left(0, 1 - \frac{|c_t - c_{t-1}|}{c_t + c_{t-1} + \epsilon}\right)$

### Decision Branches:

#### Case 1: High Similarity ($S \ge 0.75$) — Repeat Rule
* **Rule:** If $S \ge 0.75$, the physical surface has NOT changed.
* **Verdict:** 
  $$y_t = y_{t-1}$$
* **Burmese Explanation:** `"ကပ်ရပ်အကြိမ်နှင့် 75% ကျော် တူညီပါသည် (အရင်အကြိမ်က A ဆို ယခုလည်း A ပဲ)"`
* **Confidence:** $85\% - 99\%$.

#### Case 2: Dissimilarity ($S < 0.75$) — 3-of-4 Elimination Rule
* **Rule:** If $S < 0.75$, the impact has switched to a different surface:
  $$y_t \neq y_{t-1}$$
* **$3/4$ Prior Advantage:** In a 4-group system $\{A, B, C, D\}$, eliminating $y_{t-1}$ leaves exactly 3 candidates. This negative assertion carries an immediate $75\%$ ($3/4$) baseline prior accuracy!
* **Ruler Step:** Evaluate the calibrated 2x2 Ruler over the remaining candidate set $\{A, B, C, D\} \setminus \{y_{t-1}\}$.

---

## 3. The 2x2 Orthogonal Ruler Integration

The Ruler maps two independent physical axes:
1. **Decay Damping Axis ($\sigma$):** Laplace decay rate (Time domain energy dissipation).
2. **Texture / Roughness Axis ($\text{HFSR}$):** High-Frequency Spectral Ratio ($4-8\text{ kHz}$ energy vs $1-4\text{ kHz}$ energy).

### Canonical 2x2 Ruler Map:

| Group | Surface / Impact Characteristic | Laplace $\sigma$ | HFSR (Texture) |
| :---: | :--- | :---: | :---: |
| **A** | Center-Round (Unscratched Center) | **Low** ($< \sigma_{\text{split}}$) | **Low** ($< \text{HFSR}_{\text{split}}$) |
| **B** | Messy/Sharp (Rough Scratches) | **High** ($\ge \sigma_{\text{split}}$) | **High** ($\ge \text{HFSR}_{\text{split}}$) |
| **C** | Round with Tail (Delamination) | **High** ($\ge \sigma_{\text{split}}$) | **Low** ($< \text{HFSR}_{\text{split}}$) |
| **D** | Whistling Slit (Vertical Scratch) | **Low** ($< \sigma_{\text{split}}$) | **High** ($\ge \text{HFSR}_{\text{split}}$) |

*Default Calibrated Splits:* $\sigma_{\text{split}} = 18.5$, $\text{HFSR}_{\text{split}} = 0.35$.

---

## 4. Execution Workflow

When processing live audio or evaluating datasets:

```python
from .agents.skills.event_centric.scripts.event_centric_evaluator import (
    detect_transient_window,
    measure_event_features,
    evaluate_event_with_consecutive_ruler
)

# 1. Capture/load raw audio
audio, sr = load_audio()

# 2. Evaluate with previous event context
result = evaluate_event_with_consecutive_ruler(
    audio=audio,
    sr=sr,
    prev_event={"group": "A", "features": prev_feats},
    sim_threshold=0.75
)

if result is None:
    print("❌ Signal rejected: duration > 1.0s or no transient > 1kHz")
else:
    print(f"✅ Predicted Group: {result.predicted_group}")
    print(f"   Decision Path  : {result.decision_path}")
    print(f"   Similarity     : {result.similarity_with_previous*100:.1f}%")
    print(f"   Explanation    : {result.explanation_burmese}")
```

---

## 5. Verification Checklist

Before accepting any acoustic prediction:
- [ ] Was the audio high-pass filtered at $1000\text{ Hz}$?
- [ ] Is the detected transient duration strictly $\le 1.0\text{ s}$ (with physical core $< 0.5\text{ s}$)?
- [ ] Is the audio slice centered with $0.5\text{ s}$ pre and $0.5\text{ s}$ post context?
- [ ] Was consecutive similarity computed against the immediately preceding impact?
- [ ] If similarity $\ge 75\%$, was identical repetition confirmed?
- [ ] If similarity $< 75\%$, was the previous group eliminated with $3/4$ prior and the remaining candidate resolved via the Ruler?
