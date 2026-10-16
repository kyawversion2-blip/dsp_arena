# DSP Arena: Microphone and Recording Variability Investigation Report

**Date:** October 16, 2026  
**Auditor:** Antigravity Systems & Production DSP Architecture Agent  
**Environment:** Apple Silicon MacBook M4, macOS Darwin 24.6.0, Python 3.11.15, NumPy, SciPy, Plotly 7.1.0  
**Artifact Directory:** `dsp_out/recording_variability_audit/`  

---

## 1. Executive Summary

An investigation was conducted into why DSP features change when the same Teetotum repeated-impact experiment is recorded using the built-in MacBook microphone across different positions/distances, and how these recordings relate to historical datasets recorded via Bluetooth/external microphones.

### Key Audit Findings:
1. **Microphone Device Verification (BlackHole vs. Built-in Mic):**
   - The host system default input device (`device=0`) is **`BlackHole 16ch`** ($384\text{ kHz}$ virtual loopback). If `MIC_DEVICE` is not explicitly set, `mic_loop.py` connects to BlackHole and captures **pure digital silence**.
   - The physical microphone is `device=1` (**`MacBook Air Microphone`**, $48\text{ kHz}$ mono). A 1.0 s live test verified active acoustic capture (Peak = $0.1305$, RMS = $0.0203$ / $-33.9\text{ dBFS}$, $0\text{ clipped samples}$).
2. **Dataset Nature: 67 WAVs Are NOT Controlled Repeated Trials:**
   - The 67 analyzed files consist of two distinct observational folders:
     - `Session_2612` (37 files): Created **December 26, 2025** (Peak amplitude $\approx 0.11$, RMS $\approx 0.012$).
     - `Session_Audio` (30 files): Created **January 4, 2026** (Peak amplitude $\approx 0.76$, RMS $\approx 0.098$, a **$+16.5\text{ dB}$ / $7.6\times$** amplitude jump).
   - They were recorded 9 days apart under uncontrolled distances/gains. Furthermore, `labels.csv` only defines ground truth for files `11.wav` to `42.wav`. Files `1.wav` to `10.wav` in both sessions are unlabeled.
3. **Bandwidth Brick-Wall at 7.77 kHz in Historical Data:**
   - Although the WAV container declares `sr = 384,000 Hz` Float32, spectral analysis proves that all acoustic energy abruptly terminates at **$7,766\text{ Hz}$** (power drops to $-171\text{ dB}$ above $9\text{ kHz}$).
   - This proves the original physical transducer or audio pipeline was an $8\text{ kHz}$ / $16\text{ kHz}$ wideband stream (such as an mSBC Bluetooth codec) that was upsampled into a 384 kHz BlackHole virtual buffer.
4. **Resampling Implementation Audit (`resample_linear`):**
   - `resample_linear()` uses 1D linear interpolation (`np.interp`) without a pre-decimation Nyquist anti-aliasing filter. While downsampling 384 kHz to 44.1 kHz did not alias in this specific historical dataset because there was no energy above 7.8 kHz, the function is fundamentally unsuitable for downsampling true wideband audio (such as native 48 kHz MacBook recordings).
5. **Group B vs. Group D Overlap & 2x2 Ruler Breakdown:**
   - Raw features for Group B and Group D overlap almost entirely (Session 2612 centroid: B = $2559.2\text{ Hz}$ vs D = $2557.9\text{ Hz}$, difference $1.3\text{ Hz}$).
   - The 2x2 Ruler thresholds (`SIGMA_SPLIT=25.0`, `HFSR_SPLIT=0.180`) predict 100% "B" in global mode ($0\%$ accuracy on D), and 88% "C" in event-centric mode. The 2x2 Ruler cannot separate B and D on these datasets.

---

## 2. Priority 1 Audit: Python Recording Entry Point & `mic_loop.py`

### 2.1 Audio Device Probing (PortAudio / sounddevice)
Inspecting `sounddevice.query_devices()` and `query_hostapis()` on Apple Silicon M4:
- **Host API [0]:** `Core Audio` (default in=0, default out=2)
- **Input Device [0]:** `BlackHole 16ch` (16 channels, default SR = $384,000.0\text{ Hz}$) $\leftarrow$ **System Default**
- **Input Device [1]:** `MacBook Air Microphone` (1 channel mono, default SR = $48,000.0\text{ Hz}$)

### 2.2 Live Acoustic Capture Verification
A 1.0 s diagnostic recording was captured on Device [1] (`MacBook Air Microphone`):
- Duration: $1.000\text{ s}$ ($48,000\text{ samples}$ at $48,000\text{ Hz}$, dtype `float32`)
- Stream Status: `OK (no errors, no underflow/overflow)`
- Peak Amplitude: $0.130466$ ($-17.7\text{ dBFS}$)
- RMS Amplitude: $0.020277$ ($-33.9\text{ dBFS}$)
- DC Offset (Mean): $+0.000016$
- Clipped Samples: $0$ ($0.000\%$)
- Spectral Peak Frequency: $114.0\text{ Hz}$ (room ambient ventilation fundamental)
- **Verdict:** Device [1] is live physical microphone input. Device [0] (BlackHole) produces zero-amplitude digital silence.

### 2.3 Deterministic Configuration
To avoid device index instability (which shifts when USB/Bluetooth devices are plugged or unplugged):
```bash
# Robust launch command for live_server:
export MIC_DEVICE=1
export MIC_SR=48000
export VECLIB_MAXIMUM_THREADS=2
export OMP_NUM_THREADS=2
uv run uvicorn live_server:app --port 8765
```

---

## 3. Priority 2 Audit: Dataset Provenance & Variability Statistics

### 3.1 Provenance of the 67 WAVs
- **Not Controlled Repeated Takes:** The 67 WAVs were collected across two distinct sessions 9 days apart:
  - `Session_2612`: 37 files (`1.wav` to `37.wav`), recorded Dec 26, 2025.
  - `Session_Audio`: 30 files (`1.wav` to `30.wav`), recorded Jan 4, 2026.
- **Label Coverage & Ambiguity:**
  - `labels.csv` only contains 32 entries: `11.wav` through `42.wav`.
  - In `Session_2612`, files `1.wav` through `10.wav` are completely unlabeled (`UNKNOWN`).
  - In `Session_Audio`, files `1.wav` through `10.wav` are completely unlabeled (`UNKNOWN`).
  - In past reports, applying `labels.csv` to both folders assumed that `11.wav` in Dec 26 and `11.wav` in Jan 4 were the same physical top face. This was an unverified assumption.

### 3.2 Audit of CV & Aggregation Formulas
1. **Sample Standard Deviation vs Bessel Correction:**
   `scripts/analyze_recording_variability.py` computed standard deviation using `np.std(vals)` ($ddof=0$, sample dividing by $N$). For small clusters ($N=3$ to $5$), this underestimates the true population dispersion by $11\%\text{--}18\%$.
2. **Unweighted Aggregation:**
   The reported summary CV% was an unweighted average of group-level CVs across both sessions. This conflated within-session mechanical spin repeatability with between-session acoustic gain shifts.
3. **Low CV Does Not Prove Physical Invariance:**
   `mel_mean_db` ($CV = 2.80\%$) has a low percentage CV because it is on a logarithmic decibel scale ($-46\text{ dB}$ vs $-28\text{ dB}$ is a $16.5\text{ dB}$ gap, which represents a $45\times$ linear energy shift!). `hf_transient_ratio` ($CV = 3.87\%$) is bounded in $[0, 1]$ and saturated near $0.90$. Calling these features "physical invariants" based solely on low CV was mathematically misleading.

---

## 4. Priority 3 Audit: Acoustic & Resampling Claims

### 4.1 Measured WAV Formats & Cutoff Frequencies
- Container format: IEEE Float32, Mono, sample rate $384,000\text{ Hz}$.
- Empirical FFT analysis of `11.wav` across both sessions:
  - Band $1,000\text{--}3,000\text{ Hz}$: $-3.2\text{ dB}$
  - Band $3,000\text{--}5,000\text{ Hz}$: $-3.8\text{ dB}$
  - Band $5,000\text{--}7,000\text{ Hz}$: $-12.2\text{ dB}$
  - Band $7,000\text{--}9,000\text{ Hz}$: $-21.9\text{ dB}$
  - Band $9,000\text{--}11,000\text{ Hz}$: **$-171.2\text{ dB}$** (Float32 noise floor)
- **Measured Maximum Active Frequency:** Exactly **$7,766.4\text{ Hz}$**.
- **Conclusion:** Both `Session_2612` and `Session_Audio` originate from an $8\text{ kHz}$ brick-wall filtered capture (such as a 16 kHz wideband Bluetooth headset) that was routed through BlackHole 16ch.

### 4.2 Linear Resampling Behavior (`resample_linear`)
- In `dsp_inverse.py`:
  ```python
  def resample_linear(x, from_sr, to_sr):
      n = max(1, int(round(len(x) * to_sr / from_sr)))
      pos = np.arange(n) * (from_sr / to_sr)
      return np.interp(pos, np.arange(len(x)), x)
  ```
- **Filter Evaluation:** Linear interpolation has a triangular kernel in time, which corresponds to a $\text{sinc}^2(f)$ filter in frequency. It has **no anti-aliasing lowpass stopband**.
- **Context-Specific Safety:** When downsampling these specific 384 kHz files to 44.1 kHz, aliasing did not corrupt audible frequencies because the signal was already zero above 7.8 kHz.
- **Architectural Risk:** For any true wideband signal recorded via the MacBook Air microphone ($48\text{ kHz}$ containing ceramic ringdown modes up to $22\text{ kHz}$), downsampling with `resample_linear` will alias energy above $\text{to\_sr}/2$. An anti-aliasing decimation filter (e.g. `scipy.signal.resample_poly` or an FIR filter) is strictly required for production resampling.

---

## 5. Priority 4 Audit: Group B vs Group D Reversal & 2x2 Ruler

To evaluate why B and D reversed or failed in classification, we executed `scripts/audit_bd_reversal.py` on all 28 labeled B and D files across both sessions without altering the 2x2 Ruler.

### 5.1 Raw Feature Distributions: Group B vs Group D
| Metric | Session 2612 (Group B, N=7) | Session 2612 (Group D, N=9) | Session Audio (Group B, N=5) | Session Audio (Group D, N=7) |
| :--- | :---: | :---: | :---: | :---: |
| **Spectral Centroid (Hz)** | $2559.2 \pm 102.8$ | $2557.9 \pm 115.0$ | $2883.7 \pm 142.0$ | $2902.7 \pm 213.3$ |
| **Peak Amplitude** | $0.1096 \pm 0.034$ | $0.0996 \pm 0.018$ | $0.5339 \pm 0.050$ | $0.5965 \pm 0.112$ |
| **RMS Energy** | $0.0120 \pm 0.001$ | $0.0122 \pm 0.002$ | $0.0730 \pm 0.011$ | $0.0745 \pm 0.010$ |
| **Impact Freq (Hz)** | $4396.9 \pm 990.4$ | $3956.7 \pm 1515.5$ | $4053.1 \pm 1900.3$ | $3555.1 \pm 1679.1$ |
| **Global HFSR** | $0.9129 \pm 0.016$ | $0.9240 \pm 0.022$ | $0.8503 \pm 0.053$ | $0.8670 \pm 0.044$ |
| **Global Sigma** | $3.71 \pm 0.23$ | $3.53 \pm 0.58$ | $3.83 \pm 2.30$ | $4.65 \pm 1.51$ |
| **Event-Centric HFSR** | $0.0596 \pm 0.037$ | $0.0443 \pm 0.034$ | $0.0784 \pm 0.074$ | $0.1096 \pm 0.091$ |
| **Event-Centric Sigma** | $14.38 \pm 4.83$ | $14.28 \pm 9.90$ | $9.49 \pm 6.77$ | $12.56 \pm 5.93$ |

**Observation:** There is no statistically significant separation between Group B and Group D in either session. In Session 2612, the mean spectral centroids differ by only **$1.3\text{ Hz}$** ($2559.2\text{ Hz}$ vs $2557.9\text{ Hz}$).

### 5.2 2x2 Ruler Confusion Matrix (Locked Production Thresholds)

**Session 2612:**
```
Global 2x2 Ruler (Sigma split 25.0, HFSR split 0.180):
Pred:   B   D   Total
True B: 7   0     7
True D: 9   0     9  <-- 100% of D misclassified as B

Event-Centric 2x2 Ruler:
Pred:   B   C   D   Total
True B: 0   7   0     7  <-- 100% of B misclassified as C
True D: 0   7   2     9  <-- 78% of D misclassified as C
```

**Session Audio:**
```
Global 2x2 Ruler:
Pred:   B   D   Total
True B: 5   0     5
True D: 7   0     7  <-- 100% of D misclassified as B

Event-Centric 2x2 Ruler:
Pred:   B   C   D   Total
True B: 1   4   0     5
True D: 2   5   0     7  <-- 0% D classified as D
```

**Verdict:** The locked 2x2 Ruler fails completely to discriminate Group B and Group D. This is not due to a label inversion; it is an inherent failure of the static thresholds `SIGMA_SPLIT=25.0` and `HFSR_SPLIT=0.180` to separate these acoustic signatures.

---

## 6. Structured Synthesis: Facts, Observations, Hypotheses, and Open Questions

### 6.1 Verified Facts
1. `PortAudio` input device 0 is `BlackHole 16ch` ($384\text{ kHz}$) and device 1 is `MacBook Air Microphone` ($48\text{ kHz}$).
2. When `MIC_DEVICE` is omitted, `mic_loop.py` defaults to BlackHole and records digital silence. Setting `MIC_DEVICE=1` captures live acoustic audio ($RMS = 0.0203$).
3. The WAV files in both `/Users/kyawnandar/Downloads/2612` (37 files) and `/Users/kyawnandar/Downloads/audio` (30 files) have an active acoustic bandwidth ceiling of exactly **$7,766\text{ Hz}$** (power drops to $-171\text{ dB}$ above $9\text{ kHz}$).
4. `resample_linear()` implements piecewise linear interpolation without an anti-aliasing lowpass filter.
5. `labels.csv` contains labels for files 11 through 42. Files 1 through 10 in both sessions are completely unlabeled.
6. The locked 2x2 Ruler (`SIGMA_SPLIT=25.0`, `HFSR_SPLIT=0.180`) classifies 100% of Group B and Group D as "B" in global mode, and predominantly as "C" in event-centric mode.

### 6.2 Empirical Observations
1. Session 2612 (Dec 26) and Session Audio (Jan 4) differ by $+16.5\text{ dB}$ in amplitude ($0.11$ vs $0.76$ peak).
2. Spectral centroid relationship inverted between sessions: $A < C$ in 2612 ($2483\text{ Hz}$ vs $2724\text{ Hz}$), but $A > C$ in Audio ($3020\text{ Hz}$ vs $2779\text{ Hz}$).
3. Group B and Group D have virtually identical spectral centroids in Session 2612 ($2559.2\text{ Hz}$ vs $2557.9\text{ Hz}$) and Session Audio ($2883.7\text{ Hz}$ vs $2902.7\text{ Hz}$).
4. The kinematic chatter interval (`mean_impact_interval_ms`) is between $5.1\text{ ms}$ and $5.5\text{ ms}$ across all groups in both sessions.

### 6.3 Plausible but Unverified Hypotheses
1. **Chassis Shadowing:** Placing the Teetotum to the Right of the MacBook is hypothesized to attenuate frequencies $>4\text{ kHz}$ by $6\text{--}15\text{ dB}$ due to unibody/lid diffraction. While physically grounded in acoustic wave theory ($\lambda < 8.5\text{ cm}$), the exact attenuation magnitude has not yet been experimentally measured with calibrated sine sweeps or white noise on this specific chassis.
2. **Distance & Room Reflections:** The $+16.5\text{ dB}$ amplitude disparity is hypothesized to result from placing the microphone closer to the plate or from differing macOS input volume sliders / AGC settings.
3. **Manual Spin Dynamics:** Variation in initial spin velocity and tilt angle between takes is hypothesized to affect the total count of secondary impacts and the duration before settling.

### 6.4 Outstanding Questions
1. What physical microphone hardware was used to create `Session_2612` and `Session_Audio` before routing into BlackHole? (The 7.77 kHz brick-wall indicates an 8 kHz wideband speech transducer).
2. What are the true ground-truth identities of files `1.wav` to `10.wav` in both folders?
3. Should the 2x2 Ruler thresholds be retrained on controlled, wideband MacBook Air captures, or should multi-impact sequence metrics (`mean_impact_interval_ms`) replace static spectral ratios?
