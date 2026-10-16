# DSP Arena: Controlled Recording Experiment Protocol

**Version:** 1.0  
**Target Environment:** Apple Silicon MacBook (M4/M-series) & macOS Core Audio  
**Acoustic Target:** Large-size Teetotum repeated same-side impact chatter on ceramic/marble plate  

---

## 1. Executive Summary & Objective

Prior analyses demonstrate significant acoustic feature variance across historical sessions (`Session_2612` vs `Session_Audio`):
- `peak_amplitude` and `rms_energy` differed by up to **+16.5 dB** (7.6× amplitude).
- `spectral_centroid_hz` inverted between groups ($A < C$ in 2612; $A > C$ in Audio).
- `impact_freq_hz` showed a coefficient of variation of **32% to 55%** due to modal peak hopping.

This protocol standardizes the physical, hardware, and digital capture conditions to isolate physical Teetotum differences (Group A vs Group C) from acoustic capture artifacts:
1. Microphone directional chassis shadowing (Left vs Right).
2. Inverse-square law distance scaling and Direct-to-Reverberant Ratio (DRR).
3. Transducer and codec frequency response (MacBook Built-in vs Bluetooth mic).
4. Automated Gain Control (AGC) and ambient noise variations.

---

## 2. Hardware and System Setup

### 2.1 Host Configuration (macOS Darwin)
- **Tool:** macOS `Audio MIDI Setup` (`/System/Applications/Utilities/Audio MIDI Setup.app`).
- **Input Device:** `MacBook Air Microphone` (or `MacBook Pro Microphone`).
- **Sample Rate:** Set strictly to **48,000.0 Hz** (native hardware rate).
- **Format:** `1 ch 24-bit Integer` or `2 ch 24-bit Integer` (do not select BlackHole or virtual loopbacks).
- **Gain / Input Volume:**
  - Lock input slider to **50% (0 dB nominal gain)** in System Settings > Sound > Input.
  - Disable any dynamic sound processing / Voice Isolation: Set Mic Mode to **Standard** (disable "Voice Isolation" or "Wide Spectrum" which engage real-time non-linear DNN filtering and spectral subtraction).

### 2.2 Thermal & Processing Limits
- Set execution environment flags to prevent CPU throttling on Apple Silicon:
  ```bash
  export VECLIB_MAXIMUM_THREADS=2
  export OMP_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  ```

---

## 3. Physical Environment & Apparatus Setup

### 3.1 Test Surface & Isolation
- **Surface:** Smooth, hard ceramic plate (minimum diameter 20 cm, thickness >= 5 mm) or marble slab placed on a sturdy, non-resonant table.
- **Damping Isolation:** Place a thin felt mat (1 mm) under the ceramic plate to prevent table resonance from dominating plate modes.
- **Ambient Noise Floor:** Ambient acoustic noise must be $< 35\text{ dBA}$. Ensure windows are closed, HVAC/fans are off, and no human speech or typing occurs during recording.

### 3.2 Teetotum Handling
- Handle the Teetotum with dry fingers or cotton gloves.
- Use a repeatable manual spin-and-release technique:
  - Standard starting spin velocity: moderate spin with an initial vertical axis of $\approx 90^\circ$.
  - Allow the spinning top to naturally wobble, precess, lose rotational stability, fall onto its facet, and enter the repeated-impact chatter sequence until full rest.
  - Recording window must capture at least 0.5 s before impact through 1.0 s after complete cessation of sound.

---

## 4. Controlled Experiment Phases

### Experiment A: Fixed-Position Repeatability (Baseline)
**Goal:** Measure intrinsic test-retest repeatability ($CV\%$) of identical spins under zero geometric variation.
- **Microphone Distance:** Exactly **15.0 cm** from plate center.
- **Horizontal Position:** Direct line-of-sight to MacBook Left Microphone Array (angle $0^\circ$).
- **Trials:**
  - 10 valid trials for **Group A**.
  - 10 valid trials for **Group C**.
- **Metrics Collected:** Peak amplitude, RMS energy, spectral centroid, Laplace sigma ($\sigma$), HFSR, inter-impact interval ($\Delta t$).
- **Success Gate:** $CV\% \le 5\%$ for spectral centroid and HFSR; $CV\% \le 10\%$ for mean impact interval.

---

### Experiment B: Left vs. Right Chassis Shadowing (Azimuth Variation)
**Goal:** Quantify high-frequency diffraction and body-shadowing caused by the MacBook chassis.
- **Distance:** Fixed at **15.0 cm** from the laptop chassis edge.
- **Position 1 (Left):** 15 cm directly to the left of the MacBook keyboard (line-of-sight to mic array).
- **Position 2 (Right):** 15 cm directly to the right of the MacBook keyboard (sound path obstructed by open display lid and aluminum chassis).
- **Trials:**
  - 5 trials Left (Group A), 5 trials Right (Group A).
  - 5 trials Left (Group C), 5 trials Right (Group C).
- **Physical Analysis:**
  - Compute high-frequency energy ratio ($\int_{4\text{ kHz}}^{20\text{ kHz}} |X(f)|^2 df / \int_0^{4\text{ kHz}} |X(f)|^2 df$).
  - Expected phenomenon: High frequencies ($\lambda < 8\text{ cm}$, $f > 4\text{ kHz}$) undergo severe shadow attenuation ($>6\text{ dB}$) when placed on the Right, shifting spectral centroid downwards.

---

### Experiment C: Distance Scaling & Direct-to-Reverberant Ratio (DRR)
**Goal:** Determine how room multipath and geometric spreading distort exponential decay ($\sigma$) and amplitude.
- **Position:** Line-of-sight (Left array, $0^\circ$).
- **Distances:**
  1. **Near-field (10 cm):** Direct path sound dominates ($\text{DRR} > +15\text{ dB}$).
  2. **Mid-field (30 cm):** Direct path and initial tabletop reflection balanced ($\text{DRR} \approx +6\text{ dB}$).
  3. **Far-field (60 cm):** Diffuse room reverberation significantly contributes ($\text{DRR} < +3\text{ dB}$).
- **Trials:** 5 trials per distance for Group A and Group C.
- **Analysis:**
  - Fit RMS energy vs $1/r$ (verify $6\text{ dB}$ attenuation per distance doubling).
  - Track changes in fitted decay rate $\sigma$ and spurious secondary reflections.

---

### Experiment D: MacBook Built-in Mic vs. Bluetooth Transducer
**Goal:** Verify the exact cause of feature discrepancies between modern laptops and historical Bluetooth datasets.
- **Setup:**
  - Dual capture or interleaved capture:
    1. MacBook Built-in Microphone (48 kHz uncompressed 24-bit).
    2. Bluetooth Microphone / Headset (operating in HFP/HSP mSBC 16 kHz or CVSD 8 kHz).
- **Comparative Analysis:**
  - Identify brick-wall lowpass cutoff frequencies ($f_c \approx 3.8\text{ kHz}$ for CVSD, $7.5\text{ kHz}$ for mSBC).
  - Evaluate how Bluetooth filtering strips Teetotum plate ringdown modes ($8\text{–}18\text{ kHz}$), rendering HFSR and spectral centroid incomparable to built-in mic captures.

---

## 5. Metadata Logging Format

Every trial file must be accompanied by a JSON sidecar or recorded directly with `recording_metadata.py`:

```json
{
  "source_type": "builtin_mic",
  "requested_sr_hz": 48000,
  "actual_sr_hz": 48000,
  "channel_count": 1,
  "recording_position": "left",
  "microphone_distance_cm": 15.0,
  "recording_device_name": "MacBook Air Microphone",
  "recording_constraints": {
    "voice_isolation": false,
    "agc_enabled": false,
    "input_gain_pct": 50
  },
  "source_file_name": "trial_groupA_left_15cm_01.wav",
  "recording_session_id": "controlled_exp_20261016_01",
  "notes": "Ceramic plate on felt base. Standard manual spin."
}
```
