# R3 Architecture & Mathematical Contract

Status: DESIGN — IMPLEMENTATION HOLD
Date: 2026-09-29
Author: Claude Fable (Qoder) — adversarial architecture review
Handoff target: GPT-6 Sol (implementation)
Naming notice: "R3" herein is the user's DSP-system designation (Event Detection →
… → Slide Output). It is NOT remediation phase R3 ("Subprocess/optimization
controls", P1.1 plan `.gsd/evidence/P1.1-security-mitigation-plan-20260928T121914Z.md:171`).
The collision must be resolved before roadmap registration (see U1).

Inspection was read-only. No existing file was modified.

---

## PART 1 — EXISTING PROJECT FINDINGS

### 1.1 What exists today (verified by inspection)

| # | Area | Finding | Evidence |
|---|------|---------|----------|
| 1 | Event detection | Single-transient detector: Butterworth-4 HP @1000 Hz, Hilbert envelope, 1 ms boxcar, onset/offset at 20% of peak envelope, hard reject span > 1.0 s, centered FF=0.5 s / RR=0.5 s window. No refractory (single-event). Legacy multi-impact variants exist (bandpass 1k–20k, `find_peaks`, distance/prominence/merge constants). | `dsp_inverse.py:167-222`, `impact_by_impact_dsp_pipeline.py:52-158`, `event_centric_impact_dsp_pipeline.py:52-147`, `mic_loop.py:67-70,140-191` |
| 2 | DSP features | Five canonical features computed by `measure_features`: `impact` (max RMS frame 2048/512), `duration` (RMS-envelope decay to 10% of peak), `peak_time` (10→90% rise, s), `mel_mean_db` (librosa mel, n_mels=40, HTK), `impact_freq_hz` (rFFT peak + parabolic interpolation, rounded 0.1 Hz). 44k CSV extras: `Sigma, m3100, m4080, m5600`. | `dsp_inverse.py:225-376,584`; `ref_dsp.py:126-196` |
| 3 | DSP storage | `dsp_out/<run>/features.csv` (`file,group,impact,duration,peak_time,mel_mean_db,impact_freq_hz`), `params.json` (seed 42), `report.json`, `relationships.json`. In-memory event store only (UUID4, TTL 300 s, no persistence). | `dsp_inverse.py:1189-1283,1505-1547,1653-1658`; `event_store.py:84-163` |
| 4 | TR / MOOD | **TR = Training split artifacts** (frozen μ/σ stats + extracted formula; `SKILL.md:183-199`, `TV_GAME_VERDICT_SPEC.md:23,66`). **MOOD = design-time LLM conditioning only** (`/api/ai/mood`). Neither is a runtime pipeline state. Definitions confirmed — see §5. | `.agents/skills/adaptive-binary-curriculum/SKILL.md:125-210`; `live_server.py:1405-1495` |
| 5 | ABCD labels | **Hand-labelled** `labels.csv` (`File_Name,Result`), 32 events: A=6, B=8, C=6, D=12. Filename fallback `N_A.wav`. A deterministic auto-ruler exists separately: 2×2 split σ>25 / HFSR>0.18 + consecutive-Markov ≥75% repeat rule. | `dsp_inverse.py:383-413,429`; `labels.csv`; `event_centric_impact_dsp_pipeline.py:221-299`; `impact_by_impact_dsp_pipeline.py:87-93` |
| 6 | Chart rendering | Client-side Canvas 2D only. 3D scatter "particle space": X=freq **log**, Y=peak_time **linear ms**, Z=σ **linear**; constants `F_LO=500, T_WIN=3000, SIG_MAX=120`; world map `wxOf/wyOf/wzOf`, perspective `prj()`. Clamps: f∈[F_LO·1.02,190000], σ∈[1.2,SIG_MAX−0.5], tp∈[2,T_WIN−2]. `dsp_chart.py` is a WAV/manifest HTTP server, NOT a chart renderer. | `index.html:359,767-769,2169-2207,2259-2289,2566,3245-3339`; `dsp-cluster-lab.html:833,1020-1086` |
| 7 | Slide / PPT | **NONE.** No python-pptx, no deck/export endpoint anywhere. | repo-wide search, confirmed |
| 8 | Tests | 69+ tests: relationships freeze/determinism, TV-verdict stats, quiz determinism (`measure_all(θ)` exact `==`), curriculum merge provenance, DSP validator, law-cards, security suite (`tests/security/`), orchestrator contracts. Tolerances: 1e-12 tie-break, per-card 1e-6…1e-9, `pytest.approx` for stats. | `test_*.py`, `tests/security/`, `orchestrator/tests/` |
| 9 | Graphify | Knowledge graph (6254 nodes/728 communities), schema node-link JSON + freshness manifest; used as read-only code-context adapter. Not a data-visualization tool. | `graphify-out/GRAPH_REPORT.md`, `orchestrator/graphify.py` |
| 10 | Replay | `dsp_inverse.py reproduce` re-measures from `params.json` (no hash gate). `ReplayController` replays CSV rows with damped-cosine synthesis. sha256 reproducibility of `features.csv` is the de-facto replay proof. | `dsp_inverse.py:1640-1660`; `real_dsp_backend.py:432-530`; `README.md:38,53-54` |
| 11 | Model/API boundary | MCP tools: `process_wav, batch_process, classify_wav` (z-normed nearest-centroid), `get_separation` (Cohen's d), `list_wavs`. FastAPI: `POST /compute`, `/events`, `/pending`, `/engines`, mic API, WS `/ws`, `/api/ai/*`. | `mcp_server.py:150-483`; `live_server.py:613-890,1405-1547` |
| 12 | Data | `wavs/` is **EMPTY**. `recordings/latest.wav` only. `labels.csv` 32 rows. `ai_model_evaluated_wav_sessions.csv` 177 rows (`File_Name,Session,Original_Create_Time,Timestamp_Epoch,Ground_Truth_Group,File_Size_Bytes,Source_Origin`); sessions A_Training 34, A_Testing 35, B_UUID_Blind 22, C_Ultrasonic 85. | inspection of data dir |
| 13 | Invariants | Golden rule: one recipe per feature; browser/Python mathematical equivalence. Locked: feature order, τ constants, V_min=0.01, λ=1000, trial-0 baseline, quiz ≥0.7, ruler fixed physical scale. V1–V5 verification anchors; 15-min session gap; 4-way ceiling ≈71%. | `PROJECT_MEMORY.md:12-34`; `.gsd/SPEC.md:38-58` |

### 1.2 R3 integration map

| CURRENT COMPONENT | R3 CHANGE REQUIRED | FILE/MODULE | DEPENDENCIES | RISK |
|---|---|---|---|---|
| `measure_features` (feature extraction) | None (consume as source of truth) | `dsp_inverse.py:225-376` | scipy/librosa/numpy | Low — frozen recipe is an asset |
| `detect_transient_window` | None for batch; refractory param needed only if multi-event live R3 charts | `dsp_inverse.py:167-222` | scipy.signal | Low |
| `features.csv` schema | Extend, don't break: add `event_id, run_id, params_hash, tr_state, mood_state, abcd_pred, formula_version` columns (nullable, appended) | `dsp_inverse.py:1194-1206,1653-1658` | csv writers, consumers (index.html loader, `real_dsp_backend` aliases `dsp_inverse.py:432-441`) | Med — column-order consumers must tolerate append-only |
| Browser chart (live mode) | Keep as-is (live). R3 adds a separate **analysis chart** rendered from a Python-emitted ChartSpec; never feed browser-normalized values back into classification | `index.html:2169-2289,3107-3108,3245-3339` | none for R3 core | Med — two chart paths must be clearly labelled or users will conflate them |
| Auto per-file max normalization in CSV load (`sScale/eScale`) | **Forbidden in R3 analysis charts**; document as live-mode-only | `index.html:3281-3282` | — | High if reused — silently non-reproducible scales |
| ABCD labels (`labels.csv`) | Treat as hand-labels with provenance audit; freeze before formula search | `labels.csv`, `dsp_inverse.py:383-413` | user | Low |
| Session CSV (177 rows) | Adopt as the train/val/test corpus with chronological session split | `ai_model_evaluated_wav_sessions.csv` | the 177 WAVs (must be located — U5) | Med |
| Auto ruler (σ/HFSR 2×2 + Markov) | Keep as deterministic baseline classifier F0 | `event_centric_impact_dsp_pipeline.py:221-299` | none | Low |
| Replay (`reproduce`) | Add hash-gate + R3 replay ledger (as-of visibility) | `dsp_inverse.py:1640-1660` | hashlib | Low |
| MCP `classify_wav` (nearest-centroid) | Keep; register as a candidate family, not ground truth | `mcp_server.py:359-428` | numpy | Low |
| Slide output | **Net-new module** (R3 deliverable) | new: `r3/slide_render.py` | python-pptx or HTML-deck lib (U4) | Med — new dependency |
| Orchestrator contracts | Reuse for experiment tracking/approval gates | `orchestrator/schemas.py, collector.py` | existing | Low |
| Graphify | Unchanged; optional traceability aid | `orchestrator/graphify.py` | — | None |
| TR / MOOD | **Net-new causal state module** (U2 — definition required) | new: `r3/state.py` | none | High — undefined inputs poison every downstream artifact |

---

## PART 2 — R3 DATA CONTRACT

### 2.1 Canonical event record

```json
{
  "event_id": "uuid4 | <run>:<index>",
  "timestamp": {"epoch_s": 0.0, "sample_index": 0, "clock": "file|wall"},
  "event_type": "impact|transient|manual",
  "source": {"file": "11.wav", "session": "A_Training", "run_id": "run1",
             "detector": {"name": "detect_transient_window",
                          "version": "git:<sha>", "params_hash": "sha256:…"}},
  "dsp": {
    "raw": {"impact": 0.0, "duration": 0.0, "peak_time": 0.0,
            "mel_mean_db": 0.0, "impact_freq_hz": 0.0},
    "normalized": {"method": "zscore|minmax|none", "stats_hash": "sha256:…",
                   "values": {}},
    "extras": {"sigma": 0.0, "m3100": 0.0, "m4080": 0.0, "m5600": 0.0}
  },
  "chart": {"spec_id": "sha256:…", "coords_px": {"x": 0, "y": 0},
            "clipped": false},
  "train_ref": {"split": "TR:197-205", "stats_hash": "sha256:…", "formula_version": "F1/θ1190.35@…"},
  "mood": {"concept": "…|null", "source": "llm-synth|manual|null",
           "cluster_params_hash": "sha256:…", "frozen_at_session_start": true},
  "abcd": {"label_truth": "A|B|C|D|null", "label_source": "labels.csv|manual",
           "pred": null, "formula_version": null, "confidence": null},
  "trace": {"features_csv_row": 0, "ledger_seq": 0}
}
```

### 2.2 Value classes (non-negotiable separation)

1. **raw / source-of-truth** — `dsp.raw`, produced only by the frozen `measure_features` recipe. Immutable after creation. Never overwritten by any transform.
2. **transformed** — `dsp.normalized`. Always carries `method` + the hash of the normalization statistics. Recomputable, invertible only for invertible methods (z-score: yes; quantile/rank: document as non-invertible).
3. **derived** — predictions, scores, `train_ref` linkage, separations. Each carries `formula_version` / stats hash.
4. **visual-only** — `chart.coords_px`, axis tick positions, colors, marker radii. **Never readable as data by any downstream component.** The classifier input layer must structurally reject pixel values (type system or schema validation).

### 2.3 Invariants

- **I-D1** `raw ≠ normalized ≠ pixel` — three distinct fields; no aliasing.
- **I-D2** Every transformed/derived value records its transform identity and parameter hash.
- **I-D3** Transformations are reversible OR explicitly documented non-reversible with a stated reason (e.g., "mel dB floor clamp at −top_db loses sub-floor information").
- **I-D4** The chart layer is a sink: nothing in `chart.*` influences classification, formulas, or metrics.
- **I-D5** `event_id` is stable across replay; ledger append-only.

---

## PART 3 — DSP → CHART MATHEMATICAL SPECIFICATION

### 3.1 Scale definition (ChartSpec v1)

Each axis is a record:

```json
{"axis": "x", "feature": "impact_freq_hz", "scale": "log",
 "domain": [500, 190000], "range_px": [60, 1160],
 "orientation": "x-right", "clamp": "clamp-to-edge",
 "pixel_round": "half-away-from-zero"}
```

### 3.2 Mapping functions (norm64 = float64 compute)

Linear:  `t = (v − d_min) / (d_max − d_min)`
Log:     `t = (ln v − ln d_min) / (ln d_max − ln d_min)`  (requires v > 0, domain > 0)

Coordinate:
- x-axis, `x-right`:  `px = p0 + t·(p1 − p0)`
- y-axis, `y-up` data on **canvas (y-down)**: `py = p_bottom − t·(p_bottom − p_top)`

Clamping policy: `clamp-to-edge` — out-of-domain values map to the domain edge pixel and set `clipped=true` on the record. `clip-drop` is allowed only if the dropped count is recorded in the spec and asserted zero for labeled events.

Rounding: values stay float64 through all math; rounding to integer pixels happens once, at render, `round half away from zero`. Internal epsilon: `|t_roundtrip − t| ≤ 1e-12` relative.

### 3.3 Reproducibility tests

- T-S1: `∀v ∈ domain: value→norm→coord` is a pure function; same input ⇒ identical output bits.
- T-S2: `coord→value` inversion error ≤ half a pixel in data units:
  `|invert(round(coord)) − v| ≤ (d_max − d_min) / (2·|p1 − p0|)` (linear), analogously in log space for log axes.
- T-S3: monotonicity — for a monotone scale, `v1 < v2 ⇒ coord(v1) ≤ coord(v2)` always.
- T-S4: out-of-domain ⇒ `clipped=true`, coord == edge pixel.
- T-S5: domain/range changes are versioned (`spec_id` hash covers them) — a changed scale can never silently relabel old charts.

### 3.4 Conflict to resolve (flagged, not silently preserved)

`index.html:3281-3282` normalizes σ and energy by **per-file maxima** during CSV load, and `index.html:3108` uses a decaying live-energy normalization. Both are stateful/non-reproducible. These may remain in **live mode**, but R3 analysis charts must use **global, frozen, train-fitted** scales only (§9), with the scale record persisted in the ChartSpec. This is a documented behavioral divergence between live and analysis views — user must approve (U3).

---

## PART 4 — EVENT DETECTION → CHART PIPELINE

Exact sequence per event:

1. **EVENT DETECTED** — detector emits `event_id`, `timestamp`, `event_type`.
2. **CAPTURE EVENT STATE** — snapshot of as-of visibility set `V_i` (all events with index ≤ i; §10) and current TR/MOOD state.
3. **CAPTURE DSP VALUES** — run frozen recipe → `dsp.raw`.
4. **VALIDATE DSP VALUES** — schema check; finite numbers; duration ≤ 1.0 s policy; units asserted (s, Hz, dB). Failures quarantine the event, never silently coerce.
5. **TRANSFORM VALUES** — normalization with train-fitted stats only → `dsp.normalized`.
6. **MAP TO CHART COORDINATES** — ChartSpec v1 math (§3).
7. **PLOT EVENT** — render marker carrying `event_id` (hover/click reveals raw record).
8. **ATTACH METADATA** — provenance, formula version, TR/MOOD snapshot, classification.
9. **CONTINUE TIMELINE** — append ledger entry; advance state machine.

### Chart mode decision

**Primary: cumulative, event-based scatter** (every event is a permanent marker; time is a filter/animation dimension, not a window that evicts data).
**Secondary: as-of (state-based) view** — the same markers filtered by `V_i` for replay/backtesting, showing exactly what the system knew at event i.
**Explicitly rejected:** sliding-window eviction for analysis (destroys auditability) and any time-based auto-rescaling (breaks coordinate reproducibility). Rationale: auditability and reproducibility outrank visual density; the 32–177 event scale has no density problem.

---

## PART 5 — TR / MOOD INTEGRATION (DEFINITIONS CONFIRMED 2026-09-29)

### 5.1 Confirmed definitions

**TR = Training split + its frozen derived artifacts. Not a runtime state.**
Evidence: `.agents/skills/adaptive-binary-curriculum/SKILL.md:183-199` defines the
"TR Set (Training Half): 197.wav–205.wav", "TR normalization" (z-score μ/σ computed
from the TR set only), and the "Extracted TR Formula" (θ = 1190.35 Hz on
`impact_freq_hz`). TE = the held-out second half (206–214). `TV_GAME_VERDICT_SPEC.md:23,66`
uses the same TR/TE meaning ("prediction/TR-TE ref CSV"; "TR→TE မကူးနိုင်" — TR does
not transfer to TE). R3 canonicalization: TR = `train_ref` = {split id, frozen stats
hash, frozen formula version}.

**MOOD = design-time conditioning input. No runtime definition exists anywhere in the repo.**
The only implementation is `POST /api/ai/mood` (`live_server.py:1405-1495`): user
prompt → LLM (Gemini, local synthesizer fallback) → 4 target acoustic-cluster
parameter sets (A/B/C/D). It configures what to train/synthesize; it never feeds the
deterministic classifier or Gate 1/Gate 2. Confirmed by repo-wide search (only other
hits are unrelated UI/CSS tokens).

### 5.2 Data flow

```
TR events (chronological first half, e.g. 197–205)
  → measure_features → per-feature μ, σ → FROZEN + hashed   (train_ref.stats_hash)
  → formula fit/extraction on TR z-scores → FROZEN + hashed  (train_ref.formula_version)

Later event x_i (TE / live / replay) — strictly causal:
  → z-score with train_ref stats only (never corpus-wide — C5)
  → (optional, existing) Gate 1: S(x_i, x_{i−1}) = 0.40·S_freq + 0.35·S_dur + 0.25·S_hfsr
       S ≥ 0.80 → fast-path y_i = y_{i−1}
       S < 0.80 → Markov-eliminate y_{i−1}
  → Gate 2: formula(train_ref.formula_version) → super-group → reconciled class
  → honesty flag if |Score − threshold| < 0.2 × expected_margin

MOOD (design-time only, before session start):
  prompt → /api/ai/mood (or manual) → 4 cluster parameter sets
  → recorded as mood.cluster_params_hash → configures target clusters / training dashboard
  → frozen for the session; classifier never reads it at event time
```

### 5.3 Role in the formula

- **TR** enters every formula application through `train_ref` (normalization stats +
  parameters). Already mandated by C5; no new variable is added to f.
- **MOOD** default = **design-time conditioning only** (zero added leakage surface).
  Optional elevation (separate explicit decision, OFF by default): MOOD as a runtime
  class-prior at Gate 2 reconciliation — requires its own validation experiments and
  experiment records (§14) before adoption.

### 5.4 Leakage audit of the confirmed design

- TR stats are fit on the first half only → applying them to TE/live is causal by
  construction. Risk: refitting stats after seeing TE (prohibited; stats_hash gate).
- Gate 1 reads only x_{i−1} → causal. Gate 2 reads TR artifacts + current features →
  causal. MOOD is frozen at session start → causal.
- Residual risk: the SKILL.md TR formula was extracted by an LLM from a chart sweep
  (6/9 blind). R3 treats it as candidate F0-adjacent evidence only; re-derivation must
  be numerical (§7), with the 199 mis-hit exclusion and the 197–205/206–214 split
  recorded as data provenance.

---

## PART 6 — FORMAL ABCD CLASSIFICATION DEFINITION

### 6.1 What ABCD is today

Hand-labelled acoustic groups in `labels.csv` (A=6, B=8, C=6, D=12 of 32). Source of truth provenance is partial (filename fallback `N_A.wav`, `dsp_inverse.py:429`). A deterministic auto-ruler (σ>25, HFSR>0.18 → 2×2; Markov ≥75% repeat; `event_centric_impact_dsp_pipeline.py:221-299`) produces a related but **not identical** grouping — relationship unverified (U6).

### 6.2 Mathematical target (proposed, pending U6/U7 confirmation)

```
f : R^5 (normalized features) × S_state → {A, B, C, D}
```

S_state = empty by default (§5.3: TR enters via `train_ref` normalization, MOOD is
design-time only). No state variable is added to f until a validated need is shown.
Features enter **normalized** (z-score, train-fitted stats only). Chart coordinates
are **not** inputs (I-D4).

### 6.3 Label integrity requirements before formula optimization

- L-A1: provenance audit of all 32 labels (fallback-labelled files listed explicitly).
- L-A2: class imbalance recorded (6/8/6/12) → balanced metrics mandatory (§8).
- L-A3: temporal leakage audit: events within a session are correlated (15-min gap rule, `PROJECT_MEMORY.md:37`); splitting must be session-level (§9).
- L-A4: the 177-record session CSV's `Ground_Truth_Group` reconciled against `labels.csv` (U7).

**STOP condition:** if L-A1–L-A4 cannot be satisfied, formula optimization results are untrustworthy and only exploratory.

---

## PART 7 — FORMULA DISCOVERY ENGINE

### 7.1 Candidate families (justified by existing code/data)

| ID | Family | Form | Params | Fit method | Justification |
|----|--------|------|--------|-----------|---------------|
| F0 | Auto-ruler | existing σ/HFSR 2×2 + Markov | 2 + rule | none (fixed) | Already in repo; deterministic baseline |
| F1 | Single-feature threshold split | argmax over per-feature/per-split 1-D rules (TV-verdict style) | 2 per split | exhaustive scan | Mirrors existing verdict screens (`partition_scout.py`) |
| F2 | Multinomial logistic, z-scored, L2 | softmax(w·x + b) | 4(k+1) | scipy L-BFGS / sklearn | k=5 → 24 params; smallest viable multivariate model |
| F3 | Diagonal LDA | Gaussian per class, diagonal Σ | 4k means + k vars | closed form | Precedent: two-feature LDA in `partition_scout.py` |
| F4 | Quadratic/interactions (deg ≤ 2, L2) | softmax over [x, x_i·x_j] | ≤ 4·(k+C(k+1,2)) | L-BFGS | Only on 177-corpus; n=32 cannot support it (≈84 params vs 32 points — rejected by §7.3) |
| F5 | Piecewise monotone score | weighted score + thresholds, monotonicity constraints | 2k+2 | grid + isotonic | Interpretable fallback; matches "ruler" philosophy |
| F6 | Nearest-centroid (z-normed) | argmin ||x − c_j|| | 4k | closed form | Existing MCP `classify_wav` baseline (`mcp_server.py:359-428`) |

### 7.2 Per-family contract

Each candidate record must define: parameters; constraints (e.g., F5 monotone weights ≥ 0); fitting method; objective (train log-loss or 0-1 loss, never validation metrics); regularization (L2 λ, selected from a fixed grid on validation only); complexity penalty `C = #free_params + #thresholds`; validation procedure (§9 splits); acceptance criteria (§8 gates).

### 7.3 Discovery algorithm (deterministic)

1. Enumerate candidates from the registry in fixed order (F0→F6, filtered by data-size cap: `free_params ≤ max(3, n_train/10)`).
2. For each: fit on train; compute validation metrics; record experiment JSON (§14).
3. Robustness: re-fit on each session fold; jitter features by measured sensor noise σ_j (fixed, declared); report metric spread.
4. Selection: §8 objective, one-standard-error rule (prefer simplest family within 1 SE of the best).
5. Final: evaluate the single selected formula on test **once**; archive; never re-select on test.
6. A formula is rejected if it beats F0 on train but not on validation — that is memorization, not signal.

LLMs participate only in steps 0 (hypothesis proposals) and review — never in fitting, scoring, or selection.

---

## PART 8 — OBJECTIVE / SELECTION FUNCTION

Primary metric: **balanced accuracy** `BA = mean_c recall_c` (classes are imbalanced 6/8/6/12; raw accuracy is dominated by D).

```
J(f) = BA_val(f) − λ_c·C(f)/n_val − λ_s·std_folds(BA_val(f))
```

Selection: `argmax_f J`, with the one-SE simplicity rule; λ_c = 0.01, λ_s = 0.5 fixed a priori.

Mandatory reported diagnostics per candidate: full 4×4 confusion matrix; per-class precision/recall/F1; macro-F1; if probabilistic (F2/F3): log-loss + reliability summary; session-fold BA table; train-vs-val gap.

Acceptance gates for the final formula:
- G1: test BA > max(BA of F0, 4-way chance = 0.25) with session-aware evaluation.
- G2: per-class recall > 0 on validation (no silently abandoned class).
- G3: train−val BA gap ≤ 0.20 (else flagged overfit regardless of test).
- G4: all diagnostics archived; test touched exactly once.

---

## PART 9 — LEAKAGE / OVERFITTING DEFENSES

Leakage vectors identified in this repo:

| Vector | Where | Safeguard |
|---|---|---|
| Normalization using future/global stats | z-score must be fit on train sessions only; stats hash recorded | `r3/normalize.py` refuses corpus-wide fit |
| Per-file/per-window adaptive scaling | `index.html:3281-3282,3108` | forbidden in analysis charts (§3.4) |
| Labels derived from outcomes / later relabeling | `labels.csv` edits | freeze label artifact (sha256) before search |
| Session correlation | 15-min gap rule | splits are session-level, chronological |
| Replay seeing future rows | `ReplayController` CSV replay | as-of visibility set `V_i` enforced by state machine (§10) |
| Test-set selection | formula search | test evaluated once, by hash-gated script |
| Detector param drift | `params.json` per run | detector version pinned in every record |

**Split proposal (chronological, no shuffling):** train = A_Training (34), validation = A_Testing (35), test = B_UUID_Blind (22); C_Ultrasonic (85) = out-of-distribution stress set, reported separately (its ultrasonic band likely violates the 1 kHz HP detector's assumptions — that must be measured, not assumed; U5/U7). Final split is the user's call (U7).

---

## PART 10 — HISTORICAL REPLAY ENGINE

### 10.1 Determinism contract

Given identical (input snapshot sha256, code version git sha, config hash, seed), replay must produce bit-identical: events, `dsp.raw`, normalized values, chart coords, TR/MOOD states, predictions, metrics, ledger.

### 10.2 State machine

```
IDLE → LOADED(config+data hashes recorded)
     → PER_EVENT[i]: {VISIBLE=V_i, TR/MOOD update(features_{≤i}), classify_i,
                      ledger.append(hash(chain))}
     → SEALED(final metrics + manifest hashes)
```

At every event the engine materializes exactly the information available at that point: `V_i = {events_j : t_j ≤ t_i}`, train-fitted normalization stats (frozen), TR/MOOD state from event i−1. Any code path that reads index > i fails a runtime assertion.

### 10.3 Ledger

Append-only sequence: `entry_i = sha256(entry_{i−1} ‖ event_i ‖ dsp_i ‖ pred_i)`. Ledger + snapshot hashes = the replay proof; supersedes the current hash-less `reproduce` path for R3 runs (existing reproduce stays untouched for backward compatibility).

---

## PART 11 — CHART AS A VALIDATION TOOL

The R3 analysis chart is an inspection instrument:

- **Layers** (toggleable): raw values (via hover), normalized positions, event markers (colored by truth, shaped by prediction ⇒ misclassification = color/shape mismatch), TR/MOOD state band, formula decision boundaries (for F1/F5: threshold lines; for F2/F3: projected 2-D boundary contours with the projection named).
- **Trace path (required):** click ABCD result → formula version + inputs (normalized values + raw values) → event_id → source file + timestamp → chart pixel. Every marker stores `event_id`; the lookup is a pure index query.
- **Regression overlay:** replay any historical event with the current formula; divergence is highlighted.
- Validation of the chart itself: §3.3 round-trip tests + spec hash pinning (T-S5).

---

## PART 12 — SLIDE / PRESENTATION OUTPUT

### 12.1 Strict separation

```
DATA/MATH ENGINE (Python) → ChartSpec + ResultsManifest (validated, hashed)
                          → slide renderer (pure consumer; zero recomputation)
```

The renderer may read only: ChartSpec, ResultsManifest, frozen label artifact. It must not import feature extraction, formulas, or normalization. Slide build is a hash-gated script that fails if any input hash mismatches its manifest.

### 12.2 Pre-render validation checks (all must pass)

- S-1: axis domain/ticks recomputed from ChartSpec equal the source data extent (within §3 tolerance).
- S-2: marker count == event count in ResultsManifest; every marker `event_id` resolves.
- S-3: displayed labels == classification results (string equality against manifest, no re-derivation).
- S-4: `clipped == false` for all labeled events (no silent edge parking).
- S-5: scale spec version == formula/dataset version recorded in manifest (no stale data).
- S-6: slide metadata block contains dataset sha256, formula version, code git sha, timestamp.
- S-7: renderer unit test asserts it cannot run when feature-extraction modules are importable in its process (dependency firewall).

---

## PART 13 — TEST STRATEGY

### Unit
normalization (train-fitted stats, invertibility for z-score), scale mapping (§3.3 T-S1–T-S5), coordinate conversion incl. canvas y-flip, DSP transforms (recipe parity vs `ref_dsp.py`), formula evaluation (fixed vectors ⇒ fixed outputs), classification boundaries (boundary inputs map to expected sides).

### Property
identical inputs ⇒ identical outputs (bit equality); monotonic scales preserve order; out-of-domain ⇒ declared clip; z-score round-trip within 1e-12; replay ledger chain unbroken; classifier never receives pixel-typed values (schema fuzz).

### Integration
Event → DSP → normalize → chart coord → TR/MOOD update → classify → ledger entry → slide render, end-to-end on a 3-event fixture with golden hashes.

### Replay
two full replays of the same snapshot ⇒ identical ledger hash; one-row mutation ⇒ different ledger hash; as-of assertion: engine reading a future row raises.

### Regression
existing suites must stay green: `python -m pytest test_relationships.py test_tv_verdict.py test_quiz_determinism.py -v`, full `python -m pytest -v`, plus security suite (`tests/security/`). Browser/Python parity spot checks per golden rule.

### Numerical tolerance (explicit)
coordinate round-trip ≤ 0.5 px + 1e-9; normalization invertibility rel-err ≤ 1e-12; metric comparisons exact or `pytest.approx` with declared tol; sha256 equality for replay artifacts (no tolerance — exact).

---

## PART 14 — EXPERIMENT TRACKING SCHEMA

```json
{
  "experiment_id": "uuid",
  "dataset": {"version": "sha256:labels+session_csv", "splits": {"train": ["A_Training"], "val": ["A_Testing"], "test": ["B_UUID_Blind"], "ood": ["C_Ultrasonic"]}},
  "feature_set": ["impact","duration","peak_time","mel_mean_db","impact_freq_hz"],
  "normalization": {"method": "zscore", "stats_hash": "sha256:…"},
  "family": "F2", "params": {}, "hyperparameters": {"l2": 0.1},
  "metrics": {"train": {"BA": 0, "confusion": [[0]]}, "val": {}, "test": null},
  "complexity": {"free_params": 0, "thresholds": 0},
  "robustness": {"fold_BA": [], "jitter_BA_mean": 0, "jitter_BA_std": 0},
  "timestamp": "…", "code_version": "git:sha", "seed": 42,
  "verdict": "candidate|selected|rejected", "selected_on": "val"
}
```

No "best formula" is accepted without a complete, reproducible experiment record (hashes + code version). Records are append-only, content-addressed (sha256 of canonical JSON).

---

## PART 15 — ROLES OF AI MODELS

| Work | Owner | Authority? |
|---|---|---|
| Architecture, math critique, hypothesis generation, adversarial review | Claude Fable | No |
| Implementation, integration, tests, debugging, deterministic pipeline | GPT-6 Sol | No |
| Formula fitting, optimization, scoring, backtesting, statistical validation | Numerical code (scipy/sklearn/numpy, versioned) | **Yes — sole authority on correctness** |

The LLM must never declare a formula correct. Only the numerical validation system (§8 gates + §13 tests + reproducible experiment records) may. Any human/AI disagreement with the numerical system is resolved by inspecting the numerical system, not by argument.

---

## PART 16 — DELIVERABLES

### 1. R3 architecture (text diagram)

```
 WAV / mic ──► DETECTOR (frozen) ──► measure_features (frozen recipe) ──► RAW EVENT RECORD
                                                                           │
                                              ┌───────────────────────────┼───────────────────────────┐
                                              ▼                           ▼                           ▼
                                     r3/normalize.py            r3/trainref.py              r3/replay.py
                                     (train-fitted stats,      (frozen TR split stats      (as-of state machine,
                                      hashed)                  + formula; mood = design-    append-only ledger)
                                                                 time conditioning only)
                                              │                           │                           │
                                              ▼                           ▼                           ▼
                                     r3/classify.py ◄─────────────────────┘                  r3/backtest.py
                                     (formula registry F0–F6,                                   (metrics, gates G1–G4)
                                      experiment records)
                                              │
                                              ▼
                                     r3/chart_spec.py ──► ChartSpec JSON (hashed, §3 math)
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
                    ▼                         ▼                          ▼
            index.html (new            r3/slide_render.py         analyst trace UI
             analysis view,            (§12 hash-gated,            (ABCD→formula→inputs→
             consumes spec)            zero recompute)              event→timestamp→pixel)
```

### 2. Integration map — §1.2 above.
### 3. Canonical schema — §2.1; value classes §2.2.
### 4. DSP→chart math — §3.
### 5. TR/MOOD spec — §5.
### 6. ABCD formal definition — §6.
### 7. Formula families — §7.1.
### 8. Discovery algorithm — §7.3.
### 9. Objective function — §8.
### 10. Leakage defenses — §9.
### 11. Replay design — §10.
### 12. Chart validation — §11 + §3.3.
### 13. Slide validation — §12.2.
### 14. Test strategy — §13.
### 15. Experiment schema — §14.
### 16. Roles — §15.

### 17. Ordered implementation plan (GPT-6 Sol)

1. **I1** `r3/records.py` — event record + schema validation + value-class guards (I-D1…I-D5). Tests: unit + property.
2. **I2** `r3/normalize.py` — train-fitted z-score/min-max, stats hashing, refusal of corpus-wide fit. Tests: invertibility, leakage guard.
3. **I3** `r3/scale.py` + ChartSpec v1 — §3 math, spec hashing, T-S1–T-S5.
4. **I4** Label/session audit artifact — freeze `labels.csv` + session CSV hashes; produce L-A1–L-A4 report. **Gate: user review.**
5. **I5** `r3/replay.py` — as-of state machine + append-only ledger; replay determinism tests.
6. **I6** `r3/classify/registry.py` — F0, F1, F6 first (smallest); uniform candidate interface.
7. **I7** `r3/classify/fit.py` — F2, F3, F5 fitting + L2 grid; F4 only if 177-corpus approved.
8. **I8** `r3/experiments.py` — §14 schema, content-addressed store, chronological split executor.
9. **I9** Selection runner — §8 objective, gates G1–G4, one-shot test evaluation.
10. **I10** `r3/backtest.py` — replay-driven metrics over full timeline.
11. **I11** `r3/chart_spec.py` emitter + `index.html` analysis view (consumes spec; live mode untouched). Tests: S-1–S-6 analogues.
12. **I12** `r3/slide_render.py` — §12 pipeline + S-1…S-7 checks. **Final gate: user verification of a real slide.**
13. Each step: atomic commit + evidence per PROJECT_RULES; no step may touch locked invariants without explicit approval.

### 18. UNKNOWNs (must be answered before implementation of the dependent parts)

- **U1** Naming: user's R3 vs remediation R3 (subprocess controls). Register as new workstream (e.g., "R3-DSP")?
- **U2** RESOLVED 2026-09-29: TR = frozen training-split artifacts (`train_ref`, §5.1);
  MOOD = design-time conditioning only. Open sub-question **U2a**: may MOOD ever become
  a runtime Gate-2 class prior? Default NO.
- **U3** Approve the live-vs-analysis chart divergence (§3.4)?
- **U4** Slide output format/library (python-pptx? HTML deck? PDF?) and target audience/template.
- **U5** Where are the 177 session WAVs? `wavs/` is empty; without them, corpus-level work is INCONCLUSIVE.
- **U6** Is ABCD = `labels.csv` groups? How does it relate to the auto-ruler 2×2 output and to `Ground_Truth_Group` in the session CSV?
- **U7** Confirm the chronological split (§9) and whether C_Ultrasonic is OOD-stress or excluded.
- **U8** Is the 1 kHz HP / 1.0 s detector the mandated R3 detector for all sessions (incl. ultrasonic C)?
- **U9** May R3 add the 8 new nullable columns to `features.csv`, or must it write a sidecar `events.jsonl` (safer for existing consumers)?
- **U10** Confirm formula complexity cap rule (`free_params ≤ max(3, n_train/10)`) and λ_c, λ_s.

### 19. Risks ranked by concrete failure impact

1. **Corpus absence (U5).** If the 177 WAVs cannot be located, every corpus-level result is INCONCLUSIVE; the honest deliverable shrinks to the 32-event audit + engine scaffolding. Impact: formula search must not proceed past F0/F1 on 32 events.
2. **n=32 with 5 features.** Any family beyond ~3 free parameters can memorize the training set; the resulting formula would misclassify every future session while reporting glowing train accuracy. Impact: gates G1–G3 exist precisely to catch this; without them the deliverable is noise.
3. **Session correlation.** Treating 177 rows as independent inflates every metric; a formula could pass validation by learning session artifacts (e.g., C's ultrasonic band) rather than class structure. Impact: all claims void under session-aware re-test.
4. **Stateful live normalization leaking into analysis.** If `index.html`'s per-file/decaying normalization is reused, identical DSP values map to different coordinates across runs — charts become unreproducible and any visual "validation" is meaningless. Impact: §3.4 prohibition; S-5 check.
5. **Label provenance gaps.** The `N_A.wav` fallback means some "ground truth" is a filename convention, not a measurement. Impact: formula optimized against mislabeled rows encodes the naming bug.
6. **TR/MOOD undefined.** Building the formula with placeholder state bakes an arbitrary variable into the core architecture; ripping it out later invalidates every experiment record. Impact: §5.3 TR/MOOD-free mode until U2.
7. **Slide renderer recomputation.** If the renderer re-derives anything, slides silently diverge from validated data the moment code changes. Impact: S-7 dependency firewall.
8. **Detector-domain mismatch (U8).** The 1 kHz HP detector may destroy C-session (ultrasonic) information before features exist. Impact: C results meaningless unless detector policy is re-decided; must be measured, not assumed.

### 20. R3 Mathematical Contract (non-negotiable invariants)

- **C1** One recipe per feature; the frozen `measure_features` is the sole source of raw values.
- **C2** `raw ≠ normalized ≠ pixel`, structurally enforced; chart values never flow backward (I-D4).
- **C3** Every transform/derived value carries its version + parameter hash.
- **C4** Chart coordinates are pure functions of (value, scale spec); scale spec is hashed and pinned (T-S1–T-S5).
- **C5** Normalization statistics are fit on train sessions only, chronologically, and frozen.
- **C6** Splits are session-level and chronological; temporal data is never shuffled.
- **C7** TR/MOOD updates are causal (as-of `V_i`) or absent; never future-informed.
- **C8** Formula selection optimizes validation balanced accuracy with complexity penalty; test is evaluated once.
- **C9** Replay is bit-deterministic and ledger-chained; same inputs ⇒ identical hashes.
- **C10** The numerical validation system is the sole authority on correctness; no LLM or visual inspection may override it.
- **C11** Slides consume only validated, hashed artifacts; zero recomputation (S-1…S-7).
- **C12** Any conflict between existing behavior and C1–C11 is reported explicitly (like §3.4), never silently preserved and never silently changed.

---

## Next Up

**R3-DSP design spec — awaiting U1–U10 answers; I1–I3 may start immediately (no UNKNOWNs block them).**

`/clear` then hand this file to GPT-6 Sol, or answer UNKNOWNs in-session to unblock I4–I12.
