# Design Spec: Δ-Quiz Game Loop + Curriculum Loop

**Date**: 2026-08-15
**Status**: design-approved (amendments A1/A2/A3 applied)
**Domain**: DSP Arena / acoustic-event analyzer (R5 honesty layer)
**Scope**: Δ-Quiz round flow (frontend), curriculum loop (backend CLI), ruler-scales (display-only), tests + docs.

## 1. Architecture (4 subsystems)

```
User (browser)            dsp_arena/index.html
  Δ-Quiz: sr/n_fft/hop/top_db/win/stretch knob  ─┐
  Reveal: Δz bars + MCQ + XP/streak               ├─ reuses measureEvent/measureWavEvent
  Ruler: periodic|fibonacci-pisot|non-pisot       ─┘
         ┌────────────────────────┐
         │ searchlog: round record │  (θ, knob, delta, pred, actual, xp)
         └────────────────────────┘
        ↑ fetch (law_cards.json, relationships.json)
CLI: python dsp_inverse.py relationships-baseline --wavdir wavs --labels L --out <tag> --prior <approved.json> --anchor <G>
  ├─ merge: manual entries preserved, auto re-inferred against NEW baseline distances
  ├─ V_min/λ/guard_formula READ FROM --prior collapse_guard (single source of truth)  [A1]
  ├─ collapse_gate protected param: manual edges never removed (two-phase)   [A2]
  └─ freeze relationships.json with replay.features_csv_sha256 (determinism)
```

## 2. Scope B — Curriculum loop (dsp_inverse.py)

### 2.1 CLI
Add arg to `relationships-baseline` subparser (mirror `--anchor`):
```
p.add_argument("--prior", default=None,
  help="approved relationships.json to merge (manual entries preserved, auto re-inferred)")
```

### 2.2 cmd_relationships_baseline merge logic
```
rows = ds.measure_all(Params())          # fresh baseline measurement (default θ)
prior = load_frozen(args.prior) if args.prior else None
prior_rel = prior["relationships"] if prior else []
prior_guard = prior["collapse_guard"] if prior else {"V_min":0.01,"lambda_vmin":1000.0,...}
manual = [r for r in prior_rel if r.get("source")=="manual"]
auto   = [r for r in prior_rel if r.get("source")!="manual"]
protected = {(r["pair"]) for r in manual}   # [A2] manual edges never removed

# re-derive fresh centroid distances over `rows`
dm = compute_distance_matrix(centroids)     # same code path as baseline
inferred = []                               # auto edges re-evaluated on fresh dm
for r in prior_rel:
    d = dm[r["pair"]]
    if r.get("source")=="manual":
        inferred.append({**r, "d_after": d})                # immutable semantics
    else:
        tau = r.get("tau_gh") or prior["tau"] thresholds
        inferred.append(reinfer(d, tau, prior_guard, dm))   # type/distance re-derived
```

### 2.3 Guard contract [A1]
- `V_min`, `lambda_vmin`, `guard_formula` are **read from** `prior.collapse_guard` when `--prior` is given; fall back to defaults (`0.01`, `1000.0`, `v_min_regularization`) otherwise.
- Rationale: the approved JSON carries the guard contract forward (single authority). Re-deriving guard params would silently drift the curriculum.

### 2.4 collapse_gate [A2]
- `collapse_gate(relations, dm, anchor, protected=set())` — `protected` pairs (manual, source=="manual") are never removed; only auto edges below V_min are dropped.
- Manual edges remain in `relationships` and still participate in the V_min hinge (runtime guard), but the two-phase design means a manual edge won't be the sole reason for rejection on its own.
- Output provenance block:
```
"provenance": {"prior_run_id":..., "prior_sha":..., "merged_manual":N, "re_inferred_auto":M}
```

### 2.5 Determinism
- Same `--wavdir + --labels + --prior` + same pipeline → identical `replay.features_csv_sha256`.
- `replay` block written by existing `freeze_table` (csv sha, pipeline_code_hash, env).

## 3. Scope A — Δ-Quiz round flow (index.html)

### 3.1 State
```
state.quiz = {
  round: 0, knob: null, theta_before: {...}, theta_after: {...},
  prediction: {feat:null, dir:null},
  actual: {diffs:{}}, streak: 0, xp: 0
}
```

### 3.2 Round setup — one knob per round
Knob table (±1 step from current θ):
| knob | step source | shift |
|---|---|---|
| sr | slider range [44100,384000] step 100 | ±100 |
| n_fft | NFFT_POWERS (×2) | next/prev power |
| hop | [128..4096] step 16 | ±16 |
| top_db | [10..60] step 1 | ±1 |
| win | [0.1..1] step 0.05 | ±0.05 |
| stretch | Talbot ratios [1, 1.5, 1.75, 1.25] | pick adjacent ratio |

### 3.3 Prediction phase (MCQ modal)
- Question: "Which feature moves most under <knob> ±1 step?"
- 5 feature options (Impact, Duration, Peak_Time, Mel_Mean_dB, Impact_Freq_Hz) + direction radio (↑ / ↓ / ≈).
- User commits → prediction locked.

### 3.4 Reveal [A3: scratch-only, no state.rows mutation]
- **5 knobs (sr, n_fft, hop, top_db, win)**: call EXISTING `measureEvent(base, sr, nfft, hop, topdb, win)` with shifted θ on a **scratch copy** of `state.quizRows`. Compute `Δz = (meas_now - meas_base) / frozen_σ` using the **frozen pooled σ from relationships.json (scaler.std)** — NOT recomputed per reveal [A3].
- **stretch knob**: reuse existing Δ-Quiz `talbotStretch` sim transform (Duration & Peak_Time ÷ r, Impact_Freq × r, Impact/Mel invariant) — consistent with current Δ-Quiz section [stretch clarification]. `r` = adjacent Talbot ratio.
- Δz bars rendered; baseline (z=0) reference line.
- `state.baseRows`/`state.rows` **NOT modified**.

### 3.5 Reward
- XP = `7 + streak`, capped 25.
- Direction match → ×2.
- Streak +1 on correct (feat AND dir), reset on miss.
- Streak ≥5 → `addHeart()`.

### 3.6 Search log append
Each round → `state.searchHistory.push({round, knob, delta, prediction:{feat,dir}, actual:{feat,dz}, xp})`, rendered by existing `renderSearchLog` in `#tablog`.

## 4. Scope C — Ruler scale options (display-only)

Radio group in Ruler scale field:
- `periodic` (default — current behavior)
- `fibonacci-pisot` — axis ticks at log(2+τ)/log(φ) family ratios
- `non-pisot` — ticks at beta_nonpisot^n = 0.251574^n

Display-only: re-scales tick generator; **no feature recomputation** ([A3] no dot movement). Choice persisted in URL hash: `#...&ruler=fibonacci-pisot`.

Rationale [amendment #3 clarification]: ruler uses golden-ratio Pisot family `log(2+τ)/log(φ)` (musical/pitch reference), NOT `beta_nonpisot` (spectral constant). Conflating would violate DSP semantics separation.

## 5. Scope D — Tests + Docs

### 5.1 test_curriculum_consume.py (new)
1. Build an approved JSON with one manual + one auto edge, a custom `collapse_guard` (V_min=0.005, lambda=2000).
2. Run `relationships-baseline --prior approved.json --out testcur`.
3. Assert: manual edge survives verbatim (`d_after` present, type unchanged); auto edge re-inferred on fresh dm; `collapse_guard` in output == prior's (A1); `replay.features_csv_sha256` deterministic across 2 runs; V_min hinge still active for auto edges.
4. **Manual-edge protection test**: prior with a manual edge whose d_after would be <V_min → `collapse_gate` does NOT remove it (A2).

### 5.2 test_quiz_determinism.py (new)
1. Two Δ-Quiz rounds at identical `(θ, knob, prediction)` → identical reveal Δz + identical XP.
2. stretch knob uses `talbotStretch` (not `measureEvent`) per clarification.

### 5.3 Docs
- README: Δ-Quiz controls + curriculum workflow (`--prior`).
- PROJECT_MEMORY.md: session-log row.

### 5.4 Gate
- Tests ≥ 37 passing after additions (currently 37 baseline; +new tests).

## 6. Files
- Edit: `dsp_inverse.py`, `index.html`, `relationships.py` (collapse_gate signature + protected param)
- New: `test_curriculum_consume.py`, `test_quiz_determinism.py`
- `docs/superpowers/specs/2026-08-15-deltaquiz-curriculum-design.md` (this file)

## 7. Out of scope
- No new recipes / feature definitions for reveal.
- No Δ-Quiz polish beyond game loop (phase-vocoder, etc.).
- `law_cards.py`/`law_cards.json` untouched (already done).
- No curriculum *execution* approval step — only the consume test.

## 8. Open questions (resolved)
- Q: V_min from prior or defaults? → **from prior** (A1).
- Q: collapse drops manual edges? → **No** (A2, protected param).
- Q: stretch reveal uses measureEvent? → **No** (uses existing talbotStretch sim).
- Q: ruler Fibonacci-Pisot from beta_nonpisot? → **No** (golden-ratio Pisot family).

## 9. Honesty (R5)
`--verify` on law_cards.py remains `ALL CHECKS PASS`. Curriculum consume test asserts sha256 determinism (same contract as `reproduce`).
