# DSP Arena — Interactive Acoustic Event Chart

Real-time DSP chart with two modes: **CSV simulation** (paste a dataset) and **WAV mode** (real audio files). Sliders recompute features live using a hand-rolled radix-2 FFT.

## Quick start

```bash
# Option A — relationship system (recommended for optimization)
python3 dsp_inverse.py relationships-baseline --wavdir /path/to/wavs --labels labels.csv --out run1
python3 dsp_inverse.py optimize --wavdir /path/to/wavs --labels labels.csv \
  --mode relationships --relationships dsp_out/run1/relationships.json --out run1_opt

# Option B — single-file browser chart with WAV server
python3 dsp_chart.py serve --labels ~/Downloads/dataset/dataset_records.csv
# open http://127.0.0.1:8125 in browser
```

## Features

- **Own radix-2 FFT** — zero dependencies, in-browser DSP
- **5 features**: impact, duration, peak_time, mel_mean_db, impact_freq_hz
- **3 panels**: 1v3 splits, 2v2 splits, similarity (CV) — all Mann-Whitney U
- **Auto-search**: MAKE-SIMILAR / MAKE-DIFFERENT hill-climb sweeps (<8s)
- **Relationships tab**: load `relationships.json` to view baseline centroids, distance matrix, attract/neutral pairs, V_min guard, anchor. Approve for next run.
- **KAP-RAP**: click any dot → nearest-neighbor match with z-distance
- **URL hash**: share settings + mode via link (`&wav=1` for WAV mode)
- **CLI injection**: `dsp_chart.py add` → dots appear in <1s

## Relationship system

Baseline-derived relationships for attract-only DSP optimization:

- **Frozen JSON** at `dsp_out/<tag>/relationships.json` — `θ`, scaler (μ, σ, ddof), centroids, distance matrix, relationships, tau, collapse guard, replay payload (CSV sha256, env versions), provenance
- **Objective** `L = Σ w_gh · max(0, d_gh − τ_gh)² + λ · max(0, 1 − V/V_min)²`
- **V_min = 0.01**, λ = 1000 (normalized hinge — collapse penalty dominates L_attract by 10,000×)
- **Tie-break**: among trials with equal objective (within 1e-12), pick the closest to `θ_baseline`
- **Trial #0** = baseline params (always enqueued)
- **Replay verification** — same WAV + same θ + same pipeline → same CSV sha256 → same chart

### Workflow

```bash
# 1. Freeze baseline relationships
python3 dsp_inverse.py relationships-baseline --wavdir wavs --labels labels.csv --out run1
# → dsp_out/run1/relationships.json (frozen)

# 2. Optimize (attract-only)
python3 dsp_inverse.py optimize --wavdir wavs --labels labels.csv \
  --mode relationships --relationships dsp_out/run1/relationships.json --out run1_opt
# → dsp_out/run1_opt/{params.json, features.csv, report.json, ...}

# 3. Reproduce + verify replay
python3 dsp_inverse.py reproduce --out run1_opt --wavdir wavs --labels labels.csv
# → dsp_out/run1_opt/features.csv (should match frozen CSV sha256)

# 4. Curriculum loop — re-infer over fresh baseline, preserve approved entries
python3 dsp_inverse.py relationships-baseline --wavdir wavs --labels labels.csv \
  --out run2 --prior dsp_out/run1/relationships.json
# → dsp_out/run2/relationships.json (manual edges preserved verbatim; auto re-inferred;
#    V_min/λ/guard_formula read FROM prior.collapse_guard; replay.features_csv_sha256 deterministic)
```

### Δ-Quiz game loop

One-knob-per-round prediction game (browser, CSV mode). See `docs/superpowers/specs/2026-08-15-deltaquiz-curriculum-design.md`.

- **Pick a knob:** `sr`, `n_fft`, `hop`, `top_db`, `win`, or `stretch` (Talbot ratio stepped over `[1, 5/4, 3/2, 7/4]`).
- **Predict:** which of the 5 features moves most, and direction (↑/↓/≈).
- **Reveal:** 5 knobs reuse the existing `measureEvent()` recipe on a scratch copy (source rows untouched); `stretch` reuses the existing `talbotStretch` sim (Duration & Peak_Time ÷ r, Impact_Freq × r, Impact + Mel_Mean_dB invariant).
- **Reward:** `XP = min(25, (7 + streak) × (2 if dir-match else 1))`; streak ≥ 5 → heart.
- **Δz bars** normalized by the **frozen** `scaler.std` (not recomputed per reveal).
- Each round appends to the **SEARCH LOG** tab (`θ`, knob, Δ, prediction, actual, XP).
- Law cards (footer): `law_cards.json` rendered with a **verify badge** (`✅ ALL PASS (10 cards)`) — baked by `python law_cards.py --verify`.


### Browser view

After loading a `relationships.json` in the RELATIONSHIPS tab:
- baseline `θ`, distance matrix (RMS z-units)
- attract pairs highlighted on scatter with dashed cyan lines + baseline `d` labels
- centroid markers (dashed colored rings)
- V_min guard, λ, removed edges (if any)
- anchor config, provenance (if approved)
- "Approve for next run" button saves approved JSON with provenance

## TV Game Verdict Screen

Full-screen overlay (TV GAME button) giving one big answer per screen — design spec in `TV_GAME_VERDICT_SPEC.md`. All verdicts reuse the chart's own feature arrays (golden rule: one recipe per feature); the Python mirror lives in `dsp_inverse.py` §3 and is pinned by `test_tv_verdict.py`.

| Mode | Question | Math | Verdict |
|---|---|---|---|
| **MERGE** (တူလား?) | can these two groups merge? | min merged CV across features, compared against the all-6-pair CV strip (a globally-tight feature is an invariant, NOT a merge) | 🟢 `good` CV ≤ 0.05 **and** global minimum · 🟡 `warn` tight but not argmin · 🔴 `bad` |
| **SPLIT** (ကွဲလား?) | are these sides separable? | best feature by orientation-free Mann-Whitney U + exact p; Bonferroni α shown once ≥35 splits scanned; LOO % alongside | 🔴 `bad` p < α_bonf · 🟡 `warn` p < 0.1 · 🟢 `good` |
| **SINGLE** (Value တစ်ခုထဲ) | does ONE number decide? | feature mean gap in z-units (frozen `scaler.std`) + Mann-Whitney p | 🔴 `bad` p < 0.05 · 🟢 `good` |

- One-tap **တူအောင်လုပ် / MAKE-SIMILAR** and **မတူအောင်လုပ် / MAKE-DIFFERENT** buttons re-run the existing hill-climb, then re-verdict instantly.
- **Honesty card** on every verdict: n per group, source badge (🎙 WAV-real / 📋 CSV-simulated), session badge, test + p + θ settings, ceiling reminder (4-way ~71%, 1/2 ~86% — a green 100% is a bug).
- Pair buttons regenerate per mode: pairs AB/AC/AD/BC/BD/CD, splits AB·CD / AC·BD / AD·BC.
- Escape key or backdrop click closes the screen.

## Running the browser app

```bash
# Open the chart in a browser (WAV mode requires dsp_chart.py serve)
python3 dsp_chart.py serve --labels labels.csv
# Then open http://127.0.0.1:8125

# OR load a static index.html (CSV mode only):
python3 -m http.server 8000
# Then open http://localhost:8000
```

## Tests

```bash
.venv/bin/python -m pytest test_law_cards.py test_relationships.py \
  test_curriculum_consume.py test_quiz_determinism.py test_tv_verdict.py -v
```

70 tests cover centroids, distance matrix, inference (auto/manual), collapse gate (anchor + `protected` manual-edge survival), V_min guard (normalized hinge + dominance + monotonicity), L_attract zero/positive cases, freeze round-trip + determinism + schema, tie-break (closest to baseline + None skip), report self-consistency, frozen-immutable contract, end-to-end baseline table generation, law-card honesty (`--verify`), curriculum `--prior` merge (manual-preserve + V_min-from-prior + sha determinism), Δ-Quiz determinism (measure_all reproducible; talbotStretch ratio invariants; scratch-only reveal), and the TV-game verdict layer (Bonferroni guards, LOO separation/skip cases, z-gap scaler forms, merge/split/single classification incl. warn band).

- `test_curriculum_consume.py` — `--prior` merge semantics (manual preserved verbatim, auto re-inferred, V_min/λ/guard_formula from prior, protected-edge survival, sha256 determinism).
- `test_quiz_determinism.py` — `measure_all(θ)` reproducible; talbotStretch algebra (dur/pt ÷ r, freq × r, impact+mel invariant); stretch ladder sorted `[5/4, 3/2, 7/4]`; source rows never mutated by reveal.
- `test_tv_verdict.py` — Python mirror of the TV-game verdict screens (`dsp_inverse.py` §3): `bonferroni_alpha` division/guards, `loo_cv` perfect-separation + too-few-rows + absent-group skip, `mean_z_gap` scaler forms (flat dict / features list / None fallback / empty side), `verdict_merge` good/warn/bad (global-minimum rule), `verdict_split` Bonferroni threshold at n_scanned=35 + warn band + LOO field, `verdict_single` sign follows key order + p<0.05 rule.


### Frozen JSON schema (v2)

`dsp_out/<tag>/relationships.json` contains:

| Field | Meaning |
|---|---|
| `theta` | DSP params at baseline (`sr`, `n_fft`, `hop_length`, `top_db`, `window_sec`) |
| `scaler` | pooled z-score (`mean`, `std`, `ddof`, `features`, `n_samples`) |
| `centroids`, `distance_matrix` | per-group means + 6 RMS standardized pairwise distances |
| `relationships` | `{pair, type, d_baseline, tau_gh?, weight?, source}` — `source` is `auto` or `manual` |
| `tau` | `tau_similar=0.4`, `tau_different=1.0`, `alpha=0.5`, `tau_abs_max=0.5` |
| `collapse_guard` | `V_min`, `lambda_vmin`, removed edges, optional anchor |
| `replay` | wavdir manifest hash, CSV sha256, pipeline code hash, `env.python` / `env.numpy` |
| `provenance` | `approved_by`, `approved_at`, `parent_run_id` (curriculum loop) |

### Known properties

- **Pure-tone WAVs collapse after the trim recipe.** Envelope-dependent features (`impact`, `duration`, `mel_mean_db`) become nearly identical across groups when the input is a clean sine. Distances then sit in the no-decision / far-neutral zone, so auto-inference yields 0 attract pairs and optimization is **skipped** (`status: skipped`, exit 0). This is expected, not a bug. Real 44 kHz event recordings have richer variance.
- **Skip case:** 0 attract pairs → flat objective → optimizer does not run.
- **CORS:** serve `index.html` via `python3 -m http.server 8000` or `dsp_chart.py serve`, not `file://`.
- **Optuna scale:** `L_attract` is typically ~0.05–2; `L_guard` is 0–1000. TPE treats this as constraint-first (escape collapse, then shrink attract pairs).
