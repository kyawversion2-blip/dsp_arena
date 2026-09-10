# DSP Arena — Project Memory
> Read FIRST in every new session. Living document — append, never rewrite history.
> Last updated: 2026-08-21 · TV verdict screens implemented · 69 tests green

## 0. Resume protocol
1) Read this file → 2) TV_GAME_VERDICT_SPEC.md → 3) README.md → then code.
Locked values change ONLY via explicit user approval + a §8 log entry.

## 1. Identity & golden rule
- index.html (in-browser radix-2 FFT, 5 features) + dsp_inverse.py (Optuna)
  + dsp_chart.py (WAV server) + relationships.py/test_relationships.py.
- GOLDEN RULE: one recipe per feature (R1 above). Two recipes = two truths = wrong.

## 2. Locked decisions (do not re-decide)
- Chart features: impact, duration, peak_time, mel_mean_db, impact_freq_hz.
  44k CSV extras: Sigma, m3100, m4080, m5600.
- tau_similar=0.4 · tau_different=1.0 · alpha=0.5 · tau_abs_max=0.5
- V_min=0.01 · λ=1000 (normalized hinge) · tie-break 1e-12 → closest to baseline
- Trial #0 = baseline · 0 attract pairs → {status: skipped}, exit 0
- source: auto|manual (auto attract must satisfy d<tau_similar, asserted)
- Anchor opt-in · collapse gate suggestion-time only · lex tie-break
- Quiz pass ≥0.7 · ruler fixed-scale (log2-freq physical ticks + σ secondary,
  no autoscale, stretch dual scale) · one-knob-per-round quiz
- TV verdict: 3 screens + honesty layer; user verifies V1–V5 in browser FIRST

## 3. Verification anchors (build is WRONG until these reproduce)
| # | Check | Expected | Settings | Verified |
|---|---|---|---|---|
| V1 | Duration AB\|CD | p<0.05 | 384k resample, top_db≥35, win≥0.5 | p=0.0083 @384k; dies 44k/22k |
| V2 | C\|rest freq | ~85% LOO | 44.1k, fmax 8000, top_db 30, win 0.3 | 2026-08-06 |
| V3 | BD tightest Impact CV | ≈0.019 | 384k, any top_db | 2026-08-09 |
| V4 | 1/2 ceiling | ≤86% | any | exhaustive 2026-08-06 |
| V5 | Mel D\|rest | p<0.05 | same-session | p=0.0073 |
Ceilings: 4-way ~71%, 1/2 ~86%. Green 100% = bug.

## 4. Pitfalls (verified, do not relearn)
- 15-min-apart sessions don't transfer (2026-08-09).
- Pure tones collapse after trim recipe → skip expected, not a bug.
- Small n → auto warning; globally-tight feature = invariant, NOT merge.
- Mann-Whitney orientation-free; ≥7 splits×5 feats → show Bonferroni (p<0.0014).

## 5. Data inventory
- features.csv: 32 real events (A=6 B=8 C=6 D=12).
- tr_cluster1_44k.csv sample: D freq ≈1176–1179 separable; A 3087–7270; C 5604; B mixed.
- run1_opt report: θ=(44100, 512, 4096, 48, 0.9s); obj −0.00145;
  best split AB|CD p=0.1149 @ peak_time; silhouette −0.1440; Fisher duration 1.4586.

## 6. Canonical commands
relationships-baseline → optimize --mode relationships → reproduce (sha256)
→ browser load → approve → next run. Tests: `uv run python -m pytest test_relationships.py -v`

## 7. Open / future (designs approved, not implemented)
~~TV verdict screens (§3–7)~~ implemented 2026-08-21 (index.html overlay + test_tv_verdict.py) · Δ-QUIZ mode · ruler-scale chart · Doppler diagnostics
(optional) · book-derived LAW cards (DFFT⁴=I, Parseval, Talbot 3/2·7/4·5/4, Zorich λ).

## 8. Session log (append newest at bottom)
| date | run | θ | attract pairs | L_attract/L_guard | d_before→d_after | replay | V1–V5 | decisions |
|---|---|---|---|---|---|---|---|---|
| 2026-08-15 | run1_opt | (44100,512,4096,48,0.9) | (from json) | −0.00145 / — | see report | ✅ | pending | implementation complete; 26 tests green |
| 2026-08-20 | run1 | (44100,4096,1024,35,0.5) | AB(0.3808), BC(0.2238) | baseline only | — | ✅ features_csv_sha256=bb83e4a6… reproducible | — | frozen relationships.json; V_min=0.01 λ=1000; 0 removed edges; NOT skipped (2 attract pairs) |
| 2026-08-20 | run1_opt | (44100,512,2048,35,0.85) | AB(0.3808→0.3544), BC(0.2238→0.2160) | 0.03774 / 0.0 | L_attract 41% shrink | ✅ reproduce self-deterministic (5d9eff4c… stable) | V1 D\|AB p=0.0031 (peak_time); V2 B\|D p=0.0038; V3 D\|ABC p=0.0048; V4 D\|BC p=0.0069; V5 AB\|CD p=0.0215 | 60 TPE trials; best_split=D\|AB; L_guard=0 (no collapse); silhouette −0.123; peak_time is the separating feature. NOTE: earlier 'run1 reproduce' mismatch was a stale optimize params.json in run1 (θ=384000/…) — removed; canonical θ from relationships.json verified |
| 2026-08-20 | run2 (curriculum) | (44100,4096,1024,35,0.5) | same as run1 (AB, BC) | baseline | — | ✅ --prior run1 → features_csv_sha256=bb83e4a6… reproducible; V_min/λ/guard_formula READ FROM prior (A1); manual edges protected (A2) | V1 D\|AB p=0.0031; V2 B\|D p=0.0038; V3 D\|ABC p=0.0048; V4 D\|BC p=0.0069; V5 AB\|CD p=0.0215 | Δ-Quiz game loop + curriculum loop implemented; design doc at docs/superpowers/specs/2026-08-15-deltaquiz-curriculum-design.md; plan at docs/superpowers/plans/2026-08-15-deltaquiz-curriculum-plan.md; 46 tests green (37+5+4) |
| 2026-08-21 | tvgame-ui (no optimize run) | — | — | — | — | — | pending user browser check | TV-game verdict screen implemented in index.html: 3 modes (merge/split/single) + honesty layer + dynamic per-mode pair buttons + make-similar/different wiring; fixed missing closing script tag (JS syntax validated via node, 2416 lines parsed); Python mirror §3 pinned by test_tv_verdict.py — 24 tests green (69 total); V1–V5 await user browser verification per spec §5 |
