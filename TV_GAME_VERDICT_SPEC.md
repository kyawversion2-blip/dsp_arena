# DSP Arena — TV-Game Verdict Screen (Design Spec)

> Status: SPEC (not yet implemented). Written 2026-08-11 from verified session-log + skill numbers.
> User's question: "Chat ကို TV game များကဲ့သို့ တိကျ/မြန်/လွယ်ကူအောင် — app အဆင့် တက်ရင် ပိုကောင်းလား? value တစ်ခုထဲနဲ့ တူ/မတူ ဆုံးဖြတ်လို့ရလား? parameter ကစားပြီး တူအောင်/မတူအောင် လုပ်လား?"

---

## 0. The three decisions (with evidence)

| Question | Answer | Why (verified) |
|---|---|---|
| App အဆင့်တက်ရင် ပိုကောင်းလား? | **Local web app သာ — App Store/mobile/cloud မလို** | The compute engine (in-browser FFT + labels CSV) already works. UI speed does NOT fix physics: 4-way ceiling ~71%, 1/2 ceiling ~86% (2026-08-06 exhaustive search). 100% needs DATA (matched-impact n≥10/face, dual-mic), not an app. |
| Value တစ်ခုထဲနဲ့ ဆုံးဖြတ်လို့ရလား? | **ရတယ် — ဒါပေမဲ့ session-condition ပါရမယ်** | Single-feature verdicts exist: alpha5600 AC\|BD p=0.0008 @44k / 0.0036 @22k (rate-invariant!); Duration AB\|CD p=0.0083 @384k (DIES at 44k); Mel D\|rest p=0.0073. BUT 15-min-apart sessions don't transfer (pitfall 20). So: single value CAN decide, but the screen must show which session/data it came from. |
| Parameter ကစားပြီး တူအောင်/မတူအောင်? | **ရှိပြီးသား — one-tap ဖြစ်အောင်ပဲ လုပ်ရမယ်** | README: auto-search MAKE-SIMILAR / MAKE-DIFFERENT hill-climb (<8s) is implemented. TV mode just surfaces it as two big buttons + shows the verdict after landing. |

---

## 1. Golden rule (NEVER-RULE, user-corrected 2026-08-09)

**The verdict layer MUST reuse index.html's exact in-browser feature functions** (same radix-2 FFT, same 5 features: impact, duration, peak_time, mel_mean_db, impact_freq_hz, same slider params).

- NO second implementation, NO copied math, NO separate feature file.
- Two recipes = two truths. Same recipe as prediction/TR-TE ref CSV (user's CSV export rule).
- Method-specific columns must differ (FFT ≠ STFT ≠ Laplace) — irrelevant here because the chart is single-method by design; the point is one recipe per feature.

---

## 2. Data flow (already verified in dsp_chart.py, read 2026-08-11)

```
GET /manifest.json  -> [{"file":"314.wav", "group":"A"}, ...]   (groups enriched at serve time from --labels CSV)
GET /wavs/<file>    -> raw WAV bytes (16-bit PCM + float32 + stereo mixdown handled in index.html)
no /compute endpoint -> ALL DSP runs in-browser (index.html's own radix-2 FFT)
```

- WAV mode: features computed once per (params-hash) and cached in a JS Map — slider moves recompute incrementally (already the case); verdict math after features = O(n) → <50 ms per click.
- CSV mode: features come from pasted CSV; verdict math identical.

---

## 3. The TV-game surface (3 screens, one big answer each)

### Screen A — တူလား? (MERGE verdict)
- Pick any pair (A/B/C/D tiles, click two).
- Instant panel: the best feature for merging THIS pair at current settings = min CV across the 5 features, plus the all-6-pair CV strip (guards pitfall 16: a globally-tight feature is an invariant, NOT a merge).
- Verdict: 🟢 တူ (CV ≤ 0.05 AND pair is the tightest) / 🟡 သံသယ / 🔴 ကွဲ.
- One-tap **တူအောင်လုပ်** button → runs the existing MAKE-SIMILAR hill-climb → lands settings → re-verdicts instantly (TV reveal animation ~300 ms).

### Screen B — ကွဲလား? (SPLIT verdict)
- Pick any pair or 1v3 or 2v2.
- Best feature by orientation-free Mann-Whitney U (both orientations per test — pitfall 14), report exact p-value + LOO %.
- Verdict: p<0.05 → 🔴 ကွဲ; 0.05≤p<0.1 → 🟡; else 🟢 (close/indistinguishable).
- One-tap **မတူအောင်လုပ်** → MAKE-DIFFERENT hill-climb → settings + re-verdict.
- Multi-test note: if ≥7 splits × 5 features were scanned, show Bonferroni α (e.g. p<0.0014 required) — do NOT hide it.

### Screen C — Value တစ်ခုထဲ (SINGLE-VALUE verdict — the user's exact ask)
- Dropdown: feature (5) × split (1v1 / 1v3 / 2v2).
- ONE big number: the feature mean per side, the gap in z-units, exact Mann-Whitney p, verdict 🟢/🔴.
- This is the "value တစ်ခုထဲဖြင့် သိသာထင်ရှားလျှင် တူချင်းမတူခြင်း ဆုံးဖြတ်" mode — single screen, single number, single answer.

---

## 4. Honesty layer (MANDATORY on every verdict — user LAW: never fabricate)

Every verdict card shows:
1. **n per group** (e.g. A:6 B:6 C:3 D:5) — small n → auto warning "n နည်းလွန်း — TR→TE မကူးနိုင်".
2. **Source badge**: 🎙 WAV-real vs 📋 CSV-simulated (chart memory: numbers SIMULATED from CSV are not real WAV truth).
3. **Session badge**: file range shown (e.g. 249-268); session identity inferred from range is marked "ခန့်မှန်း". Cross-session verdicts get an explicit ⚠️ "15-min-apart sessions don't transfer (verified 2026-08-09)".
4. **Test + p-value + settings**: sr/fmax/top_db/win/n_fft/hop printed on the card (param-logging rule).
5. **Ceiling reminder** on any 4-way claim: honest ceilings are 4-way ~71%, 1/2 ~86% on this data — a green 100% is a bug, not a result.

---

## 5. Verification anchors (the build is WRONG until these reproduce)

Only chart-computable features listed (alpha5600 needs the 4-method backend compute — NOT in the chart's 5 features; do not fake it):

| # | Check | Expected | Settings to use | Source (verified) |
|---|---|---|---|---|
| V1 | Duration AB\|CD split | p<0.05 | 384k (resample), top_db≥35, win≥0.5 | session log 2026-08-09: p=0.0083 @384k; destroyed at 44k/22k |
| V2 | C\|rest via impact_freq_hz | ~85% LOO | 44.1k, fmax 8000, top_db 30, win 0.3 | skill benchmark 2026-08-06 |
| V3 | BD tightest pair on Impact CV | CV≈0.019 (BD < all pairs) | 384k, any top_db | session log 2026-08-09 |
| V4 | 1/2 ceiling | ≤86% (never 100%) | any feature/setting | exhaustive search 2026-08-06 |
| V5 | Mel_Mean_dB D\|rest | p<0.05 | same-session events | session log 2026-08-09: p=0.0073 |

User checks these in-browser FIRST (subagent self-report ≠ verified — user LAW). No feature is "done" until V1-V5 behave.

---

## 6. Implementation plan (when approved)

1. Backup: `cp index.html index.html.bak_tvgame` (keep phase-3 bak untouched).
2. Add `<div id="tvgame">` full-screen overlay + CSS (dark glass, same palette; compact-responsive for M4 13" 1512×982 — one column, big touch targets).
3. Add 3 mode buttons to HUD (တူ / ကွဲ / Value တစ်ခုထဲ) that swap the overlay in/out.
4. JS: verdict functions call the SAME feature arrays the scatter already uses; Mann-Whitney U + CV + orientation-free LOO helpers (~80 lines); hill-climb buttons reuse the existing auto-search functions.
5. Honesty card renderer (section 4 fields).
6. User browser-verifies V1-V5 → only then it counts (never stack features unverified).

Estimated diff: ~250 lines additive. No server change needed (dsp_chart.py untouched).

---

## 7. What an "app" WOULD change (honest) — for later

- Packaging as a desktop app (Tauri/Electron) buys: offline start, no localhost, mic recording inside the app. It does NOT buy accuracy.
- Accuracy paths that DO matter (from session log): matched-impact recordings n≥10/face (6AM protocol), dual/triple-mic 3D spatial features, 6000.csv completion (showed clean C/D duration gap 0.499 vs 0.452).
- Priority: data > app. Build the verdict screen now, keep app packaging on the shelf.

---

## 8. Relationship-driven verdict

Replaces manual MAKE-SIMILAR / MAKE-DIFFERENT as the **optimization contract**. Hill-climb buttons remain in the control panel for exploratory sweeps; the frozen relationship table is the official run-to-run curriculum.

### Flow

1. User runs `relationships-baseline` → auto-infers attract/neutral from **neutral** DSP params.
2. Browser loads `relationships.json` → distance matrix + scatter overlay (centroid rings, attract dashed lines).
3. User clicks **Approve for next run** → downloads provenance-tagged JSON.
4. Next run uses approved relationships as the new baseline (curriculum loop).

### Guard mechanisms

- **V_min guard** (in the Optuna objective): prevents all 4 centroids collapsing. `L_guard = λ · max(0, 1 − V/V_min)²` with `V_min=0.01`, `λ=1000`.
- **Collapse gate** (suggestion-time only): if the attract graph is one connected component, drop the weakest attract edge (lex tie-break). Not recomputed during optimization.
- **Anchor (opt-in):** Group D (or any group) incident attract edges demoted to neutral.
- **Tie-break:** among trials with equal `L` (within 1e-12), pick closest `||θ − θ_baseline||`. Trial #0 is always the baseline.
- **Skip:** 0 attract pairs → write `{status: skipped}` and exit 0. Do not run a flat optimizer.

### Verdict display

- Per-pair: `d_before`, `d_after`, `τ_gh`, `L_pair`, `contribution_pct`
- V_min: `V_after`, `L_guard`
- HUD still shows Mann-Whitney separation score; Relationships panel shows `L_attract`. Both stay visible.

### Known property (synthetic tones)

Pure-tone WAVs measure as nearly identical after `trim → RMS peak → window → re-trim`. Auto-inference then yields no attract pairs. Use real event recordings (or a `source: manual` approved table) for a non-skip optimize run.
