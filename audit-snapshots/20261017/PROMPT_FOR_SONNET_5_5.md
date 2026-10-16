# DSP Arena — Stage 1 Read-only Audit and Offline Verification Prompt

You are acting as a senior DSP research engineer and conservative code-change auditor. This is a curated snapshot; treat all conclusions as source-level until independently reproduced. The snapshot may not contain all runtime dependencies or data.

## Goal
Start converting the independent reviews (Sonnet 5.5, LongCat 2.5, DeepSeek Pro, Muse Spark) into a verified, staged implementation plan, beginning with the highest-risk feature-definition inconsistencies. Do NOT jump directly to production changes.

## Hard constraints
1. READ-ONLY for all production files. Do not edit, rename, delete, reformat, or overwrite any existing source, WAV, label, manifest, report, database, threshold, or config.
2. Do not change mic_loop.py, live_server.py, capture/trigger/refractory logic, replay-manifest verification, production classifiers, or thresholds.
3. Do not use .env, credentials, API tokens, or private config. Never print secrets.
4. Do not infer repository state from this snapshot: first report included/missing files and what cannot be verified.
5. New scripts and outputs, if needed, must be placed in a new isolated directory named offline_review_20261017/; never overwrite anything.
6. No production threshold tuning, no new feature wired into Live Test, no automatic commits/pushes.
7. Clearly label each statement: (A) mathematical/physical fact, (B) source-verified fact with file/function, (C) hypothesis, (D) recommendation, or (E) unknown.
8. Do not claim tests ran unless you actually ran them; record exact command, environment, exit code, and output summary.

## Stage 1 tasks
A. Reconcile the reported definitions of sigma/decay rate across pipeline, evaluator, and any available implementation. Document equation, unit, fit interval, envelope, mask, normalization, fallback, and function name. Explain why scipy.stats.laplace.fit(env) scale is not the same as a log-envelope decay rate.
B. Reconcile centroid, HFSR, duration, spectral window, sample-rate handling, resampling, Q/sharpness, and similarity definitions across included files. Do not invent missing code.
C. Build a source-of-truth feature dictionary proposal, but do not apply it to production.
D. Design synthetic tests first: exponential decay recovery at known sigma; gain sweep; SNR/noise-floor sweep; sample-rate comparison with proper anti-alias resampling; onset jitter; second bounce at 60/80 ms; HFSR band-edge tones. Put scripts/results only in offline_review_20261017/.
E. Before running any synthetic test, list assumptions and expected outcomes. Then execute only tests supported by the snapshot/environment; clearly separate simulated from real-audio evidence.
F. Propose charts: parameter recovery with identity line; sigma-bias heatmap vs SNR; feature sensitivity vs gain/SR/jitter; sigma-HFSR scatter with candidate thresholds labeled as unvalidated; event waveform/envelope/spectrogram with fit interval; per-session plots only if trustworthy session metadata exists.
G. Give a change proposal for the next stage, with exact file paths, changes, risk, rollback, and verification commands. WAIT for explicit approval before editing any protected or production file.

## Required output
1. Executive summary (max 10 bullets).
2. Verified findings table: finding, source/function, evidence level, impact, confidence, falsification test.
3. Unknowns and missing files/data.
4. Test matrix and reproducibility manifest (snapshot hash, Python/dependency versions, command, random seed, sample rate).
5. Chart plan and chart acceptance criteria.
6. Staged implementation plan: Stage 1 offline only; Stage 2 reviewable patch on a new branch/copy; Stage 3 independent review; Stage 4 only with explicit approval.
7. Explicitly state whether the snapshot supports any claim that A and C are separable or generalize across sessions. Default answer must be “not established” unless evidence in the snapshot directly supports it.

Begin with inventory and source reconciliation. Do not start by changing code.
