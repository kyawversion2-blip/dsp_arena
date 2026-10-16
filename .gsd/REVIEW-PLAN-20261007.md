# DSP Arena — Phase-by-Phase Review, Documentation, and Evidence Plan

Status: ACTIVE — REVIEW KICKOFF
Date: 2026-10-07
Scope: Code Review → DSP Math Review → Documentation → Test/Evidence
Governing contract: .gsd/SPEC.md, .gsd/ROADMAP.md, PROJECT_RULES.md

## 0. Safety and scope lock

- Preserve all existing user/uncommitted worktree changes.
- No destructive Git operation.
- No service startup, LAN exposure, cloud/OCI/Kaggle access, credential use, or package installation unless a later phase explicitly authorizes it.
- Review is evidence-first: findings are based on source, tests, command output, or reproducible artifacts.
- DSP-system R3 in .gsd/R3-ARCHITECTURE.md is distinct from remediation R3 already completed.
- Do not claim PASS where data/dependencies are missing; use INCONCLUSIVE.

## Phase 1 — Code Review / Architecture Audit

Objective: establish the actual implementation state and identify correctness, security, maintainability, and provenance risks before changing DSP behavior.

Review order:
1. Git/worktree baseline and changed/untracked path inventory.
2. Public interfaces and execution flow: dsp_chart.py, live_server.py, real_dsp_backend.py, mcp_server.py.
3. DSP pipeline modules: dsp_inverse.py, ref_dsp.py, event_centric_impact_dsp_pipeline.py, impact_by_impact_dsp_pipeline.py.
4. Supporting analysis: partition_scout.py, pca_experiment.py, pca_evaluation.py, similarity.py, orchestrator/.
5. Tests and test-to-requirement coverage.
6. Documentation/spec-to-code consistency.

Required outputs:
- Severity-ranked finding register: Critical/High/Medium/Low.
- Exact file + line/range for each actionable finding.
- Invariant/requirement affected.
- Minimal fix recommendation.
- Regression test recommendation.
- PASS/FAIL/INCONCLUSIVE disposition.

Gate:
- No DSP formula change until Phase 1 findings affecting the formula path are dispositioned.

## Phase 2 — DSP Math / Formula Review

Objective: verify the mathematical contract independently of code style and determine whether implementation, browser recipe, Python mirror, and documentation agree.

Locked canonical features:
impact, duration, peak_time, mel_mean_db, impact_freq_hz.

Review topics:
- Event detection: filter, Hilbert envelope, smoothing, onset/offset, rejection/segment policy.
- RMS and impact definition; frame length/hop; amplitude scaling.
- Duration definition and threshold.
- Peak-time definition and units.
- FFT/STFT: n_fft, hop, windowing, zero-padding, frequency resolution, Nyquist/fmax behavior.
- Spectral centroid/bandwidth/rolloff/HFSR and mel calculations.
- Laplace Sigma estimation and whether it is invariant to unrelated quantities.
- Normalization: train-only statistics, ddof, zero-variance handling, leakage.
- Feature order/schema and browser/Python numerical parity.
- TR/TE semantics, session boundaries, replay/as-of causality.
- ABCD/ruler logic, thresholds, Markov state, and whether any learned formula is being mistaken for ground truth.
- PCA/statistical formulas: balanced accuracy, permutation procedure, multiplicity correction, effect sizes, uncertainty, held-out/session-aware validation.

Required outputs:
- Formula ledger with symbol, definition, units, implementation location, test coverage, and status.
- Discrepancy register for code-vs-spec-vs-doc.
- Candidate formula risks and required numerical tests.
- Explicit separation of OBSERVED / INFERRED / HYPOTHESIS.

Gate:
- Any mathematical discrepancy that can change a scientific conclusion blocks publication-level claims until resolved or explicitly marked INCONCLUSIVE.

## Phase 3 — Documentation Alignment

Objective: make documentation describe the verified implementation rather than intended behavior.

Documents to align:
- README.md
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/STATE.md
- .gsd/R3-ARCHITECTURE.md
- docs/runbook.md
- relevant evidence records
- model-selection / experiment documentation where applicable

Documentation rules:
- Record exact commands and paths.
- Preserve uncertainty and limitations.
- Never document generated output as source-of-truth without a hash/provenance record.
- Clearly distinguish remediation R3 from DSP-system R3.
- Document missing WAV/dependency blockers instead of manufacturing fixtures.
- Keep provider/model references optional and model-independent.

Gate:
- Every documented completion claim maps to a captured evidence artifact.

## Phase 4 — Test and Evidence

Objective: turn review conclusions into reproducible proof.

Test layers:
1. Static/compile checks for changed modules.
2. Focused unit/security/orchestrator tests.
3. DSP numerical regression tests.
4. Browser/Python parity tests where available.
5. Full pytest suite.
6. Data-dependent experiments only when verified WAV/session inputs exist.
7. Reproducibility/replay/hash checks.
8. Final git diff/status and secret/generated-artifact inspection.

Evidence record schema:
- timestamp
- command
- working directory
- source revision
- input/artifact SHA-256
- environment/dependency versions
- seed
- expected result
- observed result
- exit code
- PASS / FAIL / INCONCLUSIVE
- limitations
- linked artifact paths

Known baseline limitation:
- Prior baseline found an empty wavs/ dataset and missing optional optuna/matplotlib in the selected .venv. Current review must re-measure rather than assume this remains unchanged.

Gate:
- No phase is complete without empirical proof.

## Phase 5 — Independent Final Review

Objective: challenge the completed review/fixes from an isolated perspective.

Checks:
- Reproduce key claims from frozen paths/hashes.
- Attempt negative controls for label/session/parameter mutation.
- Verify no test-set selection leakage.
- Verify no unreviewed user worktree change was overwritten.
- Confirm residual risks and explicit follow-up work.

Final disposition:
PASS, FAIL, or INCONCLUSIVE per gate; commit recommendation only after explicit user authorization.

## Execution order

Phase 1 Code Review
→ Phase 2 DSP Math Review
→ targeted fixes only when justified
→ Phase 4 Test/Evidence for each fix
→ Phase 3 Documentation alignment
→ Phase 5 Independent Final Review

Note: Documentation may be drafted early only as a plan/evidence scaffold; authoritative completion text is written after code/math verification.

## Immediate next action

Start Phase 1 with a read-only, search-first audit of the current worktree and the highest-risk/core DSP modules. Produce a compact finding register before modifying application behavior.
