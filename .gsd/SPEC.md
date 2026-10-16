# DSP Arena GSD Specification

Status: FINALIZED
Version: 1.0
Date: 2026-09-27

## 1. Purpose

DSP Arena is an interactive and batch acoustic-event analysis system. It must
produce reproducible DSP features, expose uncertainty honestly, support
controlled experiments, and prevent an optimizer or AI-assisted workflow from
silently changing the data, recipe, relationship table, or statistical claims.

This specification governs the next implementation cycle. It does not authorize
application-code changes in this planning task.

## 2. Scope and non-goals

In scope:

- Stabilize the local execution environment and capture a reproducible baseline.
- Audit OCI/container and network-facing behavior in read-only mode.
- Define an orchestrator architecture that delegates work without coupling to a
  particular model provider.
- Run one controlled DSP experiment using the existing canonical feature recipe.
- Establish statistically defensible partition analysis with session-aware
  validation and multiplicity control.
- Require independent review, integration verification, and a final audit.

Out of scope until a later approved phase:

- Replacing the browser DSP recipe or adding a second recipe per feature.
- Deploying to OCI, opening public network access, or adding credentials.
- Treating AI output as measurement, ground truth, or an approval signal.
- Reclassifying existing relationships without a frozen, reviewable artifact.
- Committing any work from this planning task.

## 3. Locked project invariants

- Feature order and names are `impact`, `duration`, `peak_time`,
  `mel_mean_db`, and `impact_freq_hz`.
- One recipe per feature is the golden rule; browser and Python mirrors must
  remain mathematically equivalent.
- Relationship defaults remain `tau_similar=0.4`, `tau_different=1.0`,
  `alpha=0.5`, `tau_abs_max=0.5`, `V_min=0.01`, and `lambda=1000` unless an
  approved prior artifact explicitly supplies inherited guard values.
- Trial zero is the baseline. Zero attract pairs is a valid skipped run with
  exit code zero, not an optimizer failure.
- Auto-attract edges must satisfy `d < tau_similar`; manual edges require
  provenance and protection from automatic collapse pruning.
- The collapse guard is suggestion-time only; tie-breaking is within `1e-12`
  and chooses the closest trial to baseline.
- A quiz pass requires at least `0.7`; ruler scales remain fixed and physical.
- Statistical claims must expose sample counts, session definition, effect
  size, uncertainty, multiplicity correction, and the exact input/settings
  hashes needed to reproduce them.
- The workflow is model-independent. Provider-specific adapters are optional
  and cannot be required for correctness.

## 4. Required outcomes

### P0: Environment stability

Record interpreter/runtime versions, dependency availability, repository
revision, platform, test command, data inventory, and a clean baseline test
result. Distinguish missing optional datasets/dependencies from product
failures. Do not alter the user's uncommitted files.

### P1: OCI safety audit, read-only

Inventory OCI/container/deployment references, network bind addresses, CORS,
file serving, uploads, subprocesses, credential/token paths, MCP tools, and
environment-variable inputs. Produce findings and mitigations without starting
services, contacting cloud APIs, changing firewall rules, or writing secrets.
At minimum, assess the current `0.0.0.0` binds, wildcard CORS, upload and file
path boundaries, OAuth/token handling, subprocess invocation, and stdio MCP
surface.

### P2: Orchestrator architecture

Define a provider-neutral orchestrator with explicit task contracts, immutable
inputs, isolated work directories, bounded resources, cancellation, structured
logs, artifact manifests, approval gates, and failure states. The orchestrator
must route research/planning/execution/verification/audit work while keeping
measurement and approval deterministic. No provider may bypass verification.

### P3: Controlled DSP experiment

Run a single pre-registered experiment against a frozen input manifest. Change
one declared DSP variable or one declared controlled factor at a time; retain a
baseline arm; preserve the exact feature recipe, labels, session boundaries,
random seed, and output hashes. Report null/skipped outcomes honestly.

### P4: Statistically defensible partition analysis

Evaluate the seven defined partitions and the five canonical features with
session-aware resampling/permutation, family-wise correction, effect sizes,
confidence intervals or empirical uncertainty intervals, and held-out
validation. A signal is not deployable solely because an uncorrected p-value is
small. Small or session-confounded samples must be marked inconclusive.

### P5: Independent audit

Have an independent reviewer or isolated verification pass inspect the spec,
experiment manifest, code diff, artifacts, statistical choices, and safety
findings. The reviewer must be able to reproduce the claimed result from paths,
commands, hashes, and seeds, and must be allowed to return FAIL or INCONCLUSIVE.

### P6: Integration

Integrate only verified artifacts and code changes. Preserve frozen
relationships and manual decisions, keep browser/Python parity, and verify CLI,
HTTP, MCP, and UI contracts that are affected. Existing unrelated worktree
changes remain untouched.

### P7: Final audit

Re-run the acceptance matrix, inspect the final diff and worktree, verify no
secret or generated artifact was unintentionally added, confirm all evidence
links, and record residual risks and explicit follow-up work. Commit is a
separate user-authorized step after this audit.

## 5. Evidence contract

Every phase must store a compact evidence record containing:

- command, working directory, timestamp, exit code, and relevant output;
- source revision and input/artifact SHA-256 hashes;
- environment/dependency versions and random seeds;
- expected result, observed result, and disposition: PASS, FAIL, or
  INCONCLUSIVE;
- links to reports, plots, screenshots, HTTP responses, or test output as
  applicable.

Required proof types are empirical: test-run output for tests, command output for
build/config checks, curl/HTTP responses for endpoints, and screenshots for UI.
"Looks correct" is not evidence.

## 6. Controlled variables and uncertainty

The experiment manifest must declare fixed and varied variables. Fixed variables
include source files, labels, session assignment, feature list/order, decoder,
resampling, trim/window recipe, statistical test, correction, seed, and output
format. Varied variables must be one factor per comparison unless a factorial
design is explicitly approved.

At minimum, report uncertainty from sampling/session variation, preprocessing
sensitivity, permutation Monte Carlo error, missing/invalid data, and multiple
comparisons. Use session-level or cluster-level resampling when observations are
not independent. Do not infer transfer across the known 15-minute session gap;
pure-tone collapse remains an expected skip case.

## 7. Acceptance criteria

- `.gsd/SPEC.md` contains `Status: FINALIZED` and this requirements contract.
- `.gsd/ROADMAP.md` maps P0–P7 in dependency order and names exact proofs.
- P0 has a reproducible environment/test baseline before implementation.
- P1 produces a read-only safety report with no service startup or cloud access.
- P2 defines provider-neutral contracts and an approval-gated execution graph.
- P3 publishes a frozen manifest and baseline-vs-one-factor result.
- P4 reports corrected statistics, uncertainty, session-aware validation, and
  inconclusive states where warranted.
- P5 and P7 are independent of the authoring pass and include artifact hashes.
- No phase claims completion without captured evidence.

## 8. Exact next implementation task

Implement **P0.1: environment and baseline evidence capture**. Add only the
smallest documentation/test-support surface needed to run a read-only command
that records Python/OS/dependency versions, git revision/status (without
mutating it), available data/artifact hashes, and the focused pytest baseline.
The command must tolerate missing optional `wavs/` or optional packages by
recording `INCONCLUSIVE` rather than fabricating success. Verify with its
captured command output plus the focused test runner. Do not begin P1 until the
P0.1 evidence record is complete and reviewed.

## 9. Evidence classification

### OBSERVED

- `PROJECT_RULES.md` requires SPEC → PLAN → EXECUTE → VERIFY → COMMIT,
  finalized-spec planning lock, search-first work, empirical proof, atomic
  commits, and model independence.
- `PROJECT_MEMORY.md` records the five features, relationship constants,
  V1–V5 anchors, known session-transfer and pure-tone pitfalls, 32-event data
  inventory, and deterministic replay expectations.
- `README.md` documents CSV/WAV modes, relationship freezing/optimization,
  Δ-Quiz, TV verdicts, test coverage, and skip semantics.
- `docs/runbook.md` requires verification, state snapshots, and a three-strike
  debugging stop.
- Existing tests cover relationships, curriculum merge/provenance, DSP
  fixtures, law cards, quiz determinism, and TV verdict math.
- The worktree already contains uncommitted modifications to
  `dsp-cluster-lab.html` and `live_server.py`, plus untracked
  `ai_model_evaluated_wav_sessions.csv` and `generate_wav_session_timestamps_csv.py`.
- `live_server.py` and `real_dsp_backend.py` expose network-facing services;
  the inspected code includes `0.0.0.0` binding, wildcard CORS, uploads,
  websockets, OAuth/token reads, and external HTTP calls.
- `mcp_server.py` exposes WAV processing, batch CSV writing, classification,
  and separation tools over stdio.
- `partition_scout.py` implements seven partitions, within-session/max-stat
  permutation logic, AUC/Cohen's d, two-feature LDA, and LOSO CV gating.
- No `.gsd/SPEC.md` or `.gsd/ROADMAP.md` existed before this task.

### INFERRED

- P0 must precede OCI audit and experiment work because missing dependencies or
  ambiguous data/session provenance can invalidate all later evidence.
- A future orchestrator needs artifact and approval contracts around existing
  CLI/server/MCP surfaces rather than embedding DSP math in the router.
- The safety audit should treat LAN exposure, wildcard CORS, credential access,
  and unrestricted file/subprocess paths as high-priority review boundaries.
- The current partition tooling is a useful starting point but requires an
  evidence schema and explicit independence/uncertainty reporting before it can
  support final claims.

### HYPOTHESIS

- The current environment may not contain the full real-event `wavs/` dataset;
  P0 must measure this instead of assuming it.
- Session structure and the 15-minute transfer limitation may dominate apparent
  partition signal; session-aware validation may reduce deployable performance.
- Restricting services to localhost, narrowing CORS, and enforcing path/tool
  allowlists will be required mitigations, but exact changes await P1 evidence.
- A one-factor experiment may show that the known high-resolution duration signal
  does not transfer to the default 44.1 kHz recipe; this must be tested, not
  asserted.
