# DSP Arena GSD Roadmap

Status: PLANNED
Protocol: SPEC → PLAN → EXECUTE → VERIFY → COMMIT
Spec: `.gsd/SPEC.md` (Status: FINALIZED)

## Dependency graph

`P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7`

P1 may inspect in parallel with P0 only after the P0 environment command is
defined; no implementation phase may start until P0 is verified. P5 depends on
P3 and P4 artifacts. P6 depends on passing P5 findings. P7 depends on the final
integration diff. Commits are omitted from this task and require separate user
authorization after P7.

## Phase acceptance matrix

| Phase | Deliverable | Depends on | Required evidence | Exit gate |
|---|---|---|---|---|
| P0 | Environment/baseline evidence | none | version report, hashes, pytest output | reproducible or explicitly inconclusive |
| P1 | Read-only OCI safety audit | P0 | static findings, bind/CORS/path/token inventory | no unreviewed critical exposure |
| P2 | Provider-neutral orchestrator design | P1 | contract tests/design review | approval and failure states defined |
| P3 | Frozen controlled DSP run | P2 | manifest, baseline/control outputs, hashes | one-factor result reproducible |
| P4 | Defensible partition report | P3 | corrected stats, uncertainty, held-out/session evidence | claims bounded by evidence |
| P5 | Independent audit report | P3, P4 | independent replay/review output | PASS or documented remediation |
| P6 | Verified integration | P5 | focused/full tests, HTTP/MCP/UI proofs | no regressions and provenance intact |
| P7 | Final audit package | P6 | acceptance matrix, final diff/status, evidence index | release/commit recommendation only |

## P0 — Environment stability

**Objective:** establish the reproducible baseline without mutating existing
work.

Tasks:

- P0.1 Implement the exact next task from SPEC: a read-only environment and
  baseline evidence capture command.
- P0.2 Inventory Python/runtime/dependency versions, optional packages, data
  files, existing `dsp_out` artifacts, and current uncommitted paths.
- P0.3 Run focused tests first, then the full available test suite; classify
  missing-data or dependency failures separately from product failures.

Acceptance:

- Evidence includes command, cwd, timestamp, revision/status, hashes, versions,
  test output, and PASS/FAIL/INCONCLUSIVE classification.
- Existing uncommitted changes are preserved byte-for-byte.

Verification:

```bash
python -m pytest test_relationships.py test_tv_verdict.py test_quiz_determinism.py -v
python -m pytest -v
git status --short
```

## P1 — OCI safety audit, read-only

**Objective:** identify and bound local/LAN/cloud execution risk before adding
orchestrator behavior.

Tasks:

- Inspect `live_server.py`, `real_dsp_backend.py`, `dsp_chart.py`,
  `mcp_server.py`, `event_store.py`, launchers, config, and docs for binds,
  CORS, uploads, file reads/writes, subprocesses, websockets, token paths,
  external URLs, and secret logging.
- Trace every request-controlled path and tool argument to its filesystem or
  subprocess effect; record missing allowlists/authentication/rate limits.
- Produce mitigations ranked by severity. Do not start servers, call OCI/cloud
  APIs, read secret contents, modify config, or change firewall settings.

Acceptance:

- Report covers current `0.0.0.0` binds, wildcard CORS, upload/path traversal
  boundaries, OAuth/token handling, subprocess scope, MCP write tools, and
  LAN QR/discovery behavior.
- Every finding has source path/line, impact, evidence, recommendation, and
  verification command; unknowns are marked INCONCLUSIVE.

Verification:

```bash
rg -n '0\.0\.0\.0|allow_origins|UploadFile|subprocess|oauth|token|open\(|Path\(|urlopen|WebSocket' live_server.py real_dsp_backend.py dsp_chart.py mcp_server.py event_store.py
```

## P2 — Orchestrator architecture

**Objective:** define safe, provider-neutral routing around deterministic work.

Tasks:

- Specify task envelope: task id, phase, immutable inputs, allowed tools,
  resource/time limits, expected artifacts, verifier, approval requirement,
  and terminal status.
- Specify artifact manifest: content hashes, provenance, parent artifacts,
  parameters, seed, environment, and redaction status.
- Specify execution isolation, cancellation/retry semantics, structured logs,
  human approval gates, and model/provider adapter boundary.
- Define failure states for missing data, dependency failure, timeout, safety
  violation, statistical inconclusive, and verification failure.

Acceptance:

- A design review can trace a request from task creation to verification without
  any provider-specific assumption or bypass of evidence gates.
- Read-only audit tasks cannot invoke write-capable tools.

Verification:

- Contract/schema validation output.
- Design review checklist with PASS/FAIL/INCONCLUSIVE per invariant.

## P3 — Controlled DSP experiment

**Objective:** measure one declared DSP factor with a frozen baseline.

Tasks:

- Freeze source manifest, labels, sessions, canonical feature recipe, and input
  hashes; publish a pre-registration record before running analysis.
- Define baseline and exactly one varied factor per comparison, with all other
  settings fixed. Include seed and trial-zero baseline.
- Run the existing baseline/replay path and one controlled comparison; retain
  raw features, parameters, reports, and replay hashes.
- Record skipped/zero-attract outcomes without forcing an optimizer result.

Acceptance:

- Independent rerun reproduces feature and report hashes.
- No feature recipe drift, label leakage, session reassignment, or unlogged
  parameter change is present.

Verification:

```bash
python3 dsp_inverse.py relationships-baseline --wavdir WAVS --labels labels.csv --out gsd_p3_baseline
python3 dsp_inverse.py reproduce --out gsd_p3_baseline --wavdir WAVS --labels labels.csv
```

Use actual verified paths from P0; do not fabricate `WAVS` if unavailable.

## P4 — Statistically defensible partition analysis

**Objective:** turn partition exploration into bounded, reproducible claims.

Tasks:

- Validate session labels or mark the fallback session construction as a
  limitation; never silently treat correlated rows as independent.
- Run all seven partitions across the five canonical features with a declared
  max-statistic family-wise correction and fixed permutation seed/count.
- Report effect sizes, sample sizes, empirical p-values, uncertainty intervals,
  missingness, and LOSO/held-out performance; retain non-deployable signals.
- Stress-test preprocessing choices only as declared sensitivity analyses; do
  not select the most favorable recipe after seeing outcomes.

Acceptance:

- No deployability claim passes without session-aware validation and the stated
  CV gate.
- Multiple comparisons, Monte Carlo uncertainty, session transfer limits, and
  the known 15-minute gap are visible in the report.

Verification:

```bash
python3 partition_scout.py FEATURES.csv --session_col SESSION --perms 2000
```

Record the actual session column, seed, row counts, and output in evidence.

## P5 — Independent audit

**Objective:** challenge the result using an isolated review context.

Tasks:

- Recompute from the frozen manifest and hashes without relying on author prose.
- Review P0–P4 evidence, safety findings, statistical assumptions, controlled
  variables, uncertainty, and claims-to-evidence mapping.
- Attempt negative controls: altered labels/session shuffle, stale artifact,
  or unauthorized parameter mutation should fail verification.

Acceptance:

- Auditor can reproduce or explicitly identify the blocking artifact.
- Findings are severity-ranked and dispositioned; INCONCLUSIVE is allowed.

Verification:

- Independent command output and signed/attributed audit record containing
  source and artifact hashes.

## P6 — Integration

**Objective:** integrate only audited changes and preserve project invariants.

Tasks:

- Apply the smallest approved implementation changes from P0–P5.
- Run focused tests, then full tests; verify CLI replay, affected HTTP routes,
  MCP tool permissions, and UI behavior with required evidence.
- Confirm frozen relationships, manual provenance, browser/Python parity, and
  unchanged unrelated worktree files.

Acceptance:

- All changed behavior has the evidence type required by PROJECT_RULES.md.
- No secrets, generated outputs, or user changes are accidentally included.

Verification:

```bash
python -m pytest -v
git diff --check
git status --short
```

Add curl, MCP harness, or screenshot proofs for affected surfaces.

## P7 — Final audit

**Objective:** produce a release/commit recommendation with residual risk.

Tasks:

- Re-run the acceptance matrix and evidence-index checks.
- Inspect final diff, artifact manifest, hashes, test output, and worktree for
  scope creep, stale reports, secret exposure, and missing proof.
- Record unresolved risks, rejected hypotheses, known data limitations, and the
  exact next task after the audit.

Acceptance:

- Every P0–P7 gate is PASS or explicitly INCONCLUSIVE with owner/action.
- Final audit recommends commit only after user authorization; this roadmap
  never performs the commit automatically.

Verification:

```bash
git diff --check
git status --short
git diff --stat
```

## Evidence ledger template

Each phase appends one record:

```markdown
### Pn evidence
- Status: PASS | FAIL | INCONCLUSIVE
- Commands:
- CWD:
- Timestamp:
- Revision:
- Inputs/artifact hashes:
- Expected:
- Observed:
- Uncertainty/limitations:
- Findings and disposition:
- Next gate:
```

## Current planning evidence

### OBSERVED

- `.gsd/SPEC.md` is finalized by this task.
- Before this task, no `.gsd/SPEC.md` or `.gsd/ROADMAP.md` existed.
- The worktree had pre-existing modifications and untracked files; they remain
  outside these planning artifacts.
- Project rules explicitly prohibit implementation before finalization and
  require empirical verification before commit.

### INFERRED

- P0 is the only safe first implementation task because later evidence depends
  on runtime, data, and dependency availability.
- P1 must remain read-only because current server/MCP surfaces include network,
  upload, token, subprocess, and file-write boundaries.
- P4 should extend and instrument the existing `partition_scout.py` behavior
  rather than inventing another statistical recipe.

### HYPOTHESIS

- The first baseline may be INCONCLUSIVE if the full WAV corpus or optional
  dependencies are absent.
- A controlled factor may not reproduce across sample rates or sessions; this
  is an intended uncertainty to measure.
