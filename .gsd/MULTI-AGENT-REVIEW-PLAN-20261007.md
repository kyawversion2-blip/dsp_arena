# DSP Arena Multi-Agent Review Pipeline

Status: ACTIVE
Date: 2026-10-07

## Objective
Accelerate review by parallelizing independent analysis while preserving a single controlled writer and numerical authority.

## Current state
Phase 1 initial review already identified:
- CR-01 HIGH: multiple DSP recipes; parity not yet proven.
- CR-02 HIGH: frequency-analysis ceilings/defaults need canonicalization.
- CR-03 HIGH: PCA normalization may fit full corpus; leakage contract requires train-only.
- CR-04 MEDIUM: synthetic session fallback can distort session claims.
- CR-05 MEDIUM: baseline tests are data-limited; quiz determinism failures are INCONCLUSIVE because wavs/labels yield no measured rows.

## Wave A — Independent review
Run Claude/architecture, Gemini/consistency, and DeepSeek/math audits READ-ONLY in parallel.
Inputs are paths, not copied file contents.
Output: raw structured findings.

## Wave B — Adjudication
Orchestrator deduplicates by invariant/path/claim. Conflicts are resolved against numerical evidence and SPEC, not model voting.
Output: adjudicated findings + approval-required task list.

## Wave C — Controlled implementation
One writer (Codex) handles approved tasks sequentially.
Each task gets focused tests and evidence.

## Wave D — Verification
Run focused suites, then regression suite where data permits. Missing optional corpus remains INCONCLUSIVE.

## Wave E — Independent final audit
A separate reviewer checks diff, evidence, hashes, acceptance matrix, and documentation alignment.

## Provider policy
Providers are optional adapters. If Claude/Gemini/DeepSeek are unavailable, use the available reviewers sequentially; never fabricate provider output.
