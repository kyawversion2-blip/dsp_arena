# DSP Arena — AI Cross-Review Prompts and Links

**Purpose:** Collect independent AI reviews for the next DSP Arena research/design plan. This document is a planning artifact only; it does not authorize implementation.

## Repository context

- Repository: [kyawversion2-blip/dsp_arena](https://github.com/kyawversion2-blip/dsp_arena)
- Branch reviewed: `backup-yolo-mode`
- Scope: real WAV → impact/event processing → DSP features → interactive chart → Training/Testing workflow.
- Primary objective: same-face/surface repeatability and similarity to the immediately preceding impact **when temporal order is verified**.
- Intended Training cohort: approximately 10–20 labeled examples.
- Explore method-wise pair relationships and the partitions `AB|CD`, `AC|BD`, `AD|BC`.
- “Make similar” and “make different” are distinct objectives.
- Freeze Training configuration and reuse it unchanged in Testing.

## Important project constraints

- Read-only review/design only. Do not edit files, run tests, install packages, restart services, change configs/data/labels/thresholds, or commit.
- Do not investigate `2012.wav`; it is explicitly out of scope.
- Do not force historical recordings into A/B/C/D separation.
- Treat the earlier circle/oval chart observation as a hypothesis, not established evidence.
- Do not infer temporal order from filename sorting. If source order is not verified, mark it unknown.
- Physical distance and frequency-dependent attenuation may be part of the phenomenon; do not automatically remove them as nuisance factors.
- Separate similarity, repeatability, classification correctness, confidence, and uncertainty.
- Distinguish verified source observations, reported/documented behavior, hypotheses, and recommendations.
- Do not claim accuracy or scientific validity without evidence.

## Suggested AI review order

1. **GPT-6 (reasoning / architecture synthesis):** establish the evidence-based plan and reconcile reviews.
2. **Claude Opus (independent architecture audit):** challenge the plan and find missing integration details.
3. **Mythos (adversarial review, if available in the user's model interface):** try to falsify assumptions and identify failure modes.
4. **Gemini (codebase and DSP method audit):** inspect implementation paths and method alternatives.
5. **DeepSeek (mathematical/statistical review):** formalize similarity and small-sample validation.
6. **Sol (UX/workflow review, if available as a selectable model):** critique Training/Testing interaction and uncertainty presentation.

Model labels may differ by product or account. Use the closest available model; do not assume Mythos or Sol have standalone public chat endpoints.

## Chat links

- [ChatGPT](https://chatgpt.com/)
- [Claude](https://claude.ai/)
- [Gemini](https://gemini.google.com/)
- [DeepSeek](https://chat.deepseek.com/)
- [Grok](https://grok.com/)
- [Qwen Chat](https://chat.qwen.ai/)

The repository is private. A model opened through a web link will not automatically receive repository access. If it cannot access the repo, paste the relevant source excerpts or provide an authorized file upload. Do not paste API keys, tokens, credentials, or private WAVs unless the service and data handling are explicitly approved.

---

# Shared project brief

DSP Arena is a real-WAV acoustic impact research application. The goal is not merely to classify A/B/C/D. The highest-priority research objective is to measure repeatability of impacts from the same face/surface, especially similarity to the immediately preceding impact when reliable temporal provenance exists.

Desired workflow:

1. Training accepts approximately 10–20 labeled examples.
2. Multiple DSP method families independently extract features and assess pairwise similarity.
3. Methods may disagree about which pairs are similar; disagreement must remain visible.
4. Explore the four groups and partitions `AB|CD`, `AC|BD`, `AD|BC`.
5. Support “make selected pair more similar” and “separate selected pair” as distinct objectives.
6. Freeze the selected DSP configuration in Training.
7. Reuse that exact configuration in Testing without test-time retuning.
8. Prioritize adjacent-impact similarity and same-face repeatability, while keeping class correctness and confidence separate.

Candidate method families include FFT/PSD, STFT, envelope/Hilbert/decay, and temporal/IPI features. These are candidates, not assumed independent or validated.

# Prompt 1 — GPT-6: Architecture synthesis and read-only audit

You are GPT-6, acting as a senior DSP engineer, applied ML researcher, and software architect.

Use the Shared project brief and constraints above.

If authorized repository access is available, inspect the current code read-only. Identify entrypoints, WAV pipeline, event detection, DSP methods, chart flow, similarity engine, optimization, persistence, Training/Testing boundaries, and tests. Cite exact files/functions/lines where possible. If you cannot access the private repo, say so clearly and request only the relevant files; do not pretend to have inspected it.

Please deliver:
1. Verified code observations.
2. Established DSP/ML principles.
3. Hypotheses.
4. Existing-vs-required capability matrix: implemented / partial / missing / unknown.
5. Repeatability-first similarity architecture, including adjacent impact, same-face repeatability, reference similarity, method disagreement, partitions, uncertainty, and abstention.
6. Small-sample validation plan for 10–20 samples with leakage/overfit controls.
7. Frozen Training config schema and immutable Testing contract.
8. Phased implementation plan with dependencies, likely files, acceptance criteria, risks, rollback, and evidence gates.
9. Open questions/blockers.

Do not implement anything. Be critical and evidence-led.

# Prompt 2 — Claude Opus: Independent architecture review

You are Claude Opus, a senior DSP researcher and software architecture reviewer. Provide an independent, critical review rather than agreeing by default.

Use the Shared project brief and constraints above.

Review the codebase read-only if available. Cite source locations. Determine what exists and what is missing. Specifically:
- Design adjacent similarity and same-face repeatability without confusing either with class correctness.
- Recommend robust method-specific normalization and cross-method comparison; raw feature scales are not interchangeable.
- Explain how to detect correlated/redundant methods.
- Address small-sample validation, test leakage, slider overfitting, and Training/Testing boundaries.
- Specify frozen configuration and Testing immutability.
- Propose phased implementation, acceptance gates, failure modes, and evidence requirements.
- List assumptions that require owner confirmation.

Use sections: Verified observations; DSP/ML principles; Hypotheses; Architecture proposal; Phased plan; Risks and unresolved questions.

No implementation or repository changes.

# Prompt 3 — Mythos: Adversarial / falsification review

You are Mythos, acting as an adversarial scientific reviewer and failure-analysis specialist.

Use the Shared project brief and constraints above. Do not assume the design is correct.

Try to falsify the approach. Examine:
- Adjacent similarity may be misleading because of sequence drift or correlated neighbors.
- Same-face repeatability may be confused with class correctness.
- 10–20 examples may not support broad parameter search.
- Slider tuning may overfit Training.
- DSP methods may share redundant information and not be independent.
- Normalization may remove meaningful distance/frequency effects.
- Circle/oval chart structure may be a visualization/scaling artifact.
- Binary partitions may lack physical/statistical interpretability.
- Frozen config and provenance may still permit leakage.

Return:
1. Critical failure modes ranked by severity.
2. Counterexamples/scenarios producing false results.
3. Minimal safeguards.
4. Evidence required before implementation.
5. Go/No-Go criteria.

Distinguish proven problems from plausible risks. Do not implement. If Mythos is not available in the current interface, use another adversarial reasoning model and preserve this prompt.

# Prompt 4 — Sol: UX and Training/Testing workflow review

You are Sol, acting as a product designer and human-in-the-loop research workflow specialist.

Use the Shared project brief and constraints above.

Design the UX—not application code—for:
1. Training and Testing screen structure.
2. Chart controls, legends, and sample provenance indicators.
3. Pairwise relationship matrix and method disagreement display.
4. Separate presentation of similarity, repeatability, confidence, and class prediction.
5. Frozen configuration review and protection against accidental test-time changes.
6. Warnings, confirmation steps, and insufficient-evidence/abstention behavior.
7. Small phased UX implementation plan.

Use concrete interaction examples. Do not imply that high similarity proves the predicted class is correct. If Sol is not selectable as a standalone model in this interface, use a product/UX model and keep this prompt.

# Prompt 5 — Gemini: Codebase and DSP method audit

You are Gemini, a senior DSP engineer and codebase auditor.

Use the Shared project brief and constraints above.

If the private repository is not accessible, explicitly state that and ask for source excerpts. Do not claim to inspect unseen code.

Please:
1. Map WAV → event detection → DSP → chart → storage.
2. Identify duplicate/inconsistent DSP feature pipelines.
3. Evaluate FFT/PSD, STFT, Hilbert/envelope, temporal/IPI, and other justified methods.
4. Propose comparable similarity measures while preserving method-specific scales and uncertainty.
5. Specify adjacent similarity and same-face repeatability.
6. Define frozen config and Testing contract.
7. Identify missing code, integration points, and verification criteria.

Separate verified, reported, hypothesized, and recommended items. Do not implement.

# Prompt 6 — DeepSeek: Mathematical/statistical review

You are DeepSeek, a quantitative DSP/ML researcher.

Use the Shared project brief and constraints above.

Formulate:
1. A mathematically clear adjacent-impact similarity metric.
2. A separate same-face repeatability metric.
3. Method-wise similarity aggregation without treating raw feature scales as interchangeable.
4. A method disagreement measure.
5. Small-sample validation and leakage prevention.
6. Principled parameter search and overfit control.
7. Uncertainty estimates and abstention conditions.
8. Limitations and assumptions.

Use equations where useful. Do not claim empirical performance without data. Do not inspect or modify files unless read-only access is explicitly available.

# Prompt 7 — Independent model / second-opinion template

You are an independent technical reviewer for DSP Arena. Use the Shared project brief and constraints above.

Do not merely summarize other models' answers. Find disagreements, unsupported assumptions, missing evidence, and risks. Give:
- your strongest alternative architecture;
- what you would reject and why;
- minimum viable milestone;
- measurable acceptance criteria;
- unresolved owner decisions.

Do not implement. Keep factual findings separate from proposals.
