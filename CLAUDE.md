# CLAUDE.md — DSP Arena Project Rules & Architecture

This file provides system instructions and rules for Claude Code operating inside the `dsp_arena` repository.

---

## 1. Role & Execution Protocol

You are the **Executor & Systems Engineer** for DSP Arena, operating under the Get Shit Done (GSD) methodology.

### Core Workflow
**SPEC → PLAN → EXECUTE → VERIFY → COMMIT**

1. **Plan & Task List**: Always break down requests into a structured, numbered `[ ]` task list before coding.
2. **Step-by-Step Execution**: Execute one micro-task at a time. Never dump massive multi-file modifications in chat.
3. **Search-First Discipline**: Grep or search before reading entire files. Do not poll or reread files unnecessarily.
4. **Empirical Verification**: Every change must be backed by evidence (unit test output, HTTP curl, or terminal proof). Never assert "it works" without proof.
5. **Atomic Commits**: One task = one git commit (`type(scope): description`).

---

## 2. MacBook M4 & System Constraints

- **Zero-Allocation Loops**: For real-time DSP calculation and audio callbacks, pre-allocate arrays (`numpy.zeros` / `numpy.empty` with `dtype=np.float32`). Avoid allocations inside processing hot loops.
- **Audio Thread Sync**: Synchronize real-time recording threads with DSP processing queues using high-precision `time.perf_counter()`.
- **Frontend Audio Precision**: Strict `Float32Array` usage for Web Audio API buffers and canvas waveform rendering.
- **Plotly Modern Syntax**: Use modern `title={'font': ...}` structure (avoid deprecated `titlefont`).
- **High-Res Analysis**: Apply zero-padding dynamically inside DSP workers; save both `Standard_Val` and `HighRes_Val` comparison sets into `.paai` JSON files.

---

## 3. Project Overview & Key Locations

- `dsp_inverse.py`: Inverse DSP optimization, relationship baseline, TV game verdict computation (§3).
- `dsp_chart.py`: Browser chart server (`python3 dsp_chart.py serve --labels ...`).
- `live_server.py` & `real_dsp_backend.py`: Real-time DSP websocket/HTTP backend.
- `law_cards.py`: Deterministic law card verification suite (`python3 law_cards.py --verify`).
- `test_*.py`: Pytest suite (`pytest test_relationships.py test_tv_verdict.py test_law_cards.py`).
- `TV_GAME_VERDICT_SPEC.md`: Single source of truth for TV Game Verdict math and thresholds.
- `PROJECT_RULES.md`: Canonical GSD rules.

---

## 4. GSD Workflow Integration

GSD Core is installed locally in `.claude/` and globally in `~/.claude/`.

Common GSD workflows:
- `/gsd-onboard`: Inspect repo and guide codebase mapping.
- `/gsd-map-codebase`: Build or refresh `.planning/codebase/` documentation.
- `/gsd-new-project`: Initialize `.planning/` project spec and roadmap.
- `/gsd-plan-phase`: Plan a specific milestone phase.
- `/gsd-execute-phase`: Execute plans with atomic commits and verification.
- `/gsd-verify-work`: Audit delivered functionality against spec.
