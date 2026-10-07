"""Read-only Graphify code-context provider for orchestrator tasks.

Design choices mirror orchestrator/adapters.py: argv execution with
shell=False, bounded timeout, allowlisted executable path, capped output.

Graphify is a query-bounded context lookup, never authoritative state.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_BUDGET = 400
MAX_BUDGET = 2000
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KiB
DEFAULT_GRAPHIFY_EXECUTABLE = Path.home() / ".local" / "bin" / "graphify"


class GraphifyError(RuntimeError):
    """Raised on Graphify invocation or output failures."""


class GraphifyBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class GraphifyContext:
    """Bounded, untrusted code context from a single Graphify query."""

    query: str
    budget: int
    truncated: bool
    output: str
    executable: str
    argv: tuple[str, ...]
    exit_code: int
    timeout_seconds: float
    required: bool  # whether the caller required Graphify context
    available: bool  # whether Graphify output was actually produced

    @property
    def is_untrusted_context(self) -> bool:
        return True


@dataclass(frozen=True)
class GraphifyEnrichment:
    """Optional enrichment attached alongside a TaskEnvelope execution.

    Does not mutate TaskGraph; passed as adapter-level context only.
    """

    task_id: str
    context: GraphifyContext | None
    skipped_reason: str | None = None


class GraphifyContextProvider:
    """Run a single bounded Graphify query via CLI only."""

    def __init__(
        self,
        graphify_executable: Path | str | None = None,
        default_budget: int = DEFAULT_BUDGET,
        max_budget: int = MAX_BUDGET,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        cwd: Path | None = None,
    ):
        self.executable = Path(graphify_executable or DEFAULT_GRAPHIFY_EXECUTABLE)
        if default_budget < 1 or default_budget > max_budget:
            raise ValueError("default_budget must be within [1, max_budget]")
        if max_budget < 1:
            raise ValueError("max_budget must be >= 1")
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be >= 1")
        self.default_budget = default_budget
        self.max_budget = max_budget
        self.max_output_bytes = max_output_bytes
        self.default_timeout_seconds = default_timeout_seconds
        self.cwd = cwd

    def query(
        self,
        question: str,
        *,
        budget: int | None = None,
        context_filter: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
        required: bool = False,
        use_dfs: bool = False,
    ) -> GraphifyContext:
        if not isinstance(question, str) or not question.strip():
            raise GraphifyBudgetError("question must be a non-empty string")
        effective_budget = self.default_budget if budget is None else int(budget)
        if effective_budget < 1:
            raise GraphifyBudgetError("budget must be >= 1")
        if effective_budget > self.max_budget:
            raise GraphifyBudgetError(f"budget {effective_budget} exceeds hard maximum {self.max_budget}")
        effective_timeout = self.default_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        if effective_timeout <= 0 or effective_timeout > MAX_TIMEOUT_SECONDS:
            raise GraphifyBudgetError(f"timeout must be in (0, {MAX_TIMEOUT_SECONDS}]")
        if not self.executable.exists():
            raise GraphifyError(f"Graphify executable not found: {self.executable}")

        argv: list[str] = [str(self.executable), "query", question.strip(), "--budget", str(effective_budget)]
        if use_dfs:
            argv.append("--dfs")
        if context_filter:
            for token in context_filter:
                if not isinstance(token, str) or not token.strip():
                    raise GraphifyBudgetError("context_filter entries must be non-empty strings")
                argv.extend(["--context", token.strip()])

        try:
            completed = subprocess.run(
                argv,
                cwd=str(self.cwd) if self.cwd is not None else None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=effective_timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GraphifyError(f"Graphify query timed out after {effective_timeout}s") from exc
        except FileNotFoundError as exc:
            raise GraphifyError(f"Graphify executable not found: {self.executable}") from exc
        except OSError as exc:
            raise GraphifyError(f"Graphify invocation failed: {exc}") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[:2000]
            raise GraphifyError(f"Graphify query failed with exit code {completed.returncode}: {detail}")

        raw = completed.stdout or ""
        truncated = False
        output = raw
        encoded_len = len(raw.encode("utf-8"))
        if encoded_len > self.max_output_bytes:
            truncated = True
            # Truncate on byte boundary safely
            encoded = raw.encode("utf-8")[: self.max_output_bytes]
            output = encoded.decode("utf-8", errors="ignore")
            output += "\n\n[truncated: output exceeded max_output_bytes]\n"

        # Treat empty output as malformed
        if not output.strip():
            raise GraphifyError("Graphify returned empty output")

        return GraphifyContext(
            query=question.strip(),
            budget=effective_budget,
            truncated=truncated,
            output=output,
            executable=str(self.executable),
            argv=tuple(argv),
            exit_code=completed.returncode,
            timeout_seconds=effective_timeout,
            required=required,
            available=True,
        )

    def enrich(
        self,
        task_id: str,
        question: str | None,
        *,
        budget: int | None = None,
        context_filter: Sequence[str] | None = None,
        timeout_seconds: float | None = None,
        required: bool = False,
        use_dfs: bool = False,
    ) -> GraphifyEnrichment:
        if not question or not question.strip():
            return GraphifyEnrichment(task_id=task_id, context=None, skipped_reason="no question provided")
        try:
            ctx = self.query(
                question,
                budget=budget,
                context_filter=context_filter,
                timeout_seconds=timeout_seconds,
                required=required,
                use_dfs=use_dfs,
            )
            return GraphifyEnrichment(task_id=task_id, context=ctx)
        except GraphifyError as exc:
            if required:
                raise
            return GraphifyEnrichment(task_id=task_id, context=None, skipped_reason=str(exc))
        except GraphifyBudgetError:
            raise

    @staticmethod
    def check_available(executable: Path | str | None = None) -> bool:
        exe = Path(executable or DEFAULT_GRAPHIFY_EXECUTABLE)
        return exe.exists()
