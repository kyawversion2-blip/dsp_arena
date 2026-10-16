"""Small local dispatcher with dry-run, bounded retries, and baseline capture.

Worktree protection is POLICY-ONLY: the post-run assertion detects mutations
outside ``.orchestrator/`` after the fact; it does not prevent them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

from .adapters import LocalSubprocessAdapter, RuntimeAdapter, RuntimeRegistry, UnknownRuntimeError
from .command_policy import CommandPolicyError, evaluate_command, redact
from .collector import ArtifactCollector
from .graph import TaskGraph
from .schemas import Claim, Evidence, EvidenceLabel, Failure, FailureClass, ResultEnvelope, TaskEnvelope, TaskStatus
from .validator import validate_result, validate_task


class Dispatcher:
    def __init__(
        self,
        root: Path,
        dry_run: bool = False,
        runtime_adapters: Mapping[str, RuntimeAdapter] | None = None,
    ):
        self.root = root.resolve()
        self.control_root = self.root / ".orchestrator"
        self.work_root = self.control_root / "workspaces"
        self.artifacts = ArtifactCollector(self.control_root / "artifacts")
        self.runtime_registry = RuntimeRegistry(runtime_adapters)
        self.adapter = LocalSubprocessAdapter()
        self._audit_lock = threading.Lock()
        self.dry_run = dry_run

    # ---------- git worktree baseline (J2) ----------

    def _git(self, *arguments: str) -> str:
        completed = subprocess.run(["git", *arguments], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return completed.stdout

    def _worktree_status_lines(self) -> list[str]:
        return [line for line in self._git("status", "--short", "--untracked-files=all").splitlines() if line.strip()]

    def _worktree_fingerprint(self) -> str:
        head = self._git("rev-parse", "HEAD").strip()
        diff = self._git("diff", "HEAD", "--", ".", ":(exclude).orchestrator")
        status = "\n".join(sorted(line for line in self._worktree_status_lines() if not self._is_orchestrator_status(line)))
        return hashlib.sha256("\x00".join((head, diff, status)).encode("utf-8")).hexdigest()

    @staticmethod
    def _status_paths(line: str) -> list[str]:
        rest = line[2:].strip() if len(line) > 2 else ""
        if not rest:
            return []
        if " -> " in rest:
            left, right = rest.split(" -> ", 1)
            return [left.strip().strip('"'), right.strip().strip('"')]
        return [rest.strip().strip('"')]

    @classmethod
    def _is_orchestrator_status(cls, line: str) -> bool:
        paths = cls._status_paths(line)
        return bool(paths) and all(path == ".orchestrator" or path.startswith(".orchestrator/") for path in paths)

    def capture_worktree_baseline(self, task_id: str = "baseline") -> dict[str, object]:
        status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        payload = {"head": head.stdout.strip(), "status": status.stdout, "status_exit_code": status.returncode, "captured_at": time.time(), "fingerprint": self._worktree_fingerprint()}
        self.artifacts.write_json(task_id, "worktree-baseline.json", payload)
        return payload

    def assert_worktree_unchanged(self, baseline: Mapping[str, object], task_id: str = "postcheck") -> list[str]:
        current = self._worktree_fingerprint()
        violations: list[str] = []
        if baseline.get("fingerprint") != current:
            base_lines = {line for line in str(baseline.get("status", "")).splitlines() if line.strip() and not self._is_orchestrator_status(line)}
            current_lines = {line for line in self._worktree_status_lines() if not self._is_orchestrator_status(line)}
            for line in sorted(current_lines ^ base_lines):
                violations.append(f"worktree status changed: {line}")
            if not violations:
                violations.append("worktree content changed outside .orchestrator/ during execution")
        self.artifacts.write_json(task_id, "worktree-postcheck.json", {"violations": violations, "fingerprint": current})
        return violations

    # ---------- audit (J4) ----------

    def _audit(self, task: TaskEnvelope, event: str, detail: Mapping[str, object]) -> None:
        record = redact({"at": time.time(), "event": event, "task_id": task.task_id, "run_id": task.run_id, **detail})
        self.control_root.mkdir(parents=True, exist_ok=True)
        with self._audit_lock:
            with (self.control_root / "audit-log.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    # ---------- execution ----------

    def _result(
        self,
        task: TaskEnvelope,
        status: TaskStatus,
        claims: tuple[Claim, ...] = (),
        evidence: tuple[Evidence, ...] = (),
        artifacts=(),
        failure: Failure | None = None,
        runtime_name: str = "local_subprocess",
        adapter_name: str | None = None,
    ) -> ResultEnvelope:
        result = ResultEnvelope(
            "result/v1",
            task.task_id,
            task.run_id,
            status,
            {"runtime": runtime_name, "adapter": adapter_name or self.adapter.name},
            claims,
            tuple(artifacts),
            evidence,
            failure,
        )
        validate_result(result, task)
        return result

    def _finalize(
        self,
        task: TaskEnvelope,
        baseline: Mapping[str, object],
        status: TaskStatus,
        evidence: tuple[Evidence, ...],
        artifacts,
        failure: Failure | None,
        claims: tuple[Claim, ...] = (),
        runtime_name: str = "local_subprocess",
        adapter_name: str | None = None,
    ) -> ResultEnvelope:
        violations = self.assert_worktree_unchanged(baseline, task.task_id)
        if violations:
            self._audit(task, "worktree_violation", {"violations": violations})
            return self._result(
                task,
                TaskStatus.FAILED,
                claims=claims,
                evidence=evidence,
                artifacts=artifacts,
                failure=Failure(FailureClass.SAFETY, "; ".join(violations), False),
                runtime_name=runtime_name,
                adapter_name=adapter_name,
            )
        return self._result(
            task,
            status,
            claims=claims,
            evidence=evidence,
            artifacts=artifacts,
            failure=failure,
            runtime_name=runtime_name,
            adapter_name=adapter_name,
        )

    def run(self, task: TaskEnvelope, command: Sequence[str] | None = None) -> ResultEnvelope:
        validate_task(task)
        workspace = self.work_root / task.task_id
        baseline = self.capture_worktree_baseline(task.task_id)
        try:
            runtime_name, adapter = self.runtime_registry.resolve(task)
        except UnknownRuntimeError as exc:
            self._audit(task, "runtime_unavailable", {"reason": str(exc), "allowed_runtimes": task.allowed_runtimes})
            return self._result(
                task, TaskStatus.BLOCKED,
                failure=Failure(FailureClass.DEPENDENCY, str(exc), False),
                runtime_name="unresolved", adapter_name="none",
            )
        if self.dry_run:
            return self._result(task, TaskStatus.SUCCEEDED, (Claim(EvidenceLabel.OBSERVED, "dry-run skipped subprocess execution", {"task_id": task.task_id}),), runtime_name=runtime_name, adapter_name=adapter.name)
        if command is None:
            return self._result(task, TaskStatus.BLOCKED, failure=Failure(FailureClass.DEPENDENCY, "local command is required", False), runtime_name=runtime_name, adapter_name=adapter.name)
        try:
            decision = evaluate_command(command, task, workspace.resolve())
        except CommandPolicyError as exc:
            self._audit(task, "command_denied", {"reason": str(exc), "command": list(command)})
            return self._result(task, TaskStatus.BLOCKED, failure=Failure(FailureClass.SAFETY, f"command policy denied: {exc}", False), runtime_name=runtime_name, adapter_name=adapter.name)
        attempts = max(1, min(task.retry.max_attempts, task.execution.max_retries + 1))
        last_output = None
        for attempt in range(attempts):
            output = adapter.run(task, command, workspace)
            last_output = output
            stdout = self.artifacts.write_text(task.task_id, f"stdout-{attempt + 1}.txt", output.stdout)
            stderr = self.artifacts.write_text(task.task_id, f"stderr-{attempt + 1}.txt", output.stderr)
            evidence = (Evidence("command", " ".join(command), output.returncode, stdout.artifact_id, stderr.artifact_id),)
            self._audit(task, "command_executed", {
                "policy": decision,
                "command": list(command),
                "attempt": attempt + 1,
                "returncode": output.returncode,
                "timed_out": output.timed_out,
                "output_truncated": output.output_truncated,
                "artifacts": [
                    {"artifact_id": stdout.artifact_id, "sha256": stdout.sha256},
                    {"artifact_id": stderr.artifact_id, "sha256": stderr.sha256},
                ],
            })
            failure = adapter.failure(output)
            artifacts = (stdout, stderr)
            if failure is None:
                return self._finalize(
                    task, baseline, TaskStatus.SUCCEEDED, evidence, artifacts, None,
                    (Claim(EvidenceLabel.OBSERVED, "runtime subprocess exited successfully", {"attempt": attempt + 1, "runtime": runtime_name}),),
                    runtime_name=runtime_name, adapter_name=adapter.name,
                )
            if not failure.retryable or attempt + 1 >= attempts:
                status = TaskStatus.BLOCKED if failure.category == FailureClass.DEPENDENCY else TaskStatus.FAILED
                return self._finalize(task, baseline, status, evidence, artifacts, failure, runtime_name=runtime_name, adapter_name=adapter.name)
        raise RuntimeError(f"unreachable dispatcher state: {last_output}")


    def run_graph(
        self,
        tasks: Sequence[TaskEnvelope],
        commands: Mapping[str, Sequence[str]],
        max_parallelism: int = 4,
    ) -> dict[str, ResultEnvelope]:
        """Run a dependency DAG, parallelizing independent read-only tasks.

        A wave is parallel only when every ready task declares mode=read_only.
        Any wave containing a mutating task is serialized, preserving the
        project rule that there is one writer at a time. Source-tree mutations
        remain subject to the existing post-run worktree assertion.
        """
        if max_parallelism < 1:
            raise ValueError("max_parallelism must be at least 1")
        graph = TaskGraph(tuple(tasks))
        pending = {
            task.task_id for task in graph.tasks()
            if task.status in (TaskStatus.PENDING, TaskStatus.READY)
        }
        results: dict[str, ResultEnvelope] = {}

        while pending:
            ready = [task for task in graph.ready() if task.task_id in pending]
            if not ready:
                for task_id in sorted(pending):
                    task = graph.get(task_id)
                    result = self._result(
                        task,
                        TaskStatus.BLOCKED,
                        failure=Failure(
                            FailureClass.DEPENDENCY,
                            "one or more dependencies did not succeed",
                            False,
                        ),
                        runtime_name="unresolved",
                        adapter_name="none",
                    )
                    results[task_id] = result
                    graph = graph.update_status(task_id, TaskStatus.BLOCKED)
                pending.clear()
                break

            if any(task.execution.mode != "read_only" for task in ready):
                wave = [ready[0]]
            else:
                wave = ready[:max_parallelism]

            wave_results: dict[str, ResultEnvelope] = {}
            if len(wave) == 1:
                task = wave[0]
                wave_results[task.task_id] = self.run(task, commands.get(task.task_id))
            else:
                with ThreadPoolExecutor(max_workers=min(max_parallelism, len(wave))) as pool:
                    future_to_task = {
                        pool.submit(self.run, task, commands.get(task.task_id)): task
                        for task in wave
                    }
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        wave_results[task.task_id] = future.result()

            for task in wave:
                result = wave_results[task.task_id]
                results[task.task_id] = result
                graph = graph.update_status(task.task_id, result.status)
                pending.discard(task.task_id)

        return results
