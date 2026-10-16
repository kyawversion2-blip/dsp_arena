import json
import sys
import time
from pathlib import Path

import pytest

from orchestrator.adapters import LocalSubprocessAdapter
from orchestrator.dispatcher import Dispatcher
from orchestrator.graph import GraphError, TaskGraph
from orchestrator.schemas import (
    Claim,
    EvidenceLabel,
    ExecutionPolicy,
    InputRef,
    ResultEnvelope,
    RetryPolicy,
    TaskEnvelope,
    TaskStatus,
    to_json_dict,
)
from orchestrator.validator import ValidationError, sha256_file, validate_result, validate_task


def task(task_id="t1", dependencies=(), **kwargs):
    return TaskEnvelope(
        "task/v1", task_id, "run-1", "P0", "integration_validation", "test task",
        tuple(dependencies), execution=ExecutionPolicy(**kwargs.pop("execution", {})),
        retry=RetryPolicy(max_attempts=kwargs.pop("max_attempts", 1)), **kwargs
    )


def test_task_is_immutable_and_serializable():
    item = task(inputs=(InputRef("input.txt", "0" * 64),))
    with pytest.raises(AttributeError):
        item.task_id = "changed"
    assert to_json_dict(item)["schema_version"] == "task/v1"


def test_graph_rejects_missing_dependency_and_cycles():
    with pytest.raises(GraphError):
        TaskGraph([task("a", ("missing",))])
    with pytest.raises(GraphError):
        TaskGraph([task("a", ("b",)), task("b", ("a",))])


def test_graph_selects_only_dependency_ready_tasks():
    graph = TaskGraph([task("a"), task("b", ("a",)), task("c")])
    assert [item.task_id for item in graph.ready()] == ["a", "c"]
    graph = graph.update_status("a", TaskStatus.SUCCEEDED)
    assert [item.task_id for item in graph.ready()] == ["b", "c"]


def test_claim_labels_and_sources_are_validated():
    result = ResultEnvelope("result/v1", "t1", "run-1", TaskStatus.SUCCEEDED, claims=(Claim(EvidenceLabel.OBSERVED, "measured", {"command": "x"}),))
    validate_result(result, task())
    with pytest.raises(ValidationError):
        validate_result(ResultEnvelope("result/v1", "t1", "run-1", TaskStatus.SUCCEEDED, claims=(Claim(EvidenceLabel.OBSERVED, "missing source"),)), task())
    validate_result(ResultEnvelope("result/v1", "t1", "run-1", TaskStatus.SUCCEEDED, claims=(Claim(EvidenceLabel.UNSUPPORTED, "not measured"),)), task())


def test_hash_and_dry_run_capture_baseline(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    assert len(sha256_file(source)) == 64
    dispatcher = Dispatcher(tmp_path, dry_run=True)
    result = dispatcher.run(task("dry"))
    assert result.status == TaskStatus.SUCCEEDED
    assert (tmp_path / ".orchestrator" / "artifacts" / "dry" / "worktree-baseline.json").exists()


def test_local_adapter_captures_output_and_filters_environment(tmp_path):
    executable = Path(sys.executable).resolve()
    current = task("run", execution={"allowed_executables": (str(executable),), "timeout_seconds": 2})
    output = LocalSubprocessAdapter(str(executable)).run(current, [str(executable), "-c", "import os; print(os.getenv('DSP_ARENA_NETWORK_POLICY')); print(os.getenv('SECRET_TOKEN'))"], tmp_path)
    assert output.returncode == 0
    assert "disabled" in output.stdout
    assert "SECRET_TOKEN" not in output.stdout


def test_local_adapter_timeout_is_reported(tmp_path):
    executable = Path(sys.executable).resolve()
    current = task("timeout", execution={"allowed_executables": (str(executable),), "timeout_seconds": 0.05})
    output = LocalSubprocessAdapter(str(executable)).run(current, [str(executable), "-c", "import time; time.sleep(1)"], tmp_path)
    assert output.timed_out is True
    assert LocalSubprocessAdapter.failure(output).retryable is True


def test_dispatcher_runs_allowlisted_subprocess(tmp_path):
    executable = Path(sys.executable).resolve()
    current = task("command", execution={"allowed_executables": (str(executable),), "timeout_seconds": 2})
    result = Dispatcher(tmp_path).run(current, [str(executable), "-c", "print('ok')"])
    assert result.status == TaskStatus.SUCCEEDED
    assert len(result.artifacts) == 2
    assert all(len(item.sha256) == 64 for item in result.artifacts)


def test_dispatcher_does_not_allow_unlisted_executable(tmp_path):
    current = task("blocked", execution={"allowed_executables": ("/bin/false",)})
    with pytest.raises(PermissionError):
        Dispatcher(tmp_path).run(current, [sys.executable, "-c", "pass"])


def test_dispatcher_uses_ordered_registered_runtime(tmp_path):
    executable = Path(sys.executable).resolve()
    runtime = LocalSubprocessAdapter(str(executable))
    dispatcher = Dispatcher(tmp_path, runtime_adapters={"reviewer": runtime})
    current = task(
        "runtime",
        allowed_runtimes=("reviewer",),
        execution={"allowed_executables": (str(executable),), "timeout_seconds": 2},
    )
    result = dispatcher.run(current, [str(executable), "-c", "print('reviewed')"])
    assert result.status == TaskStatus.SUCCEEDED
    assert result.runtime["runtime"] == "reviewer"
    assert result.runtime["adapter"] == "local_subprocess"


def test_dispatcher_blocks_unregistered_runtime(tmp_path):
    result = Dispatcher(tmp_path).run(task("missing-runtime", allowed_runtimes=("claude_code",)))
    assert result.status == TaskStatus.BLOCKED
    assert result.failure.category.value == "dependency"
    assert "claude_code" in result.failure.message


def test_run_graph_parallelizes_independent_read_only_tasks(tmp_path, monkeypatch):
    import threading

    barrier = threading.Barrier(2)
    dispatcher = Dispatcher(tmp_path)

    def fake_run(item, command=None):
        if item.task_id in {"review-a", "review-b"}:
            barrier.wait(timeout=2)
        return ResultEnvelope("result/v1", item.task_id, item.run_id, TaskStatus.SUCCEEDED)

    monkeypatch.setattr(dispatcher, "run", fake_run)
    tasks = [
        task("review-a"),
        task("review-b"),
        task("integration", ("review-a", "review-b")),
    ]
    results = dispatcher.run_graph(
        tasks,
        {"review-a": ("fake",), "review-b": ("fake",), "integration": ("fake",)},
        max_parallelism=2,
    )
    assert set(results) == {"review-a", "review-b", "integration"}
    assert all(result.status == TaskStatus.SUCCEEDED for result in results.values())


def test_run_graph_blocks_dependents_after_failed_dependency(tmp_path, monkeypatch):
    dispatcher = Dispatcher(tmp_path)

    def fake_run(item, command=None):
        status = TaskStatus.FAILED if item.task_id == "failed" else TaskStatus.SUCCEEDED
        return ResultEnvelope("result/v1", item.task_id, item.run_id, status)

    monkeypatch.setattr(dispatcher, "run", fake_run)
    results = dispatcher.run_graph(
        [task("failed"), task("independent"), task("dependent", ("failed",))],
        {"failed": ("fake",), "independent": ("fake",), "dependent": ("fake",)},
        max_parallelism=2,
    )
    assert results["failed"].status == TaskStatus.FAILED
    assert results["independent"].status == TaskStatus.SUCCEEDED
    assert results["dependent"].status == TaskStatus.BLOCKED
