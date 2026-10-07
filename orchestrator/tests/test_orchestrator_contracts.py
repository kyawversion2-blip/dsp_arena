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
