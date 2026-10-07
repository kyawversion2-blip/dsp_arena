"""Command policy, baseline protection, and subprocess containment tests
(P1.1 orchestrator gaps J1-J4). Uses temporary directories and a temporary
git repository; no external services, no network, no real project mutations.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orchestrator.adapters import LocalSubprocessAdapter
from orchestrator.command_policy import CommandPolicyError, evaluate_command
from orchestrator.dispatcher import Dispatcher
from orchestrator.schemas import ExecutionPolicy, FailureClass, RetryPolicy, TaskEnvelope, TaskStatus


def task(task_id="t1", **execution):
    return TaskEnvelope(
        "task/v1", task_id, "run-1", "P1", "integration_validation", "test task",
        execution=ExecutionPolicy(**execution),
        retry=RetryPolicy(max_attempts=1),
    )


@pytest.fixture
def git_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "seed.txt").write_text("seed", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True, env=env)
    return root


# ---------- J1: command and argument policy ----------

def test_policy_rejects_forbidden_executables(tmp_path):
    for name in ("git", "curl", "wget", "ssh", "sudo", "rm"):
        with pytest.raises(CommandPolicyError):
            evaluate_command([name, "status"], task(), tmp_path)
    with pytest.raises(CommandPolicyError):
        evaluate_command(["/usr/bin/git", "status"], task(), tmp_path)


def test_policy_rejects_credential_shaped_arguments(tmp_path):
    with pytest.raises(CommandPolicyError):
        evaluate_command([sys.executable, "--api-key=hunter2"], task(), tmp_path)
    with pytest.raises(CommandPolicyError):
        evaluate_command([sys.executable, "-c", "SECRET_TOKEN=abc"], task(), tmp_path)


def test_policy_rejects_paths_outside_workspace_and_allowlist(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    item = task(allowed_paths=(str(allowed),))
    evaluate_command([sys.executable, str(allowed / "run.py")], item, tmp_path / "ws")
    with pytest.raises(CommandPolicyError):
        evaluate_command([sys.executable, str(outside)], item, tmp_path / "ws")
    with pytest.raises(CommandPolicyError):
        evaluate_command([sys.executable, f"--input={outside}"], item, tmp_path / "ws")


def test_policy_rejects_empty_or_nul_commands(tmp_path):
    with pytest.raises(CommandPolicyError):
        evaluate_command([], task(), tmp_path)
    with pytest.raises(CommandPolicyError):
        evaluate_command([sys.executable, "a\x00b"], task(), tmp_path)


def test_dispatcher_blocks_forbidden_command_with_audit(tmp_path):
    result = Dispatcher(tmp_path).run(task("denied"), ["git", "status"])
    assert result.status == TaskStatus.BLOCKED
    assert result.failure is not None
    assert result.failure.category == FailureClass.SAFETY
    log = (tmp_path / ".orchestrator" / "audit-log.jsonl").read_text()
    assert "command_denied" in log


def test_dispatcher_denied_command_audit_is_redacted(tmp_path):
    result = Dispatcher(tmp_path).run(task("secret"), [sys.executable, "--api-key=hunter2"])
    assert result.status == TaskStatus.BLOCKED
    log = (tmp_path / ".orchestrator" / "audit-log.jsonl").read_text()
    assert "hunter2" not in log
    assert "[redacted]" in log
    assert "command_denied" in log


# ---------- J2: worktree baseline protection ----------

def test_worktree_unchanged_allows_success(git_repo):
    executable = str(Path(sys.executable).resolve())
    current = task("clean", allowed_executables=(executable,), timeout_seconds=10)
    result = Dispatcher(git_repo).run(current, [executable, "-c", "print('ok')"])
    assert result.status == TaskStatus.SUCCEEDED
    assert (git_repo / ".orchestrator" / "artifacts" / "clean" / "worktree-postcheck.json").exists()


def test_worktree_untracked_mutation_is_detected(git_repo):
    executable = str(Path(sys.executable).resolve())
    current = task("mutator", allowed_executables=(executable,), timeout_seconds=10)
    script = "import pathlib; pathlib.Path('../../../probe.txt').write_text('x')"
    result = Dispatcher(git_repo).run(current, [executable, "-c", script])
    assert result.status == TaskStatus.FAILED
    assert result.failure is not None
    assert result.failure.category == FailureClass.SAFETY
    assert (git_repo / "probe.txt").exists()  # POLICY-ONLY: detected, not prevented


def test_worktree_tracked_mutation_is_detected(git_repo):
    executable = str(Path(sys.executable).resolve())
    current = task("modifier", allowed_executables=(executable,), timeout_seconds=10)
    script = "import pathlib; pathlib.Path('../../../seed.txt').write_text('changed')"
    result = Dispatcher(git_repo).run(current, [executable, "-c", script])
    assert result.status == TaskStatus.FAILED
    assert result.failure is not None
    assert result.failure.category == FailureClass.SAFETY


def test_worktree_violation_is_audited(git_repo):
    executable = str(Path(sys.executable).resolve())
    current = task("audited-mutator", allowed_executables=(executable,), timeout_seconds=10)
    script = "import pathlib; pathlib.Path('../../../probe.txt').write_text('x')"
    Dispatcher(git_repo).run(current, [executable, "-c", script])
    records = [
        json.loads(line)
        for line in (git_repo / ".orchestrator" / "audit-log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    events = [record["event"] for record in records]
    assert "command_executed" in events
    assert "worktree_violation" in events


# ---------- J3: environment tightening ----------

def test_execution_policy_default_environment_is_minimal():
    policy = ExecutionPolicy()
    assert "PYTHONPATH" not in policy.inherit_environment
    assert set(policy.inherit_environment) == {"PATH", "HOME", "LANG", "LC_ALL"}


def test_environment_scrubs_credential_named_variables(tmp_path, monkeypatch):
    executable = str(Path(sys.executable).resolve())
    monkeypatch.setenv("ORCHESTRATOR_SECRET_TOKEN", "hunter2")
    item = task(
        "env",
        allowed_executables=(executable,),
        timeout_seconds=5,
        inherit_environment=("PATH", "ORCHESTRATOR_SECRET_TOKEN"),
    )
    output = LocalSubprocessAdapter(executable).run(
        item, [executable, "-c", "import os; print('TOKEN_PRESENT', 'ORCHESTRATOR_SECRET_TOKEN' in os.environ)"], tmp_path
    )
    assert output.returncode == 0
    assert "TOKEN_PRESENT False" in output.stdout


# ---------- J4: process-group kill, output caps, audit records ----------

def test_adapter_timeout_kills_process_group(tmp_path):
    executable = str(Path(sys.executable).resolve())
    item = task("group-kill", allowed_executables=(executable,), timeout_seconds=1)
    code = (
        "import subprocess, time\n"
        "child = subprocess.Popen(['sleep', '30'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(30)\n"
    )
    output = LocalSubprocessAdapter(executable).run(item, [executable, "-c", code], tmp_path)
    assert output.timed_out is True
    pid = int(output.stdout.strip().splitlines()[-1])
    deadline = time.monotonic() + 2
    gone = False
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.1)
    assert gone, "grandchild process survived the process-group kill"


def test_adapter_output_cap_truncates_and_fails(tmp_path):
    executable = str(Path(sys.executable).resolve())
    item = task("cap", allowed_executables=(executable,), timeout_seconds=10, max_output_bytes=8192)
    output = LocalSubprocessAdapter(executable).run(item, [executable, "-c", "print('A' * 200000)"], tmp_path)
    assert output.output_truncated is True
    assert output.timed_out is False
    assert len(output.stdout) <= 8192
    failure = LocalSubprocessAdapter.failure(output)
    assert failure is not None
    assert failure.category == FailureClass.SAFETY
    assert failure.retryable is False


def test_dispatcher_output_overflow_fails_closed(tmp_path):
    executable = str(Path(sys.executable).resolve())
    current = task("overflow", allowed_executables=(executable,), timeout_seconds=10, max_output_bytes=4096)
    result = Dispatcher(tmp_path).run(current, [executable, "-c", "print('B' * 100000)"])
    assert result.status == TaskStatus.FAILED
    assert result.failure is not None
    assert result.failure.category == FailureClass.SAFETY


def test_dispatcher_writes_audit_records_with_artifact_hashes(tmp_path):
    executable = str(Path(sys.executable).resolve())
    current = task("audited", allowed_executables=(executable,), timeout_seconds=5)
    result = Dispatcher(tmp_path).run(current, [executable, "-c", "print('ok')"])
    assert result.status == TaskStatus.SUCCEEDED
    log = tmp_path / ".orchestrator" / "audit-log.jsonl"
    assert log.exists()
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    executed = next(record for record in records if record["event"] == "command_executed")
    assert executed["policy"]["decision"] == "allowed"
    assert all(len(artifact["sha256"]) == 64 for artifact in executed["artifacts"])


def test_no_shell_true_in_adapter_source():
    from orchestrator import adapters
    assert "shell=True" not in Path(adapters.__file__).read_text()
