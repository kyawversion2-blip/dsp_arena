"""Tests for the Graphify context provider (R1-complementary, uses fakes only)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator import graphify as mod
from orchestrator.graph import TaskGraph
from orchestrator.schemas import TaskEnvelope, TaskStatus


def make_fake_run(monkeypatch, stdout="NODE x [src=foo.py]", returncode=0, stderr="", side_effect=None, capture=None):
    calls = {}

    def fake_run(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        capture["argv"] = list(argv)
        capture["shell"] = shell
        capture["timeout"] = timeout
        capture["stdin"] = stdin
        if side_effect is not None:
            raise side_effect
        class R:
            def __init__(self):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        return R()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    return capture


def provider(tmp_path, monkeypatch, **kwargs):
    exe = tmp_path / "graphify"
    exe.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    # provider checks exists()
    return mod.GraphifyContextProvider(graphify_executable=exe, cwd=tmp_path, **kwargs)


def test_argv_shell_false(monkeypatch, tmp_path):
    capture = {}
    p = provider(tmp_path, monkeypatch)
    fake_stdout = "NODE x [src=a.py]"
    # need exe to exist: already created by provider
    def fake_run(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        capture["argv"] = list(argv)
        capture["shell"] = shell
        capture["timeout"] = timeout
        capture["stdin"] = stdin
        class R:
            returncode = 0
            stdout = fake_stdout
            stderr = ""
        return R()
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    ctx = p.query("how does X work", budget=50)
    assert capture["shell"] is False
    assert capture["argv"][0] == str(p.executable)
    assert "--budget" in capture["argv"]
    assert capture["argv"][capture["argv"].index("--budget") + 1] == "50"
    assert capture["stdin"] == subprocess.DEVNULL
    assert ctx.output == fake_stdout


def test_budget_bounded_and_default(monkeypatch, tmp_path):
    p = provider(tmp_path, monkeypatch, default_budget=123, max_budget=500)
    capture = {}
    def fake_run(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        class R:
            returncode = 0
            stdout = "NODE x"
            stderr = ""
        capture["argv"] = list(argv)
        return R()
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    ctx = p.query("q")
    assert ctx.budget == 123
    assert capture["argv"][capture["argv"].index("--budget") + 1] == "123"
    p.query("q", budget=400)
    # explicit override works
    with pytest.raises(mod.GraphifyBudgetError):
        p.query("q", budget=0)
    with pytest.raises(mod.GraphifyBudgetError):
        p.query("q", budget=9999)
    # empty question rejected
    with pytest.raises(mod.GraphifyBudgetError):
        p.query("  ")


def test_timeout_enforced(monkeypatch, tmp_path):
    p = provider(tmp_path, monkeypatch, default_timeout_seconds=0.2)
    def fake_run(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        assert timeout == pytest.approx(0.2)
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(mod.GraphifyError, match="timed out"):
        p.query("q")


def test_unavailable_controlled_failure(tmp_path):
    # exe does not exist
    p = mod.GraphifyContextProvider(graphify_executable=tmp_path / "missing-graphify", cwd=tmp_path)
    with pytest.raises(mod.GraphifyError, match="not found"):
        p.query("q", budget=10)
    # enrich with required=False returns skipped, not raise
    enr = p.enrich("t1", "q", required=False)
    assert enr.context is None
    assert enr.skipped_reason is not None
    # required=True must raise
    with pytest.raises(mod.GraphifyError):
        p.enrich("t1", "q", required=True)


def test_malformed_output_controlled_failure(monkeypatch, tmp_path):
    p = provider(tmp_path, monkeypatch)
    # empty output
    def fake_run_empty(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        class R:
            returncode = 0
            stdout = "   \n"
            stderr = ""
        return R()
    monkeypatch.setattr(mod.subprocess, "run", fake_run_empty)
    with pytest.raises(mod.GraphifyError, match="empty output"):
        p.query("q")
    # non-zero exit
    def fake_run_err(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        class R:
            returncode = 2
            stdout = ""
            stderr = "graph.json missing"
        return R()
    monkeypatch.setattr(mod.subprocess, "run", fake_run_err)
    with pytest.raises(mod.GraphifyError, match="exit code 2"):
        p.query("q")


def test_output_truncation(monkeypatch, tmp_path):
    p = provider(tmp_path, monkeypatch, max_output_bytes=10)
    big = "x" * 100
    def fake_run(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        class R:
            returncode = 0
            stdout = big
            stderr = ""
        return R()
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    ctx = p.query("q")
    assert ctx.truncated is True
    assert "[truncated" in ctx.output


def test_cannot_mutate_taskgraph(monkeypatch, tmp_path):
    p = provider(tmp_path, monkeypatch)
    def fake_run(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        class R:
            returncode = 0
            stdout = "NODE x"
            stderr = ""
        return R()
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    t1 = TaskEnvelope("task/v1", "a", "run-1", "P0", "integration_validation", "obj a")
    t2 = TaskEnvelope("task/v1", "b", "run-1", "P0", "integration_validation", "obj b", dependencies=("a",))
    graph = TaskGraph([t1, t2])
    # Query does not touch graph at all
    enr = p.enrich("a", "q about a")
    assert enr.task_id == "a"
    # graph unchanged
    assert graph.get("a").status == TaskStatus.PENDING
    assert graph.get("b").status == TaskStatus.PENDING


def test_optional_by_default_and_existing_behavior_intact(tmp_path):
    p = mod.GraphifyContextProvider(graphify_executable=tmp_path / "missing", cwd=tmp_path)
    enr = p.enrich("t1", None)
    assert enr.context is None
    assert enr.skipped_reason == "no question provided"
    enr2 = p.enrich("t1", "   ")
    assert enr2.context is None
    # existing orchestrator: graph still usable without graphify
    g = TaskGraph([TaskEnvelope("task/v1", "x", "r", "P0", "integration_validation", "obj")])
    assert g.ready()[0].task_id == "x"


def test_no_network_dependency_and_context_untrusted(monkeypatch, tmp_path):
    p = provider(tmp_path, monkeypatch)
    def fake_run(argv, cwd=None, stdin=None, stdout=None, stderr=None, text=None, timeout=None, shell=None, check=None):  # noqa: A002
        # Ensure no URL/network flags leak into argv
        assert not any("http" in a for a in argv)
        assert "--budget" in argv
        class R:
            returncode = 0
            stdout = "NODE x"
            stderr = ""
        return R()
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    ctx = p.query("q", budget=50)
    assert ctx.is_untrusted_context is True
    # context_filter validated
    with pytest.raises(mod.GraphifyBudgetError):
        p.query("q", context_filter=[""])
    # valid context_filter
    p2 = provider(tmp_path, monkeypatch)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    ctx2 = p2.query("q", context_filter=["call"])
    assert "--context" in ctx2.argv


def test_smoke_real_cli_if_available():
    exe = mod.DEFAULT_GRAPHIFY_EXECUTABLE
    if not exe.exists():
        pytest.skip("local Graphify CLI not installed")
    # Do not assert graph existence; just that provider detects availability
    assert mod.GraphifyContextProvider.check_available(exe) is True
    # Optional: try a real query but allow failure if graph.json missing
    p = mod.GraphifyContextProvider(graphify_executable=exe, cwd=Path.cwd(), default_timeout_seconds=5.0)
    try:
        ctx = p.query("TaskEnvelope", budget=10)
        assert ctx.available is True
    except mod.GraphifyError as exc:
        # Graph may be missing in this cwd; still a controlled failure
        assert "failed" in str(exc).lower() or "empty" in str(exc).lower() or "exit code" in str(exc).lower()
