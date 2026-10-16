"""Deterministic /optimize policy tests (P1-05, remediation R3).

No server is started, no real optimize run occurs, no network access, and no
WAV fixtures are fabricated. Subprocess tests use short-lived local python
processes with bounded sleeps to verify timeout/kill behavior.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dsp_chart


class OptimizeProbeHandler(dsp_chart.ChartHandler):
    """Exercise /optimize decisions without binding a socket."""

    def __init__(self, path="/optimize", headers=None, peer="127.0.0.1"):
        self.path = path
        self.headers = headers or {}
        self.client_address = (peer, 51234)
        self.wfile = io.BytesIO()
        self.status = None

    def send_response(self, code, message=None, explain=None):
        self.status = code

    def send_header(self, *args):
        pass

    def end_headers(self):
        pass


VALID_TOKEN = "correct-token-0123456789abcdef"


@pytest.fixture
def audit_root(tmp_path, monkeypatch):
    root = tmp_path / "dsp_out"
    monkeypatch.setattr(dsp_chart, "OPTIMIZE_OUTPUT_ROOT", root)
    return root


def _audit_lines(root: Path) -> list[dict]:
    audit = root / "optimize-audit.jsonl"
    if not audit.exists():
        return []
    return [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]


def _body(handler: OptimizeProbeHandler) -> dict:
    return json.loads(handler.wfile.getvalue().decode())


# ---------- auth (fail closed) ----------

def test_optimize_disabled_without_token(monkeypatch, audit_root):
    monkeypatch.delenv("DSP_CHART_OPTIMIZE_TOKEN", raising=False)
    handler = OptimizeProbeHandler("/optimize?mode=generic")
    dsp_chart.ChartHandler.handle_optimize(handler)
    assert handler.status == 403
    assert _body(handler)["error"] == "forbidden"
    events = [record["event"] for record in _audit_lines(audit_root)]
    assert events == ["optimize_denied"]


def test_optimize_short_token_fails_closed(monkeypatch):
    monkeypatch.setenv("DSP_CHART_OPTIMIZE_TOKEN", "short")
    assert dsp_chart._optimize_authorized({}, ("127.0.0.1", 1)) is False


def test_optimize_wrong_token_rejected(monkeypatch, audit_root):
    monkeypatch.setenv("DSP_CHART_OPTIMIZE_TOKEN", VALID_TOKEN)
    handler = OptimizeProbeHandler(
        "/optimize?mode=generic",
        headers={"x-dsp-optimize-token": "wrong-token-0123456789abcdef"},
    )
    dsp_chart.ChartHandler.handle_optimize(handler)
    assert handler.status == 403
    assert _audit_lines(audit_root)[-1]["event"] == "optimize_denied"


def test_optimize_nonloopback_peer_rejected(monkeypatch, audit_root):
    monkeypatch.setenv("DSP_CHART_OPTIMIZE_TOKEN", VALID_TOKEN)
    handler = OptimizeProbeHandler(
        "/optimize?mode=generic",
        headers={"x-dsp-optimize-token": VALID_TOKEN},
        peer="192.168.1.9",
    )
    dsp_chart.ChartHandler.handle_optimize(handler)
    assert handler.status == 403
    assert _audit_lines(audit_root)[-1]["event"] == "optimize_denied"


def test_optimize_authorized_loopback_with_matching_token(monkeypatch):
    monkeypatch.setenv("DSP_CHART_OPTIMIZE_TOKEN", VALID_TOKEN)
    assert dsp_chart._optimize_authorized(
        {"x-dsp-optimize-token": VALID_TOKEN}, ("127.0.0.1", 1)
    ) is True


def test_optimize_bearer_header_accepted(monkeypatch):
    monkeypatch.setenv("DSP_CHART_OPTIMIZE_TOKEN", VALID_TOKEN)
    assert dsp_chart._optimize_authorized(
        {"authorization": f"Bearer {VALID_TOKEN}"}, ("::1", 1)
    ) is True


# ---------- strict schema ----------

def test_optimize_query_accepts_only_known_fields():
    with pytest.raises(ValueError):
        dsp_chart.validate_optimize_query({"mode": ["generic"], "token": ["x"]})
    with pytest.raises(ValueError):
        dsp_chart.validate_optimize_query({"path": ["/etc/passwd"]})


def test_optimize_query_mode_trials_bounds():
    with pytest.raises(ValueError):
        dsp_chart.validate_optimize_query({"mode": ["evil"]})
    for bad in ("abc", "1e3", "9999", "-1", ""):
        with pytest.raises(ValueError):
            dsp_chart.validate_optimize_query({"trials": [bad]})
    for out_of_range in ("0", "201"):
        with pytest.raises(ValueError):
            dsp_chart.validate_optimize_query({"trials": [out_of_range]})
    assert dsp_chart.validate_optimize_query({"trials": ["30"]})["trials"] == 30


def test_optimize_query_split_pair_rules():
    with pytest.raises(ValueError):
        dsp_chart.validate_optimize_query({"split": ["X|Y"]})
    with pytest.raises(ValueError):
        dsp_chart.validate_optimize_query({"pair": ["ABC"]})
    with pytest.raises(ValueError):
        dsp_chart.validate_optimize_query({"mode": ["diff"]})
    with pytest.raises(ValueError):
        dsp_chart.validate_optimize_query({"mode": ["sim"]})
    params = dsp_chart.validate_optimize_query(
        {"mode": ["diff"], "split": ["A|B"], "trials": ["5"]}
    )
    assert params == {"mode": "diff", "trials": 5, "split": "A|B", "pair": ""}


def test_optimize_handler_rejects_unknown_field_with_audit(monkeypatch, audit_root):
    monkeypatch.setenv("DSP_CHART_OPTIMIZE_TOKEN", VALID_TOKEN)
    handler = OptimizeProbeHandler(
        "/optimize?mode=generic&token=leaked-value",
        headers={"x-dsp-optimize-token": VALID_TOKEN},
    )
    dsp_chart.ChartHandler.handle_optimize(handler)
    assert handler.status == 400
    assert _body(handler)["error"] == "invalid_request"
    record = _audit_lines(audit_root)[-1]
    assert record["event"] == "optimize_rejected"
    assert record["reason"] == "[redacted]"
    assert "token" not in (audit_root / "optimize-audit.jsonl").read_text()


# ---------- concurrency ----------

def test_optimize_busy_returns_429(monkeypatch, audit_root):
    monkeypatch.setenv("DSP_CHART_OPTIMIZE_TOKEN", VALID_TOKEN)
    semaphore = threading.Semaphore(1)
    semaphore.acquire()
    monkeypatch.setattr(dsp_chart, "_OPTIMIZE_SEM", semaphore)
    handler = OptimizeProbeHandler(
        "/optimize?mode=generic&trials=5",
        headers={"x-dsp-optimize-token": VALID_TOKEN},
    )
    dsp_chart.ChartHandler.handle_optimize(handler)
    assert handler.status == 429
    assert _body(handler)["error"] == "busy"
    assert _audit_lines(audit_root)[-1]["event"] == "optimize_rejected"


# ---------- argv policy ----------

def test_build_optimize_argv_rejects_malformed_out_tag():
    params = {"mode": "generic", "trials": 5, "split": "", "pair": ""}
    for bad in ("../evil", "webopt_bad", "webopt_1_zzzzzzzz", "webopt_1_0000000x"):
        with pytest.raises(ValueError):
            dsp_chart.build_optimize_argv(params, bad)


def test_build_optimize_argv_pinned_elements(monkeypatch, tmp_path):
    fake_backend = tmp_path / "backend-python"
    fake_backend.write_text("", encoding="utf-8")
    monkeypatch.setattr(dsp_chart, "BACKEND_PY", str(fake_backend))
    params = {"mode": "diff", "trials": 7, "split": "A|B", "pair": ""}
    argv = dsp_chart.build_optimize_argv(params, "webopt_1_00000000")
    assert argv[0] == str(fake_backend.resolve())
    assert argv[1] == str(dsp_chart.OPTIMIZE_SCRIPT.resolve())
    assert argv[argv.index("--wavdir") + 1] == os.path.realpath(dsp_chart.WAVS)
    assert argv[argv.index("--labels") + 1] == dsp_chart.DEFAULT_LABELS
    assert argv[argv.index("--trials") + 1] == "7"
    assert argv[argv.index("--split") + 1] == "A|B"
    assert "--pair" not in argv
    assert all("\x00" not in item for item in argv)


def test_optimize_python_falls_back_to_current_interpreter(monkeypatch):
    monkeypatch.setattr(dsp_chart, "BACKEND_PY", "/nonexistent/backend/python")
    assert dsp_chart._optimize_python() == os.path.realpath(sys.executable)


def test_optimize_python_fails_closed_when_allowlist_unusable(monkeypatch):
    monkeypatch.setattr(dsp_chart, "BACKEND_PY", "/nonexistent/backend/python")
    monkeypatch.setattr(dsp_chart, "sys", SimpleNamespace(executable="/nonexistent/current/python"))
    with pytest.raises(PermissionError):
        dsp_chart._optimize_python()


def test_optimize_env_is_minimal_and_credential_free(monkeypatch):
    monkeypatch.setenv("DSP_CHART_OPTIMIZE_TOKEN", VALID_TOKEN)
    env = dsp_chart._optimize_env()
    assert set(env) <= {"PATH", "HOME", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE"}
    assert VALID_TOKEN not in env.values()


# ---------- bounded subprocess execution ----------

def test_run_optimize_subprocess_timeout_kills_and_reports():
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    returncode, output, timed_out, truncated = dsp_chart.run_optimize_subprocess(argv, timeout=1.0)
    assert timed_out is True
    assert truncated is False
    assert returncode == -1


def test_run_optimize_subprocess_output_cap(monkeypatch):
    monkeypatch.setattr(dsp_chart, "OPTIMIZE_MAX_OUTPUT_BYTES", 8192)
    argv = [sys.executable, "-c", "print('A' * 200000)"]
    returncode, output, timed_out, truncated = dsp_chart.run_optimize_subprocess(argv, timeout=30)
    assert truncated is True
    assert timed_out is False
    assert returncode != 0
    assert len(output) <= 8192


def test_run_optimize_subprocess_success_returns_output():
    argv = [sys.executable, "-c", "print('optimize-ok')"]
    returncode, output, timed_out, truncated = dsp_chart.run_optimize_subprocess(argv, timeout=30)
    assert returncode == 0
    assert "optimize-ok" in output
    assert timed_out is False
    assert truncated is False


# ---------- output confinement ----------

def test_resolve_report_success_inside_root(audit_root):
    tag = "webopt_1_00000000"
    report_dir = audit_root / tag
    report_dir.mkdir(parents=True)
    report = report_dir / "report.json"
    report.write_text("{}", encoding="utf-8")
    resolved = dsp_chart._resolve_report(tag)
    assert resolved == report.resolve()
    assert audit_root.resolve() in resolved.parents


def test_resolve_report_missing_file(audit_root):
    with pytest.raises(FileNotFoundError):
        dsp_chart._resolve_report("webopt_1_00000000")


def test_resolve_report_oversize_rejected(monkeypatch, audit_root):
    monkeypatch.setattr(dsp_chart, "OPTIMIZE_MAX_REPORT_BYTES", 8)
    report_dir = audit_root / "webopt_1_00000000"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text("0123456789", encoding="utf-8")
    with pytest.raises(PermissionError):
        dsp_chart._resolve_report("webopt_1_00000000")


def test_resolve_report_symlinked_file_rejected(audit_root, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    report_dir = audit_root / "webopt_1_00000000"
    report_dir.mkdir(parents=True)
    os.symlink(outside, report_dir / "report.json")
    with pytest.raises(PermissionError):
        dsp_chart._resolve_report("webopt_1_00000000")


def test_resolve_report_symlinked_parent_rejected(audit_root, tmp_path):
    audit_root.mkdir(parents=True, exist_ok=True)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "report.json").write_text("{}", encoding="utf-8")
    os.symlink(outside_dir, audit_root / "webopt_1_00000000")
    with pytest.raises(PermissionError):
        dsp_chart._resolve_report("webopt_1_00000000")


def test_resolve_report_traversal_tag_rejected(audit_root):
    with pytest.raises(PermissionError):
        dsp_chart._resolve_report("../escape")


# ---------- audit records ----------

def test_audit_records_events(audit_root):
    dsp_chart._audit_optimize(
        {"event": "optimize_denied", "reason": "auth_or_disabled", "peer": "127.0.0.1", "at": 1.0}
    )
    record = _audit_lines(audit_root)[-1]
    assert record["event"] == "optimize_denied"
    assert record["reason"] == "auth_or_disabled"


def test_audit_redacts_credential_values_but_keeps_event(audit_root):
    dsp_chart._audit_optimize(
        {"event": "optimize_rejected", "reason": "token=abc123", "peer": "127.0.0.1", "at": 1.0}
    )
    text = (audit_root / "optimize-audit.jsonl").read_text()
    assert "token=abc123" not in text
    assert "abc123" not in text
    assert "[redacted]" in text
    record = _audit_lines(audit_root)[-1]
    assert record["event"] == "optimize_rejected"


# ---------- static guarantees ----------

def test_no_shell_true_in_sources():
    for module in (dsp_chart,):
        assert "shell=True" not in Path(module.__file__).read_text()
    adapter_source = (ROOT / "orchestrator" / "adapters.py").read_text()
    assert "shell=True" not in adapter_source
