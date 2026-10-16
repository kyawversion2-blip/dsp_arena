from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import dsp_chart
from path_policy import PathPolicyError, validate_manifest_entries


class ProbeHandler(dsp_chart.ChartHandler):
    """Exercise route/path decisions without binding a socket."""
    def __init__(self, path: str):
        self.path = path
        self.wfile = io.BytesIO()
        self.responses = []

    def send_error(self, code, message=None, explain=None):
        self.responses.append((code, message, explain))

    def send_response(self, code, message=None):
        self.responses.append((code, message, None))

    def send_header(self, *args):
        pass

    def end_headers(self):
        pass


def test_chart_safe_route_parser_rejects_traversal_and_unsafe_extension():
    h = ProbeHandler("/")
    assert h._safe_relative_path("/") == "index.html"
    assert h._safe_relative_path("/wavs/sample.wav") == "wavs/sample.wav"
    with pytest.raises(PathPolicyError):
        h._safe_relative_path("//evil.example/file.wav")
    for bad in ("../index.html", "wavs/../index.html", ".git/config", "live_server.py"):
        from path_policy import validate_read_path, MAX_WAV_BYTES
        with pytest.raises(PathPolicyError):
            validate_read_path(
                bad,
                allowed_roots=dsp_chart.CHART_READ_ROOTS,
                extensions=dsp_chart.CHART_READ_EXTENSIONS,
                maximum=MAX_WAV_BYTES,
                label="chart path",
            )


def test_chart_manifest_and_clear_are_fail_closed(monkeypatch, tmp_path):
    assert validate_manifest_entries([{"file": "ok.wav", "group": "A"}]) == [{"file": "ok.wav", "group": "A"}]
    with pytest.raises(PathPolicyError):
        validate_manifest_entries([{"file": "../escape.wav"}])
    with pytest.raises(SystemExit) as exc:
        dsp_chart.cmd_clear(SimpleNamespace(yes=False))
    assert exc.value.code == 2


def test_chart_static_boundary_and_no_wildcard_cors():
    source = (ROOT / "dsp_chart.py").read_text()
    assert 'Access-Control-Allow-Origin", "*"' not in source
    assert "validate_read_path" in source
    assert "validate_write_path" in source
    assert "validate_manifest_entries" in source
    assert "requires --yes" in source
    assert "directory=DIR" in source  # fallback server root is guarded by pre-validation
