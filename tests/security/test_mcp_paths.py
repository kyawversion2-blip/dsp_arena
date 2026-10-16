from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest

import path_policy
from path_policy import (
    PathPolicyError,
    validate_directory,
    validate_manifest_entries,
    validate_read_path,
    validate_write_path,
)


def _mcp():
    text = (ROOT / "mcp_server.py").read_text()
    namespace: dict = {"Path": Path, "__name__": "mcp_pathcheck", "__file__": str(ROOT / "mcp_server.py")}
    compiled = compile(text, str(ROOT / "mcp_server.py"), "exec")
    try:
        exec(compiled, namespace)
    except ModuleNotFoundError as exc:
        pytest.skip(f"mcp runtime dependency unavailable: {exc}")
    return namespace


def make_tree(tmp_path: Path) -> dict:
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    (wavs / "ok.wav").write_bytes(b"RIFF-TEST-DATA")
    (wavs / "unsafe name.wav").write_text("unsafe")
    (tmp_path / "labels.csv").write_text("File,Group\nok,A\n")
    (tmp_path / "large.csv").write_bytes(b"x" * 32)
    (tmp_path / "out").mkdir()
    return {"wavs": wavs, "out": tmp_path / "out"}


def test_traversal_absolute_hidden_and_case_paths(tmp_path):
    tree = make_tree(tmp_path)
    roots = (tree["wavs"],)
    for bad in ("/etc/passwd", "../secret.wav", "wavs/../secret.wav", "", ".hidden.wav", "C:\\evil.wav", "\\\\host\\share.wav"):
        with pytest.raises(PathPolicyError):
            validate_read_path(bad, allowed_roots=roots, extensions={".wav"}, maximum=1024, label="wav")
    with pytest.raises(PathPolicyError):
        validate_read_path("ok.WAV", allowed_roots=roots, extensions={".txt"}, maximum=1024, label="wav")
    assert validate_read_path("ok.wav", allowed_roots=roots, extensions={".wav"}, maximum=1024, label="wav") == tree["wavs"] / "ok.wav"


def test_symlink_escape_is_blocked(tmp_path):
    tree = make_tree(tmp_path)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    link = tree["wavs"] / "link.wav"
    link.symlink_to(outside)
    with pytest.raises(PathPolicyError):
        validate_read_path("link.wav", allowed_roots=(tree["wavs"],), extensions={".wav"}, maximum=1024, label="wav")
    (tree["wavs"] / "subdir").mkdir()
    nested = tree["wavs"] / "subdir" / "nested.wav"
    nested.write_bytes(b"RIFF")
    assert validate_read_path("subdir/nested.wav", allowed_roots=(tree["wavs"],), extensions={".wav"}, maximum=1024, label="wav") == nested


def test_size_cap_and_write_root_separation(tmp_path):
    tree = make_tree(tmp_path)
    with pytest.raises(PathPolicyError):
        validate_read_path("../labels.csv", allowed_roots=(tree["wavs"],), extensions={".csv"}, maximum=1024, label="csv")
    with pytest.raises(PathPolicyError):
        validate_read_path("labels.csv", allowed_roots=(tree["wavs"],), extensions={".csv"}, maximum=1024, label="csv")
    with pytest.raises(PathPolicyError):
        validate_read_path("large.csv", allowed_roots=(tmp_path,), extensions={".csv"}, maximum=8, label="csv")
    with pytest.raises(PathPolicyError):
        validate_write_path("../escape.csv", output_root=tree["out"], extensions={".csv"}, label="output")
    with pytest.raises(PathPolicyError):
        validate_write_path("/tmp/absolute.csv", output_root=tree["out"], extensions={".csv"}, label="output")
    with pytest.raises(PathPolicyError):
        validate_write_path("bad.exe", output_root=tree["out"], extensions={".csv"}, label="output")
    assert validate_write_path("report.csv", output_root=tree["out"], extensions={".csv"}, label="output") == tree["out"] / "report.csv"
    assert validate_directory("wavs", allowed_roots=(tmp_path,), label="wav_dir").resolve() == tree["wavs"].resolve()
    with pytest.raises(PathPolicyError):
        validate_directory("/etc", allowed_roots=(tmp_path,), label="wav_dir")


def test_manifest_schema_validation():
    assert validate_manifest_entries([{"file": "a.wav", "group": "b"}]) == [{"file": "a.wav", "group": "B"}]
    for bad in ([{"file": "../escape.wav"}], [{"file": ".hidden.wav"}], [{"file": "a.wav", "group": "Z"}], [{"file": "a.wav"}, {"file": "a.wav"}], "not-a-list", [{"file": "a.mp3"}]):
        with pytest.raises(PathPolicyError):
            validate_manifest_entries(bad)


def test_mcp_and_chart_use_policy():
    mcp = _mcp()
    live = (ROOT / "mcp_server.py").read_text()
    assert "validate_read_path" in live and "validate_write_path" in live and "validate_directory" in live
    assert "Absolute path to WAV file" not in live
    assert "MCP_OUTPUT_ROOT" in live
    chart = (ROOT / "dsp_chart.py").read_text()
    assert "validate_read_path" in chart and "validate_write_path" in chart and "validate_manifest_entries" in chart
    assert 'Access-Control-Allow-Origin", "*"' not in chart
    assert 'clear requires --yes confirmation' in chart
    assert path_policy.MAX_WAV_BYTES > 0
    assert mcp["_validate_mcp_dir"]("wavs").resolve() == (ROOT / "wavs").resolve()
    assert mcp["_process_wav"]({"wav_path": "/etc/passwd"})["ok"] is False
    assert mcp["_process_wav"]({"wav_path": "../wavs/evil.wav"})["ok"] is False
    assert mcp["_classify_wav"]({"wav_path": "/etc/passwd", "training_csv": "labels.csv"})["ok"] is False
    assert mcp["_get_separation"]({"csv_path": "/etc/passwd"})["ok"] is False
    assert mcp["_batch_process"]({"wav_dir": "/etc", "output_csv": "../escape.csv"})["ok"] is False
