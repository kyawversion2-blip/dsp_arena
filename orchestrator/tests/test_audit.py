import json
from pathlib import Path

import pytest

from orchestrator.audit import audit, validate_audit_task, write_artifacts
from orchestrator.schemas import ExecutionPolicy, TaskEnvelope
from orchestrator.validator import ValidationError


def test_audit_detects_surfaces_and_preserves_evidence(tmp_path):
    source = tmp_path / "app.py"
    source.write_text("""\nfrom fastapi import UploadFile\nimport subprocess\napp.run(host='0.0.0.0', port=8080)\ndef run(value):\n    return subprocess.run([value], shell=False)\n""", encoding="utf-8")
    report = audit(tmp_path)
    by_category = {item.category: item for item in report.findings if item.status != "UNSUPPORTED"}
    assert by_category["network_binds"].line == 4
    assert by_category["uploads"].status == "OBSERVED"
    assert by_category["subprocess"].evidence
    assert report.read_only is True
    assert report.result["status"] == "SUCCEEDED"


def test_audit_marks_unmatched_surface_unsupported_not_vulnerable(tmp_path):
    (tmp_path / "README.md").write_text("No runtime implementation here.\n", encoding="utf-8")
    report = audit(tmp_path)
    finding = next(item for item in report.findings if item.category == "oauth")
    assert finding.status == "UNSUPPORTED"
    assert finding.severity == "info"
    assert "not a vulnerability" in finding.rationale


def test_artifacts_are_machine_and_human_readable(tmp_path):
    (tmp_path / "service.py").write_text("import os\nTOKEN = os.environ.get('TOKEN')\n", encoding="utf-8")
    report = audit(tmp_path)
    json_path, markdown_path = write_artifacts(report, tmp_path / "out")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "oci-safety-audit/v1"
    assert payload["read_only"] is True
    assert "## Findings" in markdown_path.read_text(encoding="utf-8")
    assert not (tmp_path / "service.py").read_text(encoding="utf-8").endswith("\n\n")


def test_validator_rejects_network_or_remote_execution():
    task = TaskEnvelope(
        "task/v1", "audit", "run", "P1", "independent_audit", "audit",
        execution=ExecutionPolicy(plane="oci", mode="read_only", network="enabled"),
    )
    with pytest.raises(ValidationError):
        validate_audit_task(task)
