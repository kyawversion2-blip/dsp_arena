"""Read-only static safety audit for application and OCI-adjacent surfaces."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .schemas import (
    Claim,
    EvidenceLabel,
    ResultEnvelope,
    TaskStatus,
    TaskEnvelope,
    to_json_dict,
)
from .validator import ValidationError, validate_task

CATEGORIES = (
    "network_binds", "cors", "uploads", "filesystem", "oauth", "secrets",
    "subprocess", "mcp_surfaces", "git_mutation", "oci_mutation_deployment",
)
SEVERITIES = ("critical", "high", "medium", "low", "info")
LABELS = {item.value for item in EvidenceLabel}
TEXT_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".yaml", ".yml", ".json", ".toml", ".ini", ".conf", ".md", ".sh"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".orchestrator", "node_modules", "graphify-out", "dsp_out", "recordings"}


@dataclass(frozen=True)
class AuditFinding:
    category: str
    severity: str
    status: str
    evidence: str
    file: str
    line: int
    rationale: str
    remediation: str
    verification_method: str
    control: str = "POLICY_ONLY"

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown audit category: {self.category}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity: {self.severity}")
        if self.status not in LABELS:
            raise ValueError(f"unknown evidence status: {self.status}")
        if self.control not in {"ENFORCED", "POLICY_ONLY"}:
            raise ValueError(f"unknown control type: {self.control}")
        if not self.evidence or not self.rationale or not self.remediation or not self.verification_method:
            raise ValueError("audit findings require evidence, rationale, remediation, and verification")


@dataclass(frozen=True)
class AuditReport:
    schema_version: str
    root: str
    read_only: bool
    findings: tuple[AuditFinding, ...] = field(default_factory=tuple)
    scanned_files: tuple[str, ...] = field(default_factory=tuple)
    result: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root": self.root,
            "read_only": self.read_only,
            "scanned_files": list(self.scanned_files),
            "findings": [asdict(item) for item in self.findings],
            "result": self.result,
        }


def _files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def _line_for(text: str, pattern: str) -> tuple[int, str] | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for number, line in enumerate(text.splitlines(), 1):
        if regex.search(line):
            return number, line.strip()[:300]
    return None


def _finding(category: str, severity: str, status: str, evidence: str, path: Path | None, line: int, rationale: str, remediation: str, verification: str, control: str = "POLICY_ONLY") -> AuditFinding:
    return AuditFinding(category, severity, status, evidence, str(path) if path is not None else "", line, rationale, remediation, verification, control)


def validate_audit_task(task: TaskEnvelope) -> None:
    """Reject contracts that could authorize remote or mutating audit work."""
    validate_task(task)
    if task.execution.plane != "local":
        raise ValidationError("safety audit execution plane must be local")
    if task.execution.mode != "read_only":
        raise ValidationError("safety audit mode must be read_only")
    if task.execution.network != "disabled":
        raise ValidationError("safety audit network must be disabled")
    if task.approval_required:
        raise ValidationError("safety audit must not imply mutation approval")


def validate_audit_report(report: AuditReport) -> None:
    if report.schema_version != "oci-safety-audit/v1" or not report.read_only:
        raise ValidationError("invalid or non-read-only audit report")
    for finding in report.findings:
        if finding.status == "UNSUPPORTED" and finding.severity != "info":
            raise ValidationError("unsupported findings must use info severity")
        if finding.status != "UNSUPPORTED" and (not finding.file or finding.line < 1):
            raise ValidationError("supported findings require file and line evidence")


def _source_findings(root: Path, path: Path, text: str) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    rel = path.relative_to(root)

    def add(category: str, severity: str, status: str, match: tuple[int, str] | None, rationale: str, remediation: str, verification: str, control: str = "POLICY_ONLY") -> None:
        if match:
            findings.append(_finding(category, severity, status, match[1], rel, match[0], rationale, remediation, verification, control))

    add("network_binds", "medium", "OBSERVED", _line_for(text, r"(?:0\.0\.0\.0|host\s*=\s*[\"']localhost[\"']|--host\b)"), "A wildcard or localhost bind is present in source; exposure depends on the serving network and proxy configuration.", "Bind production services to an explicit private interface and document ingress allowlists.", "Run the deployed process configuration audit and inspect socket listeners with an approved read-only command.")
    add("network_binds", "medium", "OBSERVED", _line_for(text, r"(?:uvicorn\.run|app\.run|listen\s*\(|port\s*=|--port\b)"), "A service port or listener is configured in source; static inspection cannot establish reachability.", "Document intended exposure and enforce ingress rules at the deployment boundary.", "Verify the deployment manifest and listener/ingress state without changing it.")
    add("cors", "medium", "OBSERVED", _line_for(text, r"(?:CORSMiddleware|Access-Control-Allow-Origin|allow_origins|cors\s*=)"), "CORS configuration is present and must be checked for origin and credential restrictions.", "Use an explicit origin allowlist; reject wildcard origins when credentials are allowed.", "Exercise preflight and credentialed requests against a test deployment.")
    add("uploads", "medium", "OBSERVED", _line_for(text, r"(?:UploadFile|multipart/form-data|File\s*\(|\.read\(\)|content_length|MAX_UPLOAD|upload)"), "An upload-related surface is present; path, size, and content validation require endpoint-level review.", "Require bounded streaming size checks, generated destination names, and content sniffing plus extension allowlists.", "Run focused endpoint tests with traversal names, oversized bodies, mismatched content, and polyglots.")
    add("filesystem", "medium", "OBSERVED", _line_for(text, r"(?:Path\s*\(|open\s*\(|read_text|write_text|send_file|FileResponse|\.resolve\(\))"), "Filesystem access is present; safety depends on canonicalization and an allowlist at every user-controlled path boundary.", "Canonicalize input, reject traversal and unexpected absolute paths, and enforce an allowlist after resolution.", "Test encoded traversal, symlinks, absolute paths, and paths outside the permitted root.")
    add("oauth", "medium", "OBSERVED", _line_for(text, r"(?:oauth|OAuth|authorization|callback|PKCE|code_verifier|state|redirect_uri|innerHTML|localStorage)"), "OAuth or browser-auth terms are present; static evidence alone does not prove state, PKCE, origin, escaping, or token safety.", "Require server-side state validation, PKCE, exact redirect URI matching, contextual escaping, and short-lived protected token storage.", "Complete an authorization-code integration test including CSRF, callback-origin, token-redaction, and XSS cases.")
    add("secrets", "medium", "OBSERVED", _line_for(text, r"(?:os\.environ|getenv\(|SECRET|TOKEN|API_KEY|credential|\.env|password)"), "Credential or environment access is present; leakage depends on storage, logging, and response paths.", "Use narrowly scoped environment access, secret managers, redacted logs, and never return credentials in responses or artifacts.", "Scan logs, responses, artifacts, and process environments in a controlled test run.")
    add("subprocess", "medium", "OBSERVED", _line_for(text, r"(?:subprocess|os\.system|Popen\(|shell\s*=|exec\()"), "Subprocess or execution capability is present; command construction and executable restrictions require review.", "Use argv execution with shell disabled, a strict executable allowlist, bounded timeout, safe working directory, and sanitized environment.", "Unit-test shell metacharacters, executable allowlist rejection, timeout, cwd, stdin, and environment filtering.")
    add("mcp_surfaces", "medium", "OBSERVED", _line_for(text, r"(?:MCP|mcp|filesystem|read_file|write_file|list_directory)"), "MCP or filesystem-tool vocabulary is present; tool registration and path restrictions require review.", "Expose only explicit read operations by default, canonicalize paths, and enforce workspace/allowlist boundaries before every tool call.", "Invoke each declared tool with traversal, absolute, symlink, and write-operation cases under a deny-by-default harness.")
    add("git_mutation", "medium", "OBSERVED", _line_for(text, r"(?:git\s+(?:commit|reset|clean|checkout|push|merge)|subprocess.*git)"), "Git command capability is present; static evidence does not establish whether mutation is reachable from application input.", "Keep audit execution read-only and deny mutation verbs in the adapter or policy layer.", "Attempt each mutation through the task surface and verify denial plus unchanged HEAD/worktree.", "POLICY_ONLY")
    add("oci_mutation_deployment", "medium", "OBSERVED", _line_for(text, r"(?:oci|oracle cloud|terraform|kubectl|docker|podman|deploy|ssh|scp)"), "OCI or deployment vocabulary is present; no conclusion about remote access is made from vocabulary alone.", "Separate OCI adapters from audit tasks and require explicit approval plus a mutation denylist for deployment operations.", "Review manifests and perform a separately approved dry-run/read-only validation; do not infer remote state from source.")
    return findings


def _enforced_findings(root: Path, texts: dict[Path, str]) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for path, text in texts.items():
        rel = path.relative_to(root)
        if "shell=False" in text and "subprocess.run" in text:
            line = _line_for(text, r"shell\s*=\s*False")
            findings.append(_finding("subprocess", "info", "OBSERVED", line[1], rel, line[0], "The local adapter explicitly disables shell expansion for its subprocess call.", "Retain shell=False and test the invariant when changing the adapter.", "Inspect adapter tests and verify shell metacharacters are not interpreted.", "ENFORCED"))
        if "allowed_executables" in text and "PermissionError" in text:
            line = _line_for(text, r"allowed_executables|PermissionError")
            findings.append(_finding("subprocess", "info", "OBSERVED", line[1], rel, line[0], "The local adapter validates commands against an executable allowlist or its default interpreter.", "Retain deny-by-default executable validation.", "Run the allowlist rejection test with an unlisted executable.", "ENFORCED"))
        if "git" in text and "commit" in text and "never" in text.lower():
            line = _line_for(text, r"git|commit")
            findings.append(_finding("git_mutation", "info", "OBSERVED", line[1], rel, line[0], "The orchestrator documentation states that dispatcher operations do not mutate Git.", "Keep mutation commands unavailable to audit tasks.", "Run a read-only task and compare HEAD and worktree status before and after.", "ENFORCED"))
    return findings


def audit(root: str | Path) -> AuditReport:
    root_path = Path(root).resolve()
    texts = {path: path.read_text(encoding="utf-8", errors="replace") for path in _files(root_path)}
    findings: list[AuditFinding] = []
    for path, text in texts.items():
        findings.extend(_source_findings(root_path, path, text))
    findings.extend(_enforced_findings(root_path, texts))
    observed_categories = {item.category for item in findings}
    for category in CATEGORIES:
        if category not in observed_categories:
            findings.append(_finding(category, "info", "UNSUPPORTED", "No static evidence matched the audit rule set.", None, 0, "Absence of a match is not a vulnerability finding and does not establish runtime safety.", "Review the deployed or configured surface if this capability exists outside the scanned source.", "Confirm scope with runtime/deployment inventory and add a focused test if the surface exists."))
    findings.sort(key=lambda item: (CATEGORIES.index(item.category), item.file, item.line, item.severity))
    task = TaskEnvelope("task/v1", "p1-oci-safety-audit", "audit", "P1", "independent_audit", "Static read-only OCI safety audit")
    validate_audit_task(task)
    claims = tuple(Claim(EvidenceLabel(item.status), f"{item.category}: {item.rationale}", {"file": item.file, "line": item.line}) for item in findings if item.status != "UNSUPPORTED")
    result = ResultEnvelope("result/v1", task.task_id, task.run_id, TaskStatus.SUCCEEDED, {"read_only": True, "network": "not used"}, claims=claims, next_action="Review confirmed risks and perform approved runtime verification")
    structured_result = to_json_dict(result)
    structured_result.update({
        "confirmed_risks": [asdict(item) for item in findings if item.status == "OBSERVED" and item.severity in {"critical", "high"}],
        "policy_only_limitations": [asdict(item) for item in findings if item.control == "POLICY_ONLY" and item.status != "UNSUPPORTED"],
        "unknowns_requiring_runtime_verification": [asdict(item) for item in findings if item.status in {"INFERRED", "HYPOTHESIS", "UNSUPPORTED"}],
        "next_recommended_action": result.next_action,
    })
    report = AuditReport("oci-safety-audit/v1", str(root_path), True, tuple(findings), tuple(str(path.relative_to(root_path)) for path in texts), structured_result)
    validate_audit_report(report)
    return report


def write_artifacts(report: AuditReport, output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "oci-safety-audit.json"
    markdown_path = destination / "oci-safety-audit.md"
    json_path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# OCI Safety Audit", "", "- Schema: `oci-safety-audit/v1`", f"- Root: `{report.root}`", "- Read-only: `true`", "", "## Structured Result", "", f"- Confirmed risks: `{len(report.result['confirmed_risks'])}`", f"- Policy-only limitations: `{len(report.result['policy_only_limitations'])}`", f"- Runtime unknowns: `{len(report.result['unknowns_requiring_runtime_verification'])}`", f"- Next recommended action: {report.result['next_recommended_action']}", "", "## Findings", ""]
    for item in report.findings:
        location = f"{item.file}:{item.line}" if item.file else "(no matching file)"
        lines.extend([f"### {item.category} — {item.severity} — {item.status} — {item.control}", f"- Evidence: `{item.evidence}`", f"- File/line: `{location}`", f"- Rationale: {item.rationale}", f"- Remediation: {item.remediation}", f"- Verification: {item.verification_method}", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a read-only static OCI safety audit")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--output-dir", default="audit-artifacts")
    args = parser.parse_args()
    report = audit(args.root)
    write_artifacts(report, args.output_dir)
    print(json.dumps({"schema_version": report.schema_version, "findings": len(report.findings), "read_only": report.read_only}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
