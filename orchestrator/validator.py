"""Contract validators for task graphs, results, artifacts, and policies."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from .schemas import Claim, EvidenceLabel, ResultEnvelope, TaskEnvelope, TaskStatus

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(ValueError):
    """Raised when an orchestration contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256(value: str) -> None:
    if not SHA256_RE.fullmatch(value):
        raise ValidationError("sha256 must be 64 lowercase hexadecimal characters")


def validate_task(task: TaskEnvelope) -> None:
    if not task.task_id or not task.run_id or not task.objective:
        raise ValidationError("task_id, run_id, and objective are required")
    if task.execution.network not in {"disabled", "enabled", "policy_only"}:
        raise ValidationError("execution.network must be disabled, enabled, or policy_only")
    if task.execution.timeout_seconds <= 0:
        raise ValidationError("timeout_seconds must be positive")
    if task.retry.max_attempts < 1 or task.retry.max_attempts > 10:
        raise ValidationError("retry.max_attempts must be between 1 and 10")
    for item in task.inputs:
        validate_sha256(item.sha256)


def validate_claim(claim: Claim) -> None:
    if not isinstance(claim.label, EvidenceLabel):
        raise ValidationError("claim label is not an allowed evidence label")
    if not claim.claim.strip():
        raise ValidationError("claim text is required")
    if claim.label != EvidenceLabel.UNSUPPORTED and not claim.source:
        raise ValidationError(f"{claim.label.value} claims require a source")


def validate_result(result: ResultEnvelope, task: TaskEnvelope | None = None) -> None:
    if task is not None and (result.task_id != task.task_id or result.run_id != task.run_id):
        raise ValidationError("result task identity does not match task envelope")
    if result.status == TaskStatus.SUCCEEDED and result.failure is not None:
        raise ValidationError("successful result cannot contain a failure")
    for claim in result.claims:
        validate_claim(claim)
    for artifact in result.artifacts:
        validate_sha256(artifact.sha256)
        if not artifact.immutable:
            raise ValidationError(f"artifact is not immutable: {artifact.artifact_id}")


def validate_artifact_paths(paths: Iterable[Path], workspace: Path, allowed_paths: Iterable[Path] = ()) -> None:
    workspace = workspace.resolve()
    allowed = tuple(item.resolve() for item in allowed_paths)
    for path in paths:
        resolved = path.resolve()
        if resolved == workspace or workspace in resolved.parents:
            continue
        if any(resolved == item or item in resolved.parents for item in allowed):
            continue
        raise ValidationError(f"path escapes workspace and allowlist: {path}")
