"""Immutable task, result, evidence, and artifact contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class EvidenceLabel(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    HYPOTHESIS = "HYPOTHESIS"
    UNSUPPORTED = "UNSUPPORTED"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"
    CANCELLED = "CANCELLED"


class FailureClass(str, Enum):
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    MALFORMED_RESULT = "malformed_result"
    DEPENDENCY = "dependency"
    SAFETY = "safety"
    MISSING_DATA = "missing_data"
    STATISTICAL_INCONCLUSIVE = "statistical_inconclusive"
    PROVIDER = "provider"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class InputRef:
    path: str
    sha256: str


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    path: str
    sha256: str
    media_type: str = "application/octet-stream"
    immutable: bool = True
    producer_task_id: str = ""


@dataclass(frozen=True)
class ExecutionPolicy:
    plane: str = "local"
    mode: str = "read_only"
    network: str = "disabled"
    workspace: str = ""
    # Generous defaults: reviewers must tighten per task; unreviewed commands
    # must never rely on them.
    timeout_seconds: float = 900.0
    max_retries: int = 0
    allowed_executables: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    inherit_environment: tuple[str, ...] = ("PATH", "HOME", "LANG", "LC_ALL")
    max_output_bytes: int = 10 * 1024 * 1024


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retryable_failures: tuple[FailureClass, ...] = (
        FailureClass.TRANSIENT,
        FailureClass.TIMEOUT,
        FailureClass.PROVIDER,
    )


@dataclass(frozen=True)
class TaskEnvelope:
    schema_version: str
    task_id: str
    run_id: str
    phase: str
    role: str
    objective: str
    dependencies: tuple[str, ...] = ()
    inputs: tuple[InputRef, ...] = ()
    allowed_runtimes: tuple[str, ...] = ()
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    required_artifacts: tuple[str, ...] = ()
    validator: str = ""
    approval_required: bool = False
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    status: TaskStatus = TaskStatus.PENDING


@dataclass(frozen=True)
class Claim:
    label: EvidenceLabel
    claim: str
    source: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Evidence:
    kind: str
    command: str
    exit_code: int
    stdout_artifact: str = ""
    stderr_artifact: str = ""


@dataclass(frozen=True)
class Failure:
    category: FailureClass
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class ResultEnvelope:
    schema_version: str
    task_id: str
    run_id: str
    status: TaskStatus
    runtime: Mapping[str, Any] = field(default_factory=dict)
    claims: tuple[Claim, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    failure: Failure | None = None
    next_action: str = ""


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def to_json_dict(value: Any) -> dict[str, Any]:
    result = _json_value(value)
    if not isinstance(result, dict):
        raise TypeError("expected a dataclass or mapping")
    return result


def task_from_dict(data: Mapping[str, Any]) -> TaskEnvelope:
    execution = ExecutionPolicy(**data.get("execution", {}))
    retry_data = data.get("retry", {})
    retry = RetryPolicy(
        max_attempts=int(retry_data.get("max_attempts", 1)),
        retryable_failures=tuple(FailureClass(item) for item in retry_data.get("retryable_failures", [item.value for item in RetryPolicy().retryable_failures])),
    )
    return TaskEnvelope(
        schema_version=str(data["schema_version"]),
        task_id=str(data["task_id"]),
        run_id=str(data["run_id"]),
        phase=str(data["phase"]),
        role=str(data["role"]),
        objective=str(data["objective"]),
        dependencies=tuple(data.get("dependencies", ())),
        inputs=tuple(InputRef(**item) for item in data.get("inputs", ())),
        allowed_runtimes=tuple(data.get("allowed_runtimes", ())),
        execution=execution,
        required_artifacts=tuple(data.get("required_artifacts", ())),
        validator=str(data.get("validator", "")),
        approval_required=bool(data.get("approval_required", False)),
        retry=retry,
        status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
    )
