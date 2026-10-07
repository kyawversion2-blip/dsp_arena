"""Argument-level command policy for dispatcher-run subprocesses.

POLICY-ONLY: these are application-level validation checks, not an OS sandbox;
no OS-enforced isolation is claimed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .schemas import TaskEnvelope

FORBIDDEN_EXECUTABLES = frozenset({
    "git", "curl", "wget", "ssh", "scp", "sftp", "nc", "ncat", "netcat",
    "telnet", "ftp", "tftp", "rsync", "ping", "nmap", "sudo", "su", "doas",
    "rm", "dd", "shred", "mkfs",
})

_CREDENTIAL_ARGUMENT_RE = re.compile(
    r"(token|secret|password|passwd|credential|api[_-]?key|private[_-]?key)", re.I
)


class CommandPolicyError(PermissionError):
    """A command or argument violates the dispatch policy."""


def redact(value: object) -> object:
    """Recursively replace credential-shaped strings for logs and audit records."""
    if isinstance(value, str):
        return "[redacted]" if _CREDENTIAL_ARGUMENT_RE.search(value) else value
    if isinstance(value, dict):
        return {redact(key): redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def _contained(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def evaluate_command(command: Sequence[str], task: TaskEnvelope, workspace: Path) -> dict[str, object]:
    """Validate every argv element and return the policy decision record.

    Raises CommandPolicyError when the command is denied. Absolute-path
    arguments (including the value side of ``--flag=/abs/path``) must stay
    inside the task workspace or a declared allowed_path root.
    """
    if not command or any("\x00" in item for item in command):
        raise CommandPolicyError("command must be a non-empty argv sequence without NUL bytes")
    executable = Path(command[0])
    if executable.name.lower() in FORBIDDEN_EXECUTABLES:
        raise CommandPolicyError("forbidden executable")
    workspace = workspace.resolve()
    allowed_roots = [workspace] + [Path(item).resolve() for item in task.execution.allowed_paths]
    for argument in command[1:]:
        if _CREDENTIAL_ARGUMENT_RE.search(argument):
            raise CommandPolicyError("argument matches a credential pattern")
        for piece in argument.split("="):
            if piece.startswith("/"):
                resolved = Path(piece).resolve()
                if not _contained(resolved, allowed_roots):
                    raise CommandPolicyError("argument path escapes workspace and allowlist")
    return {"decision": "allowed", "executable": str(executable)}
