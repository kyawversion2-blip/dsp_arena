"""Local subprocess adapter. Remote runtimes are intentionally not implemented.

Environment filtering, output caps, and timeout kills are POLICY-ONLY controls
(application validation); no OS sandbox or OS-enforced rlimit is claimed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .schemas import Failure, FailureClass, TaskEnvelope

_POLL_INTERVAL_SECONDS = 0.25
_CREDENTIAL_ENV_RE = re.compile(
    r"(token|secret|password|passwd|credential|api[_-]?key|private[_-]?key)", re.I
)


@dataclass(frozen=True)
class CommandOutput:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    output_truncated: bool = False


class LocalSubprocessAdapter:
    """Run an argv command in a task workspace without shell expansion.

    Each command runs in its own process group; the whole group is killed on
    timeout or output-cap overflow. Captured stdout and stderr are each capped
    at task.execution.max_output_bytes.
    """

    name = "local_subprocess"

    def __init__(self, python_executable: str | None = None):
        self.python_executable = python_executable or sys.executable

    def _environment(self, task: TaskEnvelope) -> Mapping[str, str]:
        inherited = set(task.execution.inherit_environment)
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in inherited and not _CREDENTIAL_ENV_RE.search(key)
        }
        environment.update({"DSP_ARENA_TASK_ID": task.task_id, "DSP_ARENA_NETWORK_POLICY": task.execution.network})
        return environment

    def _validate_command(self, command: Sequence[str], task: TaskEnvelope) -> None:
        if not command or any("\x00" in item for item in command):
            raise ValueError("command must be a non-empty argv sequence")
        executable = Path(command[0]).resolve()
        allowed = tuple(Path(item).resolve() for item in task.execution.allowed_executables)
        if allowed and executable not in allowed:
            raise PermissionError(f"executable is outside declared allowlist: {command[0]}")
        if not allowed and command[0] != self.python_executable:
            raise PermissionError("command executable must be explicitly allowlisted")

    def run(self, task: TaskEnvelope, command: Sequence[str], workspace: Path) -> CommandOutput:
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        self._validate_command(command, task)
        max_output = max(1, int(task.execution.max_output_bytes))
        proc = subprocess.Popen(
            list(command),
            cwd=workspace,
            env=self._environment(task),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            start_new_session=True,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        truncated = threading.Event()

        def _pump(stream, sink: list[str]) -> None:
            total = 0
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    return
                total += len(chunk)
                if total > max_output:
                    truncated.set()
                    return
                sink.append(chunk)

        readers = [
            threading.Thread(target=_pump, args=(proc.stdout, stdout_chunks), daemon=True),
            threading.Thread(target=_pump, args=(proc.stderr, stderr_chunks), daemon=True),
        ]
        for reader in readers:
            reader.start()
        timed_out = False
        deadline = time.monotonic() + task.execution.timeout_seconds
        while True:
            try:
                proc.wait(timeout=_POLL_INTERVAL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
            if truncated.is_set():
                self._kill_group(proc)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                self._kill_group(proc)
                break
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill_group(proc)
        for reader in readers:
            reader.join(timeout=5)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        if timed_out:
            returncode = -1
        else:
            returncode = proc.returncode if proc.returncode is not None else -1
        return CommandOutput(
            tuple(command), returncode, "".join(stdout_chunks), "".join(stderr_chunks), timed_out, truncated.is_set()
        )

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()

    @staticmethod
    def failure(output: CommandOutput) -> Failure | None:
        if output.timed_out:
            return Failure(FailureClass.TIMEOUT, "subprocess exceeded task timeout", retryable=True)
        if output.output_truncated:
            return Failure(FailureClass.SAFETY, "subprocess output exceeded the configured cap", retryable=False)
        if output.returncode != 0:
            return Failure(FailureClass.PROVIDER, f"subprocess exited with code {output.returncode}", retryable=True)
        return None
