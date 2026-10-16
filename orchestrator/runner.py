"""CLI entry point for dependency-aware local task orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import LocalSubprocessAdapter
from .dispatcher import Dispatcher
from .schemas import TaskEnvelope, TaskStatus, task_from_dict, to_json_dict


def load_plan(path: Path) -> tuple[tuple[TaskEnvelope, ...], dict[str, tuple[str, ...]]]:
    """Load a JSON plan containing immutable task envelopes and argv commands."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("plan must be a JSON object with a tasks array")
    raw_commands = payload.get("commands", {})
    if not isinstance(raw_commands, dict):
        raise ValueError("commands must be an object mapping task IDs to argv arrays")
    tasks = tuple(task_from_dict(item) for item in payload["tasks"])
    commands: dict[str, tuple[str, ...]] = {}
    for task_id, argv in raw_commands.items():
        if not isinstance(task_id, str) or not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
            raise ValueError("each command must map a task ID to a string argv array")
        commands[task_id] = tuple(argv)
    return tasks, commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="JSON task graph plan")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository root")
    parser.add_argument("--max-parallelism", type=int, default=4)
    parser.add_argument(
        "--runtime-alias", action="append", default=[],
        help="explicitly register a CLI runtime name to the local subprocess adapter (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and record tasks without running commands")
    args = parser.parse_args(argv)

    tasks, commands = load_plan(args.plan)
    runtime_adapters = {name: LocalSubprocessAdapter() for name in args.runtime_alias}
    results = Dispatcher(args.root, dry_run=args.dry_run, runtime_adapters=runtime_adapters).run_graph(
        tasks, commands, max_parallelism=args.max_parallelism
    )
    print(json.dumps({task_id: to_json_dict(result) for task_id, result in results.items()}, indent=2))
    return 0 if all(result.status == TaskStatus.SUCCEEDED for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
