"""Dependency-aware task graph operations."""

from __future__ import annotations

from dataclasses import replace

from .schemas import TaskEnvelope, TaskStatus


class GraphError(ValueError):
    """Raised when a task graph is malformed."""


class TaskGraph:
    def __init__(self, tasks: list[TaskEnvelope] | tuple[TaskEnvelope, ...]):
        self._tasks = {task.task_id: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise GraphError("task IDs must be unique")
        self.validate()

    def validate(self) -> None:
        for task in self._tasks.values():
            missing = set(task.dependencies) - self._tasks.keys()
            if missing:
                raise GraphError(f"{task.task_id} references missing tasks: {sorted(missing)}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise GraphError("task graph contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in self._tasks[task_id].dependencies:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in self._tasks:
            visit(task_id)

    def get(self, task_id: str) -> TaskEnvelope:
        return self._tasks[task_id]

    def tasks(self) -> tuple[TaskEnvelope, ...]:
        return tuple(self._tasks.values())

    def ready(self) -> tuple[TaskEnvelope, ...]:
        ready = []
        for task in self._tasks.values():
            if task.status not in (TaskStatus.PENDING, TaskStatus.READY):
                continue
            if all(self._tasks[item].status == TaskStatus.SUCCEEDED for item in task.dependencies):
                ready.append(replace(task, status=TaskStatus.READY))
        return tuple(ready)

    def update_status(self, task_id: str, status: TaskStatus) -> "TaskGraph":
        updated = [replace(task, status=status) if task.task_id == task_id else task for task in self.tasks()]
        return TaskGraph(updated)
