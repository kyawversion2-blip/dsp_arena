"""Artifact and subprocess-output collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import ArtifactRef
from .validator import sha256_file, validate_artifact_paths


class ArtifactCollector:
    def __init__(self, artifact_root: Path):
        self.artifact_root = artifact_root.resolve()

    def collect_file(self, task_id: str, path: Path, artifact_id: str, media_type: str = "application/octet-stream", producer_task_id: str = "") -> ArtifactRef:
        path = path.resolve()
        task_root = (self.artifact_root / task_id).resolve()
        task_root.mkdir(parents=True, exist_ok=True)
        validate_artifact_paths((path,), path.parent)
        destination = task_root / artifact_id
        destination.write_bytes(path.read_bytes())
        return ArtifactRef(artifact_id, str(destination), sha256_file(destination), media_type, True, producer_task_id or task_id)

    def write_text(self, task_id: str, artifact_id: str, content: str, media_type: str = "text/plain") -> ArtifactRef:
        task_root = (self.artifact_root / task_id).resolve()
        task_root.mkdir(parents=True, exist_ok=True)
        destination = task_root / artifact_id
        destination.write_text(content, encoding="utf-8")
        return ArtifactRef(artifact_id, str(destination), sha256_file(destination), media_type, True, task_id)

    def write_json(self, task_id: str, artifact_id: str, value: object) -> ArtifactRef:
        return self.write_text(task_id, artifact_id, json.dumps(value, indent=2, sort_keys=True) + "\n", "application/json")
