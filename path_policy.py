"""Fail-closed filesystem/path policy for DSP Arena R2.

These are application-level POLICY-ONLY checks, not an OS sandbox.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

class PathPolicyError(ValueError):
    """Caller-selected path violates the approved R2 boundary."""

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_WAV_BYTES = 200 * 1024 * 1024
MAX_CSV_BYTES = 50 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_FILENAME_LEN = 128
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def require_relative(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PathPolicyError(f"{label}: empty path")
    text = value.strip()
    if "\x00" in text or text.startswith("~"):
        raise PathPolicyError(f"{label}: invalid path")
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\") or os.path.isabs(text):
        raise PathPolicyError(f"{label}: absolute paths are not allowed")
    return text


def _parts(value: str) -> list[str]:
    parts = re.split(r"[\\/]+", value)
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise PathPolicyError(f"path traversal or empty segment: {value!r}")
    for part in parts:
        if len(part) > MAX_FILENAME_LEN or part.startswith(".") or not _SAFE.fullmatch(part):
            raise PathPolicyError(f"unsafe path segment: {part!r}")
    return parts


def validate_basename(name: str, extensions: set[str], *, label: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise PathPolicyError(f"{label}: empty filename")
    name = name.strip()
    if "/" in name or "\\" in name or len(name) > MAX_FILENAME_LEN or not _SAFE.fullmatch(name):
        raise PathPolicyError(f"{label}: unsafe filename")
    if os.path.splitext(name)[1].lower() not in extensions:
        raise PathPolicyError(f"{label}: extension is not allowed")
    return name


def _resolve(root: Path, relative: str, *, label: str) -> Path:
    parts = _parts(require_relative(relative, label=label))
    root = root.resolve()
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise PathPolicyError(f"{label}: symlinks are not allowed")
    candidate = current.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise PathPolicyError(f"{label}: escapes approved root") from None
    if candidate.is_symlink():
        raise PathPolicyError(f"{label}: symlink target is not allowed")
    return candidate


def _size(path: Path, maximum: int, *, label: str) -> None:
    if path.exists() and path.stat().st_size > maximum:
        raise PathPolicyError(f"{label}: exceeds size cap")


def validate_read_path(caller_path: str, *, allowed_roots: tuple[Path, ...], extensions: set[str], maximum: int, label: str) -> Path:
    text = require_relative(caller_path, label=label)
    parts = _parts(text)
    validate_basename(parts[-1], extensions, label=label)
    for root in allowed_roots:
        try:
            path = _resolve(root, text, label=label)
        except PathPolicyError:
            continue
        if not path.exists():
            continue
        if path.is_dir():
            raise PathPolicyError(f"{label}: directory is not readable")
        _size(path, maximum, label=label)
        return path
    raise PathPolicyError(f"{label}: outside approved roots or file not found")


def validate_write_path(caller_path: str, *, output_root: Path, extensions: set[str], label: str) -> Path:
    text = require_relative(caller_path, label=label)
    parts = _parts(text)
    validate_basename(parts[-1], extensions, label=label)
    path = _resolve(output_root, text, label=label)
    if path.exists() and path.is_dir():
        raise PathPolicyError(f"{label}: directory is not writable")
    return path


def validate_manifest_entries(entries: object) -> list[dict]:
    if not isinstance(entries, list):
        raise PathPolicyError("manifest: expected a list")
    result: list[dict] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            raise PathPolicyError("manifest: entry must be an object")
        filename = validate_basename(item.get("file", ""), {".wav"}, label="manifest.file")
        if filename in seen:
            raise PathPolicyError("manifest: duplicate file")
        seen.add(filename)
        entry = {"file": filename}
        if item.get("group") not in (None, ""):
            group = str(item["group"]).strip().upper()
            if group not in {"A", "B", "C", "D"}:
                raise PathPolicyError("manifest: invalid group")
            entry["group"] = group
        result.append(entry)
    return result


def validate_directory(caller_path: str, *, allowed_roots: tuple[Path, ...], label: str) -> Path:
    text = require_relative(caller_path, label=label)
    for root in allowed_roots:
        try:
            path = _resolve(root, text, label=label)
        except PathPolicyError:
            continue
        if not path.is_dir():
            raise PathPolicyError(f"{label}: directory not found")
        return path
    raise PathPolicyError(f"{label}: outside approved roots")
