"""Deterministic replay manifest/hash verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping

from relationships import load_frozen


FEATURE_FIELDS = (
    "impact",
    "duration",
    "peak_time",
    "mel_mean_db",
    "impact_freq_hz",
)
REQUIRED_REPLAY_FIELDS = (
    "wavdir_manifest_hash",
    "features_csv_sha256",
    "pipeline_code_hash",
)
VERIFICATION_SCOPE = "deterministic_manifest_hash_only"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def features_csv_semantic_sha256(path: Path | str) -> str:
    csv_path = Path(path)
    values: list[str] = []
    row_count = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("features CSV has no header")
        missing = [field for field in FEATURE_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"features CSV missing fields: {', '.join(missing)}")
        for row in reader:
            row_count += 1
            for field in FEATURE_FIELDS:
                raw = (row.get(field) or "").strip()
                if not raw:
                    values.append("")
                    continue
                try:
                    number = float(raw)
                except ValueError:
                    values.append(raw)
                else:
                    if not math.isfinite(number):
                        raise ValueError(f"features CSV contains non-finite {field}")
                    values.append(f"{number:.6f}")
    if row_count == 0:
        raise ValueError("features CSV has no data rows")
    return _sha256_bytes("|".join(values).encode("utf-8"))


def features_csv_file_sha256(path: Path | str) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def wavdir_manifest_hash(path: Path | str) -> str:
    directory = Path(path)
    if not directory.is_dir():
        return "missing"
    files = sorted(
        (entry.name for entry in directory.iterdir() if entry.name.lower().endswith(".wav")),
        key=str,
    )
    parts: list[str] = []
    for filename in files:
        file_path = directory / filename
        try:
            size = file_path.stat().st_size
        except OSError:
            size = -1
        try:
            file_hash = _sha256_bytes(file_path.read_bytes())
        except OSError:
            file_hash = "?"
        parts.append(f"{filename}:{size}:{file_hash}")
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def _check(status: str, expected: Any = None, actual: Any = None, detail: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {"status": status}
    if expected is not None:
        result["expected"] = expected
    if actual is not None:
        result["actual"] = actual
    if detail:
        result["detail"] = detail
    return result


def verify_replay_manifest(
    manifest_path: Path | str,
    features_csv_path: Path | str,
    *,
    expected_manifest_sha256: str | None = None,
    expected_features_file_sha256: str | None = None,
    wavdir: Path | str | None = None,
    expected_pipeline_code_hash: str | None = None,
) -> Dict[str, Any]:
    """Verify only manifest/file hashes; no as-of or pipeline-state claims."""
    checks: Dict[str, Dict[str, Any]] = {}
    try:
        manifest = load_frozen(str(manifest_path))
    except Exception as exc:
        checks["manifest_load"] = _check("FAIL", detail=str(exc))
        return {"scope": VERIFICATION_SCOPE, "status": "FAIL", "checks": checks}

    schema_version = manifest.get("schema_version")
    checks["schema_version"] = (
        _check("PASS", 2, schema_version)
        if schema_version == 2
        else _check("FAIL", 2, schema_version)
    )
    frozen = manifest.get("frozen")
    checks["frozen"] = _check("PASS", True, frozen) if frozen is True else _check("FAIL", True, frozen)
    run_id = manifest.get("run_id")
    checks["run_id"] = (
        _check("PASS", actual=run_id)
        if isinstance(run_id, str) and run_id
        else _check("FAIL", detail="run_id must be a non-empty string")
    )

    replay = manifest.get("replay")
    if not isinstance(replay, Mapping):
        checks["replay_fields"] = _check("FAIL", detail="replay must be an object")
        replay = {}
    else:
        missing = [field for field in REQUIRED_REPLAY_FIELDS if not isinstance(replay.get(field), str) or not replay.get(field)]
        checks["replay_fields"] = (
            _check("PASS", actual=list(REQUIRED_REPLAY_FIELDS))
            if not missing
            else _check("FAIL", detail=f"missing or invalid replay fields: {', '.join(missing)}")
        )

    if expected_manifest_sha256 is None:
        checks["manifest_sha256"] = _check("SKIPPED", detail="expected_manifest_sha256 not supplied")
    else:
        actual_manifest_hash = _sha256_bytes(Path(manifest_path).read_bytes())
        checks["manifest_sha256"] = (
            _check("PASS", expected_manifest_sha256, actual_manifest_hash)
            if actual_manifest_hash == expected_manifest_sha256
            else _check("FAIL", expected_manifest_sha256, actual_manifest_hash)
        )

    recorded_features_semantic = replay.get("features_csv_sha256")
    try:
        actual_features_semantic = features_csv_semantic_sha256(features_csv_path)
    except Exception as exc:
        checks["features_csv_semantic_sha256"] = _check("FAIL", detail=str(exc))
    else:
        checks["features_csv_semantic_sha256"] = (
            _check("PASS", recorded_features_semantic, actual_features_semantic)
            if actual_features_semantic == recorded_features_semantic
            else _check("FAIL", recorded_features_semantic, actual_features_semantic)
        )

    recorded_features_file = expected_features_file_sha256 or replay.get("features_csv_file_sha256")
    try:
        actual_features_file = features_csv_file_sha256(features_csv_path)
    except Exception as exc:
        checks["features_csv_file_sha256"] = _check("FAIL", detail=str(exc))
    else:
        if recorded_features_file is None:
            checks["features_csv_file_sha256"] = _check(
                "SKIPPED",
                actual=actual_features_file,
                detail="expected file hash not supplied",
            )
        else:
            checks["features_csv_file_sha256"] = (
                _check("PASS", recorded_features_file, actual_features_file)
                if actual_features_file == recorded_features_file
                else _check("FAIL", recorded_features_file, actual_features_file)
            )

    if wavdir is None:
        checks["wavdir_manifest_hash"] = _check("SKIPPED", detail="wavdir not supplied")
    else:
        recorded_wavdir_hash = replay.get("wavdir_manifest_hash")
        try:
            actual_wavdir_hash = wavdir_manifest_hash(wavdir)
        except Exception as exc:
            checks["wavdir_manifest_hash"] = _check("FAIL", detail=str(exc))
        else:
            checks["wavdir_manifest_hash"] = (
                _check("PASS", recorded_wavdir_hash, actual_wavdir_hash)
                if actual_wavdir_hash == recorded_wavdir_hash
                else _check("FAIL", recorded_wavdir_hash, actual_wavdir_hash)
            )

    if expected_pipeline_code_hash is None:
        checks["pipeline_code_hash"] = _check("SKIPPED", detail="expected_pipeline_code_hash not supplied")
    else:
        recorded_pipeline_hash = replay.get("pipeline_code_hash")
        checks["pipeline_code_hash"] = (
            _check("PASS", expected_pipeline_code_hash, recorded_pipeline_hash)
            if recorded_pipeline_hash == expected_pipeline_code_hash
            else _check("FAIL", expected_pipeline_code_hash, recorded_pipeline_hash)
        )

    failed = any(check.get("status") == "FAIL" for check in checks.values())
    return {
        "scope": VERIFICATION_SCOPE,
        "status": "FAIL" if failed else "PASS",
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a deterministic replay manifest and feature hashes.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-features-file-sha256")
    parser.add_argument("--wavdir", type=Path)
    parser.add_argument("--expected-pipeline-code-hash")
    args = parser.parse_args(argv)
    result = verify_replay_manifest(
        args.manifest,
        args.features,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_features_file_sha256=args.expected_features_file_sha256,
        wavdir=args.wavdir,
        expected_pipeline_code_hash=args.expected_pipeline_code_hash,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
