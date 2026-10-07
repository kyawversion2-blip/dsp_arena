from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relationships import freeze_table
from replay_manifest import (
    features_csv_file_sha256,
    features_csv_semantic_sha256,
    verify_replay_manifest,
)


FEATURES = [
    "impact",
    "duration",
    "peak_time",
    "mel_mean_db",
    "impact_freq_hz",
]


def _write_features(path: Path) -> str:
    header = "file,group," + ",".join(FEATURES) + "\n"
    rows = [
        "11.wav,A,0.5000012,0.3000008,0.2500001,-42.1234567,3200.1234567\n",
        "12.wav,B,0.5100012,0.3100008,0.2600001,-41.9876543,3500.1234567\n",
        "13.wav,C,0.5200012,0.3200008,0.2700001,-43.1111111,3100.1234567\n",
    ]
    path.write_text(header + "".join(rows), encoding="utf-8")
    return header + "".join(rows)


def _make_manifest(features_path: Path) -> dict:
    return {
        "schema_version": 2,
        "run_id": "p5-replay",
        "frozen": True,
        "replay": {
            "wavdir_manifest_hash": "not-verified-in-this-scope",
            "features_csv_sha256": features_csv_semantic_sha256(features_path),
            "pipeline_code_hash": "git:test",
        },
    }


def test_replay_manifest_e2e_is_deterministic_and_passes(tmp_path):
    features_a = tmp_path / "a" / "features.csv"
    features_b = tmp_path / "b" / "features.csv"
    features_a.parent.mkdir()
    features_b.parent.mkdir()
    _write_features(features_a)
    _write_features(features_b)
    manifest_a = tmp_path / "a" / "relationships.json"
    manifest_b = tmp_path / "b" / "relationships.json"
    table = _make_manifest(features_a)
    manifest_hash_a = freeze_table(table, str(manifest_a))
    manifest_hash_b = freeze_table(table, str(manifest_b))
    file_hash_a = features_csv_file_sha256(features_a)
    file_hash_b = features_csv_file_sha256(features_b)

    assert manifest_hash_a == manifest_hash_b
    assert file_hash_a == file_hash_b
    result = verify_replay_manifest(
        manifest_a,
        features_a,
        expected_manifest_sha256=manifest_hash_a,
        expected_features_file_sha256=file_hash_a,
        expected_pipeline_code_hash="git:test",
    )
    assert result["status"] == "PASS"
    assert result["scope"] == "deterministic_manifest_hash_only"
    assert result["checks"]["manifest_sha256"]["status"] == "PASS"
    assert result["checks"]["features_csv_semantic_sha256"]["status"] == "PASS"
    assert result["checks"]["features_csv_file_sha256"]["status"] == "PASS"
    assert result["checks"]["wavdir_manifest_hash"]["status"] == "SKIPPED"


def test_replay_manifest_detects_feature_mutation(tmp_path):
    features = tmp_path / "features.csv"
    original = _write_features(features)
    manifest = tmp_path / "relationships.json"
    table = _make_manifest(features)
    manifest_hash = freeze_table(table, str(manifest))
    file_hash = features_csv_file_sha256(features)
    features.write_text(original.replace("0.5000012", "0.6000012"), encoding="utf-8")

    result = verify_replay_manifest(
        manifest,
        features,
        expected_manifest_sha256=manifest_hash,
        expected_features_file_sha256=file_hash,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["features_csv_semantic_sha256"]["status"] == "FAIL"
    assert result["checks"]["features_csv_file_sha256"]["status"] == "FAIL"


def test_replay_manifest_detects_manifest_mutation(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)
    manifest = tmp_path / "relationships.json"
    table = _make_manifest(features)
    manifest_hash = freeze_table(table, str(manifest))
    mutated = json.loads(manifest.read_text(encoding="utf-8"))
    mutated["run_id"] = "mutated"
    manifest.write_text(json.dumps(mutated, indent=2), encoding="utf-8")

    result = verify_replay_manifest(
        manifest,
        features,
        expected_manifest_sha256=manifest_hash,
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["manifest_sha256"]["status"] == "FAIL"


def test_replay_manifest_missing_features_file_fails_cleanly(tmp_path):
    features = tmp_path / "features.csv"
    _write_features(features)
    table = _make_manifest(features)
    manifest = tmp_path / "relationships.json"
    freeze_table(table, str(manifest))
    features.unlink()

    result = verify_replay_manifest(manifest, features)
    assert result["status"] == "FAIL"
    assert result["checks"]["features_csv_semantic_sha256"]["status"] == "FAIL"
    assert result["checks"]["features_csv_file_sha256"]["status"] == "FAIL"
