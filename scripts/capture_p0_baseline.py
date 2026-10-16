#!/usr/bin/env python3
"""Capture repository baseline evidence without changing application inputs."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / ".gsd" / "evidence"
DEPENDENCIES = (
    "pytest", "numpy", "scipy", "librosa", "pandas", "optuna",
    "soundfile", "scikit-learn", "matplotlib", "fastapi", "uvicorn",
)
COMMANDS = {
    "git_head": ["git", "rev-parse", "HEAD"],
    "git_branch": ["git", "branch", "--show-current"],
    "git_status": ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
    "git_diff_check": ["git", "diff", "--check"],
    "orchestrator_tests": [sys.executable, "-m", "pytest", "orchestrator/tests", "-v", "--tb=short", "-p", "no:cacheprovider"],
    # Security tests are deliberately excluded until the user authorizes the final security stage.
    "functional_tests": [sys.executable, "-m", "pytest", "tests", "--ignore=tests/security", "-v", "--tb=short", "-p", "no:cacheprovider"],
}


def run_command(argv):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        result = subprocess.run(argv, cwd=ROOT, env=environment, capture_output=True, text=True, errors="replace")
        return {"command": argv, "cwd": str(ROOT), "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except OSError as error:
        return {"command": argv, "cwd": str(ROOT), "exit_code": None,
                "stdout": "", "stderr": str(error)}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory():
    paths = [ROOT / "labels.csv", ROOT / "ai_model_evaluated_wav_sessions.csv"]
    for directory in (ROOT / "wavs", ROOT / "dsp_out"):
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file() and path.name != ".DS_Store")
    files = {}
    for path in sorted(paths):
        if path.is_file():
            try:
                files[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            except OSError as error:
                files[str(path.relative_to(ROOT))] = {"error": str(error)}
    wavs = ROOT / "wavs"
    external_audio = Path.home() / "Downloads" / "audio"
    external_wav_files = {}
    if external_audio.is_dir():
        for path in sorted(external_audio.iterdir()):
            if path.is_file() and path.suffix.lower() == ".wav":
                try:
                    external_wav_files[path.name] = {
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                except OSError as error:
                    external_wav_files[path.name] = {"error": str(error)}
    return {
        "wavs_directory": wavs.is_dir(),
        "wav_count": sum(1 for path in wavs.rglob("*") if path.is_file() and path.suffix.lower() == ".wav") if wavs.is_dir() else 0,
        "labels_available": (ROOT / "labels.csv").is_file(),
        "dsp_out_directory": (ROOT / "dsp_out").is_dir(),
        "files": files,
        "external_audio_directory": str(external_audio),
        "external_audio_available": external_audio.is_dir(),
        "external_audio_wav_count": len(external_wav_files),
        "external_audio_wavs": external_wav_files,
    }


def classify_tests(result, missing_wavs, full_suite=False):
    if result["exit_code"] == 0:
        return {"status": "VERIFIED", "reason": "pytest exited successfully"}
    output = result["stdout"] + result["stderr"]
    failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", output, re.MULTILINE)
    missing_fixture_tests = (
        "test_curriculum_consume.py::test_manual_edge_preserved_and_provenance",
        "test_curriculum_consume.py::test_V_min_and_guard_formula_read_from_prior",
        "test_curriculum_consume.py::test_frozen_features_sha256_deterministic",
        "test_quiz_determinism.py::test_measure_is_deterministic",
        "test_quiz_determinism.py::test_talbot_stretch_formula_invariants",
        "test_quiz_determinism.py::test_reveal_does_not_mutate_source_rows",
    )
    if full_suite and missing_wavs and failures and all(
        failure in missing_fixture_tests for failure in failures
    ) and output.count("no rows measured") >= len(failures):
        return {"status": "INCONCLUSIVE", "reason": "failures limited to WAV-dependent tests with absent wavs/; inspect raw output", "failed_nodes": failures}
    return {"status": "FAILED", "reason": "pytest failed or could not run; inspect raw output", "failed_nodes": failures}


def main():
    now = datetime.now(timezone.utc)
    captured = {name: run_command(argv) for name, argv in COMMANDS.items() if not name.endswith("tests")}
    data = inventory()
    dependencies = {}
    for name in DEPENDENCIES:
        try:
            dependencies[name] = {"status": "VERIFIED", "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = {"status": "INCONCLUSIVE", "reason": "not installed in selected interpreter"}
    captured["orchestrator_tests"] = run_command(COMMANDS["orchestrator_tests"])
    captured["functional_tests"] = run_command(COMMANDS["functional_tests"])
    classifications = {
        "git": "VERIFIED" if all(captured[name]["exit_code"] == 0 for name in ("git_head", "git_branch", "git_status")) else "FAILED",
        "worktree_diff_check": "VERIFIED" if captured["git_diff_check"]["exit_code"] == 0 else "INCONCLUSIVE",
        "dependencies": "VERIFIED" if all(item["status"] == "VERIFIED" for item in dependencies.values()) else "INCONCLUSIVE",
        "dataset": "VERIFIED" if data["wav_count"] and data["labels_available"] else "INCONCLUSIVE",
        "orchestrator_tests": classify_tests(captured["orchestrator_tests"], data["wav_count"] == 0),
        "functional_tests": classify_tests(captured["functional_tests"], data["wav_count"] == 0, full_suite=True),
    }
    if any(isinstance(value, dict) and value["status"] == "FAILED" or value == "FAILED" for value in classifications.values()):
        overall = "FAILED"
    elif any(isinstance(value, dict) and value["status"] == "INCONCLUSIVE" or value == "INCONCLUSIVE" for value in classifications.values()):
        overall = "INCONCLUSIVE"
    else:
        overall = "VERIFIED"
    evidence = {
        "schema": "p0.1-baseline/v1", "captured_at_utc": now.isoformat(),
        "capture_command": [sys.executable, "scripts/capture_p0_baseline.py"],
        "cwd": str(ROOT), "python": {"version": sys.version, "executable": sys.executable},
        "os": {"platform": platform.platform(), "uname": list(platform.uname())},
        "scope": {"security_tests": "DEFERRED_BY_USER", "security_tests_run": False},
        "commands": captured, "dependencies": dependencies, "dataset": data,
        "classification": classifications, "overall": overall,
        "observed": ["Command outputs, dependency versions, and file hashes are captured above."],
        "inferred": ["Absent WAV files limit WAV-dependent test conclusions; existing DSP artifacts are not substitutes for source data."],
        "hypothesis": [],
        "limitations": ["Package availability is determined for the selected Python interpreter, not every system environment.",
                        "The external Downloads/audio corpus is inventoried separately and is not treated as the labelled experiment corpus without label/session reconciliation.",
                        "Security tests are intentionally deferred to the final security stage by user request.",
                        "Tests may create temporary files as defined by existing tests; the capture script does not modify source data."],
    }
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    destination = EVIDENCE_DIR / f"P0.1-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with destination.open("x", encoding="utf-8") as output:
        json.dump(evidence, output, indent=2, ensure_ascii=True)
        output.write("\n")
    print(f"Evidence: {destination.relative_to(ROOT)}")
    print(f"Classification: {overall}")
    for name, value in classifications.items():
        print(f"{name}: {value['status'] if isinstance(value, dict) else value}")
    return 1 if overall == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
