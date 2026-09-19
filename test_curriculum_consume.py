"""Curriculum consume tests (D1).

Validates relationships-baseline --prior merge semantics:
- manual entries preserved verbatim (immutable type/tau_gh/weight)
- auto entries re-inferred on fresh dm (single code path)
- V_min/lambda/guard_formula read FROM prior.collapse_guard (A1)
- manual edges PROTECTED: collapse_gate never removes them (A2)
- determinism: same --wavdir + --prior -> same replay.features_csv_sha256
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
VENV_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), ".venv", "bin", "python")
if not os.path.exists(VENV_PY):
    VENV_PY = sys.executable  # fallback


def _approved_with_manual_fixture(tmp_path, prior_path):
    """Build an approved JSON fixture where pair BC is marked manual."""
    approved = json.load(open(prior_path))
    for r in approved["relationships"]:
        if r["pair"] == "BC":
            r["source"] = "manual"
            r["type"] = "attract"   # immutable manual type
            r["weight"] = 2.0       # immutable manual weight
    approved["collapse_guard"]["V_min"] = 0.005
    approved["collapse_guard"]["lambda_vmin"] = 2000.0
    approved["collapse_guard"]["guard_formula"] = "normalized_hinge_v1"
    out = os.path.join(tmp_path, "approved_manual.json")
    json.dump(approved, open(out, "w"), indent=2)
    return out


def _run_cli(out_tag, prior=None):
    _cleanup(out_tag)
    cmd = [
        VENV_PY, "dsp_inverse.py", "relationships-baseline",
        "--wavdir", "wavs", "--labels", "labels.csv", "--out", out_tag,
    ]
    if prior:
        cmd += ["--prior", prior]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE, timeout=300)
    assert r.returncode == 0, f"CLI failed:\n{r.stdout}\n{r.stderr}"
    return r


def _load(out_tag):
    return json.load(open(os.path.join(HERE, "dsp_out", out_tag, "relationships.json")))


def _cleanup(tag):
    import shutil
    d = os.path.join(HERE, "dsp_out", tag)
    if os.path.isdir(d):
        shutil.rmtree(d)


@pytest.fixture(autouse=True)
def _clean_tags():
    """Remove any stale artifact dirs from prior runs before each test."""
    for tag in ("testcur_manual", "testcur_guard", "testcur_detA",
                "testcur_detB", "testcur_noprior", "testcur_protected"):
        _cleanup(tag)
    yield
    for tag in ("testcur_manual", "testcur_guard", "testcur_detA",
                "testcur_detB", "testcur_noprior", "testcur_protected"):
        _cleanup(tag)


def test_manual_edge_preserved_and_provenance(tmp_path):
    prior = os.path.join(HERE, "dsp_out", "run1", "relationships.json")
    approved = _approved_with_manual_fixture(tmp_path, prior)
    _run_cli("testcur_manual", prior=approved)
    t = _load("testcur_manual")

    # provenance block
    assert t["provenance"]["prior_run_id"] == "run1"
    assert t["provenance"]["merged_manual"] == 1
    assert t["provenance"]["re_inferred_auto"] == 5
    assert t["provenance"]["prior_sha"] is not None

    # manual BC edge preserved verbatim
    bc = next(r for r in t["relationships"] if r["pair"] == "BC")
    assert bc["source"] == "manual"
    assert bc["type"] == "attract"           # immutable manual type
    assert bc["weight"] == 2.0               # immutable manual weight
    assert "d_after" in bc                  # fresh distance stamped


def test_V_min_and_guard_formula_read_from_prior(tmp_path):
    prior = os.path.join(HERE, "dsp_out", "run1", "relationships.json")
    approved = _approved_with_manual_fixture(tmp_path, prior)  # sets V_min=0.005
    _run_cli("testcur_guard", prior=approved)
    t = _load("testcur_guard")
    g = t["collapse_guard"]
    assert g["V_min"] == 0.005               # read from prior, NOT default 0.01
    assert g["lambda_vmin"] == 2000.0         # read from prior
    assert g["guard_formula"] == "normalized_hinge_v1"


def test_manual_edge_protected_from_collapse(tmp_path):
    """A manual attractor below V_min must NOT be dropped by collapse_gate."""
    import relationships as R
    rels = [
        {"pair": "AB", "type": "attract", "d_baseline": 0.05, "tau_gh": 0.1,
         "weight": 1.0, "source": "manual"},
        {"pair": "AC", "type": "attract", "d_baseline": 0.05, "tau_gh": 0.1,
         "weight": 1.0, "source": "auto"},
        {"pair": "BC", "type": "attract", "d_baseline": 0.05, "tau_gh": 0.1,
         "weight": 1.0, "source": "auto"},
        {"pair": "AD", "type": "neutral", "d_baseline": 0.9, "source": "auto"},
        {"pair": "BD", "type": "neutral", "d_baseline": 0.9, "source": "auto"},
        {"pair": "CD", "type": "neutral", "d_baseline": 0.9, "source": "auto"},
    ]
    dm = {"AB": 0.05, "AC": 0.05, "BC": 0.05, "AD": 0.9, "BD": 0.9, "CD": 0.9}
    # with BC protected, it must survive even if d < V_min
    pruned_p, removed_p = R.collapse_gate(rels, dm, anchor=None, protected={"BC"})
    bc_kept = any(r["pair"] == "BC" and r["type"] == "attract" for r in pruned_p)
    assert bc_kept, "protected manual edge BC must survive collapse_gate"
    assert "BC" not in removed_p, "BC must not appear in removed list"


def test_frozen_features_sha256_deterministic(tmp_path):
    """Determinism gate (per spec): same --wavdir + --prior -> same sha.
    Also: --prior replay matches a fresh (no --prior) baseline over the same wavs."""
    prior = os.path.join(HERE, "dsp_out", "run1", "relationships.json")
    _run_cli("testcur_detA", prior=prior)
    _run_cli("testcur_detB", prior=prior)
    sha_a = _load("testcur_detA")["replay"]["features_csv_sha256"]
    sha_b = _load("testcur_detB")["replay"]["features_csv_sha256"]
    assert sha_a == sha_b, f"non-deterministic across --prior runs: {sha_a} != {sha_b}"
    # replay lock: a fresh baseline (no --prior) over the same wavs must match.
    _run_cli("testcur_noprior")
    sha_baseline = _load("testcur_noprior")["replay"]["features_csv_sha256"]
    assert sha_a == sha_baseline, f"--prior replay != fresh baseline: {sha_a} != {sha_baseline}"
