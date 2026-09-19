"""Tests for law_cards.py (T4).

Each card's pinned value is checked within its tolerance, and the Gauss-sum
card is re-derived from first principles (|G(p)| = sqrt(p)) to validate the
pure-math honesty layer (R5), not just the inlined constant.
"""
import math

import law_cards as L


# Pinned references (from the design spec / documented constants).
REF = {
    "dfft4_gauss": math.sqrt(199),
    "talbot_3_2": 1.5,
    "talbot_7_4": 1.75,
    "talbot_5_4": 1.25,
    "tri_forms": 3.0,
    "tilings_forbidden": 7.0,
    "beta_nonpisot": 0.251574,
    "kontsevich_sum": 53.0 / 28.0,
    "kontsevich_lambda_1_3": 1.0 / 3.0,
    "kontsevich_lambda_1_2": 1.0 / 2.0,
}


def test_card_count():
    assert len(L.cards()) == 10


def test_every_card_value_within_tolerance():
    by_id = {c["id"]: c for c in L.cards()}
    assert set(by_id) == set(REF)
    for cid, ref in REF.items():
        c = by_id[cid]
        assert abs(c["value"] - ref) < c["tol"], f"{cid}: {c['value']} vs {ref} (tol {c['tol']})"


def test_gauss_sum_recomputed_from_first_principles():
    # Honest math: |G(199)| == sqrt(199)  (not just the inlined constant).
    assert abs(L.gauss_sum_magnitude(199) - math.sqrt(199)) < 1e-6


def test_talbot_ratios_exact():
    assert L.TALBOT == {"3/2": 1.5, "7/4": 1.75, "5/4": 1.25}


def test_triangular_tiling_structure():
    t = L.TRIANGULAR
    assert t["n"] == 3
    assert t["forms"] == ["A2", "B3", "G2"]
    assert t["vertex_angle_sum"] == 180.0


def test_tilings_forbidden_n_is_seven():
    assert L.TILINGS_FORBIDDEN == {"5": 7, "8": 7, "10": 7, "12": 7}


def test_kontsevich_lambda_set():
    assert sorted(L.KONTSEVICH_LAMBDA) == sorted([1.0 / 3.0, 1.0 / 2.0])


def test_kontsevich_sum_value():
    assert abs(L.KONTSEVICH_SUM - 53.0 / 28.0) < 1e-9


def test_to_json_roundtrip():
    import json
    parsed = json.loads(L.to_json())
    assert isinstance(parsed, list)
    assert len(parsed) == 10
    ids = {c["id"] for c in parsed}
    assert "dfft4_gauss" in ids and "kontsevich_lambda_1_2" in ids


def test_verify_all_pass():
    report = L.verify()
    assert len(report) == 10
    bad = [r["id"] for r in report if not r["ok"]]
    assert bad == [], f"honesty layer failures: {bad}"


def test_beta_derived_from_luck_eq73():
    tau = (1.0 + 5.0 ** 0.5) / 2.0
    luck_formula = math.log(3.0 - tau) / math.log(2.0 + tau)
    pin_separates_from_formula = abs(L.BETA - luck_formula) > 1e-12
    assert pin_separates_from_formula
    assert abs(L.nice_beta() - luck_formula) < 1e-12
    assert abs(L.nice_beta() - L.BETA) < L.BETA_TOL
