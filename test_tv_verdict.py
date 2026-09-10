"""Tests for the TV-game verdict layer (dsp_inverse.py §3).

Covers the six functions mirrored by index.html's TV game screen:
  bonferroni_alpha, loo_cv, mean_z_gap, verdict_merge, verdict_split,
  verdict_single.

Expected p-values below are derived from the chart-parity Mann-Whitney
implementation in dsp_inverse.mann_whitney_p (normal approximation,
tie-corrected, continuity-corrected, two-sided on min(U, n1*n2-U)):
  * fully separated 8v8, no ties      -> U=0, z=31.5/sqrt(90.667)~=3.308, p~=0.00097
  * fully separated 4v4, no ties      -> U=0, z=7.5/sqrt(12)~=2.165,     p~=0.030
  * interleaved 4v4                   -> U=6, z=1.5/sqrt(12)~=0.433,     p~=0.665
  * engineered 4v4 with U=1           -> z=6.5/sqrt(12)~=1.876,          p~=0.060
"""

import pytest

from dsp_inverse import (
    bonferroni_alpha,
    loo_cv,
    mean_z_gap,
    verdict_merge,
    verdict_split,
    verdict_single,
)

FEATS = ["duration"]


def _rows(spec, feat="duration"):
    """{group: [values]} -> list of row dicts."""
    return [{"group": g, feat: v} for g, vals in spec.items() for v in vals]


# ---------------------------------------------------------------------------
# 1. bonferroni_alpha
# ---------------------------------------------------------------------------

class TestBonferroniAlpha:
    def test_divides_by_n_tests(self):
        assert bonferroni_alpha(35) == pytest.approx(0.05 / 35)

    def test_single_test_unchanged(self):
        assert bonferroni_alpha(1) == pytest.approx(0.05)

    def test_zero_n_tests_guarded(self):
        # max(n_tests, 1) prevents division by zero.
        assert bonferroni_alpha(0) == pytest.approx(0.05)

    def test_custom_alpha(self):
        assert bonferroni_alpha(10, alpha=0.01) == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# 2. loo_cv
# ---------------------------------------------------------------------------

class TestLooCv:
    def test_separable_two_groups_perfect(self):
        rows = _rows({"A": [0.0, 0.5, 1.0, 0.2], "B": [10.0, 10.5, 11.0, 10.2]})
        assert loo_cv(rows, FEATS, ["A", "B"]) == pytest.approx(1.0)

    def test_separable_three_groups_perfect(self):
        rows = _rows({"A": [0.0, 0.5], "B": [10.0, 10.5], "C": [20.0, 20.5]})
        assert loo_cv(rows, FEATS, ["A", "B", "C"]) == pytest.approx(1.0)

    def test_too_few_rows_returns_zero(self):
        rows = _rows({"A": [1.0]})
        assert loo_cv(rows, FEATS, ["A"]) == 0.0

    def test_absent_group_skipped(self):
        # Asking for a group with no rows must not crash.
        rows = _rows({"A": [0.0, 0.5, 1.0], "B": [10.0, 10.5, 11.0]})
        assert loo_cv(rows, FEATS, ["A", "B", "C"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. mean_z_gap
# ---------------------------------------------------------------------------

class TestMeanZGap:
    def test_flat_scaler_dict(self):
        ra = _rows({"A": [10.0, 12.0]})
        rb = _rows({"B": [20.0, 22.0]})
        r = mean_z_gap(ra, rb, "duration", {"duration": 5.0})
        assert r["mean_a"] == pytest.approx(11.0)
        assert r["mean_b"] == pytest.approx(21.0)
        assert r["z_gap"] == pytest.approx(-2.0)
        assert r["n_a"] == 2
        assert r["n_b"] == 2
        assert 0.0 < r["p"] <= 1.0

    def test_scaler_dict_with_features_list(self):
        # Mirrors frozenStd(): {'std': [...], 'features': [...]}
        ra = _rows({"A": [8.0, 10.0]})
        rb = _rows({"B": [13.0, 15.0]})
        scaler = {"std": [2.0, 4.0], "features": ["impact", "duration"]}
        r = mean_z_gap(ra, rb, "duration", scaler)
        assert r["z_gap"] == pytest.approx((9.0 - 14.0) / 4.0)

    def test_none_scaler_falls_back_to_one(self):
        ra = _rows({"A": [10.0, 12.0]})
        rb = _rows({"B": [20.0, 22.0]})
        r = mean_z_gap(ra, rb, "duration", None)
        assert r["z_gap"] == pytest.approx(11.0 - 21.0)

    def test_empty_side_is_safe(self):
        rb = _rows({"B": [20.0, 22.0]})
        r = mean_z_gap([], rb, "duration", {"duration": 5.0})
        assert r["n_a"] == 0
        assert r["mean_a"] == 0.0
        assert r["p"] == 1.0  # n<2 per side -> uninformative


# ---------------------------------------------------------------------------
# 4. verdict_merge
# ---------------------------------------------------------------------------

class TestVerdictMerge:
    def test_tight_global_minimum_pair_is_good(self):
        rows = _rows({
            "A": [99.9, 100.0, 100.1, 100.0],
            "B": [100.1, 99.9, 100.0, 100.0],
            "C": [200.0, 200.1, 199.9, 200.0],
            "D": [300.0, 300.1, 299.9, 300.0],
        })
        r = verdict_merge(rows, "AB", FEATS)
        assert r["verdict"] == "good"
        assert r["best_feat"] == "duration"
        assert r["cv"] <= 0.05
        assert len(r["all_cv"]) == 6
        assert r["all_cv"]["AB"] == min(r["all_cv"].values())
        assert r["n_a"] == 4 and r["n_b"] == 4

    def test_low_cv_but_not_global_minimum_is_warn(self):
        # C|D merged with zero spread is the global minimum (CV=0);
        # A|B is still tight (CV<=0.05) but loses the argmin.
        rows = _rows({
            "A": [99.9, 100.0, 100.1, 100.0],
            "B": [100.1, 99.9, 100.0, 100.0],
            "C": [100.0, 100.0, 100.0, 100.0],
            "D": [100.0, 100.0, 100.0, 100.0],
        })
        r = verdict_merge(rows, "AB", FEATS)
        assert r["cv"] <= 0.05
        assert r["all_cv"]["CD"] < r["all_cv"]["AB"]
        assert r["verdict"] == "warn"

    def test_loose_pair_is_bad(self):
        rows = _rows({
            "A": [100.0, 100.1, 99.9, 100.0],
            "B": [300.0, 300.1, 299.9, 300.0],
            "C": [200.0, 200.1, 199.9, 200.0],
            "D": [250.0, 250.1, 249.9, 250.0],
        })
        r = verdict_merge(rows, "AB", FEATS)
        assert r["cv"] > 0.05
        assert r["verdict"] == "bad"


# ---------------------------------------------------------------------------
# 5. verdict_split
# ---------------------------------------------------------------------------

SEP_8V8 = {
    "A": [100.0, 101.0, 102.0, 103.0],
    "B": [104.0, 105.0, 106.0, 107.0],
    "C": [500.0, 501.0, 502.0, 503.0],
    "D": [504.0, 505.0, 506.0, 507.0],
}


class TestVerdictSplit:
    def test_separated_2v2_with_bonferroni_is_bad(self):
        # 8v8 fully separated -> p~=0.00097 < 0.05/35 ~= 0.00143.
        rows = _rows(SEP_8V8)
        r = verdict_split(rows, "AB|CD", FEATS, n_scanned=35)
        assert r["p"] < r["bonf_alpha"]
        assert r["bonf_alpha"] == pytest.approx(0.05 / 35)
        assert r["verdict"] == "bad"
        assert r["best_feat"] == "duration"

    def test_below_scan_threshold_keeps_plain_alpha(self):
        rows = _rows(SEP_8V8)
        r = verdict_split(rows, "AB|CD", FEATS, n_scanned=34)
        assert r["bonf_alpha"] == pytest.approx(0.05)
        assert r["verdict"] == "bad"  # p~=0.00097 < 0.05

    def test_interleaved_split_is_good(self):
        rows = _rows({
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [1.5, 2.5, 3.5, 4.5],
            "C": [1.2, 2.2, 3.2, 4.2],
            "D": [1.7, 2.7, 3.7, 4.7],
        })
        r = verdict_split(rows, "AB|CD", FEATS)
        assert r["p"] >= 0.1
        assert r["verdict"] == "good"

    def test_warn_band(self):
        # Engineered 4v4 with U=1 -> z~=1.876 -> p~=0.060 in [0.05, 0.1).
        rows = _rows({"A": [4.0, 6.0, 7.0, 8.0], "B": [1.0, 2.0, 3.0, 5.0]})
        r = verdict_split(rows, "A|B", FEATS, n_scanned=35)
        assert 0.05 <= r["p"] < 0.1
        assert r["verdict"] == "warn"

    def test_result_fields(self):
        rows = _rows(SEP_8V8)
        r = verdict_split(rows, "AB|CD", FEATS, n_scanned=35)
        assert set(r["pvals"].keys()) == set(FEATS)
        assert r["n"] == len(rows)
        assert 0.0 <= r["loo"] <= 1.0

    def test_perfectly_separable_loo_is_one(self):
        rows = _rows(SEP_8V8)
        r = verdict_split(rows, "AB|CD", FEATS)
        assert r["loo"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. verdict_single
# ---------------------------------------------------------------------------

class TestVerdictSingle:
    def test_separated_pair_is_bad(self):
        rows = _rows({
            "A": [100.0, 101.0, 102.0, 103.0],
            "B": [200.0, 201.0, 202.0, 203.0],
        })
        r = verdict_single(rows, "duration", "AB", {"duration": 10.0})
        # z_gap = (101.5 - 201.5) / 10 = -10
        assert r["z_gap"] == pytest.approx(-10.0)
        assert r["p"] < 0.05
        assert r["verdict"] == "bad"
        assert r["n_a"] == 4 and r["n_b"] == 4

    def test_overlapping_pair_is_good(self):
        rows = _rows({
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [1.5, 2.5, 3.5, 4.5],
        })
        r = verdict_single(rows, "duration", "AB", {"duration": 1.0})
        assert r["z_gap"] == pytest.approx((2.5 - 3.0) / 1.0)
        assert r["p"] >= 0.05
        assert r["verdict"] == "good"

    def test_sign_follows_key_order(self):
        rows = _rows({
            "A": [200.0, 201.0, 202.0, 203.0],
            "B": [100.0, 101.0, 102.0, 103.0],
        })
        r = verdict_single(rows, "duration", "AB", {"duration": 10.0})
        assert r["z_gap"] > 0  # mean_a > mean_b -> positive gap
