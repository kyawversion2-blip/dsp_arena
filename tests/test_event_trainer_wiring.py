"""tests/test_event_trainer_wiring.py — Event Trainer static wiring.

The Event Trainer is a standalone, pure-client page (event_trainer.html) that
turns labeled events into an interpretable one-feature threshold rule. These
tests are static (no browser): they assert only the guarantees the slice claims.

  1. the canonical 5 features + units are reused verbatim (no re-invention)
  2. the scatter chart carries 4 distinct group colors and labelled axes/ticks
  3. a one-feature threshold rule is searched for two selected groups
  4. a threshold slider re-computes the REAL evaluation (not just a marker)
  5. a second-feature (AND) evaluation exists as a fallback
  6. training accuracy is never presented as generalization proof; small-sample
     limits are disclosed and leave-one-out CV is reported
  7. misclassified events + sample counts are surfaced
  8. no animal analogies and no ranking/"best" language in user-facing copy
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HTML = ROOT / "event_trainer.html"

CANONICAL = ["impact", "duration_s", "peak_time_ms", "impact_freq_hz", "mel_mean_db"]
UNITS = {"impact": "RMS", "duration_s": "s", "peak_time_ms": "ms",
         "impact_freq_hz": "Hz", "mel_mean_db": "dB"}


def _src() -> str:
    return HTML.read_text(encoding="utf-8")


def test_page_exists_and_is_self_contained():
    src = _src()
    assert "<canvas" in src
    assert src.count("<script>") == 1
    # pure client: no external script/style fetches
    assert "src=\"http" not in src and "href=\"http" not in src


def test_canonical_features_and_units_reused_verbatim():
    src = _src()
    line = [l for l in src.splitlines() if l.startswith("const FEATURES =")]
    assert len(line) == 1, "expected exactly one FEATURES declaration"
    for feat in CANONICAL:
        assert "'" + feat + "'" in line[0], f"feature {feat} missing from FEATURES"
    for feat, unit in UNITS.items():
        assert re.search(r"\b" + feat + r":\s*'" + re.escape(unit) + r"'", src), \
            f"unit {unit} for {feat} not declared"


def test_four_distinct_group_colors_defined_and_used():
    src = _src()
    m = re.search(r"const GROUP_COLORS = \{([^}]*)\}", src)
    assert m, "GROUP_COLORS missing"
    colors = re.findall(r"#[0-9a-fA-F]{6}", m.group(1))
    assert len(colors) == 4 and len(set(colors)) == 4, "need 4 distinct colors"
    for g in ("A", "B", "C", "D"):
        assert re.search(g + r":\s*'#", m.group(1)), f"group {g} has no color"
    # the renderer must actually paint points with the palette
    assert "GROUP_COLORS[p.group]" in src


def test_scatter_axes_have_labels_units_and_ticks():
    src = _src()
    assert src.count('id="trainer-canvas"') == 1
    # axis labels include the feature name plus its unit
    assert "UNITS[axisX]" in src and "UNITS[axisY]" in src
    # ticks come from the mapping (normalised 0..1 positions)
    assert "ticks" in src and "t.pos" in src and "t.label" in src
    # 4 group legend swatches rendered from the palette
    assert "chart-legend" in src


def test_one_feature_threshold_rule_is_searched():
    src = _src()
    assert "function searchBestThreshold(" in src
    assert "function searchRules(" in src
    # the search is bound to a user action
    assert "$('bt-search').addEventListener('click', runSearch)" in src
    assert 'id="bt-search"' in src
    # the learned rule is rendered as an IF/THEN expression
    assert "IF ' + ruleString(currentRule)" in src


def test_slider_recomputes_real_evaluation_not_marker_only():
    src = _src()
    assert src.count('id="sl-threshold"') == 1
    # slider input must drive the recompute handler
    assert "$('sl-threshold').addEventListener('input', onThresholdChange)" in src
    # the handler must call the true evaluator and refresh both result + chart
    body = src[src.index("function onThresholdChange("):]
    body = body[: body.index("function onDirectionChange(")]
    assert "recomputeRule(" in body, "slider does not recompute the rule"
    assert "evaluateRule(" in src, "evaluation entry point missing"
    assert "renderRuleResult()" in body, "slider does not refresh metrics"
    assert "drawChart()" in body, "slider does not refresh chart annotation"
    # the chart annotation is a threshold line driven by currentRule.threshold
    assert "currentRule.threshold" in src
    assert "ctx.moveTo" in src


def test_second_feature_fallback_present():
    src = _src()
    assert "function searchTwoFeature(" in src
    assert 'id="bt-two-feature"' in src
    assert "$('bt-two-feature').addEventListener('click', runTwoFeature)" in src


def test_training_accuracy_not_presented_as_generalization():
    src = _src()
    lower = src.lower()
    # leave-one-out CV must be computed and shown
    assert "looAccuracy" in src and "looCorrect" in src
    assert "leave-one-out" in lower
    # LOO must score only the two selected groups (binary problem), never other groups
    assert "events.filter(e => e.group === g0 || e.group === g1)" in src, \
        "LOO must be scoped to the two selected groups"
    # an explicit small-sample caveat must be present
    assert "not" in lower and "evidence of generalization" in lower
    assert "exploratory" in lower
    # no over-claiming language in rendered copy
    for bad in ("best rule", "optimal", "ground truth", "generalizes"):
        assert bad not in lower, f"over-claiming copy: {bad}"


def test_misclassified_table_and_sample_counts_surfaced():
    src = _src()
    assert 'id="misclass-tbody"' in src
    assert "misclassifiedIds" in src
    assert 'id="kpi-n"' in src and "groupCounts" in src


def test_no_animal_analogies_or_ranking():
    src = _src()
    lower = src.lower()
    for bad in ("frog", "pig", "chicken", "snake"):
        assert bad not in lower, f"forbidden animal analogy: {bad}"
    # no ranking/leaderboard language in user-facing strings (HTML text only)
    text = re.sub(r"<script>.*?</script>", "", src, flags=re.S).lower()
    for bad in ("best", "optimal", "rank", "leaderboard", "winner"):
        assert bad not in text, f"ranking language in UI copy: {bad}"
