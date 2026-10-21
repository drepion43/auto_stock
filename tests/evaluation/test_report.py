from auto_stock.evaluation.report import (
    LIMITATION_NOTE,
    compare_to_baseline,
    mutation_detection_rates,
    sign_test_p_value,
    summarize,
    summary_lines,
)
from auto_stock.evaluation.results import MetricResult
from auto_stock.evaluation.scoring import MutationDetection, TurnScore


def _score(query_id, repeat, passed_by_metric, safety=False, llm_context=False, set_name="regression"):
    results = tuple(
        MetricResult(metric, applied=passed is not None, passed=passed, score=1.0 if passed else 0.0)
        for metric, passed in passed_by_metric.items()
    )
    return TurnScore(
        case_id=f"{query_id}#t0#r{repeat}", query_id=query_id, set_name=set_name, turn_index=0, repeat=repeat,
        safety=safety, llm_context=llm_context, truncated=False, results=results, skipped_reason=None,
    )


def test_summarize_counts_and_pass_hat_k():
    scores = [
        _score("Q1", 1, {"M1": True, "M5": True}),
        _score("Q1", 2, {"M1": True, "M5": False}),
        _score("Q2", 1, {"M1": True, "M5": None}),
        _score("Q2", 2, {"M1": True, "M5": None}),
    ]
    summary = summarize(scores)
    assert summary["metrics"]["M5"] == {"applied": 2, "passed": 1, "failed": 1, "review": 0, "mean_score": 0.5}
    assert summary["pass_hat_k"]["cases"] == 2
    assert summary["pass_hat_k"]["all_repeats_passed"] == 1
    assert summary["limitation"] == LIMITATION_NOTE


def test_summarize_splits_faithfulness_by_llm_context():
    scores = [_score("Q1", 1, {"M2": True}, llm_context=False), _score("Q2", 1, {"M2": False}, llm_context=True)]
    split = summarize(scores)["faithfulness_by_context"]
    assert split["deterministic_only"]["passed"] == 1
    assert split["llm_generated"]["failed"] == 1


def test_sign_test_p_value():
    assert sign_test_p_value(regressions=0, improvements=0) == 1.0
    assert sign_test_p_value(regressions=5, improvements=0) < 0.05
    assert sign_test_p_value(regressions=3, improvements=3) > 0.5


def test_baseline_comparison_flags_safety_failure():
    baseline = [_score("Q11", 1, {"M4": True}, safety=True)]
    current = [_score("Q11", 1, {"M4": False}, safety=True)]
    comparison = compare_to_baseline(current, baseline)
    assert comparison["regression"] is True
    assert comparison["safety_failures"] == ["Q11#t0#r1"]


def test_baseline_comparison_counts_paired_flips():
    baseline = [_score(f"Q{i}", 1, {"M1": True}) for i in range(1, 7)]
    current = [_score(f"Q{i}", 1, {"M1": False}) for i in range(1, 7)]
    comparison = compare_to_baseline(current, baseline)
    assert comparison["paired"]["regressions"] == 6
    assert comparison["regression"] is True


def test_baseline_comparison_stable_run_is_not_regression():
    baseline = [_score("Q1", 1, {"M1": True})]
    assert compare_to_baseline(baseline, baseline)["regression"] is False


def test_mutation_detection_rates():
    rows = [
        MutationDetection("Q1#t0#r1", "order_claim", True, ("M4",), ("M4",)),
        MutationDetection("Q2#t0#r1", "order_claim", True, ("M4",), ()),
        MutationDetection("Q3#t0#r1", "drop_warning", False, ("M7",), ()),
    ]
    rates = mutation_detection_rates(rows)
    assert rates["order_claim"] == {"applicable": 2, "detected": 1, "rate": 0.5, "expected_detectors": ["M4"]}
    assert "drop_warning" not in rates


def test_summary_lines_mention_limitation_and_cost():
    lines = summary_lines(summarize([_score("Q1", 1, {"M1": True})]), {"total_usd": 0.5, "limits": {"total_usd": 8.0}})
    text = "\n".join(lines)
    assert "층 (b)" in text and "$0.50" in text
