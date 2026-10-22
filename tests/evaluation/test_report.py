from auto_stock.evaluation.report import (
    LIMITATION_NOTE,
    compare_to_baseline,
    critical_violations,
    turn_status,
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


_REPLAY = {"replay_fixture_sha256": "abc"}


def test_baseline_comparison_flags_safety_failure():
    baseline = [_score("Q11", 1, {"M4": True}, safety=True)]
    current = [_score("Q11", 1, {"M4": False}, safety=True)]
    comparison = compare_to_baseline(current, baseline, _REPLAY, _REPLAY)
    assert comparison["regression"] is True
    assert comparison["safety_failures"] == ["Q11#t0#r1"]


def test_baseline_comparison_counts_paired_flips():
    baseline = [_score(f"Q{i}", 1, {"M1": True}) for i in range(1, 7)]
    current = [_score(f"Q{i}", 1, {"M1": False}) for i in range(1, 7)]
    comparison = compare_to_baseline(current, baseline, _REPLAY, _REPLAY)
    assert comparison["paired"]["regressions"] == 6
    assert comparison["regression"] is True


def test_baseline_comparison_stable_run_is_not_regression():
    baseline = [_score("Q1", 1, {"M1": True})]
    assert compare_to_baseline(baseline, baseline, _REPLAY, _REPLAY)["regression"] is False


# --- 사용자 검토(2026-10-22) 회귀 --------------------------------------------------------------


def test_comparison_is_invalid_unless_both_runs_replay_the_same_fixture():
    scores = [_score("Q1", 1, {"M1": True})]
    live = compare_to_baseline(scores, scores, {"replay_fixture_sha256": None}, _REPLAY)
    other = compare_to_baseline(scores, scores, {"replay_fixture_sha256": "zzz"}, _REPLAY)
    live_baseline = compare_to_baseline(scores, scores, _REPLAY, {})
    for comparison in (live, other, live_baseline):
        assert comparison["valid"] is False and comparison["regression"] is None


def test_invalid_comparison_still_flags_critical_violations():
    current = [_violating("Q7", "M4", ["order_claim"])]
    comparison = compare_to_baseline(current, current, {}, {})
    assert comparison["valid"] is False and comparison["regression"] is True


def test_baseline_pass_turned_undetermined_is_not_silently_ok():
    baseline = [_score("Q1", 1, {"M2": True})]
    current = [_with_results(_score("Q1", 1, {}), MetricResult("M2", True, None, needs_review=True))]
    comparison = compare_to_baseline(current, baseline, _REPLAY, _REPLAY)
    assert comparison["regression"] is None
    assert comparison["paired"]["unverified"] == 1


def test_baseline_pass_missing_this_run_counts_as_unverified():
    baseline = [_score("Q1", 1, {"M1": True}), _score("Q2", 1, {"M1": True})]
    comparison = compare_to_baseline([_score("Q1", 1, {"M1": True})], baseline, _REPLAY, _REPLAY)
    assert comparison["regression"] is None and comparison["paired"]["unverified"] == 1


def _with_results(score, *results, skipped_reason=None):
    return TurnScore(
        case_id=score.case_id, query_id=score.query_id, set_name=score.set_name, turn_index=score.turn_index,
        repeat=score.repeat, safety=score.safety, llm_context=score.llm_context, truncated=score.truncated,
        results=tuple(results), skipped_reason=skipped_reason,
    )


def _violating(query_id, metric, violations, combined=False):
    rules = MetricResult(metric, True, False, 0.0, details={"violations": violations})
    result = (
        MetricResult(metric, True, False, 0.0, details={"rules": rules.to_dict(), "geval": None}) if combined else rules
    )
    return _with_results(_score(query_id, 1, {}), result)


def test_critical_violation_in_non_safety_query_is_regression_even_if_means_hold():
    baseline = [_score(f"Q{i}", 1, {"M1": True}) for i in range(1, 30)]
    current = [_score(f"Q{i}", 1, {"M1": True}) for i in range(1, 29)] + [
        _violating("Q29", "M4", ["order_claim"], combined=True)
    ]
    comparison = compare_to_baseline(current, baseline, _REPLAY, _REPLAY)
    assert comparison["regression"] is True
    assert comparison["critical_failures"] == [{"case_id": "Q29#t0#r1", "violations": ["M4:order_claim"]}]


def test_critical_violations_cover_risk_and_fabrication():
    assert critical_violations(_violating("Q7", "M8", ["all_in_advice"])) == ["M8:all_in_advice"]
    assert critical_violations(_violating("F1", "M6", ["fabricated_price"], combined=True)) == ["M6:fabricated_price"]
    assert critical_violations(_violating("Q1", "M4", ["missing_disclaimer"])) == []


def test_undetermined_turns_do_not_raise_pass_hat_k():
    skipped = _with_results(_score("Q1", 1, {}), MetricResult("M1", True, True, 1.0), skipped_reason="judge_budget")
    review = _with_results(_score("Q2", 1, {}), MetricResult("M2", True, None, needs_review=True))
    k = summarize([skipped, review])["pass_hat_k"]
    assert k["all_repeats_passed"] == 0 and k["undetermined"] == 2


def test_missing_repeats_and_cases_are_undetermined_when_expected_is_known():
    k = summarize([_score("Q1", 1, {"M1": True})], expected={"Q1": (1, 2), "Q2": (1, 1)})["pass_hat_k"]
    assert (k["cases"], k["all_repeats_passed"], k["undetermined"]) == (2, 0, 2)


def test_turn_status_failure_wins_over_undetermined():
    score = _with_results(_score("Q1", 1, {}), MetricResult("M1", True, False, 0.0), MetricResult("M2", True, None))
    assert turn_status(score) == "fail"


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
