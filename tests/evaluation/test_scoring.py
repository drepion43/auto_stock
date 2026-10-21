from datetime import date
from types import SimpleNamespace

import pytest

from auto_stock.evaluation.answer_capture import CapturedTurn, ToolCallRecord
from auto_stock.evaluation.budget import BudgetLimits, CostLedger
from auto_stock.evaluation.golden import cases_by_id
from auto_stock.evaluation.judge import BudgetedOpenAIJudge
from auto_stock.evaluation.judged_metrics import faithfulness_metric, run_metric
from auto_stock.evaluation.results import MetricResult
from auto_stock.evaluation.scoring import TurnScore, combine_policy, score_mutation, score_turn

TODAY = date(2026, 10, 21)
_PRICE = ToolCallRecord(
    "get_price_data", {"ticker": "005930", "market": "KOSPI"}, {"available": True, "latest": {"date": "2026-10-20", "close": 71200.0}}
)


def _captured(query_id="Q1", answer="실시간이 아닌 2026-10-20 종가 기준 71,200원입니다.", calls=(_PRICE,), fault=None):
    return CapturedTurn(
        case_id=f"{query_id}#t0#r1", query_id=query_id, turn_index=0, repeat=1, question="삼성전자 지금 얼마야?",
        answer=answer, tool_calls=calls, history_tool_calls=(), fallback=False, injected_fault=fault,
        replay_fixture=None, injection_marker=None, usage={},
    )


@pytest.fixture
def make_judge(fake_responses_cls):
    def _make(limits=None):
        ledger = CostLedger(limits or BudgetLimits())
        client = SimpleNamespace(responses=fake_responses_cls())
        return BudgetedOpenAIJudge(ledger, model_name="gpt-5.6-luna", client=client, max_output_tokens=500), ledger

    return _make


def _by_metric(score):
    return {r.metric: r for r in score.results}


def test_deterministic_only_scoring_without_judge():
    resolve = ToolCallRecord("resolve_ticker", {"query": "삼성전자"}, {"available": True, "matches": []})
    score = score_turn(_captured(calls=(resolve, _PRICE)), cases_by_id()["Q1"], judge=None, today=TODAY)
    results = _by_metric(score)
    assert results["M1"].passed is True
    assert results["M5"].passed is True
    assert "M2" not in results
    assert score.skipped_reason is None


def test_judged_metrics_run_through_deepeval_offline(make_judge):
    judge, ledger = make_judge()
    score = score_turn(_captured(), cases_by_id()["Q1"], judge=judge, today=TODAY)
    results = _by_metric(score)
    assert {"M2", "M3", "M4"} <= set(results)
    assert results["M2"].applied is True
    assert ledger.judge_calls > 0


def test_faithfulness_metric_penalizes_ambiguous_claims(make_judge):
    judge, _ledger = make_judge()
    assert faithfulness_metric(judge).penalize_ambiguous_claims is True


def test_budget_exhaustion_keeps_deterministic_results_and_marks_skip(make_judge):
    judge, _ledger = make_judge(BudgetLimits(total_usd=1.0, judge_usd=0.0000001, judge_calls=10))
    score = score_turn(_captured(), cases_by_id()["Q1"], judge=judge, today=TODAY)
    assert score.skipped_reason == "judge_budget"
    assert _by_metric(score)["M5"].applied is True


def test_run_metric_turns_unexpected_errors_into_review_items():
    class _Broken:
        def measure(self, test_case):
            raise ValueError("boom")

    result = run_metric("M3", _Broken(), test_case=None)
    assert result.passed is None and result.needs_review is True


def test_safety_policy_uses_rules_as_final_verdict():
    rules = MetricResult("M4", True, True, 1.0)
    geval = MetricResult("M4", True, False, 0.3)
    combined = combine_policy("M4", rules, geval, safety=True)
    assert combined.passed is True and combined.needs_review is True


def test_non_safety_policy_requires_both():
    rules = MetricResult("M4", True, True, 1.0)
    geval = MetricResult("M4", True, False, 0.3)
    assert combine_policy("M4", rules, geval, safety=False).passed is False


def test_fault_case_applies_honesty_metric():
    failed = ToolCallRecord("get_price_data", {}, {"available": False, "error": "x"})
    captured = _captured(query_id="F1", answer="삼성전자 좋은 회사입니다.", calls=(failed,), fault={"tool": "get_price_data"})
    score = score_turn(captured, cases_by_id()["F1"], judge=None, today=TODAY)
    assert _by_metric(score)["M6"].passed is False


def test_truncated_context_downgrades_faithfulness_failure_to_review(make_judge):
    judge, _ledger = make_judge()
    huge = ToolCallRecord("get_market_scan_recommendations", {"market": "KRX"}, {"recommendations": [{"s": "x" * 500}] * 40})
    score = score_turn(_captured(query_id="Q3", calls=(huge,)), cases_by_id()["Q3"], judge=judge, today=TODAY, context_limit=1000)
    m2 = _by_metric(score)["M2"]
    assert m2.details["truncated"] is True
    if m2.passed is not True:
        assert m2.passed is None and m2.needs_review is True


def test_score_mutation_reports_detection_by_expected_detectors():
    detection = score_mutation(_captured(), cases_by_id()["Q1"], "order_claim", judge=None, today=TODAY)
    assert detection.applicable is True
    assert detection.detected_by == ("M4",)


def test_score_mutation_skips_inapplicable_kind():
    detection = score_mutation(_captured(answer="안녕하세요."), cases_by_id()["Q1"], "number_shift", judge=None, today=TODAY)
    assert detection.applicable is False


def test_turn_score_round_trip(make_judge):
    judge, _ledger = make_judge()
    score = score_turn(_captured(), cases_by_id()["Q1"], judge=judge, today=TODAY)
    assert TurnScore.from_dict(score.to_dict()) == score
