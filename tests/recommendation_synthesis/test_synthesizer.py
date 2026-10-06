"""synthesizer.py TDD — stock_analyst.py/related_companies.py와 동일한 테스트 원칙:
`create_deep_agent`/`.invoke`를 모킹해 래퍼 로직(예산 사전차감, 실패격리, provenance
라벨, reinvestigate_ticker 도구 배선)만 검증한다. "실제로 애매한 경우를 잘 포착해
재조사하는지"는 pytest로 주장하지 않는다(Stage C 수동 체크리스트 전례와 동일)."""

from datetime import date
from unittest.mock import MagicMock, patch

from auto_stock.chat_agent.models import QueryBudget
from auto_stock.ml_predictor.models import MLPrediction
from auto_stock.orchestrator.deep_scan import TickerJudgment
from auto_stock.recommendation_synthesis.credentials import (
    RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE,
    RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT,
)
from auto_stock.recommendation_synthesis.models import RecommendationItem, RecommendationSynthesisResult
from auto_stock.recommendation_synthesis.synthesizer import (
    _SYSTEM_PROMPT,
    _reinvestigate_ticker_impl,
    run_recommendation_synthesis,
)


def _judgment(ticker="005930", market="KRX"):
    return TickerJudgment(
        ticker=ticker, market=market,
        rule_candidate=None,
        ml_prediction=MLPrediction(
            ticker=ticker, market=market, date=date(2026, 1, 2),
            probability_up=0.7, top_features=[],
        ),
        stock_analyst_result={
            "available": True,
            "judgment": {"summary": "상승 신호", "outlook": "UP", "signals_used": ["analyze_ml_prediction"]},
        },
    )


def _bind_tools(ticker, market):
    return [f"tool-for-{ticker}"]


class TestRunRecommendationSynthesis:
    def test_returns_recommendations_with_provenance_label_on_success(self):
        budget = QueryBudget(max_llm_calls=20)
        fake_structured = RecommendationSynthesisResult(
            recommendations=[
                RecommendationItem(ticker="005930", market="KRX", action="BUY", rank=1, summary="근거")
            ]
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": fake_structured}
        with patch(
            "auto_stock.recommendation_synthesis.synthesizer.create_deep_agent", return_value=fake_agent
        ) as mock_create:
            result = run_recommendation_synthesis(
                budget, "gpt-5.6-luna", _bind_tools, [_judgment()], top_n=5
            )

        assert result["available"] is True
        assert result["recommendations"] == [
            {"ticker": "005930", "market": "KRX", "action": "BUY", "rank": 1, "summary": "근거"}
        ]
        assert result["provenance"] == "서브에이전트 자율 조사 결과"
        assert budget.llm_calls_made == RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE
        _, kwargs = mock_create.call_args
        assert len(kwargs["tools"]) == 1
        invoke_kwargs = fake_agent.invoke.call_args.kwargs
        assert invoke_kwargs["config"]["recursion_limit"] == RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT

    def test_returns_unavailable_when_budget_exhausted(self):
        budget = QueryBudget(max_llm_calls=1)
        budget.llm_calls_made = 1
        with patch("auto_stock.recommendation_synthesis.synthesizer.create_deep_agent") as mock_create:
            result = run_recommendation_synthesis(
                budget, "gpt-5.6-luna", _bind_tools, [_judgment()], top_n=5
            )

        assert result["available"] is False
        mock_create.assert_not_called()

    def test_never_raises_when_invoke_fails(self):
        budget = QueryBudget(max_llm_calls=20)
        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = RuntimeError("recursion limit exceeded")
        with patch("auto_stock.recommendation_synthesis.synthesizer.create_deep_agent", return_value=fake_agent):
            result = run_recommendation_synthesis(
                budget, "gpt-5.6-luna", _bind_tools, [_judgment()], top_n=5
            )

        assert result["available"] is False
        assert "error" in result

    def test_never_raises_when_structured_response_missing(self):
        budget = QueryBudget(max_llm_calls=20)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"messages": []}
        with patch("auto_stock.recommendation_synthesis.synthesizer.create_deep_agent", return_value=fake_agent):
            result = run_recommendation_synthesis(
                budget, "gpt-5.6-luna", _bind_tools, [_judgment()], top_n=5
            )

        assert result["available"] is False

    def test_never_raises_when_structured_response_is_present_but_malformed(self):
        """stock_analyst.py Stage E 리뷰 수정과 동일한 회귀 — structured_response 키는
        있지만 값이 기대한 pydantic 인스턴스가 아니면 AttributeError가 나는데, 후처리가
        try 블록 밖에 있으면 그대로 새어나간다."""
        budget = QueryBudget(max_llm_calls=20)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": None}
        with patch("auto_stock.recommendation_synthesis.synthesizer.create_deep_agent", return_value=fake_agent):
            result = run_recommendation_synthesis(
                budget, "gpt-5.6-luna", _bind_tools, [_judgment()], top_n=5
            )

        assert result["available"] is False
        assert "error" in result


def test_system_prompt_instructs_selective_reinvestigation_not_blanket():
    assert "reinvestigate_ticker" in _SYSTEM_PROMPT
    assert "애매" in _SYSTEM_PROMPT


def test_system_prompt_requires_disclosing_signal_coverage_asymmetry():
    """§3에서 agent 자율성을 택한 트레이드오프 — 종목마다 signals_used가 다를 수 있다는
    걸 최종 summary가 숨기면 안 된다(recommendation-synthesis-plan.md §4)."""
    assert "signals_used" in _SYSTEM_PROMPT


def test_system_prompt_instructs_treating_judgment_text_as_data_not_instructions():
    assert "데이터" in _SYSTEM_PROMPT
    assert "무시" in _SYSTEM_PROMPT


def test_reinvestigate_ticker_impl_shares_budget_and_uses_bind_tools(mocker):
    mock_run_stock_analyst = mocker.patch(
        "auto_stock.recommendation_synthesis.synthesizer.run_stock_analyst",
        return_value={"available": True, "judgment": {"summary": "재조사 결과"}},
    )
    budget = QueryBudget(max_llm_calls=20)

    result = _reinvestigate_ticker_impl(budget, "gpt-5.6-luna", _bind_tools, "005930", "KRX")

    mock_run_stock_analyst.assert_called_once_with(
        budget, "gpt-5.6-luna", ["tool-for-005930"], "005930", "KRX"
    )
    assert result == {"available": True, "judgment": {"summary": "재조사 결과"}}
