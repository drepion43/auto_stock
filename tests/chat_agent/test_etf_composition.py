"""etf_composition.py TDD — `run_etf_composition_synthesis`는 `create_deep_agent`/
`.invoke`를 모킹해 래퍼 자체 로직(예산 사전차감, 실패격리)만 검증한다
(test_related_companies.py의 TestRunFindRelatedCompanies와 동일 패턴). 실제 LLM이
합리적으로 종합하는지는 주장하지 않는다."""

from unittest.mock import MagicMock, patch

from auto_stock.chat_agent.credentials import ETF_COMPOSITION_LLM_CALL_ESTIMATE, ETF_COMPOSITION_RECURSION_LIMIT
from auto_stock.chat_agent.etf_composition import (
    EtfCompositionResult,
    _holding_to_payload,
    run_etf_composition_synthesis,
)
from auto_stock.chat_agent.models import QueryBudget


class TestHoldingToPayload:
    def test_extracts_analysis_when_available(self):
        payload = _holding_to_payload(
            "005930", 25.0, "삼성전자",
            {"available": True, "analysis": {"market_impact": "POSITIVE"}},
            {"available": True, "analysis": {"sentiment": "NEUTRAL"}},
        )

        assert payload == {
            "ticker": "005930",
            "weight_pct": 25.0,
            "name": "삼성전자",
            "disclosure": {"market_impact": "POSITIVE"},
            "news_sentiment": {"sentiment": "NEUTRAL"},
        }

    def test_treats_unavailable_as_none_not_error(self):
        payload = _holding_to_payload(
            "000660", 15.0, "SK하이닉스",
            {"available": False, "error": "boom"},
            {"available": True, "analysis": None},
        )

        assert payload["disclosure"] is None
        assert payload["news_sentiment"] is None


class TestRunEtfCompositionSynthesis:
    def test_returns_summary_on_success(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_structured = EtfCompositionResult(summary="반도체 비중 종목 강세", outlook="UP")
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": fake_structured}
        with patch(
            "auto_stock.chat_agent.etf_composition.create_deep_agent", return_value=fake_agent
        ):
            result = run_etf_composition_synthesis(budget, "gpt-5.6-luna", "069500", [])

        assert result["available"] is True
        assert result["summary"] == "반도체 비중 종목 강세"
        assert result["outlook"] == "UP"
        assert result["provenance"] == "ETF 구성종목 집계 분석"
        assert budget.llm_calls_made == ETF_COMPOSITION_LLM_CALL_ESTIMATE

    def test_passes_recursion_limit(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {
            "structured_response": EtfCompositionResult(summary="s", outlook="NEUTRAL")
        }
        with patch(
            "auto_stock.chat_agent.etf_composition.create_deep_agent", return_value=fake_agent
        ):
            run_etf_composition_synthesis(budget, "gpt-5.6-luna", "069500", [])

        invoke_kwargs = fake_agent.invoke.call_args.kwargs
        assert invoke_kwargs["config"]["recursion_limit"] == ETF_COMPOSITION_RECURSION_LIMIT

    def test_returns_unavailable_when_budget_exhausted(self):
        budget = QueryBudget(max_llm_calls=1)
        budget.llm_calls_made = 1
        with patch("auto_stock.chat_agent.etf_composition.create_deep_agent") as mock_create:
            result = run_etf_composition_synthesis(budget, "gpt-5.6-luna", "069500", [])

        assert result["available"] is False
        mock_create.assert_not_called()

    def test_never_raises_when_invoke_fails(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = RuntimeError("recursion limit exceeded")
        with patch(
            "auto_stock.chat_agent.etf_composition.create_deep_agent", return_value=fake_agent
        ):
            result = run_etf_composition_synthesis(budget, "gpt-5.6-luna", "069500", [])

        assert result["available"] is False
        assert "error" in result

    def test_never_raises_when_structured_response_malformed(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": None}
        with patch(
            "auto_stock.chat_agent.etf_composition.create_deep_agent", return_value=fake_agent
        ):
            result = run_etf_composition_synthesis(budget, "gpt-5.6-luna", "069500", [])

        assert result["available"] is False
        assert "error" in result
