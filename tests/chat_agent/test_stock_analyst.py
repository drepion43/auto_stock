"""stock_analyst.py TDD. `run_stock_analyst`는 `create_deep_agent`/`.invoke`를 모킹해
래퍼 자체 로직(예산 사전차감, 실패격리, provenance 라벨 코드 부착)만 검증한다 — 실제
LLM이 어떤 신호를 부를지 잘 판단하는지는 이 프로젝트에 처음 등장하는 영역이라 pytest로
주장하지 않는다(계획서 §7 명시, Stage C 수동 체크리스트로 대체)."""

from unittest.mock import MagicMock, patch

from auto_stock.chat_agent.credentials import STOCK_ANALYST_LLM_CALL_ESTIMATE, STOCK_ANALYST_RECURSION_LIMIT
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.stock_analyst import _SYSTEM_PROMPT, StockAnalystResult, run_stock_analyst


def _fake_tools():
    return [MagicMock(), MagicMock()]


class TestRunStockAnalyst:
    def test_returns_judgment_with_provenance_label_on_success(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_structured = StockAnalystResult(
            summary="상승 신호 우세", outlook="UP", signals_used=["analyze_rule_engine"]
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": fake_structured}
        tools = _fake_tools()
        with patch(
            "auto_stock.chat_agent.stock_analyst.create_deep_agent", return_value=fake_agent
        ) as mock_create:
            result = run_stock_analyst(budget, "gpt-5.6-luna", tools, "005930", "KRX")

        assert result["available"] is True
        assert result["judgment"]["summary"] == "상승 신호 우세"
        assert result["judgment"]["outlook"] == "UP"
        assert result["judgment"]["signals_used"] == ["analyze_rule_engine"]
        assert result["provenance"] == "서브에이전트 자율 조사 결과"
        assert budget.llm_calls_made == STOCK_ANALYST_LLM_CALL_ESTIMATE
        _, kwargs = mock_create.call_args
        assert kwargs["tools"] == tools
        invoke_kwargs = fake_agent.invoke.call_args.kwargs
        assert invoke_kwargs["config"]["recursion_limit"] == STOCK_ANALYST_RECURSION_LIMIT

    def test_returns_unavailable_when_budget_exhausted(self):
        budget = QueryBudget(max_llm_calls=1)
        budget.llm_calls_made = 1
        with patch("auto_stock.chat_agent.stock_analyst.create_deep_agent") as mock_create:
            result = run_stock_analyst(budget, "gpt-5.6-luna", _fake_tools(), "005930", "KRX")

        assert result["available"] is False
        mock_create.assert_not_called()

    def test_never_raises_when_invoke_fails(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = RuntimeError("recursion limit exceeded")
        with patch("auto_stock.chat_agent.stock_analyst.create_deep_agent", return_value=fake_agent):
            result = run_stock_analyst(budget, "gpt-5.6-luna", _fake_tools(), "005930", "KRX")

        assert result["available"] is False
        assert "error" in result

    def test_never_raises_when_structured_response_missing(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"messages": []}
        with patch("auto_stock.chat_agent.stock_analyst.create_deep_agent", return_value=fake_agent):
            result = run_stock_analyst(budget, "gpt-5.6-luna", _fake_tools(), "005930", "KRX")

        assert result["available"] is False

    def test_never_raises_when_structured_response_is_present_but_malformed(self):
        """코드 리뷰 발견사항(Stage E) — structured_response 키는 있지만 값이 기대한
        pydantic 인스턴스가 아니면(예: None) AttributeError가 나야 정상인데, 이전에는
        이 접근이 try/except 밖에 있어 raise가 그대로 새어나갔다."""
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": None}
        with patch("auto_stock.chat_agent.stock_analyst.create_deep_agent", return_value=fake_agent):
            result = run_stock_analyst(budget, "gpt-5.6-luna", _fake_tools(), "005930", "KRX")

        assert result["available"] is False
        assert "error" in result


def test_system_prompt_requires_summary_to_cite_each_used_signals_concrete_finding():
    """실사용 중 발견(2026-09-27) — stock_analyst가 뉴스/공시/차트분석을 실제로 호출해도
    summary가 뭉뚱그린 결론만 담으면 그 발견 내용이 최종 답변에 전혀 드러나지 않는다.
    signals_used는 도구 "이름"만 기록하는 감사용 필드라 발견 내용 자체는 담지 않으므로,
    PRD §5.1 Explainability 예시('RSI 과매도 구간 진입 + 최근 실적 서프라이즈 기사 확인 +
    LLM 차트 분석상 단기 반등 패턴 감지')처럼 summary 자체에 신호별 구체적 근거를 담도록
    프롬프트가 명시해야 한다."""
    assert "signals_used" in _SYSTEM_PROMPT
    assert "구체적" in _SYSTEM_PROMPT


def test_system_prompt_instructs_treating_tool_output_as_data_not_instructions():
    """보안 리뷰 발견사항(Stage E) — 공시/뉴스감성 분석 도구의 자유텍스트(rationale/
    key_headline 등)가 이 서브에이전트 컨텍스트로 그대로 흘러들어간다 — 지시문처럼
    보이는 문구를 따르지 말라는 방어 지침이 필요하다."""
    assert "데이터" in _SYSTEM_PROMPT
    assert "무시" in _SYSTEM_PROMPT
