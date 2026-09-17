"""related_companies.py TDD. `verify_companies_co_mentioned_in_news`는 search_news/
search_articles를 모킹해 결정론적으로 검증한다(계획서 §7 "결정론적으로 커버" 범위).
`run_find_related_companies`는 `create_deep_agent`/`.invoke`를 모킹해 래퍼 자체의
로직(예산 사전차감, 실패격리, 최대 개수 절단)만 검증한다 — 실제 LLM이 합리적인
관련기업을 찾는지는 이 프로젝트에 처음 등장하는 영역이라 pytest로 주장하지 않는다
(계획서 §7 명시, Stage C 수동 체크리스트로 대체)."""

from datetime import date
from unittest.mock import MagicMock, patch

from auto_stock.chat_agent.credentials import (
    FIND_RELATED_LLM_CALL_ESTIMATE,
    FIND_RELATED_RECURSION_LIMIT,
    MAX_TICKERS_PER_QUERY,
)
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.related_companies import (
    _SYSTEM_PROMPT,
    RelatedCompaniesResult,
    RelatedCompanyItem,
    run_find_related_companies,
    verify_companies_co_mentioned_in_news,
)
from auto_stock.data.models import NewsArticle


def _article(title: str = "관련 기사") -> NewsArticle:
    return NewsArticle(
        ticker="", market="KRX", title=title, url="http://x", published_at=date.today(), source="x.com"
    )


class TestVerifyCompaniesCoMentionedInNews:
    def test_krx_co_mentioned_true_when_articles_found(self):
        with patch(
            "auto_stock.chat_agent.related_companies.search_news", return_value=[_article()]
        ) as mock_search:
            result = verify_companies_co_mentioned_in_news.invoke(
                {"company_a": "삼성전자", "company_b": "삼성SDI", "market": "KRX"}
            )

        assert result["co_mentioned"] is True
        assert result["headline"] == "관련 기사"
        mock_search.assert_called_once()

    def test_krx_co_mentioned_false_when_no_articles(self):
        with patch("auto_stock.chat_agent.related_companies.search_news", return_value=[]):
            result = verify_companies_co_mentioned_in_news.invoke(
                {"company_a": "삼성전자", "company_b": "없는회사", "market": "KRX"}
            )

        assert result["co_mentioned"] is False

    def test_nasdaq_uses_gdelt_search_articles(self):
        with patch(
            "auto_stock.chat_agent.related_companies.search_articles",
            return_value=[_article("Related")],
        ) as mock_search:
            result = verify_companies_co_mentioned_in_news.invoke(
                {"company_a": "NVIDIA", "company_b": "TSMC", "market": "NASDAQ"}
            )

        assert result["co_mentioned"] is True
        mock_search.assert_called_once()

    def test_unsupported_market_returns_not_co_mentioned_without_raising(self):
        result = verify_companies_co_mentioned_in_news.invoke(
            {"company_a": "A", "company_b": "B", "market": "TOKYO"}
        )

        assert result["co_mentioned"] is False
        assert "error" in result

    def test_source_failure_never_raises(self):
        with patch(
            "auto_stock.chat_agent.related_companies.search_news", side_effect=RuntimeError("boom")
        ):
            result = verify_companies_co_mentioned_in_news.invoke(
                {"company_a": "삼성전자", "company_b": "삼성SDI", "market": "KRX"}
            )

        assert result["co_mentioned"] is False
        assert "error" in result


class TestRunFindRelatedCompanies:
    def test_returns_related_list_on_success(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_structured = RelatedCompaniesResult(
            related=[
                RelatedCompanyItem(
                    ticker="006400", market="KRX", name="삼성SDI", relation="계열사", confidence="confirmed"
                )
            ]
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": fake_structured}
        with patch(
            "auto_stock.chat_agent.related_companies.create_deep_agent", return_value=fake_agent
        ) as mock_create:
            result = run_find_related_companies(budget, "gpt-5.6-luna", "005930", "KRX", "삼성전자")

        assert result["available"] is True
        assert result["related"] == [
            {
                "ticker": "006400",
                "market": "KRX",
                "name": "삼성SDI",
                "relation": "계열사",
                "confidence": "confirmed",
            }
        ]
        assert budget.llm_calls_made == FIND_RELATED_LLM_CALL_ESTIMATE
        _, kwargs = mock_create.call_args
        assert kwargs["tools"] == [verify_companies_co_mentioned_in_news]
        fake_agent.invoke.assert_called_once()
        invoke_kwargs = fake_agent.invoke.call_args.kwargs
        assert invoke_kwargs["config"]["recursion_limit"] == FIND_RELATED_RECURSION_LIMIT

    def test_truncates_to_max_tickers_per_query_minus_self(self):
        budget = QueryBudget(max_llm_calls=10)
        items = [
            RelatedCompanyItem(ticker=str(i), market="KRX", name=f"C{i}", relation="r", confidence="inferred")
            for i in range(10)
        ]
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": RelatedCompaniesResult(related=items)}
        with patch("auto_stock.chat_agent.related_companies.create_deep_agent", return_value=fake_agent):
            result = run_find_related_companies(budget, "gpt-5.6-luna", "005930", "KRX", "삼성전자")

        assert len(result["related"]) == MAX_TICKERS_PER_QUERY - 1

    def test_returns_unavailable_when_budget_exhausted(self):
        budget = QueryBudget(max_llm_calls=1)
        budget.llm_calls_made = 1
        with patch("auto_stock.chat_agent.related_companies.create_deep_agent") as mock_create:
            result = run_find_related_companies(budget, "gpt-5.6-luna", "005930", "KRX", "삼성전자")

        assert result["available"] is False
        mock_create.assert_not_called()

    def test_never_raises_when_invoke_fails(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = RuntimeError("recursion limit exceeded")
        with patch("auto_stock.chat_agent.related_companies.create_deep_agent", return_value=fake_agent):
            result = run_find_related_companies(budget, "gpt-5.6-luna", "005930", "KRX", "삼성전자")

        assert result["available"] is False
        assert "error" in result

    def test_never_raises_when_structured_response_missing(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"messages": []}
        with patch("auto_stock.chat_agent.related_companies.create_deep_agent", return_value=fake_agent):
            result = run_find_related_companies(budget, "gpt-5.6-luna", "005930", "KRX", "삼성전자")

        assert result["available"] is False

    def test_never_raises_when_structured_response_is_present_but_malformed(self):
        """코드 리뷰 발견사항(Stage E) — structured_response 키는 있지만 값이 기대한
        pydantic 인스턴스가 아니면(예: None) AttributeError가 나야 정상인데, 이전에는
        이 접근이 try/except 밖에 있어 raise가 그대로 새어나갔다."""
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": None}
        with patch("auto_stock.chat_agent.related_companies.create_deep_agent", return_value=fake_agent):
            result = run_find_related_companies(budget, "gpt-5.6-luna", "005930", "KRX", "삼성전자")

        assert result["available"] is False
        assert "error" in result


def test_system_prompt_instructs_treating_verification_tool_output_as_data_not_instructions():
    """보안 리뷰 발견사항(Stage E) — verify_companies_co_mentioned_in_news가 반환하는
    headline/url은 뉴스 색인(사실상 공개 웹)이 통제하는 텍스트라 서브에이전트 컨텍스트에
    그대로 흘러들어간다 — 지시문처럼 보이는 문구를 따르지 말라는 방어 지침이 필요하다."""
    assert "데이터" in _SYSTEM_PROMPT
    assert "무시" in _SYSTEM_PROMPT
