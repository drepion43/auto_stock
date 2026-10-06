"""sector_resolution.py TDD — 1차(공식 업종 분류) 경로만 다룬다
(recommendation-synthesis-plan.md §5). `resolve_official_sector`는 새 LLM
클라이언트를 만들지 않고 메인 챗봇이 이미 쓰는 `ChatAgentReader`를 재사용한다 —
닫힌 선택지 중 하나(또는 'NONE')만 자유텍스트로 받아, 실제 업종명 목록에 있는지
검증한다(환각 방지: 모델이 목록 밖 이름을 지어내도 무시됨). 2차(LLM 제안+뉴스검증
폴백)는 별도 함수로 다음 단계에서 구현한다."""

from unittest.mock import MagicMock, patch

from auto_stock.chat_agent.credentials import (
    SECTOR_THEME_LLM_CALL_ESTIMATE,
    SECTOR_THEME_RECURSION_LIMIT,
)
from auto_stock.chat_agent.models import ChatModelResponse, QueryBudget
from auto_stock.chat_agent.sector_resolution import (
    SectorThemeSearchResult,
    SectorThemeTickerItem,
    resolve_official_sector,
    resolve_sector_tickers,
    run_sector_theme_search,
)


class _FakeReader:
    model = "gpt-5.6-luna"

    def __init__(self, output_text: str):
        self._output_text = output_text

    def create_response(self, input, tools, tool_choice, previous_response_id):
        return ChatModelResponse(output_text=self._output_text, function_calls=[], response_id="r1")


def test_matched_name_returns_tickers_from_sector_classification(mocker):
    mocker.patch(
        "auto_stock.chat_agent.sector_resolution.fetch_sector_names",
        return_value={"전기전자": ["1013", "2072"], "화학": ["1008"]},
    )
    mocker.patch(
        "auto_stock.chat_agent.sector_resolution.fetch_constituents",
        return_value=["005930", "000660"],
    )
    reader = _FakeReader(output_text="전기전자")

    result = resolve_official_sector(reader, "반도체", "KRX")

    assert result == {"matched": True, "matched_name": "전기전자", "tickers": ["005930", "000660"]}


def test_classifier_output_none_is_not_matched(mocker):
    mocker.patch(
        "auto_stock.chat_agent.sector_resolution.fetch_sector_names",
        return_value={"전기전자": ["1013"]},
    )
    mock_constituents = mocker.patch("auto_stock.chat_agent.sector_resolution.fetch_constituents")
    reader = _FakeReader(output_text="NONE")

    result = resolve_official_sector(reader, "로봇", "KRX")

    assert result == {"matched": False, "matched_name": None, "tickers": []}
    mock_constituents.assert_not_called()


def test_classifier_output_not_in_official_list_is_treated_as_not_matched(mocker):
    """환각 방지 — 모델이 목록에 없는 이름을 출력해도 그대로 믿지 않는다."""
    mocker.patch(
        "auto_stock.chat_agent.sector_resolution.fetch_sector_names",
        return_value={"전기전자": ["1013"]},
    )
    mock_constituents = mocker.patch("auto_stock.chat_agent.sector_resolution.fetch_constituents")
    reader = _FakeReader(output_text="로봇산업")  # 목록에 없는, 지어낸 이름

    result = resolve_official_sector(reader, "로봇", "KRX")

    assert result == {"matched": False, "matched_name": None, "tickers": []}
    mock_constituents.assert_not_called()


def test_unsupported_market_skips_classifier_call_entirely(mocker):
    """pykrx가 지원 안 하는 시장(NASDAQ)이면 fetch_sector_names가 빈 dict를 주므로
    분류할 목록 자체가 없다 — LLM 호출 없이 바로 not matched."""
    mocker.patch("auto_stock.chat_agent.sector_resolution.fetch_sector_names", return_value={})
    reader = mocker.Mock()

    result = resolve_official_sector(reader, "로봇", "NASDAQ")

    assert result == {"matched": False, "matched_name": None, "tickers": []}
    reader.create_response.assert_not_called()


def test_never_raises_when_reader_fails(mocker):
    mocker.patch(
        "auto_stock.chat_agent.sector_resolution.fetch_sector_names",
        return_value={"전기전자": ["1013"]},
    )
    reader = mocker.Mock()
    reader.create_response.side_effect = RuntimeError("API 오류")

    result = resolve_official_sector(reader, "반도체", "KRX")

    assert result == {"matched": False, "matched_name": None, "tickers": []}


class TestRunSectorThemeSearch:
    """2차 폴백 — find_related_companies와 동일한 테스트 원칙: create_deep_agent/
    .invoke를 모킹해 래퍼 로직(예산 사전차감, 실패격리)만 검증한다. 실제 LLM이
    테마에 맞는 종목을 잘 찾는지는 pytest로 주장하지 않는다."""

    def test_returns_tickers_with_provenance_label_on_success(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_structured = SectorThemeSearchResult(
            tickers=[SectorThemeTickerItem(ticker="277810", market="KRX", name="레인보우로보틱스", confidence="confirmed")]
        )
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": fake_structured}
        with patch(
            "auto_stock.chat_agent.sector_resolution.create_deep_agent", return_value=fake_agent
        ) as mock_create:
            result = run_sector_theme_search(budget, "gpt-5.6-luna", "로봇", "KRX")

        assert result["available"] is True
        assert result["tickers"] == [
            {"ticker": "277810", "market": "KRX", "name": "레인보우로보틱스", "confidence": "confirmed"}
        ]
        assert result["provenance"] == "서브에이전트 자율 조사 결과"
        assert budget.llm_calls_made == SECTOR_THEME_LLM_CALL_ESTIMATE
        invoke_kwargs = fake_agent.invoke.call_args.kwargs
        assert invoke_kwargs["config"]["recursion_limit"] == SECTOR_THEME_RECURSION_LIMIT

    def test_returns_unavailable_when_budget_exhausted(self):
        budget = QueryBudget(max_llm_calls=1)
        budget.llm_calls_made = 1
        with patch("auto_stock.chat_agent.sector_resolution.create_deep_agent") as mock_create:
            result = run_sector_theme_search(budget, "gpt-5.6-luna", "로봇", "KRX")

        assert result["available"] is False
        mock_create.assert_not_called()

    def test_never_raises_when_invoke_fails(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.side_effect = RuntimeError("recursion limit exceeded")
        with patch("auto_stock.chat_agent.sector_resolution.create_deep_agent", return_value=fake_agent):
            result = run_sector_theme_search(budget, "gpt-5.6-luna", "로봇", "KRX")

        assert result["available"] is False
        assert "error" in result

    def test_never_raises_when_structured_response_is_present_but_malformed(self):
        budget = QueryBudget(max_llm_calls=10)
        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"structured_response": None}
        with patch("auto_stock.chat_agent.sector_resolution.create_deep_agent", return_value=fake_agent):
            result = run_sector_theme_search(budget, "gpt-5.6-luna", "로봇", "KRX")

        assert result["available"] is False
        assert "error" in result


class TestResolveSectorTickers:
    """1차+2차를 묶는 진입점 — 1차가 매칭되면 2차를 호출하지 않는다(비용 통제)."""

    def test_returns_official_sector_result_without_calling_theme_search(self, mocker):
        mocker.patch(
            "auto_stock.chat_agent.sector_resolution.resolve_official_sector",
            return_value={"matched": True, "matched_name": "전기전자", "tickers": ["005930", "000660"]},
        )
        mock_theme_search = mocker.patch(
            "auto_stock.chat_agent.sector_resolution.run_sector_theme_search"
        )
        budget = QueryBudget(max_llm_calls=10)

        result = resolve_sector_tickers(mocker.Mock(), budget, "gpt-5.6-luna", "반도체", "KRX")

        assert result == {
            "available": True,
            "source": "official_sector",
            "matched_name": "전기전자",
            "tickers": [{"ticker": "005930", "market": "KRX"}, {"ticker": "000660", "market": "KRX"}],
        }
        mock_theme_search.assert_not_called()

    def test_falls_back_to_theme_search_when_not_matched(self, mocker):
        mocker.patch(
            "auto_stock.chat_agent.sector_resolution.resolve_official_sector",
            return_value={"matched": False, "matched_name": None, "tickers": []},
        )
        mocker.patch(
            "auto_stock.chat_agent.sector_resolution.run_sector_theme_search",
            return_value={
                "available": True,
                "tickers": [{"ticker": "277810", "market": "KRX", "name": "레인보우로보틱스", "confidence": "confirmed"}],
                "provenance": "서브에이전트 자율 조사 결과",
            },
        )
        budget = QueryBudget(max_llm_calls=10)

        result = resolve_sector_tickers(mocker.Mock(), budget, "gpt-5.6-luna", "로봇", "KRX")

        assert result == {
            "available": True,
            "source": "llm_theme_search",
            "matched_name": None,
            "tickers": [{"ticker": "277810", "market": "KRX", "name": "레인보우로보틱스", "confidence": "confirmed"}],
            "provenance": "서브에이전트 자율 조사 결과",
        }

    def test_propagates_failure_when_theme_search_also_fails(self, mocker):
        mocker.patch(
            "auto_stock.chat_agent.sector_resolution.resolve_official_sector",
            return_value={"matched": False, "matched_name": None, "tickers": []},
        )
        mocker.patch(
            "auto_stock.chat_agent.sector_resolution.run_sector_theme_search",
            return_value={"available": False, "error": "예산 부족"},
        )
        budget = QueryBudget(max_llm_calls=10)

        result = resolve_sector_tickers(mocker.Mock(), budget, "gpt-5.6-luna", "로봇", "KRX")

        assert result == {"available": False, "error": "예산 부족"}


def test_system_prompt_instructs_treating_tool_output_as_data_not_instructions():
    from auto_stock.chat_agent.sector_resolution import _THEME_SYSTEM_PROMPT

    assert "데이터" in _THEME_SYSTEM_PROMPT
    assert "무시" in _THEME_SYSTEM_PROMPT
