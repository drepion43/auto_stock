"""ticker_resolution.py — 사용자가 채팅에 입력한 문자열(종목코드 또는 회사명)을 (ticker,
market) 후보로 바꾼다. dart_source/edgar_source의 기존 캐시 조회 함수를 모킹해 새 API
호출 없이 테스트한다."""

import pytest

from auto_stock.chat_agent import ticker_resolution
from auto_stock.chat_agent.ticker_resolution import TickerMatch, resolve_ticker

_ETF_LISTING = [("069500", "KODEX 200"), ("091160", "KODEX 반도체"), ("091230", "TIGER 반도체")]


@pytest.fixture(autouse=True)
def _no_live_etf_listing(mocker):
    """ETF 폴백 경로가 테스트 중 실제 FDR을 호출하지 않도록 기본은 빈 목록으로 막는다
    (프로세스 수명 캐시도 매 테스트마다 비운다) — ETF 테스트만 값을 덮어쓴다."""
    ticker_resolution._etf_listing.cache_clear()
    mocker.patch("auto_stock.chat_agent.ticker_resolution.fetch_etf_listing", return_value=[])
    yield
    ticker_resolution._etf_listing.cache_clear()


def test_resolve_ticker_direct_krx_code_match(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_corp_name", return_value="삼성전자")
    mock_name_search = mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name")

    result = resolve_ticker("005930")

    assert result.matches == [TickerMatch(ticker="005930", market="KRX", name="삼성전자")]
    mock_name_search.assert_not_called()  # 직접매치 성공 시 이름검색 폴백을 타지 않음


def test_resolve_ticker_direct_nasdaq_symbol_match(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_corp_name", return_value=None)
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_company_title", return_value="Apple Inc."
    )
    mock_dart_search = mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name")
    mock_edgar_search = mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name")

    result = resolve_ticker("aapl")

    assert result.matches == [TickerMatch(ticker="AAPL", market="NASDAQ", name="Apple Inc.")]
    mock_dart_search.assert_not_called()
    mock_edgar_search.assert_not_called()


def test_resolve_ticker_korean_alias_match_for_nasdaq_stock(mocker):
    """EDGAR의 이름검색은 영문 title만 대상이라 '애플' 같은 한글 질의는 원래 매칭되지
    않는다(실사용 중 발견) — 한글 별칭 표로 직접매치처럼 처리해야 한다."""
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_company_title", return_value="Apple Inc."
    )
    mock_dart_search = mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name")
    mock_edgar_search = mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name")

    result = resolve_ticker("애플")

    assert result.matches == [TickerMatch(ticker="AAPL", market="NASDAQ", name="Apple Inc.")]
    mock_dart_search.assert_not_called()
    mock_edgar_search.assert_not_called()


def test_resolve_ticker_korean_alias_falls_through_when_title_lookup_fails(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_company_title", return_value=None)
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name", return_value=[])
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name", return_value=[])

    result = resolve_ticker("애플")

    assert result.is_not_found


def test_resolve_ticker_unknown_korean_word_is_not_an_alias(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name", return_value=[])
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name", return_value=[])

    result = resolve_ticker("삼성전자")

    assert result.is_not_found


def test_resolve_ticker_falls_back_to_name_search_when_not_a_ticker_code(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name", return_value=[])
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name",
        return_value=[("AAPL", "Apple Inc.")],
    )

    result = resolve_ticker("Apple")

    assert result.matches == [TickerMatch(ticker="AAPL", market="NASDAQ", name="Apple Inc.")]


def test_resolve_ticker_merges_krx_and_nasdaq_matches_from_both_sources(mocker):
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name",
        return_value=[("005930", "삼성전자")],
    )
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name",
        return_value=[("SSNLF", "Samsung Electronics Co Ltd")],
    )

    result = resolve_ticker("삼성")

    assert sorted(result.matches, key=lambda m: m.market) == sorted(
        [
            TickerMatch(ticker="005930", market="KRX", name="삼성전자"),
            TickerMatch(ticker="SSNLF", market="NASDAQ", name="Samsung Electronics Co Ltd"),
        ],
        key=lambda m: m.market,
    )


def test_resolve_ticker_returns_no_matches_for_empty_query():
    result = resolve_ticker("   ")

    assert result.matches == []
    assert result.is_not_found


def test_ticker_resolution_status_properties(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name", return_value=[])
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name", return_value=[])
    not_found = resolve_ticker("존재하지않는회사아무거나")
    assert not_found.is_not_found and not not_found.is_resolved and not not_found.is_ambiguous

    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name",
        return_value=[("005930", "삼성전자")],
    )
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name", return_value=[])
    resolved = resolve_ticker("삼성전자")
    assert resolved.is_resolved and not resolved.is_not_found and not resolved.is_ambiguous

    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name",
        return_value=[("005930", "삼성전자"), ("006400", "삼성SDI")],
    )
    ambiguous = resolve_ticker("삼성")
    assert ambiguous.is_ambiguous and not ambiguous.is_resolved and not ambiguous.is_not_found


# --- ETF 폴백 (DART corpCode.xml은 상장"법인" 전용이라 ETF(펀드)가 없다) ---


def test_resolve_ticker_etf_code_falls_back_to_etf_listing(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_corp_name", return_value=None)
    mocker.patch("auto_stock.chat_agent.ticker_resolution.fetch_etf_listing", return_value=_ETF_LISTING)
    mock_name_search = mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name")

    result = resolve_ticker("069500")

    assert result.matches == [TickerMatch(ticker="069500", market="KRX", name="KODEX 200")]
    mock_name_search.assert_not_called()


def test_resolve_ticker_stock_code_does_not_consult_etf_listing(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_corp_name", return_value="삼성전자")
    mock_etf = mocker.patch("auto_stock.chat_agent.ticker_resolution.fetch_etf_listing")

    resolve_ticker("005930")

    mock_etf.assert_not_called()


def test_resolve_ticker_etf_name_exact_match_when_dart_has_none(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name", return_value=[])
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name", return_value=[])
    mocker.patch("auto_stock.chat_agent.ticker_resolution.fetch_etf_listing", return_value=_ETF_LISTING)

    result = resolve_ticker("kodex 반도체")

    # 완전일치(대소문자 무시)가 있으면 부분일치("TIGER 반도체" 등)를 섞지 않는다 — DART와 동일 규칙
    assert result.matches == [TickerMatch(ticker="091160", market="KRX", name="KODEX 반도체")]


def test_resolve_ticker_etf_name_substring_match_is_ambiguous(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name", return_value=[])
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name", return_value=[])
    mocker.patch("auto_stock.chat_agent.ticker_resolution.fetch_etf_listing", return_value=_ETF_LISTING)

    result = resolve_ticker("반도체")

    assert {m.ticker for m in result.matches} == {"091160", "091230"}
    assert result.is_ambiguous


def test_resolve_ticker_skips_etf_name_search_when_dart_matches(mocker):
    """'삼성전자' 질의에 'KODEX 삼성전자채권혼합' 같은 ETF가 끼어들어 이미 확정되던
    일반 종목 질의를 모호하게 만들면 안 된다."""
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name",
        return_value=[("005930", "삼성전자")],
    )
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name", return_value=[])
    mock_etf = mocker.patch("auto_stock.chat_agent.ticker_resolution.fetch_etf_listing")

    result = resolve_ticker("삼성전자")

    assert result.is_resolved
    mock_etf.assert_not_called()


def test_resolve_ticker_etf_listing_failure_does_not_break_edgar_name_matches(mocker):
    """ETF 목록(FDR)은 보조 경로일 뿐 — 그 장애가 ETF와 무관한 나스닥 이름 질의까지
    `available=false`로 만들면 안 된다(코드 리뷰 발견사항)."""
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name", return_value=[])
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name",
        return_value=[("NVDA", "NVIDIA CORP")],
    )
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.fetch_etf_listing", side_effect=RuntimeError("fdr down")
    )

    result = resolve_ticker("nvidia")

    assert result.matches == [TickerMatch(ticker="NVDA", market="NASDAQ", name="NVIDIA CORP")]


def test_resolve_ticker_etf_listing_failure_on_code_is_not_found_not_error(mocker):
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_corp_name", return_value=None)
    mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.fetch_etf_listing", side_effect=RuntimeError("fdr down")
    )
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_dart_ticker_by_name", return_value=[])
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_edgar_ticker_by_name", return_value=[])

    assert resolve_ticker("069500").is_not_found


def test_etf_listing_is_fetched_once_per_process(mocker):
    mock_etf = mocker.patch(
        "auto_stock.chat_agent.ticker_resolution.fetch_etf_listing", return_value=_ETF_LISTING
    )
    mocker.patch("auto_stock.chat_agent.ticker_resolution.resolve_corp_name", return_value=None)

    resolve_ticker("069500")
    resolve_ticker("091160")

    mock_etf.assert_called_once()
