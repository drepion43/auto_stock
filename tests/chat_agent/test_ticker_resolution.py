"""ticker_resolution.py — 사용자가 채팅에 입력한 문자열(종목코드 또는 회사명)을 (ticker,
market) 후보로 바꾼다. dart_source/edgar_source의 기존 캐시 조회 함수를 모킹해 새 API
호출 없이 테스트한다."""

from auto_stock.chat_agent.ticker_resolution import TickerMatch, resolve_ticker


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
