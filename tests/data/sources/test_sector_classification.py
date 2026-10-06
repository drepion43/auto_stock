"""sector_classification.py TDD — recommendation-synthesis-plan.md §5 1차(공식 업종)
경로. pykrx의 지수 API를 모킹해 래퍼 로직(코스피+코스닥 병합, 구성종목 합집합)만
검증한다 — 2026-10-05 실제 pykrx 호출로 `load_dotenv()` 선행이 필요함을 확인했다
(KRX_ID/KRX_PW 로그인, pykrx_source.py와 동일 패턴). NASDAQ은 pykrx 대상이 아니라
이 모듈은 market="KRX"만 지원한다(사용자 확인)."""

from auto_stock.data.sources import sector_classification


def test_fetch_sector_names_merges_kospi_and_kosdaq(mocker):
    """코스피/코스닥에 동명 업종(예: "전기전자")이 둘 다 있을 수 있다 — 코드 목록으로
    합쳐둬야 ②단계에서 양쪽 구성종목을 모두 조회할 수 있다."""

    def fake_ticker_list(market):
        return {"KOSPI": ["1013"], "KOSDAQ": ["2072"]}[market]

    def fake_ticker_name(code):
        return {"1013": "전기전자", "2072": "전기전자"}[code]

    mocker.patch.object(sector_classification.stock, "get_index_ticker_list", side_effect=fake_ticker_list)
    mocker.patch.object(sector_classification.stock, "get_index_ticker_name", side_effect=fake_ticker_name)

    names = sector_classification.fetch_sector_names("KRX")

    assert names == {"전기전자": ["1013", "2072"]}


def test_fetch_sector_names_keeps_distinct_names_separate(mocker):
    def fake_ticker_list(market):
        return {"KOSPI": ["1008"], "KOSDAQ": ["2031"]}[market]

    def fake_ticker_name(code):
        return {"1008": "화학", "2031": "금융"}[code]

    mocker.patch.object(sector_classification.stock, "get_index_ticker_list", side_effect=fake_ticker_list)
    mocker.patch.object(sector_classification.stock, "get_index_ticker_name", side_effect=fake_ticker_name)

    names = sector_classification.fetch_sector_names("KRX")

    assert names == {"화학": ["1008"], "금융": ["2031"]}


def test_fetch_sector_names_returns_empty_for_unsupported_market(mocker):
    """pykrx는 KRX 전용이다 — NASDAQ은 2차 LLM 폴백으로만 처리된다(사용자 확인,
    2026-10-05)."""
    mock_list = mocker.patch.object(sector_classification.stock, "get_index_ticker_list")

    names = sector_classification.fetch_sector_names("NASDAQ")

    assert names == {}
    mock_list.assert_not_called()


def test_fetch_constituents_unions_multiple_index_codes_without_duplicates(mocker):
    def fake_deposit_file(code):
        return {"1013": ["005930", "000660"], "2072": ["000660", "123456"]}[code]

    mocker.patch.object(
        sector_classification.stock, "get_index_portfolio_deposit_file", side_effect=fake_deposit_file
    )

    tickers = sector_classification.fetch_constituents(["1013", "2072"])

    assert tickers == ["005930", "000660", "123456"]  # 중복(000660) 1번만, 순서 보존


def test_fetch_constituents_empty_list_returns_empty_without_calling_pykrx(mocker):
    mock_deposit = mocker.patch.object(sector_classification.stock, "get_index_portfolio_deposit_file")

    tickers = sector_classification.fetch_constituents([])

    assert tickers == []
    mock_deposit.assert_not_called()
