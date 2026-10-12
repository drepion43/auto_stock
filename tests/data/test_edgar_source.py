"""edgar_source.py 테스트 — requests를 모킹해 실제 SEC EDGAR API를 호출하지 않는다."""

import json
from datetime import date

import pytest
import requests

from auto_stock.data.models import DisclosureRecord
from auto_stock.data.sources.edgar_source import (
    COMPANY_TICKERS_EXCHANGE_URL,
    COMPANY_TICKERS_URL,
    EdgarApiError,
    fetch_filings,
    resolve_cik,
    resolve_company_title,
    resolve_exchange,
    resolve_ticker_by_name,
)


@pytest.fixture(autouse=True)
def _dummy_edgar_user_agent(monkeypatch, mocker):
    # 실제 .env에 SEC_EDGAR_USER_AGENT가 생기더라도 테스트가 영향받지 않도록 load_dotenv 자체를 모킹
    mocker.patch("auto_stock.data.sources.edgar_source.load_dotenv")
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "auto_stock-test test@example.com")


def _company_tickers_payload(entries: list[tuple[str, int, str]]) -> dict:
    """entries: (ticker, cik_int, title) 목록으로 company_tickers.json 형태를 만든다.
    (실제 SEC 응답과 동일하게) exchange 필드는 포함하지 않는다 — exchange는 별도
    company_tickers_exchange.json에서만 온다."""
    return {
        str(i): {"cik_str": cik, "ticker": ticker, "title": title}
        for i, (ticker, cik, title) in enumerate(entries)
    }


def _exchange_payload(exchanges: dict[str, str]) -> dict:
    """{ticker: exchange} 목록으로 company_tickers_exchange.json 형태(fields/data
    배열)를 만든다 — 실제 SEC 응답과 동일한 스키마."""
    return {
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [[0, "", ticker, exchange] for ticker, exchange in exchanges.items()],
    }


def _mock_edgar_get(mocker, company_tickers_payload: dict, exchange_payload: dict | None = None):
    """`requests.get`을 URL별로 분기하는 FakeResponse로 모킹한다 —
    `_download_cik_map`이 company_tickers.json과 company_tickers_exchange.json을
    각각 호출하므로 둘 다 응답해야 한다."""
    exchange_payload = exchange_payload if exchange_payload is not None else _exchange_payload({})

    def fake_get(url, **kwargs):
        if url == COMPANY_TICKERS_URL:
            return _FakeResponse(json_data=company_tickers_payload)
        if url == COMPANY_TICKERS_EXCHANGE_URL:
            return _FakeResponse(json_data=exchange_payload)
        raise AssertionError(f"unexpected URL: {url}")

    return mocker.patch("auto_stock.data.sources.edgar_source.requests.get", side_effect=fake_get)


def _submissions_payload(ticker: str, filings: list[tuple[str, str]]) -> dict:
    """filings: (form, filingDate) 목록으로 submissions API 응답(컬럼형 배열)을 만든다."""
    return {
        "cik": "0000320193",
        "tickers": [ticker],
        "filings": {
            "recent": {
                "form": [form for form, _ in filings],
                "filingDate": [filing_date for _, filing_date in filings],
                "accessionNumber": [f"0001-{i}" for i in range(len(filings))],
            }
        },
    }


class _FakeResponse:
    def __init__(self, json_data: dict | None = None):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_resolve_cik_downloads_and_caches_when_cache_missing(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("AAPL", 320193, "Apple Inc."), ("NVDA", 1045810, "NVIDIA CORP")])
    mock_get = _mock_edgar_get(mocker, payload)

    result = resolve_cik("AAPL", cache_path=cache_path)

    assert result == "0000320193"
    assert cache_path.exists()
    assert mock_get.call_count == 2  # company_tickers.json + company_tickers_exchange.json


def test_resolve_cik_pads_cik_to_ten_digits(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("NVDA", 1045810, "NVIDIA CORP")])
    _mock_edgar_get(mocker, payload)

    result = resolve_cik("NVDA", cache_path=cache_path)

    assert result == "0001045810"
    assert len(result) == 10


def test_resolve_cik_returns_none_for_ticker_not_in_edgar(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("AAPL", 320193, "Apple Inc.")])
    _mock_edgar_get(mocker, payload)

    result = resolve_cik("005930", cache_path=cache_path)  # 한국 상장사 — EDGAR 미등록

    assert result is None


def test_resolve_cik_uses_cache_without_calling_api_when_fresh(mocker, tmp_path):
    import json

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(json.dumps({"AAPL": "0000320193"}), encoding="utf-8")
    mock_get = mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    result = resolve_cik("AAPL", cache_path=cache_path)

    assert result == "0000320193"
    mock_get.assert_not_called()


def test_resolve_cik_redownloads_when_cache_is_stale(mocker, tmp_path):
    import json
    import os
    import time

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(json.dumps({"AAPL": "OLD_CIK"}), encoding="utf-8")
    stale_time = time.time() - 8 * 86400  # 기본 갱신주기(7일) 초과
    os.utime(cache_path, (stale_time, stale_time))

    payload = _company_tickers_payload([("AAPL", 320193, "Apple Inc.")])
    _mock_edgar_get(mocker, payload)

    result = resolve_cik("AAPL", cache_path=cache_path)

    assert result == "0000320193"


def test_resolve_cik_leaves_no_leftover_temp_file_after_cache_write(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("AAPL", 320193, "Apple Inc.")])
    _mock_edgar_get(mocker, payload)

    resolve_cik("AAPL", cache_path=cache_path)

    leftover_files = {p.name for p in tmp_path.iterdir()} - {cache_path.name}
    assert leftover_files == set()


def test_resolve_cik_wraps_connection_error_as_edgar_api_error(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    mocker.patch(
        "auto_stock.data.sources.edgar_source.requests.get",
        side_effect=requests.exceptions.ConnectionError("Max retries exceeded"),
    )

    with pytest.raises(EdgarApiError) as exc_info:
        resolve_cik("AAPL", cache_path=cache_path)

    assert "Max retries exceeded" not in str(exc_info.value)


def test_resolve_cik_requires_edgar_user_agent(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)

    with pytest.raises(KeyError, match="SEC_EDGAR_USER_AGENT"):
        resolve_cik("AAPL", cache_path=tmp_path / "edgar_cik_map.json")


def test_fetch_filings_returns_records_within_date_range(mocker):
    payload = _submissions_payload(
        "AAPL",
        [
            ("10-K", "2026-01-15"),
            ("8-K", "2025-06-01"),  # 조회 기간 밖
        ],
    )
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get", return_value=_FakeResponse(json_data=payload))

    records = fetch_filings("0000320193", date(2026, 1, 1), date(2026, 1, 31))

    assert records == [
        DisclosureRecord(
            corp_code="0000320193",
            ticker="AAPL",
            report_name="연차보고서(10-K)",
            filed_date=date(2026, 1, 15),
            remark="",
        )
    ]


def test_fetch_filings_excludes_form_3_4_5(mocker):
    payload = _submissions_payload(
        "AAPL",
        [
            ("4", "2026-01-10"),
            ("3", "2026-01-11"),
            ("5", "2026-01-12"),
            ("8-K", "2026-01-15"),
        ],
    )
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get", return_value=_FakeResponse(json_data=payload))

    records = fetch_filings("0000320193", date(2026, 1, 1), date(2026, 1, 31))

    assert len(records) == 1
    assert records[0].report_name == "주요사항보고(8-K)"


def test_fetch_filings_excludes_form_3_4_5_amendments(mocker):
    """코드 리뷰 MEDIUM: "3/A"/"4/A"/"5/A"(정정판)는 정확 일치 필터를 우회해 통과했었다."""
    payload = _submissions_payload(
        "AAPL",
        [
            ("4/A", "2026-01-10"),
            ("3/A", "2026-01-11"),
            ("5/A", "2026-01-12"),
            ("8-K", "2026-01-15"),
        ],
    )
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get", return_value=_FakeResponse(json_data=payload))

    records = fetch_filings("0000320193", date(2026, 1, 1), date(2026, 1, 31))

    assert len(records) == 1
    assert records[0].report_name == "주요사항보고(8-K)"


def test_fetch_filings_uses_raw_form_code_when_no_label(mocker):
    payload = _submissions_payload("AAPL", [("SC TO-T", "2026-01-15")])
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get", return_value=_FakeResponse(json_data=payload))

    records = fetch_filings("0000320193", date(2026, 1, 1), date(2026, 1, 31))

    assert records[0].report_name == "SC TO-T"


def test_fetch_filings_returns_empty_list_when_no_filings_in_range(mocker):
    payload = _submissions_payload("AAPL", [("10-K", "2020-01-01")])
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get", return_value=_FakeResponse(json_data=payload))

    records = fetch_filings("0000320193", date(2026, 1, 1), date(2026, 1, 31))

    assert records == []


def test_fetch_filings_requires_edgar_user_agent(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)

    with pytest.raises(KeyError, match="SEC_EDGAR_USER_AGENT"):
        fetch_filings("0000320193", date(2026, 1, 1), date(2026, 1, 31))


def test_fetch_filings_wraps_connection_error_as_edgar_api_error(mocker):
    mocker.patch(
        "auto_stock.data.sources.edgar_source.requests.get",
        side_effect=requests.exceptions.ConnectionError("Max retries exceeded"),
    )

    with pytest.raises(EdgarApiError) as exc_info:
        fetch_filings("0000320193", date(2026, 1, 1), date(2026, 1, 31))

    assert "Max retries exceeded" not in str(exc_info.value)


def test_fetch_filings_wraps_malformed_filing_date_as_edgar_api_error(mocker):
    """보안/코드 리뷰 LOW: date.fromisoformat이 다른 실패 경로처럼 정제되지 않고
    있었다 — SEC가 예상 밖 날짜 형식을 보내면 원본 ValueError가 그대로 새어나갔다."""
    payload = _submissions_payload("AAPL", [("10-K", "not-a-date")])
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get", return_value=_FakeResponse(json_data=payload))

    with pytest.raises(EdgarApiError):
        fetch_filings("0000320193", date(2026, 1, 1), date(2026, 1, 31))


# --- market 세분화(2026-10) 확장: resolve_exchange — 영문 심볼이 실제로 NASDAQ인지
# NYSE인지 역조회한다. resolve_cik/resolve_company_title와 캐시를 공유한다. ---


def test_resolve_exchange_normalizes_nasdaq(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("AAPL", 320193, "Apple Inc.")])
    _mock_edgar_get(mocker, payload, _exchange_payload({"AAPL": "Nasdaq"}))

    assert resolve_exchange("AAPL", cache_path=cache_path) == "NASDAQ"


def test_resolve_exchange_normalizes_nyse(mocker, tmp_path):
    """JPM은 실제로 NYSE 상장이다 — ticker_resolution.py의 한글 별칭표가 과거
    NASDAQ으로 하드코딩했던 오분류 사례(market 세분화 작업에서 발견)."""
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("JPM", 19617, "JPMORGAN CHASE & CO")])
    _mock_edgar_get(mocker, payload, _exchange_payload({"JPM": "NYSE"}))

    assert resolve_exchange("JPM", cache_path=cache_path) == "NYSE"


def test_resolve_exchange_returns_none_for_unrecognized_exchange_name(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("SPY", 884394, "SPDR S&P 500 ETF TRUST")])
    _mock_edgar_get(mocker, payload, _exchange_payload({"SPY": "batsglobal"}))

    assert resolve_exchange("SPY", cache_path=cache_path) is None


def test_resolve_exchange_returns_none_when_ticker_missing_from_exchange_map(mocker, tmp_path):
    """`company_tickers.json`에는 있지만 `company_tickers_exchange.json`에는 없는
    티커(두 파일의 커버리지가 완전히 같지 않을 수 있음)는 조용히 None — 2026-10
    버그 수정(exchange가 애초에 다른 파일에 있었다는 사실이 드러난 계기)."""
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("AAPL", 320193, "Apple Inc.")])
    _mock_edgar_get(mocker, payload, _exchange_payload({}))  # exchange 맵에 AAPL 없음

    assert resolve_exchange("AAPL", cache_path=cache_path) is None


def test_resolve_exchange_returns_none_for_legacy_flat_cache_schema(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(json.dumps({"AAPL": "0000320193"}), encoding="utf-8")  # 구 스키마
    mock_get = mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    assert resolve_exchange("AAPL", cache_path=cache_path) is None
    mock_get.assert_not_called()


def test_resolve_exchange_returns_none_for_intermediate_schema_without_exchange_field(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(
        json.dumps({"AAPL": {"cik": "0000320193", "title": "Apple Inc."}}), encoding="utf-8"
    )  # exchange 필드 추가 이전의 중간 스키마
    mock_get = mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    assert resolve_exchange("AAPL", cache_path=cache_path) is None
    mock_get.assert_not_called()


# --- 뉴스 감성분석(#4-뉴스) 확장: resolve_company_title — 뉴스 검색 쿼리를 만들기
# 위한 회사명 조회. resolve_cik와 캐시를 공유한다(docs/design/news-sentiment.md
# 핵심 설계 결정 2). ---


def test_resolve_company_title_downloads_and_caches_when_cache_missing(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("AAPL", 320193, "Apple Inc.")])
    mock_get = _mock_edgar_get(mocker, payload)

    result = resolve_company_title("AAPL", cache_path=cache_path)

    assert result == "Apple Inc."
    assert mock_get.call_count == 2


def test_resolve_company_title_returns_none_for_ticker_not_in_edgar(mocker, tmp_path):
    cache_path = tmp_path / "edgar_cik_map.json"
    payload = _company_tickers_payload([("AAPL", 320193, "Apple Inc.")])
    _mock_edgar_get(mocker, payload)

    result = resolve_company_title("005930", cache_path=cache_path)

    assert result is None


def test_resolve_company_title_uses_cache_without_calling_api_when_fresh(mocker, tmp_path):
    import json

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(
        json.dumps({"AAPL": {"cik": "0000320193", "title": "Apple Inc."}}), encoding="utf-8"
    )
    mock_get = mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    result = resolve_company_title("AAPL", cache_path=cache_path)

    assert result == "Apple Inc."
    mock_get.assert_not_called()


def test_resolve_company_title_returns_none_gracefully_for_legacy_flat_cache_schema(mocker, tmp_path):
    """뉴스 감성분석 확장 전(title이 없던 시절)에 저장된 평면 캐시를 만나도 크래시하지
    않고 None을 반환해야 한다."""
    import json

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(json.dumps({"AAPL": "0000320193"}), encoding="utf-8")  # 구 스키마
    mock_get = mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    title_result = resolve_company_title("AAPL", cache_path=cache_path)
    cik_result = resolve_cik("AAPL", cache_path=cache_path)

    assert title_result is None
    assert cik_result == "0000320193"  # cik 조회는 구 스키마에서도 여전히 동작
    mock_get.assert_not_called()


# --- 대화형 챗봇(Stage B) 확장: resolve_ticker_by_name — 사용자가 채팅에 입력한
# 회사명 문자열로 티커를 역조회한다(resolve_company_title의 반대 방향). 기존 캐시를
# 그대로 재사용하며 새 EDGAR API 호출은 없다. ---


def test_resolve_ticker_by_name_exact_match_is_case_insensitive(mocker, tmp_path):
    import json

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(
        json.dumps(
            {
                "AAPL": {"cik": "0000320193", "title": "Apple Inc."},
                "MSFT": {"cik": "0000789019", "title": "Microsoft Corp"},
            }
        ),
        encoding="utf-8",
    )
    mock_get = mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    result = resolve_ticker_by_name("microsoft corp", cache_path=cache_path)

    assert result == [("MSFT", "Microsoft Corp")]
    mock_get.assert_not_called()


def test_resolve_ticker_by_name_falls_back_to_substring_match(mocker, tmp_path):
    import json

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(
        json.dumps(
            {
                "AAPL": {"cik": "0000320193", "title": "Apple Inc."},
                "APLE": {"cik": "0001628106", "title": "Apple Hospitality REIT Inc."},
            }
        ),
        encoding="utf-8",
    )
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    result = resolve_ticker_by_name("Apple", cache_path=cache_path)

    assert sorted(result) == [
        ("AAPL", "Apple Inc."),
        ("APLE", "Apple Hospitality REIT Inc."),
    ]


def test_resolve_ticker_by_name_exact_match_excludes_broader_substring_hits(mocker, tmp_path):
    """완전일치가 하나라도 있으면 부분일치 후보는 섞지 않는다."""
    import json

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(
        json.dumps(
            {
                "AAPL": {"cik": "0000320193", "title": "Apple Inc."},
                "APLE": {"cik": "0001628106", "title": "Apple Hospitality REIT Inc."},
            }
        ),
        encoding="utf-8",
    )
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    result = resolve_ticker_by_name("Apple Inc.", cache_path=cache_path)

    assert result == [("AAPL", "Apple Inc.")]


def test_resolve_ticker_by_name_returns_empty_list_when_no_match(mocker, tmp_path):
    import json

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(
        json.dumps({"AAPL": {"cik": "0000320193", "title": "Apple Inc."}}), encoding="utf-8"
    )
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    result = resolve_ticker_by_name("존재하지않는회사아무거나", cache_path=cache_path)

    assert result == []


def test_resolve_ticker_by_name_excludes_legacy_flat_cache_entries(mocker, tmp_path):
    """이름이 없는 구 스키마(문자열) 캐시 항목은 이름 매칭 대상에서 제외한다."""
    import json

    cache_path = tmp_path / "edgar_cik_map.json"
    cache_path.write_text(
        json.dumps(
            {
                "AAPL": "0000320193",  # 구 스키마 — title 없음
                "MSFT": {"cik": "0000789019", "title": "Microsoft Corp"},
            }
        ),
        encoding="utf-8",
    )
    mocker.patch("auto_stock.data.sources.edgar_source.requests.get")

    result = resolve_ticker_by_name("Microsoft Corp", cache_path=cache_path)

    assert result == [("MSFT", "Microsoft Corp")]
