"""edgar_nport_source.py 테스트 — requests를 모킹해 실제 SEC EDGAR API를 호출하지
않는다. N-PORT XML 네임스페이스(`http://www.sec.gov/edgar/nport`)는 2026-10 라이브
호출로 확인한 실제 SPY 공시 구조를 그대로 재현한다."""

import pytest
import requests

from auto_stock.data.sources import edgar_nport_source
from auto_stock.data.sources.edgar_nport_source import (
    EdgarNportError,
    fetch_latest_nport_holdings,
    resolve_etf_cik,
)

_NPORT_NS = "http://www.sec.gov/edgar/nport"


@pytest.fixture(autouse=True)
def _dummy_edgar_user_agent(monkeypatch, mocker):
    mocker.patch("auto_stock.data.sources.edgar_nport_source.load_dotenv")
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "auto_stock-test test@example.com")


def _mf_tickers_payload(entries: list[tuple[str, int]]) -> dict:
    """entries: (symbol, cik_int) 목록으로 company_tickers_mf.json 형태를 만든다."""
    return {
        "fields": ["cik", "seriesId", "classId", "symbol"],
        "data": [[cik, "S000000001", "C000000001", symbol] for symbol, cik in entries],
    }


def _submissions_payload(forms_and_accessions: list[tuple[str, str]]) -> dict:
    return {
        "filings": {
            "recent": {
                "form": [f for f, _ in forms_and_accessions],
                "accessionNumber": [a for _, a in forms_and_accessions],
            }
        }
    }


def _nport_xml(holdings: list[tuple[str, float]]) -> bytes:
    """holdings: (name, pctVal) 목록으로 N-PORT primary_doc.xml(간소화) 바이트를
    만든다 — 라이브 응답과 동일한 네임스페이스."""
    from xml.sax.saxutils import escape

    secs = "".join(
        f'<invstOrSec xmlns="{_NPORT_NS}"><name>{escape(name)}</name><pctVal>{pct}</pctVal></invstOrSec>'
        for name, pct in holdings
    )
    xml = (
        f'<edgarSubmission xmlns="{_NPORT_NS}">'
        f"<formData><invstOrSecs>{secs}</invstOrSecs></formData>"
        f"</edgarSubmission>"
    )
    return xml.encode("utf-8")


class _FakeResponse:
    def __init__(self, json_data=None, content: bytes | None = None):
        self._json_data = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


# --- resolve_etf_cik ---


def test_resolve_etf_cik_hits_equity_map_first(mocker, tmp_path):
    """SPY/DIA/GLD처럼 equity 맵에 있는 UIT형 ETF는 뮤추얼펀드 맵을 조회하지 않는다."""
    mocker.patch("auto_stock.data.sources.edgar_nport_source.resolve_cik", return_value="0000884394")
    mock_mf_get = mocker.patch("auto_stock.data.sources.edgar_nport_source.requests.get")

    result = resolve_etf_cik("SPY", mf_cache_path=tmp_path / "mf.json")

    assert result == "0000884394"
    mock_mf_get.assert_not_called()


def test_resolve_etf_cik_falls_back_to_mutual_fund_map(mocker, tmp_path):
    """QQQ/VOO처럼 equity 맵에 없는 ETF는 뮤추얼펀드 series/class 맵에서 찾는다."""
    mocker.patch("auto_stock.data.sources.edgar_nport_source.resolve_cik", return_value=None)
    payload = _mf_tickers_payload([("QQQ", 1067839)])
    mocker.patch(
        "auto_stock.data.sources.edgar_nport_source.requests.get",
        return_value=_FakeResponse(json_data=payload),
    )

    result = resolve_etf_cik("QQQ", mf_cache_path=tmp_path / "mf.json")

    assert result == "0001067839"


def test_resolve_etf_cik_returns_none_when_absent_from_both_maps(mocker, tmp_path):
    mocker.patch("auto_stock.data.sources.edgar_nport_source.resolve_cik", return_value=None)
    payload = _mf_tickers_payload([("QQQ", 1067839)])
    mocker.patch(
        "auto_stock.data.sources.edgar_nport_source.requests.get",
        return_value=_FakeResponse(json_data=payload),
    )

    result = resolve_etf_cik("NOT_AN_ETF", mf_cache_path=tmp_path / "mf.json")

    assert result is None


def test_resolve_etf_cik_mf_map_uses_cache_without_calling_api_when_fresh(mocker, tmp_path):
    import json

    mocker.patch("auto_stock.data.sources.edgar_nport_source.resolve_cik", return_value=None)
    cache_path = tmp_path / "mf.json"
    cache_path.write_text(json.dumps({"QQQ": "0001067839"}), encoding="utf-8")
    mock_get = mocker.patch("auto_stock.data.sources.edgar_nport_source.requests.get")

    result = resolve_etf_cik("QQQ", mf_cache_path=cache_path)

    assert result == "0001067839"
    mock_get.assert_not_called()


# --- fetch_latest_nport_holdings ---


def test_fetch_latest_nport_holdings_returns_sorted_top_n(mocker):
    submissions = _submissions_payload([("NPORT-P", "0001752724-26-089410")])
    xml_bytes = _nport_xml([("Aflac Inc", 1.5), ("AT&T Inc", 10.0), ("AES Corp/The", 5.0)])

    def fake_get(url, **kwargs):
        if "data.sec.gov" in url:
            return _FakeResponse(json_data=submissions)
        return _FakeResponse(content=xml_bytes)

    mocker.patch("auto_stock.data.sources.edgar_nport_source.requests.get", side_effect=fake_get)

    result = fetch_latest_nport_holdings("0000884394", top_n=2)

    assert result == [("AT&T Inc", 10.0), ("AES Corp/The", 5.0)]  # pctVal 내림차순, top 2


def test_fetch_latest_nport_holdings_returns_none_when_no_nport_filing(mocker):
    submissions = _submissions_payload([("10-K", "0001752724-26-000001")])
    mocker.patch(
        "auto_stock.data.sources.edgar_nport_source.requests.get",
        return_value=_FakeResponse(json_data=submissions),
    )

    result = fetch_latest_nport_holdings("0000320193")

    assert result is None


def test_fetch_latest_nport_holdings_wraps_connection_error(mocker):
    mocker.patch(
        "auto_stock.data.sources.edgar_nport_source.requests.get",
        side_effect=requests.exceptions.ConnectionError("Max retries exceeded"),
    )

    with pytest.raises(EdgarNportError) as exc_info:
        fetch_latest_nport_holdings("0000884394")

    assert "Max retries exceeded" not in str(exc_info.value)


def test_fetch_latest_nport_holdings_wraps_malformed_xml_as_error(mocker):
    submissions = _submissions_payload([("NPORT-P", "0001752724-26-089410")])

    def fake_get(url, **kwargs):
        if "data.sec.gov" in url:
            return _FakeResponse(json_data=submissions)
        return _FakeResponse(content=b"not valid xml <<<")

    mocker.patch("auto_stock.data.sources.edgar_nport_source.requests.get", side_effect=fake_get)

    with pytest.raises(EdgarNportError):
        fetch_latest_nport_holdings("0000884394")


def test_fetch_latest_nport_holdings_skips_entries_missing_name_or_pct(mocker):
    submissions = _submissions_payload([("NPORT-P", "0001752724-26-089410")])
    xml_bytes = _nport_xml([("Valid Co", 2.0)])
    # 이름 없는 holding(현금 등 비상장 구성요소에 대응) 수동 삽입
    xml_bytes = xml_bytes.replace(
        b"</invstOrSecs>",
        f'<invstOrSec xmlns="{_NPORT_NS}"><pctVal>99.0</pctVal></invstOrSec></invstOrSecs>'.encode(),
    )

    def fake_get(url, **kwargs):
        if "data.sec.gov" in url:
            return _FakeResponse(json_data=submissions)
        return _FakeResponse(content=xml_bytes)

    mocker.patch("auto_stock.data.sources.edgar_nport_source.requests.get", side_effect=fake_get)

    result = fetch_latest_nport_holdings("0000884394")

    assert result == [("Valid Co", 2.0)]
