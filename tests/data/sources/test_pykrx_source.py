import json
from datetime import date

import pandas as pd

from auto_stock.data.sources import pykrx_source


def test_get_ticker_list_delegates_to_pykrx(mocker):
    mock_list = mocker.patch.object(
        pykrx_source.stock, "get_market_ticker_list", return_value=["005930", "000660"]
    )

    tickers = pykrx_source.get_ticker_list(date(2026, 1, 2), market="KOSPI")

    assert tickers == ["005930", "000660"]
    mock_list.assert_called_once_with("20260102", market="KOSPI")


def test_get_market_cap_returns_latest_value(mocker):
    df = pd.DataFrame(
        {"시가총액": [450_000_000_000_000], "거래량": [10_000_000]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=df)

    cap = pykrx_source.get_market_cap("005930", date(2026, 1, 2))

    assert cap == 450_000_000_000_000


def test_get_market_cap_returns_none_when_no_data(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=pd.DataFrame())

    cap = pykrx_source.get_market_cap("000000", date(2026, 1, 2))

    assert cap is None


def test_resolve_market_downloads_and_caches_on_miss(mocker, tmp_path):
    cache_path = tmp_path / "krx_ticker_market.json"

    def fake_ticker_list(as_of_str, market):
        return {"KOSPI": ["005930"], "KOSDAQ": ["091990"]}[market]

    mock_list = mocker.patch.object(
        pykrx_source.stock, "get_market_ticker_list", side_effect=fake_ticker_list
    )

    kospi_market = pykrx_source.resolve_market("005930", cache_path=cache_path)
    kosdaq_market = pykrx_source.resolve_market("091990", cache_path=cache_path)

    assert kospi_market == "KOSPI"
    assert kosdaq_market == "KOSDAQ"
    assert mock_list.call_count == 2  # KOSPI + KOSDAQ, cached across both resolve_market calls
    assert cache_path.exists()


def test_resolve_market_returns_none_for_unknown_ticker(mocker, tmp_path):
    cache_path = tmp_path / "krx_ticker_market.json"
    mocker.patch.object(pykrx_source.stock, "get_market_ticker_list", return_value=["005930"])

    result = pykrx_source.resolve_market("999999", cache_path=cache_path)

    assert result is None


def test_resolve_market_reads_fresh_cache_without_calling_pykrx(mocker, tmp_path):
    cache_path = tmp_path / "krx_ticker_market.json"
    cache_path.write_text(json.dumps({"005930": "KOSPI"}), encoding="utf-8")
    mock_list = mocker.patch.object(pykrx_source.stock, "get_market_ticker_list")

    result = pykrx_source.resolve_market("005930", cache_path=cache_path)

    assert result == "KOSPI"
    mock_list.assert_not_called()


# --- 빈 매핑 캐시 회귀(2026-10-13 발견) ---
# 일시적 KRX 실패로 빈 목록을 받은 결과가 `{}`로 7일간 캐시돼, 그동안 모든 국내 종목의 소속
# 시장을 모른다고 판단했다(삼성전자가 KOSPI/KOSDAQ 양쪽 후보로 나와 챗봇이 매번 되물음).


def test_resolve_market_does_not_cache_an_empty_mapping(mocker, tmp_path):
    cache_path = tmp_path / "krx_ticker_market.json"
    mocker.patch.object(pykrx_source.stock, "get_market_ticker_list", return_value=[])

    assert pykrx_source.resolve_market("005930", cache_path=cache_path) is None
    assert not cache_path.exists()


def test_resolve_market_refreshes_a_fresh_but_empty_cache(mocker, tmp_path):
    cache_path = tmp_path / "krx_ticker_market.json"
    cache_path.write_text("{}", encoding="utf-8")

    def fake_ticker_list(as_of_str, market):
        return {"KOSPI": ["005930"], "KOSDAQ": ["091990"]}[market]

    mocker.patch.object(pykrx_source.stock, "get_market_ticker_list", side_effect=fake_ticker_list)

    assert pykrx_source.resolve_market("005930", cache_path=cache_path) == "KOSPI"
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"005930": "KOSPI", "091990": "KOSDAQ"}
