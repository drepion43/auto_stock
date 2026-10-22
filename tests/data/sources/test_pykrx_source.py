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


def _market_cap_df(market_cap=450_000_000_000_000, volume=10_000_000, trading_value=5_000_000_000, shares=5_969_782_550):
    return pd.DataFrame(
        {
            "시가총액": [market_cap],
            "거래량": [volume],
            "거래대금": [trading_value],
            "상장주식수": [shares],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )


def _full_market_cap_df():
    return pd.DataFrame(
        {"종가": [70000, 500], "시가총액": [400, 3], "거래량": [10, 1], "거래대금": [700, 0], "상장주식수": [5, 6]},
        index=pd.Index(["005930", "000020"], name="티커"),
    )


def test_get_market_cap_table_returns_rows_by_ticker(mocker):
    call = mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=_full_market_cap_df())
    table = pykrx_source.get_market_cap_table(date(2021, 10, 25), "KOSPI")
    assert table == {
        "005930": {"market_cap": 400, "trading_value": 700},
        "000020": {"market_cap": 3, "trading_value": 0},
    }
    call.assert_called_once_with("20211025", market="KOSPI")


def test_get_market_cap_table_walks_back_over_non_trading_days(mocker):
    call = mocker.patch.object(
        pykrx_source.stock, "get_market_cap", side_effect=[pd.DataFrame(), pd.DataFrame(), _full_market_cap_df()]
    )
    table = pykrx_source.get_market_cap_table(date(2021, 10, 24), "KOSDAQ")
    assert "005930" in table
    assert [c.args[0] for c in call.call_args_list] == ["20211024", "20211023", "20211022"]


def test_get_market_cap_table_treats_all_zero_table_as_non_trading_day(mocker):
    # pykrx는 휴장일에 빈 표 대신 전부 0인 표를 준다(2026-10-22 실측)
    zeros = _full_market_cap_df().assign(시가총액=0, 거래대금=0)
    call = mocker.patch.object(pykrx_source.stock, "get_market_cap", side_effect=[zeros, _full_market_cap_df()])
    table = pykrx_source.get_market_cap_table(date(2021, 10, 23), "KOSPI")
    assert table["005930"]["market_cap"] == 400
    assert [c.args[0] for c in call.call_args_list] == ["20211023", "20211022"]


def test_get_market_cap_table_gives_up_after_lookback(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=pd.DataFrame())
    assert pykrx_source.get_market_cap_table(date(2021, 10, 24), "KOSPI", max_lookback_days=3) == {}


def test_get_market_cap_returns_latest_value(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=_market_cap_df())

    cap = pykrx_source.get_market_cap("005930", date(2026, 1, 2))

    assert cap == 450_000_000_000_000


def test_get_market_cap_returns_none_when_no_data(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=pd.DataFrame())

    cap = pykrx_source.get_market_cap("000000", date(2026, 1, 2))

    assert cap is None


def test_get_market_cap_detail_returns_all_fields(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=_market_cap_df())

    detail = pykrx_source.get_market_cap_detail("005930", date(2026, 1, 2))

    assert detail == {
        "market_cap": 450_000_000_000_000,
        "volume": 10_000_000,
        "trading_value": 5_000_000_000,
        "shares_outstanding": 5_969_782_550,
    }


def test_get_market_cap_detail_returns_none_when_no_data(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_cap", return_value=pd.DataFrame())

    assert pykrx_source.get_market_cap_detail("000000", date(2026, 1, 2)) is None


def test_get_market_fundamental_returns_all_fields(mocker):
    df = pd.DataFrame(
        {
            "BPS": [37528],
            "PER": [26.21875],
            "PBR": [2.210938],
            "EPS": [3166],
            "DIV": [1.709961],
            "DPS": [1416],
        },
        index=pd.to_datetime(["2026-01-02"]),
    )
    mocker.patch.object(pykrx_source.stock, "get_market_fundamental", return_value=df)

    fundamental = pykrx_source.get_market_fundamental("005930", date(2026, 1, 2))

    assert fundamental == {
        "per": 26.21875,
        "pbr": 2.210938,
        "eps": 3166.0,
        "bps": 37528.0,
        "div": 1.709961,
        "dps": 1416.0,
    }


def test_get_market_fundamental_returns_none_when_no_data(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_fundamental", return_value=pd.DataFrame())

    assert pykrx_source.get_market_fundamental("000000", date(2026, 1, 2)) is None


def test_get_market_fundamental_range_keys_by_date(mocker):
    df = pd.DataFrame(
        {
            "BPS": [37528, 37528],
            "PER": [26.21875, 26.5],
            "PBR": [2.210938, 2.240234],
            "EPS": [3166, 3166],
            "DIV": [1.709961, 1.690430],
            "DPS": [1416, 1416],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )
    mocker.patch.object(pykrx_source.stock, "get_market_fundamental_by_date", return_value=df)

    result = pykrx_source.get_market_fundamental_range("005930", date(2026, 1, 2), date(2026, 1, 5))

    assert set(result.keys()) == {date(2026, 1, 2), date(2026, 1, 5)}
    assert result[date(2026, 1, 2)]["per"] == 26.21875


def test_get_market_fundamental_range_returns_empty_when_no_data(mocker):
    mocker.patch.object(pykrx_source.stock, "get_market_fundamental_by_date", return_value=pd.DataFrame())

    result = pykrx_source.get_market_fundamental_range("000000", date(2026, 1, 2), date(2026, 1, 5))

    assert result == {}


def test_get_market_cap_detail_range_keys_by_date(mocker):
    df = pd.DataFrame(
        {
            "시가총액": [450_000_000_000_000, 451_000_000_000_000],
            "거래량": [10_000_000, 11_000_000],
            "거래대금": [5_000_000_000, 5_100_000_000],
            "상장주식수": [5_969_782_550, 5_969_782_550],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )
    mocker.patch.object(pykrx_source.stock, "get_market_cap_by_date", return_value=df)

    result = pykrx_source.get_market_cap_detail_range("005930", date(2026, 1, 2), date(2026, 1, 5))

    assert set(result.keys()) == {date(2026, 1, 2), date(2026, 1, 5)}
    assert result[date(2026, 1, 5)]["volume"] == 11_000_000


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
