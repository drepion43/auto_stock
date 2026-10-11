"""etf_source.py — pykrx(holdings)/FDR(universe/listing) 모킹 테스트. 라이브 호출
없음(fetch_etf_holdings의 실제 로그인 필요성은 scripts/verify_sources_individually.py로
별도 검증, test_investor_flow.py와 동일한 모킹 패턴)."""

from datetime import date

import pandas as pd

from auto_stock.data.sources import etf_source


def _holdings_df() -> pd.DataFrame:
    """2026-10-11 라이브 실측 형태(KODEX 200): 컬럼은 구성종목명/계약수/금액/시가총액/비중.
    pykrx의 `비중`은 `시가총액` 기준으로 계산되는데 주식 행의 시가총액이 전부 0이라
    현금(원화예금) 행만 100.0, 나머지는 0.0으로 나온다 — 실제 비중은 `금액` 기준이다."""
    return pd.DataFrame(
        {
            "구성종목명": ["원화예금", "삼성전자", "SK하이닉스", "LG에너지솔루션"],
            "계약수": [0.0, 10.0, 5.0, 1.0],
            "금액": [0, 500, 300, 200],
            "시가총액": [2791099, 0, 0, 0],
            "비중": [100.0, 0.0, 0.0, 0.0],
        },
        index=["010010", "005930", "000660", "373220"],
    )


def test_fetch_etf_holdings_weights_by_amount_not_broken_weight_column(mocker):
    mocker.patch.object(
        etf_source.stock, "get_etf_portfolio_deposit_file", return_value=_holdings_df()
    )

    result = etf_source.fetch_etf_holdings("069500", top_n=2)

    assert result == [("005930", 50.0), ("000660", 30.0)]


def test_fetch_etf_holdings_passes_explicit_date(mocker):
    """date를 생략하면 pykrx가 `get_index_ohlcv_by_date`로 최근 영업일을 찾는데, 이 환경에서
    그 지수 API가 "Expecting value"로 실패한다(2026-10-11 실측) — 명시 날짜를 넘기면 KRX가
    그 시점의 최신 PDF를 반환한다(일요일 날짜로도 확인)."""
    mock_pdf = mocker.patch.object(
        etf_source.stock, "get_etf_portfolio_deposit_file", return_value=_holdings_df()
    )

    etf_source.fetch_etf_holdings("069500", as_of=date(2026, 10, 11))

    mock_pdf.assert_called_once_with("069500", "20261011")


def test_fetch_etf_holdings_returns_empty_list_when_amounts_are_all_zero(mocker):
    df = _holdings_df().assign(금액=0)
    mocker.patch.object(etf_source.stock, "get_etf_portfolio_deposit_file", return_value=df)

    assert etf_source.fetch_etf_holdings("069500") == []


def test_fetch_etf_holdings_returns_empty_list_when_no_data(mocker):
    mocker.patch.object(
        etf_source.stock, "get_etf_portfolio_deposit_file", return_value=pd.DataFrame()
    )

    result = etf_source.fetch_etf_holdings("999999")

    assert result == []


def test_fetch_etf_holdings_returns_empty_list_when_none(mocker):
    mocker.patch.object(etf_source.stock, "get_etf_portfolio_deposit_file", return_value=None)

    result = etf_source.fetch_etf_holdings("999999")

    assert result == []


def test_fetch_etf_universe_returns_symbol_list(mocker):
    mocker.patch.object(
        etf_source.fdr,
        "StockListing",
        return_value=pd.DataFrame({"Symbol": ["069500", "091160"], "Name": ["KODEX 200", "KODEX 반도체"]}),
    )

    result = etf_source.fetch_etf_universe()

    assert result == ["069500", "091160"]


def test_fetch_etf_listing_returns_symbol_name_pairs(mocker):
    mocker.patch.object(
        etf_source.fdr,
        "StockListing",
        return_value=pd.DataFrame({"Symbol": ["069500", "091160"], "Name": ["KODEX 200", "KODEX 반도체"]}),
    )

    result = etf_source.fetch_etf_listing()

    assert result == [("069500", "KODEX 200"), ("091160", "KODEX 반도체")]
