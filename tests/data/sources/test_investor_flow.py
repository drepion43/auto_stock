"""investor_flow.py — pykrx 기반 외국인/기관 순매수, 공매도 데이터. pykrx의
`stock` 모듈을 모킹해 실제 KRX API 호출 없이 테스트한다(pykrx_source.py 테스트와
동일 패턴)."""

from datetime import date

import pandas as pd

from auto_stock.data.sources import investor_flow


def _net_purchases_df(ticker: str, net_volume: int, net_value: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "종목명": ["테스트종목"],
            "매도거래량": [0],
            "매수거래량": [0],
            "순매수거래량": [net_volume],
            "매도거래대금": [0],
            "매수거래대금": [0],
            "순매수거래대금": [net_value],
        },
        index=[ticker],
    )


def _shorting_balance_df(ticker: str, balance_shares: int, balance_value: int, ratio: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "공매도잔고": [balance_shares],
            "상장주식수": [1_000_000],
            "공매도금액": [balance_value],
            "시가총액": [1_000_000_000.0],
            "비중": [ratio],
        },
        index=[ticker],
    )


def test_fetch_net_purchases_returns_foreign_and_institution_values(mocker):
    def fake_net_purchases(from_date, to_date, market, investor):
        if investor == "외국인":
            return _net_purchases_df("005930", net_volume=1000, net_value=50_000_000)
        return _net_purchases_df("005930", net_volume=-500, net_value=-20_000_000)

    mocker.patch.object(
        investor_flow.stock,
        "get_market_net_purchases_of_equities_by_ticker",
        side_effect=fake_net_purchases,
    )

    result = investor_flow.fetch_net_purchases("005930", "KOSPI", date(2026, 1, 1), date(2026, 1, 5))

    assert result == {
        "foreign_net_volume": 1000,
        "foreign_net_value": 50_000_000,
        "institution_net_volume": -500,
        "institution_net_value": -20_000_000,
    }


def test_fetch_net_purchases_returns_none_for_unsupported_market(mocker):
    mock_fetch = mocker.patch.object(investor_flow.stock, "get_market_net_purchases_of_equities_by_ticker")

    result = investor_flow.fetch_net_purchases("AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 5))

    assert result is None
    mock_fetch.assert_not_called()


def test_fetch_net_purchases_returns_none_when_ticker_not_in_response(mocker):
    mocker.patch.object(
        investor_flow.stock,
        "get_market_net_purchases_of_equities_by_ticker",
        return_value=_net_purchases_df("000660", net_volume=100, net_value=1000),
    )

    result = investor_flow.fetch_net_purchases("005930", "KOSPI", date(2026, 1, 1), date(2026, 1, 5))

    assert result is None


def test_fetch_shorting_balance_returns_values(mocker):
    mocker.patch.object(
        investor_flow.stock,
        "get_shorting_balance_by_ticker",
        return_value=_shorting_balance_df("005930", balance_shares=10_000, balance_value=500_000_000, ratio=2.5),
    )

    result = investor_flow.fetch_shorting_balance("005930", "KOSPI", date(2026, 1, 2))

    assert result == {
        "balance_shares": 10_000,
        "balance_value": 500_000_000,
        "market_cap_ratio_pct": 2.5,
    }


def test_fetch_shorting_balance_returns_none_for_unsupported_market(mocker):
    mock_fetch = mocker.patch.object(investor_flow.stock, "get_shorting_balance_by_ticker")

    result = investor_flow.fetch_shorting_balance("AAPL", "NYSE", date(2026, 1, 2))

    assert result is None
    mock_fetch.assert_not_called()


def test_fetch_shorting_balance_returns_none_when_ticker_not_in_response(mocker):
    mocker.patch.object(
        investor_flow.stock,
        "get_shorting_balance_by_ticker",
        return_value=_shorting_balance_df("000660", balance_shares=1, balance_value=1, ratio=0.1),
    )

    result = investor_flow.fetch_shorting_balance("005930", "KOSPI", date(2026, 1, 2))

    assert result is None
