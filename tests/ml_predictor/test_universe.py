from datetime import date

import pytest

import auto_stock.ml_predictor.universe as universe
from auto_stock.ml_predictor.universe import is_common_stock, select_krx_universe, select_universe

AS_OF = date(2021, 10, 25)


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("005930", True),  # 삼성전자 보통주
        ("005935", False),  # 삼성전자우
        ("001465", False),  # BYC우
        ("00104K", False),  # CJ4우(전환)
        ("0126Z0", True),  # 영숫자 신규 코드 보통주(끝자리 0)
        ("12345", False),
    ],
)
def test_is_common_stock(ticker, expected):
    assert is_common_stock(ticker) is expected


def _table(rows):
    return {ticker: {"market_cap": cap, "trading_value": value} for ticker, cap, value in rows}


def test_select_krx_universe_ranks_common_stocks_by_market_cap():
    table = _table(
        [
            ("000010", 100, 5),
            ("005930", 900, 50),
            ("005935", 800, 40),  # preferred — excluded
            ("000020", 500, 0),  # no trading that day (suspended) — excluded
            ("000660", 700, 30),
        ]
    )
    picked = select_krx_universe("KOSPI", AS_OF, size=2, cap_table=lambda as_of, market: table)
    assert picked == ["005930", "000660"]


def test_select_krx_universe_fails_loudly_on_empty_table():
    with pytest.raises(RuntimeError, match="KOSDAQ"):
        select_krx_universe("KOSDAQ", AS_OF, size=10, cap_table=lambda as_of, market: {})


def test_select_universe_routes_krx_markets_to_point_in_time_selection(monkeypatch):
    seen = {}

    def fake_krx(market, as_of, size):
        seen.update(market=market, as_of=as_of, size=size)
        return ["005930"]

    monkeypatch.setattr(universe, "select_krx_universe", fake_krx)
    assert select_universe("KOSDAQ", AS_OF, 3) == ["005930"]
    assert seen == {"market": "KOSDAQ", "as_of": AS_OF, "size": 3}


def test_select_universe_uses_current_listing_for_us(monkeypatch):
    monkeypatch.setattr(universe, "get_universe", lambda market: ["NVDA", "AAPL", "MSFT"])
    assert select_universe("NASDAQ", AS_OF, 2) == ["NVDA", "AAPL"]
