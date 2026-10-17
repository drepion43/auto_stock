import logging
from datetime import date

import pytest

import auto_stock.ml_predictor.collection as collection
from auto_stock.ml_predictor.collection import collect_universe_records, format_duration

LOGGER = "auto_stock.ml_predictor.collection"
START, END = date(2024, 1, 1), date(2024, 12, 31)


@pytest.fixture
def fake_sources(monkeypatch):
    """Network-free stand-ins: every ticker returns one record; behaviour is tweakable per test."""
    state = {"fail": set(), "fundamentals": {date(2024, 1, 2): {"per": 10.0}}}

    def fake_get_ohlcv(cache, ticker, start, end, market):
        if ticker in state["fail"]:
            raise RuntimeError(f"boom {ticker}")
        return [f"record-{ticker}"]

    def fake_fundamentals(ticker, market, start, end):
        return state["fundamentals"] if market in ("KOSPI", "KOSDAQ") else None

    monkeypatch.setattr(collection, "get_ohlcv", fake_get_ohlcv)
    monkeypatch.setattr(collection, "build_fundamentals_by_date", fake_fundamentals)
    return state


def _progress_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.INFO]


def test_progress_is_logged_every_n_tickers_and_at_the_end(fake_sources, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        collect_universe_records(None, ["A", "B", "C", "D", "E"], START, END, "NASDAQ", log_every=2)

    lines = _progress_lines(caplog)
    assert [line.split()[1] for line in lines] == ["2/5", "4/5", "5/5"]
    assert lines[-1].startswith("[NASDAQ] 5/5 로드=5 스킵=0")
    assert "경과=" in lines[-1] and "종목당=" in lines[-1] and "남은≈" in lines[-1]


def test_skipped_ticker_is_logged_as_warning_with_reason(fake_sources, caplog):
    fake_sources["fail"].add("B")

    with caplog.at_level(logging.INFO, logger=LOGGER):
        records, _ = collect_universe_records(None, ["A", "B", "C"], START, END, "NASDAQ", log_every=10)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["[NASDAQ] 스킵 B: boom B"]
    assert len(records) == 2
    assert "로드=2 스킵=1" in _progress_lines(caplog)[-1]


def test_domestic_ticker_without_fundamentals_is_counted(fake_sources, caplog):
    # pykrx failures return {} silently — the progress line must surface them.
    fake_sources["fundamentals"] = {}

    with caplog.at_level(logging.INFO, logger=LOGGER):
        collect_universe_records(None, ["005930", "000660"], START, END, "KOSPI", log_every=10)

    assert "펀더멘털없음=2" in _progress_lines(caplog)[-1]


def test_us_tickers_are_not_counted_as_missing_fundamentals(fake_sources, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        collect_universe_records(None, ["AAPL"], START, END, "NASDAQ", log_every=10)

    assert "펀더멘털없음=0" in _progress_lines(caplog)[-1]


def test_empty_ticker_list_logs_nothing(fake_sources, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert collect_universe_records(None, [], START, END, "NASDAQ") == ([], [])

    assert _progress_lines(caplog) == []


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(45, "45초"), (125, "2분"), (3600, "1시간0분"), (7080, "1시간58분")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected
