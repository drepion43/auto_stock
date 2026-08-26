from datetime import datetime
from zoneinfo import ZoneInfo

from auto_stock.data import scheduler


def test_krx_open_on_weekday_during_hours():
    now = datetime(2026, 1, 5, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))  # Monday
    assert scheduler.is_market_open("KRX", now) is True


def test_krx_closed_on_weekday_before_open():
    now = datetime(2026, 1, 5, 8, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    assert scheduler.is_market_open("KRX", now) is False


def test_krx_closed_on_weekend():
    saturday = datetime(2026, 1, 3, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    assert scheduler.is_market_open("KRX", saturday) is False


def test_nasdaq_open_during_us_hours():
    now = datetime(2026, 1, 5, 10, 0, tzinfo=ZoneInfo("America/New_York"))  # Monday, 10:00 ET
    assert scheduler.is_market_open("NASDAQ", now) is True


def test_nasdaq_closed_outside_us_hours():
    now = datetime(2026, 1, 5, 20, 0, tzinfo=ZoneInfo("America/New_York"))
    assert scheduler.is_market_open("NASDAQ", now) is False


def test_poll_market_skips_refresh_when_market_closed(mocker):
    cache = mocker.Mock()
    refresh = mocker.patch("auto_stock.data.scheduler.service.refresh_recent")
    closed_time = datetime(2026, 1, 3, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))  # Saturday

    scheduler.poll_market(cache, "KRX", get_tickers=lambda market: ["005930"], now=closed_time)

    refresh.assert_not_called()


def test_poll_market_refreshes_when_market_open(mocker):
    cache = mocker.Mock()
    refresh = mocker.patch("auto_stock.data.scheduler.service.refresh_recent")
    open_time = datetime(2026, 1, 5, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))  # Monday

    scheduler.poll_market(cache, "KRX", get_tickers=lambda market: ["005930"], now=open_time)

    refresh.assert_called_once_with(cache, ["005930"], market="KRX")
