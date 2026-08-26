from datetime import datetime, time
from typing import Callable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler

from auto_stock.data import service
from auto_stock.data.cache import OHLCVCache

MARKET_HOURS = {
    "KRX": {"tz": "Asia/Seoul", "open": time(9, 0), "close": time(15, 30)},
    "NASDAQ": {"tz": "America/New_York", "open": time(9, 30), "close": time(16, 0)},
}


def is_market_open(market: str, now: datetime | None = None) -> bool:
    """시장별 운영시간(평일 + 정규장 시간) 여부 판단 (design: docs/design/data-collection-layer.md §6)."""
    cfg = MARKET_HOURS[market]
    tz = ZoneInfo(cfg["tz"])
    local_now = (now or datetime.now(tz)).astimezone(tz)

    if local_now.weekday() >= 5:  # 토(5)/일(6)
        return False
    return cfg["open"] <= local_now.time() <= cfg["close"]


def poll_market(
    cache: OHLCVCache,
    market: str,
    get_tickers: Callable[[str], list[str]],
    now: datetime | None = None,
) -> None:
    """장 시간에만 대상 종목을 갱신한다. APScheduler job이 주기적으로 호출."""
    if not is_market_open(market, now):
        return
    tickers = get_tickers(market)
    service.refresh_recent(cache, tickers, market=market)


def register_polling_jobs(
    scheduler: BackgroundScheduler,
    cache: OHLCVCache,
    get_tickers: Callable[[str], list[str]],
    interval_minutes: int = 5,
) -> None:
    """시장별로 별도 폴링 job을 등록한다. 간격의 정확한 수치는 실제 API 응답을 보며 튜닝한다."""
    for market in MARKET_HOURS:
        scheduler.add_job(
            poll_market,
            trigger="interval",
            minutes=interval_minutes,
            args=[cache, market, get_tickers],
            id=f"poll_{market}",
            replace_existing=True,
        )
