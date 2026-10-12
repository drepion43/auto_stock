import contextlib
import io
import json
import os
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# pykrx는 import되는 순간(모듈 최상위 `website/comm/webio.py:12`의
# `_session = build_krx_session()`) KRX 로그인을 시도하고, KRX_ID/KRX_PW가 없으면
# "KRX 로그인 실패..." 노이즈를 stdout에 출력한다 — 이 프로젝트는 그 로그인이 필요한
# 프리미엄 엔드포인트를 쓰지 않고 공개 OHLCV/티커 목록만 쓰므로 실제 동작에는 영향이
# 없다(사용자가 실제 챗봇 실행 중 이 메시지를 보고 문의해 확인함, 2026-09-06). import
# 시점의 출력만 억제하고, 이후 실제 함수 호출 시의 출력(있다면)은 그대로 둔다.
with contextlib.redirect_stdout(io.StringIO()):
    from pykrx import stock

TICKER_MARKET_CACHE_PATH = Path("data/krx_ticker_market.json")
TICKER_MARKET_REFRESH_DAYS = 7

_SUB_MARKETS = ("KOSPI", "KOSDAQ")  # KOSPI/KOSDAQ 동시 상장 종목은 없다(KRX 상장 규정) —
# 그래서 역방향 매핑이 1:1로 성립한다.


def get_ticker_list(as_of: date, market: str = "ALL") -> list[str]:
    """전체 상장종목 코드 리스트 (KRX)."""
    load_dotenv()
    return stock.get_market_ticker_list(as_of.strftime("%Y%m%d"), market=market)


def get_market_cap(ticker: str, as_of: date) -> int | None:
    """특정 종목의 특정일자 시가총액. 데이터가 없으면 None."""
    load_dotenv()
    date_str = as_of.strftime("%Y%m%d")
    df = stock.get_market_cap(date_str, date_str, ticker)
    if df.empty:
        return None
    return int(df.iloc[-1]["시가총액"])


def _download_ticker_market_map() -> dict[str, str]:
    """KOSPI/KOSDAQ 각각의 종목코드 리스트를 받아 {ticker: sub_market} 역방향
    매핑으로 뒤집는다. 동시 상장 종목이 없다는 KRX 상장 규정이 전제다."""
    load_dotenv()
    mapping: dict[str, str] = {}
    as_of_str = date.today().strftime("%Y%m%d")
    for sub_market in _SUB_MARKETS:
        for ticker in stock.get_market_ticker_list(as_of_str, market=sub_market):
            mapping[ticker] = sub_market
    return mapping


def _is_cache_fresh(cache_path: Path, refresh_days: int) -> bool:
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < refresh_days * 86400


def _load_or_refresh_ticker_market_map(cache_path: Path, refresh_days: int) -> dict[str, str]:
    if _is_cache_fresh(cache_path, refresh_days):
        return json.loads(cache_path.read_text(encoding="utf-8"))
    mapping = _download_ticker_market_map()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # DART/EDGAR 캐시와 동일한 원자적 쓰기 패턴(임시파일 + os.replace) — 쓰기 도중
    # 프로세스가 죽어도 손상된 캐시가 refresh_days 동안 남지 않는다.
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(mapping), encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return mapping


def resolve_market(
    ticker: str,
    cache_path: Path | str = TICKER_MARKET_CACHE_PATH,
    refresh_days: int = TICKER_MARKET_REFRESH_DAYS,
) -> str | None:
    """국내 종목코드가 KOSPI/KOSDAQ 중 어디에 상장돼 있는지 역조회한다. 캐시에 없는
    티커(코넥스, 상장폐지 등)는 None을 반환한다(에러 아님)."""
    mapping = _load_or_refresh_ticker_market_map(Path(cache_path), refresh_days)
    return mapping.get(ticker)
