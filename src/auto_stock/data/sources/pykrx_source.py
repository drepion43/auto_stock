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
    detail = get_market_cap_detail(ticker, as_of)
    return None if detail is None else detail["market_cap"]


def get_market_cap_detail(ticker: str, as_of: date) -> dict | None:
    """특정 종목의 특정일자 시가총액/거래량/거래대금/상장주식수. 데이터가 없으면 None.

    회전율(거래량/상장주식수)을 계산하려면 거래량과 상장주식수가 모두 필요한데,
    `get_market_cap`이 이미 같은 pykrx 호출로 둘 다 받고 있던 걸 시가총액만 뽑고
    버리고 있었다 — 이 함수가 그 나머지 컬럼까지 노출한다(ml_predictor 신규
    펀더멘털 피처용). 일별로 바뀌는 값이라 캐싱하지 않는다(investor_flow.py와
    동일한 "일별 변동 데이터는 캐싱 안 함" 관례)."""
    load_dotenv()
    date_str = as_of.strftime("%Y%m%d")
    df = stock.get_market_cap(date_str, date_str, ticker)
    if df.empty:
        return None
    row = df.iloc[-1]
    return {
        "market_cap": int(row["시가총액"]),
        "volume": int(row["거래량"]),
        "trading_value": int(row["거래대금"]),
        "shares_outstanding": int(row["상장주식수"]),
    }


def get_market_fundamental(ticker: str, as_of: date) -> dict | None:
    """특정 종목의 특정일자 PER/PBR/EPS/BPS/DIV/DPS. 데이터가 없으면 None(에러 아님).

    DIV(배당수익률)/PER/PBR은 당일 종가 기준으로 매일 바뀌므로 캐싱하지 않는다
    (get_market_cap_detail과 동일 원칙)."""
    load_dotenv()
    date_str = as_of.strftime("%Y%m%d")
    df = stock.get_market_fundamental(date_str, date_str, ticker)
    if df.empty:
        return None
    row = df.iloc[-1]
    return {
        "per": float(row["PER"]),
        "pbr": float(row["PBR"]),
        "eps": float(row["EPS"]),
        "bps": float(row["BPS"]),
        "div": float(row["DIV"]),
        "dps": float(row["DPS"]),
    }


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
    # 빈 매핑은 정상 결과일 수 없다(KOSPI+KOSDAQ 상장종목은 늘 수천 개) — 일시적 KRX 실패(로그인
    # 실패·서버 장애)의 흔적이다. 캐시하면 refresh_days 동안 모든 국내 종목의 소속 시장을 모르게
    # 되므로(2026-10-13 실제 발생: 삼성전자가 KOSPI/KOSDAQ 양쪽 후보로 나옴), 빈 캐시는 오래된
    # 것으로 보고 다시 받고, 빈 다운로드 결과는 저장하지 않는다(다음 호출이 다시 시도).
    if _is_cache_fresh(cache_path, refresh_days):
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached:
            return cached
    mapping = _download_ticker_market_map()
    if not mapping:
        return mapping
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # DART/EDGAR 캐시와 동일한 원자적 쓰기 패턴(임시파일 + os.replace) — 쓰기 도중
    # 프로세스가 죽어도 손상된 캐시가 refresh_days 동안 남지 않는다.
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(mapping), encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return mapping


def get_market_fundamental_range(ticker: str, start: date, end: date) -> dict[date, dict]:
    """`[start, end]` 구간 전체의 PER/PBR/EPS/BPS/DIV/DPS를 날짜별 dict로 반환한다.

    ml_predictor가 종목의 과거 전체 시퀀스를 만들 때 날짜마다 반복 호출(단일일자 조회)하는
    대신 한 번에 구간 전체를 받는다 — pykrx가 날짜 범위 조회를 기간조회 함수로 따로 제공하므로
    그걸 그대로 쓴다. 거래일이 아닌 날짜(휴장일)는 결과에 없다(에러 아님) — 호출부가 날짜별로
    `.get(d)`로 조회해 없으면 None으로 취급해야 한다."""
    load_dotenv()
    df = stock.get_market_fundamental_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker)
    if df.empty:
        return {}
    return {
        row_date.date(): {
            "per": float(row["PER"]),
            "pbr": float(row["PBR"]),
            "eps": float(row["EPS"]),
            "bps": float(row["BPS"]),
            "div": float(row["DIV"]),
            "dps": float(row["DPS"]),
        }
        for row_date, row in df.iterrows()
    }


def get_market_cap_detail_range(ticker: str, start: date, end: date) -> dict[date, dict]:
    """`[start, end]` 구간 전체의 시가총액/거래량/거래대금/상장주식수를 날짜별 dict로 반환한다.
    `get_market_fundamental_range`와 동일한 이유로 기간조회 함수를 쓴다."""
    load_dotenv()
    df = stock.get_market_cap_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker)
    if df.empty:
        return {}
    return {
        row_date.date(): {
            "market_cap": int(row["시가총액"]),
            "volume": int(row["거래량"]),
            "trading_value": int(row["거래대금"]),
            "shares_outstanding": int(row["상장주식수"]),
        }
        for row_date, row in df.iterrows()
    }


def resolve_market(
    ticker: str,
    cache_path: Path | str = TICKER_MARKET_CACHE_PATH,
    refresh_days: int = TICKER_MARKET_REFRESH_DAYS,
) -> str | None:
    """국내 종목코드가 KOSPI/KOSDAQ 중 어디에 상장돼 있는지 역조회한다. 캐시에 없는
    티커(코넥스, 상장폐지 등)는 None을 반환한다(에러 아님)."""
    mapping = _load_or_refresh_ticker_market_map(Path(cache_path), refresh_days)
    return mapping.get(ticker)
