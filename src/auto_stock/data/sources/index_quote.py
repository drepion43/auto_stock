"""지수 자체의 시세 조회 — 코스피/코스닥/코스피200/VIX/다우존스/나스닥종합/S&P500/
러셀2000. 종목(ticker)이 아니라 지수 자체를 묻는 챗봇 질의("VIX 지금 어때",
"코스피 지금 어때") 전용 데이터 소스다.

`fdr.DataReader(symbol)`이 Yahoo Finance를 거쳐 지수의 OHLCV+Adj Close를 그대로
반환한다(FDR 소스코드 확인: `data.py`의 `yahoo_index_symbol_map`을 거쳐
`YahooDailyReader`로 라우팅됨) — `fdr_source.fetch_ohlcv`와 달리 `OHLCVRecord`로
감싸지 않는다. 지수는 "종목"이 아니라 market 필드를 가질 이유가 없고(`OHLCVRecord`의
`VALID_MARKETS` 검증과 무관한 별개의 데이터 형태), 챗봇에는 최근 종가/전일대비
변화율만 보여주면 충분하기 때문이다.

**국내 지수(KS11/KQ11/KS200)는 FDR이 아니라 pykrx(KRX 공식)로 조회한다** — FDR은 이 심볼들을
Yahoo가 아니라 KRX 소스로 보내는데, 그 소스가 약 4주 늦은 데이터만 돌려줘서(2026-10-13 실측:
9/17까지) 최근 구간 조회가 늘 비어 국내 지수 질의가 전부 "데이터 없음"이었다. pykrx가 실패하면
Yahoo 심볼(^KS11/^KQ11)로 폴백한다(코스피200은 Yahoo에 데이터가 없어 폴백 없음). pykrx는
`pykrx_source.py`와 같은 이유로 호출 시점에 `load_dotenv()`를 불러 KRX 로그인을 보장한다.

**신선도 가드**: 최신 데이터가 `_MAX_STALENESS_DAYS`보다 오래됐으면 None — 몇 주 지난 값을 현재
시세처럼 보여주지 않기 위함(연휴+주말을 넘길 만큼 여유를 둔다)."""

import contextlib
import io
from dataclasses import dataclass
from datetime import date, timedelta

import FinanceDataReader as fdr
import pandas as pd
from dotenv import load_dotenv

with contextlib.redirect_stdout(io.StringIO()):
    from pykrx import stock

# fdr.DataReader가 받는 심볼 — 챗봇 tool enum(tool_schemas.py의 get_index_quote)과
# 반드시 동기화해야 한다.
SUPPORTED_INDICES = frozenset({"KS11", "KQ11", "KS200", "VIX", "DJI", "IXIC", "S&P500", "RUT"})

_LOOKBACK_DAYS = 10  # 전일대비 변화율 계산에 필요한 최소 2거래일 + 공휴일 여유
_MAX_STALENESS_DAYS = 10  # 이보다 오래된 최신 데이터는 "데이터 없음"으로 본다

_KRX_INDEX_CODES = {"KS11": "1001", "KQ11": "2001", "KS200": "1028"}  # KRX 지수 코드(코스피/코스닥/코스피200)
_YAHOO_FALLBACK = {"KS11": "^KS11", "KQ11": "^KQ11"}


def _today() -> date:
    return date.today()


class UnsupportedIndexError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class IndexQuote:
    index: str
    date: date
    close: float
    change_pct: float | None  # 직전 거래일 대비 변화율(%). 직전 데이터가 없으면 None


def _closes_from_krx(code: str, start: date, end: date) -> pd.Series:
    load_dotenv()
    df = stock.get_index_ohlcv_by_date(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code)
    return df["종가"] if df is not None and not df.empty else pd.Series(dtype=float)


def _closes_from_fdr(symbol: str, start: date, end: date) -> pd.Series:
    df = fdr.DataReader(symbol, start, end)
    return df["Close"] if df is not None and not df.empty else pd.Series(dtype=float)


def _closes(index: str, start: date, end: date) -> pd.Series:
    if index not in _KRX_INDEX_CODES:
        return _closes_from_fdr(index, start, end)
    try:
        closes = _closes_from_krx(_KRX_INDEX_CODES[index], start, end)
    except Exception:  # pykrx 실패(로그인·KRX 장애)는 폴백으로 넘긴다
        closes = pd.Series(dtype=float)
    if closes.empty and index in _YAHOO_FALLBACK:
        closes = _closes_from_fdr(_YAHOO_FALLBACK[index], start, end)
    return closes


def fetch_index_quote(index: str) -> IndexQuote | None:
    """지수의 최신 종가와 전일대비 변화율을 반환한다. 데이터가 없거나 너무 오래됐으면 None
    (에러 아님 — 신규/휴장/소스 지연 등 정상적으로 발생 가능)."""
    if index not in SUPPORTED_INDICES:
        raise UnsupportedIndexError(f"unsupported index: {index!r}")

    end = _today()
    start = end - timedelta(days=_LOOKBACK_DAYS)
    closes = _closes(index, start, end).dropna()
    if closes.empty:
        return None

    latest_date = closes.index[-1].date()
    if (end - latest_date).days > _MAX_STALENESS_DAYS:
        return None
    latest_close = float(closes.iloc[-1])

    change_pct = None
    if len(closes) >= 2:
        prior_close = float(closes.iloc[-2])
        if prior_close != 0:
            change_pct = (latest_close / prior_close - 1) * 100

    return IndexQuote(index=index, date=latest_date, close=latest_close, change_pct=change_pct)
