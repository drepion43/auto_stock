"""지수 자체의 시세 조회 — 코스피/코스닥/코스피200/VIX/다우존스/나스닥종합/S&P500/
러셀2000. 종목(ticker)이 아니라 지수 자체를 묻는 챗봇 질의("VIX 지금 어때",
"코스피 지금 어때") 전용 데이터 소스다.

`fdr.DataReader(symbol)`이 Yahoo Finance를 거쳐 지수의 OHLCV+Adj Close를 그대로
반환한다(FDR 소스코드 확인: `data.py`의 `yahoo_index_symbol_map`을 거쳐
`YahooDailyReader`로 라우팅됨) — `fdr_source.fetch_ohlcv`와 달리 `OHLCVRecord`로
감싸지 않는다. 지수는 "종목"이 아니라 market 필드를 가질 이유가 없고(`OHLCVRecord`의
`VALID_MARKETS` 검증과 무관한 별개의 데이터 형태), 챗봇에는 최근 종가/전일대비
변화율만 보여주면 충분하기 때문이다."""

from dataclasses import dataclass
from datetime import date, timedelta

import FinanceDataReader as fdr

# fdr.DataReader가 받는 심볼 — 챗봇 tool enum(tool_schemas.py의 get_index_quote)과
# 반드시 동기화해야 한다.
SUPPORTED_INDICES = frozenset({"KS11", "KQ11", "KS200", "VIX", "DJI", "IXIC", "S&P500", "RUT"})

_LOOKBACK_DAYS = 10  # 전일대비 변화율 계산에 필요한 최소 2거래일 + 공휴일 여유


class UnsupportedIndexError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class IndexQuote:
    index: str
    date: date
    close: float
    change_pct: float | None  # 직전 거래일 대비 변화율(%). 직전 데이터가 없으면 None


def fetch_index_quote(index: str) -> IndexQuote | None:
    """지수의 최신 종가와 전일대비 변화율을 반환한다. 데이터가 전혀 없으면 None
    (에러 아님 — 신규/휴장 등 정상적으로 발생 가능)."""
    if index not in SUPPORTED_INDICES:
        raise UnsupportedIndexError(f"unsupported index: {index!r}")

    end = date.today()
    start = end - timedelta(days=_LOOKBACK_DAYS)
    df = fdr.DataReader(index, start, end)
    if df.empty:
        return None

    latest = df.iloc[-1]
    latest_date = df.index[-1].date()
    latest_close = float(latest["Close"])

    change_pct = None
    if len(df) >= 2:
        prior_close = float(df.iloc[-2]["Close"])
        if prior_close != 0:
            change_pct = (latest_close / prior_close - 1) * 100

    return IndexQuote(index=index, date=latest_date, close=latest_close, change_pct=change_pct)
