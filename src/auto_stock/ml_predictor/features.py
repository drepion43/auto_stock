"""OHLCV -> 코어 11종(스케일 프리) + 국내전용 펀더멘털 5종(+결측 플래그) 피처.

**코어(all-or-nothing)**: 전부 비율/정규화값이라 KRX(원)와 NASDAQ(달러) 종목을 하나의
통합 모델에 섞어도 스케일 왜곡이 없다. 11개 중 하나라도 워밍업 미충족(None)이면 그
시점 전체를 None으로 처리한다(반쪽짜리 벡터를 만들지 않음). `rule_engine.indicators`의
sma/rsi/macd/atr는 이미 후행(trailing) 윈도우만 쓰는 causal 구현이라 그대로 재사용한다
— lookahead 방어의 1번째 축.

**펀더멘털(국내전용, 결측 허용)**: PER/PBR/배당수익률/회전율/PSR은 pykrx+DART 재무제표
전용 데이터라 미국 종목엔 존재하지 않는다. 통합 모델이 KRX+NASDAQ을 함께 학습하므로,
이 5개를 코어와 같은 all-or-nothing 규칙으로 묶으면 미국 종목이 영구히 탈락한다 —
그래서 결측 시 중립값(모델이 "신호 없음"으로 해석하게 되는 값: earnings_yield/
dividend_yield/turnover_rate/sales_yield=0.0, pbr=1.0)으로 채우고, "이 날짜에 실제
펀더멘털 데이터가 있었는지"를 나타내는 `fundamental_available` 플래그 1채널을 둔다
(국내 종목이라도 매출액 계정 매칭 실패 등으로 일부만 결측일 수 있음 — 그 경우도
플래그는 1.0 유지, 결측된 개별 필드만 중립값).

PER/PSR은 역수(earnings_yield=1/PER, sales_yield=1/PSR)로 저장한다 — 손실기업의
음수/발산값을 완화하는 퀀트 투자 관례(Earnings Yield/Sales Yield)다. PBR은 발산
위험이 적어 역수를 쓰지 않고 그대로 쓴다."""

from datetime import date

from auto_stock.data.models import OHLCVRecord
from auto_stock.ml_predictor.models import FeatureVector
from auto_stock.rule_engine.indicators import atr, macd, rsi, sma, volume_ratio

CORE_FEATURE_NAMES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "rsi_14",
    "macd_hist_norm",
    "close_to_sma20",
    "close_to_sma60",
    "sma20_to_sma60",
    "atr_14_pct",
    "volume_ratio_20",
    "channel_position_20",
]

FUNDAMENTAL_FEATURE_NAMES = [
    "earnings_yield",
    "pbr",
    "dividend_yield",
    "turnover_rate",
    "sales_yield",
    "fundamental_available",
]

FEATURE_NAMES = CORE_FEATURE_NAMES + FUNDAMENTAL_FEATURE_NAMES

_FUNDAMENTAL_NEUTRAL = {
    "earnings_yield": 0.0,
    "pbr": 1.0,
    "dividend_yield": 0.0,
    "turnover_rate": 0.0,
    "sales_yield": 0.0,
}

CHANNEL_WINDOW = 20
VOLUME_WINDOW = 20


def _returns(closes: list[float], k: int) -> list[float | None]:
    n = len(closes)
    out: list[float | None] = [None] * n
    for i in range(k, n):
        prior = closes[i - k]
        out[i] = None if prior == 0 else closes[i] / prior - 1
    return out


def _channel_position(highs: list[float], lows: list[float], closes: list[float], window: int) -> list[float | None]:
    n = len(closes)
    out: list[float | None] = [None] * n
    for i in range(window - 1, n):
        high_slice = highs[i - window + 1 : i + 1]
        low_slice = lows[i - window + 1 : i + 1]
        channel_high, channel_low = max(high_slice), min(low_slice)
        span = channel_high - channel_low
        out[i] = 0.5 if span == 0 else (closes[i] - channel_low) / span
    return out


def _fundamental_values(entry: dict | None) -> dict[str, float]:
    """`entry`(그 날짜의 펀더멘털 raw dict, pykrx+DART 혼합: per/pbr/div/market_cap/
    volume/shares_outstanding/revenue 중 일부 또는 전부) -> 5개 피처 + available 플래그.

    `entry`가 None이면(미국 종목, 또는 그 날짜에 조회 자체가 안 된 경우) 전부 중립값 +
    available=0.0. `entry`가 있으면 available=1.0이고, 개별 필드가 없거나(매출액 계정
    매칭 실패 등) 0으로 나누게 되는 경우만 그 필드만 중립값으로 떨어진다."""
    if entry is None:
        return {**_FUNDAMENTAL_NEUTRAL, "fundamental_available": 0.0}

    per = entry.get("per")
    earnings_yield = 0.0 if per is None or per == 0 else 1 / per

    pbr = entry.get("pbr")
    pbr_value = _FUNDAMENTAL_NEUTRAL["pbr"] if pbr is None else pbr

    div = entry.get("div")
    dividend_yield = 0.0 if div is None else div / 100

    volume = entry.get("volume")
    shares_outstanding = entry.get("shares_outstanding")
    turnover_rate = (
        0.0 if volume is None or shares_outstanding is None or shares_outstanding == 0
        else volume / shares_outstanding
    )

    revenue = entry.get("revenue")
    market_cap = entry.get("market_cap")
    sales_yield = (
        0.0 if revenue is None or market_cap is None or market_cap == 0
        else revenue / market_cap
    )

    return {
        "earnings_yield": earnings_yield,
        "pbr": pbr_value,
        "dividend_yield": dividend_yield,
        "turnover_rate": turnover_rate,
        "sales_yield": sales_yield,
        "fundamental_available": 1.0,
    }


def build_feature_vectors(
    records: list[OHLCVRecord], fundamentals_by_date: dict[date, dict] | None = None
) -> list[FeatureVector | None]:
    """records와 index-aligned된 피처 벡터 리스트. 코어 워밍업(최장: SMA60) 미충족
    구간은 None(all-or-nothing). 펀더멘털은 날짜별로 `fundamentals_by_date.get(date)`를
    조회해 결측 시 중립값으로 채운다(all-or-nothing 아님) — `fundamentals_by_date`가
    통째로 None이면(해외 종목 등) 전 구간 중립값."""
    if not records:
        return []

    closes = [r.close for r in records]
    highs = [r.high for r in records]
    lows = [r.low for r in records]
    volumes = [r.volume for r in records]

    return_1d = _returns(closes, 1)
    return_5d = _returns(closes, 5)
    return_20d = _returns(closes, 20)
    rsi_14 = rsi(closes, period=14)
    _, _, macd_hist = macd(closes)
    sma_20 = sma(closes, window=20)
    sma_60 = sma(closes, window=60)
    atr_14 = atr(highs, lows, closes, period=14)
    volume_ratio_20 = volume_ratio(volumes, VOLUME_WINDOW)
    channel_position_20 = _channel_position(highs, lows, closes, CHANNEL_WINDOW)

    result: list[FeatureVector | None] = []
    for i in range(len(records)):
        core: dict[str, float | None] = {
            "return_1d": return_1d[i],
            "return_5d": return_5d[i],
            "return_20d": return_20d[i],
            "rsi_14": None if rsi_14[i] is None else rsi_14[i] / 100,
            "macd_hist_norm": None if macd_hist[i] is None or closes[i] == 0 else macd_hist[i] / closes[i],
            "close_to_sma20": None if sma_20[i] is None or sma_20[i] == 0 else closes[i] / sma_20[i] - 1,
            "close_to_sma60": None if sma_60[i] is None or sma_60[i] == 0 else closes[i] / sma_60[i] - 1,
            "sma20_to_sma60": (
                None if sma_20[i] is None or sma_60[i] is None or sma_60[i] == 0 else sma_20[i] / sma_60[i] - 1
            ),
            "atr_14_pct": None if atr_14[i] is None or closes[i] == 0 else atr_14[i] / closes[i],
            "volume_ratio_20": volume_ratio_20[i],
            "channel_position_20": channel_position_20[i],
        }
        if any(v is None for v in core.values()):
            result.append(None)
            continue

        record = records[i]
        fundamental_entry = fundamentals_by_date.get(record.date) if fundamentals_by_date else None
        values = {**core, **_fundamental_values(fundamental_entry)}
        result.append(
            FeatureVector(ticker=record.ticker, market=record.market, date=record.date, values=values)  # type: ignore[arg-type]
        )
    return result


def latest_feature_vector(
    records: list[OHLCVRecord], fundamentals_by_date: dict[date, dict] | None = None
) -> FeatureVector | None:
    """단일 시점 피처 스냅샷 전용 호출부(`llm_chart_analyst/snapshot.py`)를 위해 유지한다
    — Mamba 추론 경로(`predictor.py`) 자신은 시퀀스 전체를 쓰므로 이 함수를 호출하지
    않는다. 내부적으로 `build_feature_vectors`를 그대로 호출하고 마지막 원소만 꺼낸다."""
    vectors = build_feature_vectors(records, fundamentals_by_date)
    return vectors[-1] if vectors else None
