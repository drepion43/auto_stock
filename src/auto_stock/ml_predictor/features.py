"""OHLCV -> 스케일 프리(비율/정규화) 피처 11종 (docs/design/ml-predictor-plan.md 피처 정의표).

전부 비율/정규화값이라 KRX(원)와 NASDAQ(달러) 종목을 하나의 pooled 모델에 섞어도
스케일 왜곡이 없다. rule_engine.indicators의 sma/rsi/macd/atr는 이미 후행(trailing)
윈도우만 쓰는 causal 구현이라 그대로 재사용한다 — lookahead 방어의 1번째 축.
"""

from auto_stock.data.models import OHLCVRecord
from auto_stock.ml_predictor.models import FeatureVector
from auto_stock.rule_engine.indicators import atr, macd, rsi, sma

FEATURE_NAMES = [
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

CHANNEL_WINDOW = 20
VOLUME_WINDOW = 20


def _returns(closes: list[float], k: int) -> list[float | None]:
    n = len(closes)
    out: list[float | None] = [None] * n
    for i in range(k, n):
        prior = closes[i - k]
        out[i] = None if prior == 0 else closes[i] / prior - 1
    return out


def _volume_ratio(volumes: list[int], window: int) -> list[float | None]:
    n = len(volumes)
    out: list[float | None] = [None] * n
    for i in range(window - 1, n):
        window_slice = volumes[i - window + 1 : i + 1]
        mean_volume = sum(window_slice) / window
        out[i] = None if mean_volume == 0 else volumes[i] / mean_volume
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


def build_feature_vectors(records: list[OHLCVRecord]) -> list[FeatureVector | None]:
    """records와 index-aligned된 피처 벡터 리스트. 워밍업(최장: SMA60) 미충족 구간은 None."""
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
    volume_ratio_20 = _volume_ratio(volumes, VOLUME_WINDOW)
    channel_position_20 = _channel_position(highs, lows, closes, CHANNEL_WINDOW)

    result: list[FeatureVector | None] = []
    for i in range(len(records)):
        raw: dict[str, float | None] = {
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
        if any(v is None for v in raw.values()):
            result.append(None)
        else:
            record = records[i]
            result.append(
                FeatureVector(ticker=record.ticker, market=record.market, date=record.date, values=raw)  # type: ignore[arg-type]
            )
    return result


def latest_feature_vector(records: list[OHLCVRecord]) -> FeatureVector | None:
    """추론 경로 전용: 가장 최근 시점의 피처 벡터, 워밍업 미충족이면 None."""
    vectors = build_feature_vectors(records)
    return vectors[-1] if vectors else None
