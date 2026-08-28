"""OHLCV -> `ChartSnapshot`(정규화 가격/거래량비 봉 + 지표) 변환. API 호출 전혀 없는 순수 계층.

지표(`indicators`)는 `ml_predictor.features.latest_feature_vector`를 그대로 재사용한다 —
지표 수식을 새로 짜지 않는다(lookahead 방어를 이미 검증받은 로직 재사용). 가격은 구간 첫
종가를 100으로 하는 지수로, 거래량은 20일 평균 대비 배율로 변환해 절대 스케일(원/달러)을
없앤다 — KRX/NASDAQ을 하나의 프롬프트 형식으로 다룰 수 있는 이유이자 환각 방어의 전제다.
"""

from auto_stock.data.models import OHLCVRecord
from auto_stock.llm_chart_analyst.models import BarSummary, ChartSnapshot
from auto_stock.ml_predictor.features import latest_feature_vector

RECENT_BARS = 30
SNAPSHOT_BASE_INDEX = 100.0
VOLUME_AVG_WINDOW = 20


def _volume_ratios(volumes: list[int], window: int = VOLUME_AVG_WINDOW) -> list[float | None]:
    """`ml_predictor.features._volume_ratio`와 동일한 후행(trailing) 윈도우 계산."""
    n = len(volumes)
    out: list[float | None] = [None] * n
    for i in range(window - 1, n):
        window_slice = volumes[i - window + 1 : i + 1]
        mean_volume = sum(window_slice) / window
        out[i] = None if mean_volume == 0 else volumes[i] / mean_volume
    return out


def build_snapshot(records: list[OHLCVRecord], recent_bars: int = RECENT_BARS) -> ChartSnapshot | None:
    """워밍업(최장 SMA60) 미충족이거나 기준 종가(구간 첫 봉)가 0이면 None —
    `latest_feature_vector`와 동일한 "0 나눗셈 대신 None" 관례."""
    vector = latest_feature_vector(records)
    if vector is None:
        return None

    window = records[-recent_bars:]
    base_close = window[0].close
    if base_close == 0:
        return None

    volume_ratios = _volume_ratios([r.volume for r in records])
    window_volume_ratios = volume_ratios[-len(window) :]

    bar_count = len(window)
    bars = [
        BarSummary(
            offset=bar_count - 1 - position,
            open=record.open / base_close * SNAPSHOT_BASE_INDEX,
            high=record.high / base_close * SNAPSHOT_BASE_INDEX,
            low=record.low / base_close * SNAPSHOT_BASE_INDEX,
            close=record.close / base_close * SNAPSHOT_BASE_INDEX,
            volume_ratio=volume_ratio if volume_ratio is not None else 0.0,
        )
        for position, (record, volume_ratio) in enumerate(zip(window, window_volume_ratios))
    ]

    latest = records[-1]
    return ChartSnapshot(
        ticker=latest.ticker,
        market=latest.market,
        date=latest.date,
        bars=bars,
        indicators=dict(vector.values),
    )
