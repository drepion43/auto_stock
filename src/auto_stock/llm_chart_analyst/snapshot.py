"""OHLCV -> `ChartSnapshot`(정규화 가격/거래량비 봉 + 지표) 변환. API 호출 전혀 없는 순수 계층.

지표(`indicators`)는 `ml_predictor.features.latest_feature_vector`를 그대로 재사용한다 —
지표 수식을 새로 짜지 않는다(lookahead 방어를 이미 검증받은 로직 재사용). 가격은 구간 첫
종가를 100으로 하는 지수로, 거래량은 20일 평균 대비 배율로 변환해 절대 스케일(원/달러)을
없앤다 — KRX/NASDAQ을 하나의 프롬프트 형식으로 다룰 수 있는 이유이자 환각 방어의 전제다.
"""

from auto_stock.data.models import OHLCVRecord
from auto_stock.llm_chart_analyst.models import BarSummary, ChartSnapshot
from auto_stock.ml_predictor.features import latest_feature_vector
from auto_stock.rule_engine.indicators import volume_ratio

RECENT_BARS = 30
SNAPSHOT_BASE_INDEX = 100.0
VOLUME_AVG_WINDOW = 20


def build_snapshot(records: list[OHLCVRecord], recent_bars: int = RECENT_BARS) -> ChartSnapshot | None:
    """워밍업(최장 SMA60) 미충족, 기준 종가(구간 첫 봉)가 0, 또는 거래량비 윈도우가
    구간 내 일부라도 미충족이면 None — `latest_feature_vector`와 동일한
    "정의 안 된 값을 지어내지 않는다" 관례. 실제 오케스트레이터 경로는 SMA60
    워밍업이 거래량비 윈도우(20일)보다 항상 길어 이 조건에 걸리지 않지만,
    `recent_bars`를 호출자가 크게 지정하는 경우를 위한 방어다."""
    vector = latest_feature_vector(records)
    if vector is None:
        return None

    window = records[-recent_bars:]
    base_close = window[0].close
    if base_close == 0:
        return None

    volume_ratios = volume_ratio([r.volume for r in records], VOLUME_AVG_WINDOW)
    window_volume_ratios = volume_ratios[-len(window) :]
    if any(r is None for r in window_volume_ratios):
        return None

    bar_count = len(window)
    bars = [
        BarSummary(
            offset=bar_count - 1 - position,
            open=record.open / base_close * SNAPSHOT_BASE_INDEX,
            high=record.high / base_close * SNAPSHOT_BASE_INDEX,
            low=record.low / base_close * SNAPSHOT_BASE_INDEX,
            close=record.close / base_close * SNAPSHOT_BASE_INDEX,
            volume_ratio=ratio,
        )
        for position, (record, ratio) in enumerate(zip(window, window_volume_ratios))
    ]

    latest = records[-1]
    return ChartSnapshot(
        ticker=latest.ticker,
        market=latest.market,
        date=latest.date,
        bars=bars,
        indicators=dict(vector.values),
    )
