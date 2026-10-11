"""7개 기술지표를 종목 1개 단위로 모아 "숫자 그대로" 반환하는 조립 레이어.

`rule_engine.indicators`는 순수 지표 계산(단일 책임)만 담당하고, 이 모듈이 여러
지표를 호출해 하나의 결과로 묶는다 — `ml_predictor.features`가 `rule_engine.indicators`를
조립해 `FeatureVector`를 만드는 것과 동일한 레이어 구조.

이 모듈은 BUY/SELL 판단을 내리지 않는다(`rule_engine.engine.generate_candidates`와
다른 책임) — 해석은 `chat_agent.stock_analyst`(LLM)에게 그대로 넘긴다. 워밍업
미충족 등으로 7개 지표 중 하나라도 계산이 안 되면 `None`을 반환한다
(`ml_predictor.features.build_feature_vectors`의 all-or-nothing 관례와 동일 —
일부만 채워진 반쪽 결과를 만들지 않는다)."""

from dataclasses import dataclass

from auto_stock.data.models import OHLCVRecord
from auto_stock.rule_engine.indicators import adx, aroon, bbands, obv, rsi, rsi_divergence, squeeze, stoch

RSI_DIVERGENCE_LOOKBACK = 20


@dataclass(frozen=True, slots=True)
class TechnicalContext:
    ticker: str
    market: str
    adx: float
    plus_di: float
    minus_di: float
    bb_percent_b: float
    bb_bandwidth: float
    stoch_k: float
    stoch_d: float
    obv_trend: str  # "RISING" | "FALLING" | "FLAT" — 최근 5개 OBV 값의 방향
    aroon_up: float
    aroon_down: float
    aroon_oscillator: float
    squeeze_on: bool
    rsi_divergence: str | None  # "bullish" | "bearish" | None


def _obv_trend(obv_values: list[float | None], window: int = 5) -> str:
    """최근 `window`개 OBV 값의 방향 — 절대 레벨이 아니라 변화 방향만 본다(OBV는
    임의 기준점에서 시작하는 누적값이라 레벨 자체는 의미가 없다)."""
    recent = obv_values[-window:]
    if recent[-1] > recent[0]:
        return "RISING"
    if recent[-1] < recent[0]:
        return "FALLING"
    return "FLAT"


def build_technical_context(records: list[OHLCVRecord]) -> TechnicalContext | None:
    """records와 정렬된 가장 최근 시점의 기술지표 컨텍스트. 7개 지표 중 하나라도
    워밍업 미충족(None)이면 전체를 None으로 처리한다."""
    if not records:
        return None

    closes = [r.close for r in records]
    highs = [r.high for r in records]
    lows = [r.low for r in records]
    volumes = [r.volume for r in records]

    rsi_values = rsi(closes)
    adx_values, plus_di_values, minus_di_values = adx(highs, lows, closes)
    percent_b_values, bandwidth_values, _, _ = bbands(closes)
    stoch_k_values, stoch_d_values = stoch(highs, lows, closes)
    obv_values = obv(closes, volumes)
    aroon_up_values, aroon_down_values, aroon_osc_values = aroon(highs, lows)
    squeeze_on_values = squeeze(highs, lows, closes)
    divergence_values = rsi_divergence(closes, rsi_values, lookback=RSI_DIVERGENCE_LOOKBACK)

    latest = -1
    required = [
        adx_values[latest], plus_di_values[latest], minus_di_values[latest],
        percent_b_values[latest], bandwidth_values[latest],
        stoch_k_values[latest], stoch_d_values[latest],
        aroon_up_values[latest], aroon_down_values[latest], aroon_osc_values[latest],
        squeeze_on_values[latest],
    ]
    if any(v is None for v in required) or len(obv_values) < 5 or obv_values[-5] is None:
        return None

    record = records[-1]
    return TechnicalContext(
        ticker=record.ticker,
        market=record.market,
        adx=adx_values[latest],
        plus_di=plus_di_values[latest],
        minus_di=minus_di_values[latest],
        bb_percent_b=percent_b_values[latest],
        bb_bandwidth=bandwidth_values[latest],
        stoch_k=stoch_k_values[latest],
        stoch_d=stoch_d_values[latest],
        obv_trend=_obv_trend(obv_values),
        aroon_up=aroon_up_values[latest],
        aroon_down=aroon_down_values[latest],
        aroon_oscillator=aroon_osc_values[latest],
        squeeze_on=squeeze_on_values[latest] == 1.0,
        rsi_divergence=divergence_values[latest],
    )
