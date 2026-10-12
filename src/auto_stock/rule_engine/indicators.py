"""Technical indicator functions, backed by the battle-tested `pandas-ta` library
rather than hand-rolled formulas (see docs/design/rule-engine.md for rationale).

Inputs/outputs stay plain float lists (index-aligned with the input) so callers
(engine.py) don't need to know pandas-ta is involved underneath."""

import pandas as pd
import pandas_ta as ta


def _to_list(series: pd.Series | None, length: int) -> list[float | None]:
    # pandas-ta returns None (not a Series/DataFrame of NaN) when the input is
    # shorter than the indicator's required length.
    if series is None:
        return [None] * length
    return [None if pd.isna(v) else float(v) for v in series]


def sma(closes: list[float], window: int) -> list[float | None]:
    return _to_list(ta.sma(pd.Series(closes, dtype=float), length=window), len(closes))


def ema(values: list[float], period: int) -> list[float | None]:
    return _to_list(ta.ema(pd.Series(values, dtype=float), length=period), len(values))


def rsi(closes: list[float], period: int = 14) -> list[float | None]:
    return _to_list(ta.rsi(pd.Series(closes, dtype=float), length=period), len(closes))


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> list[float | None]:
    series = ta.atr(
        pd.Series(highs, dtype=float),
        pd.Series(lows, dtype=float),
        pd.Series(closes, dtype=float),
        length=period,
    )
    return _to_list(series, len(closes))


def volume_ratio(volumes: list[int], window: int = 20) -> list[float | None]:
    """Latest volume divided by its trailing `window`-period average.

    Hand-rolled rather than pandas-ta-backed (no direct pandas-ta equivalent for
    "volume vs its own trailing average"). Shared by ml_predictor.features and
    llm_chart_analyst.snapshot so the two auxiliary signals agree on this number
    for the same date."""
    n = len(volumes)
    out: list[float | None] = [None] * n
    for i in range(window - 1, n):
        window_slice = volumes[i - window + 1 : i + 1]
        mean_volume = sum(window_slice) / window
        out[i] = None if mean_volume == 0 else volumes[i] / mean_volume
    return out


def macd(
    closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (macd_line, signal_line, histogram), index-aligned with `closes`."""
    n = len(closes)
    df = ta.macd(pd.Series(closes, dtype=float), fast=fast, slow=slow, signal=signal)
    if df is None:
        return [None] * n, [None] * n, [None] * n
    macd_col = f"MACD_{fast}_{slow}_{signal}"
    hist_col = f"MACDh_{fast}_{slow}_{signal}"
    signal_col = f"MACDs_{fast}_{slow}_{signal}"
    return _to_list(df[macd_col], n), _to_list(df[signal_col], n), _to_list(df[hist_col], n)


# --- rule_engine/technical_context.py가 조립하는 지표 7종 ---
#
# pandas-ta의 결과 컬럼명은 버전마다 달라진다(예: 0.4.71b0의 bbands는 `BBL_20_2.0_2.0`처럼 표준편차
# 접미사가 두 번 붙는다) — 컬럼명을 하드코딩하지 않고 접두사로 고른다.


def _column(df: pd.DataFrame, prefix: str) -> pd.Series:
    return df[next(name for name in df.columns if name.startswith(prefix))]


def adx(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (adx, plus_di, minus_di) — 추세 강도와 방향(+DI > -DI면 상승 우위)."""
    n = len(closes)
    df = ta.adx(pd.Series(highs, dtype=float), pd.Series(lows, dtype=float), pd.Series(closes, dtype=float), length=period)
    if df is None:
        return [None] * n, [None] * n, [None] * n
    return _to_list(_column(df, "ADX_"), n), _to_list(_column(df, "DMP_"), n), _to_list(_column(df, "DMN_"), n)


def bbands(
    closes: list[float], period: int = 20, std: float = 2.0
) -> tuple[list[float | None], list[float | None], list[float | None], list[float | None]]:
    """Returns (percent_b, bandwidth, lower, upper). percent_b: 0=하단밴드, 1=상단밴드.
    bandwidth: (상단-하단)/중심 × 100(pandas-ta 정의)."""
    n = len(closes)
    df = ta.bbands(pd.Series(closes, dtype=float), length=period, std=std)
    if df is None:
        return [None] * n, [None] * n, [None] * n, [None] * n
    return (
        _to_list(_column(df, "BBP_"), n),
        _to_list(_column(df, "BBB_"), n),
        _to_list(_column(df, "BBL_"), n),
        _to_list(_column(df, "BBU_"), n),
    )


def stoch(
    highs: list[float], lows: list[float], closes: list[float], k: int = 14, d: int = 3, smooth_k: int = 3
) -> tuple[list[float | None], list[float | None]]:
    """Returns (%K, %D), 0~100."""
    n = len(closes)
    df = ta.stoch(
        pd.Series(highs, dtype=float), pd.Series(lows, dtype=float), pd.Series(closes, dtype=float),
        k=k, d=d, smooth_k=smooth_k,
    )
    if df is None:
        return [None] * n, [None] * n
    return _to_list(_column(df, "STOCHk_"), n), _to_list(_column(df, "STOCHd_"), n)


def obv(closes: list[float], volumes: list[int]) -> list[float | None]:
    """On-Balance Volume — 임의 기준점에서 시작하는 누적값이라 레벨이 아니라 방향만 의미가 있다."""
    return _to_list(ta.obv(pd.Series(closes, dtype=float), pd.Series(volumes, dtype=float)), len(closes))


def aroon(
    highs: list[float], lows: list[float], period: int = 14
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (aroon_up, aroon_down, aroon_oscillator). up 100 = 기간 내 최고가가 방금 나옴."""
    n = len(highs)
    df = ta.aroon(pd.Series(highs, dtype=float), pd.Series(lows, dtype=float), length=period)
    if df is None:
        return [None] * n, [None] * n, [None] * n
    return _to_list(_column(df, "AROONU_"), n), _to_list(_column(df, "AROOND_"), n), _to_list(_column(df, "AROONOSC_"), n)


def squeeze(highs: list[float], lows: list[float], closes: list[float]) -> list[float | None]:
    """TTM Squeeze 켜짐 여부(1.0=볼린저밴드가 켈트너채널 안쪽 — 변동성 수축, 0.0=아님).
    pandas-ta는 워밍업 구간에도 플래그를 0으로 채울 수 있어, 모멘텀 값이 없는 구간은 None으로 둔다."""
    n = len(closes)
    df = ta.squeeze(pd.Series(highs, dtype=float), pd.Series(lows, dtype=float), pd.Series(closes, dtype=float))
    if df is None:
        return [None] * n
    momentum = df[next(name for name in df.columns if name.startswith("SQZ_") and name not in ("SQZ_ON", "SQZ_OFF", "SQZ_NO"))]
    on = df["SQZ_ON"]
    return [None if pd.isna(m) or pd.isna(flag) else float(flag) for m, flag in zip(momentum, on)]


def rsi_divergence(
    closes: list[float], rsi_values: list[float | None], lookback: int = 20
) -> list[str | None]:
    """가격-RSI 다이버전스. pandas-ta에 직접 대응 함수가 없어 직접 계산한다.

    i 시점 가격이 직전 `lookback`개 봉의 최저가보다 낮은데 RSI는 그 최저가 시점의 RSI보다 높으면
    "bullish"(하락 모멘텀 약화), 최고가보다 높은데 RSI는 그 최고가 시점보다 낮으면 "bearish".
    어느 쪽도 아니거나, 이력·RSI 값이 부족하면 None."""
    n = len(closes)
    out: list[str | None] = [None] * n
    for i in range(lookback, n):
        if rsi_values[i] is None:
            continue
        window = range(i - lookback, i)
        low_idx = min(window, key=lambda j: closes[j])
        high_idx = max(window, key=lambda j: closes[j])
        if closes[i] < closes[low_idx] and rsi_values[low_idx] is not None and rsi_values[i] > rsi_values[low_idx]:
            out[i] = "bullish"
        elif closes[i] > closes[high_idx] and rsi_values[high_idx] is not None and rsi_values[i] < rsi_values[high_idx]:
            out[i] = "bearish"
    return out
