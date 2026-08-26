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
