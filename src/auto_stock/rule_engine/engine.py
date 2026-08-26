from auto_stock.data.models import OHLCVRecord
from auto_stock.rule_engine.indicators import rsi, sma
from auto_stock.rule_engine.models import Candidate

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 60


def generate_candidates(records: list[OHLCVRecord]) -> list[Candidate]:
    """1차 규칙 기반 필터링: RSI 과매수/과매도 + SMA 골든/데드크로스.
    두 신호가 상충하면 후보를 만들지 않는다 (보수적 원칙)."""
    if not records:
        return []

    closes = [r.close for r in records]
    rsi_values = rsi(closes)
    sma_short = sma(closes, SMA_SHORT_WINDOW)
    sma_long = sma(closes, SMA_LONG_WINDOW)

    buy_reasons: list[str] = []
    sell_reasons: list[str] = []

    latest_rsi = rsi_values[-1]
    if latest_rsi is not None:
        if latest_rsi < RSI_OVERSOLD:
            buy_reasons.append(f"RSI(14)={latest_rsi:.1f} 과매도 구간 진입")
        elif latest_rsi > RSI_OVERBOUGHT:
            sell_reasons.append(f"RSI(14)={latest_rsi:.1f} 과매수 구간 진입")

    if len(records) >= 2:
        short_now, short_prev = sma_short[-1], sma_short[-2]
        long_now, long_prev = sma_long[-1], sma_long[-2]
        if None not in (short_now, short_prev, long_now, long_prev):
            crossed_up = short_prev <= long_prev and short_now > long_now
            crossed_down = short_prev >= long_prev and short_now < long_now
            if crossed_up:
                buy_reasons.append(f"골든크로스(SMA{SMA_SHORT_WINDOW}이 SMA{SMA_LONG_WINDOW}을 상향 돌파)")
            elif crossed_down:
                sell_reasons.append(f"데드크로스(SMA{SMA_SHORT_WINDOW}이 SMA{SMA_LONG_WINDOW}을 하향 돌파)")

    latest = records[-1]
    if buy_reasons and not sell_reasons:
        return [Candidate(ticker=latest.ticker, market=latest.market, action="BUY", reasons=buy_reasons)]
    if sell_reasons and not buy_reasons:
        return [Candidate(ticker=latest.ticker, market=latest.market, action="SELL", reasons=sell_reasons)]
    return []
