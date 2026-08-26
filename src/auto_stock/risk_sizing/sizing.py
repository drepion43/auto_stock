"""Reference-only position sizing for MVP-0 (PRD §6, IMPLEMENTATION_PLAN.md §5).

Computes a suggested quantity/allocation, ATR-based stop-loss/take-profit, and
a non-blocking limit check against PRD §6's risk caps. No enforcement (no
order blocking, no daily/monthly loss-limit halts) — that's MVP-1 (#8) scope.

The ATR multiples and baseline volatility below are placeholder values; PRD §11
marks them as TBD pending backtest confirmation.
"""

from auto_stock.data.models import OHLCVRecord
from auto_stock.risk_sizing.models import AccountState, SizingSuggestion
from auto_stock.rule_engine.indicators import atr
from auto_stock.rule_engine.models import Candidate

MAX_POSITION_PCT = 0.05
MAX_CONCURRENT_POSITIONS = 10
MAX_TOTAL_EXPOSURE_PCT = 0.50
STOP_LOSS_ATR_MULT = 1.5
TAKE_PROFIT_ATR_MULT = 3.0
BASELINE_ATR_PCT = 0.02
_MIN_PRICE = 0.01  # stop-loss floor so ATR-derived prices never go to zero/negative


def _not_applicable(candidate: Candidate, reason: str) -> SizingSuggestion:
    return SizingSuggestion(
        ticker=candidate.ticker,
        market=candidate.market,
        action=candidate.action,
        suggested_quantity=None,
        suggested_allocation_pct=None,
        stop_loss_price=None,
        take_profit_price=None,
        limit_check="NOT_APPLICABLE",
        notes=[reason],
    )


def suggest_position(
    candidate: Candidate, records: list[OHLCVRecord], account: AccountState
) -> SizingSuggestion:
    if candidate.action != "BUY":
        return _not_applicable(candidate, "매도 후보는 기존 보유분 청산 개념이라 신규 매수 사이징 대상이 아닙니다.")

    highs = [r.high for r in records]
    lows = [r.low for r in records]
    closes = [r.close for r in records]
    latest_atr = atr(highs, lows, closes)[-1]
    if latest_atr is None:
        return _not_applicable(candidate, "ATR 계산에 필요한 가격 데이터가 부족합니다.")

    close = closes[-1]
    if close <= 0:
        return _not_applicable(candidate, "종가가 유효하지 않아(0 이하) 사이징을 계산할 수 없습니다.")

    atr_pct = latest_atr / close
    allocation_pct = MAX_POSITION_PCT if atr_pct <= 0 else min(
        MAX_POSITION_PCT, MAX_POSITION_PCT * BASELINE_ATR_PCT / atr_pct
    )
    stop_loss_price = max(_MIN_PRICE, close - STOP_LOSS_ATR_MULT * latest_atr)
    take_profit_price = close + TAKE_PROFIT_ATR_MULT * latest_atr
    suggested_quantity = int(account.equity * allocation_pct // close)

    notes: list[str] = []
    is_new_position = candidate.ticker not in account.held_tickers
    if is_new_position and len(account.held_tickers) >= MAX_CONCURRENT_POSITIONS:
        limit_check = "EXCEEDS_MAX_POSITIONS"
        notes.append(f"최대 동시 보유 종목 수({MAX_CONCURRENT_POSITIONS}) 한도를 초과합니다.")
    elif account.total_exposure_pct + allocation_pct > MAX_TOTAL_EXPOSURE_PCT:
        limit_check = "EXCEEDS_EXPOSURE_CAP"
        notes.append(f"총 익스포저 한도({MAX_TOTAL_EXPOSURE_PCT:.0%})를 초과합니다.")
    else:
        limit_check = "PASS"

    return SizingSuggestion(
        ticker=candidate.ticker,
        market=candidate.market,
        action=candidate.action,
        suggested_quantity=suggested_quantity,
        suggested_allocation_pct=allocation_pct,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        limit_check=limit_check,
        notes=notes,
    )
