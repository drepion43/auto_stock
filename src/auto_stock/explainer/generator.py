"""Deterministic template-based explanation generator (PRD §5.1, Explainability).

No LLM call for MVP-0 — the only signal sources implemented so far (rule engine
#1, risk sizing #5) are already structured data, so a template is sufficient
and needs no API key. `extra_reasons` is the extension point for future signal
layers (ML #2, LLM chart #3, news/disclosure #4): their generated reason
strings can be passed straight through without redesigning this function.
"""

from auto_stock.explainer.models import Explanation
from auto_stock.risk_sizing.models import SizingSuggestion
from auto_stock.rule_engine.models import Candidate


def generate_explanation(
    candidate: Candidate, sizing: SizingSuggestion, extra_reasons: list[str] | None = None
) -> Explanation:
    reasons = list(candidate.reasons) + list(extra_reasons or [])
    reason_text = " + ".join(reasons) if reasons else "신호 없음"
    parts = [f"{reason_text}."]

    if sizing.suggested_quantity is not None and sizing.suggested_allocation_pct is not None:
        parts.append(
            f"참고용 매수 제안: {sizing.suggested_quantity}주 "
            f"(계좌 자산의 {sizing.suggested_allocation_pct:.1%})."
        )

    if sizing.stop_loss_price is not None and sizing.take_profit_price is not None:
        parts.append(
            f"손절 {sizing.stop_loss_price:,.0f} / 익절 {sizing.take_profit_price:,.0f} (ATR 기반, 참고용)."
        )

    if sizing.notes:
        parts.append(" ".join(sizing.notes))

    summary = f"{candidate.ticker}: " + " ".join(parts)
    return Explanation(ticker=candidate.ticker, market=candidate.market, action=candidate.action, summary=summary)
