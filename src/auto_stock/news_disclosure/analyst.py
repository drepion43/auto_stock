"""추론 경로: 공시 목록 -> LLM 판독 -> 규칙엔진 후보에 대한 보조 문구.

`analyze`는 `action`을 받지 않고 독립적으로 영향을 판단하며(동조 방어), `ticker`/`market`는
호출자가 넘긴 값을 그대로 라벨링에만 쓰고 프롬프트에는 넣지 않는다(환각 방어, prompt.py
참고). 공시 목록이 비어 있으면 `None`을 반환하고 `client.read_disclosures`를 호출하지
않는다 — API 호출 자체가 발생하지 않는 1층 방어(news-disclosure-plan.md 에러 처리 1층과
동일). `to_reasons`가 코드에서 동의/상충/중립을 비교한다 — llm_chart_analyst.analyst와
정확히 같은 구조다.
"""

from datetime import date

from auto_stock.data.models import DisclosureRecord
from auto_stock.news_disclosure.credentials import MAX_DISCLOSURES_PER_QUERY
from auto_stock.news_disclosure.models import DisclosureAnalysis, DisclosureReader, DisclosureSummary
from auto_stock.news_disclosure.prompt import SYSTEM_PROMPT, build_user_prompt

NEWS_DISCLAIMER = "(공시해석은 백테스트 미검증 정성 신호입니다)"
CONFIDENCE_LABELS = {"LOW": "낮음", "MEDIUM": "보통", "HIGH": "높음"}
IMPACT_LABELS = {"POSITIVE": "호재", "NEGATIVE": "악재", "NEUTRAL": "중립"}


def _build_summary(
    disclosures: list[DisclosureRecord], ticker: str, market: str, as_of: date
) -> DisclosureSummary:
    recent = sorted(disclosures, key=lambda item: item.filed_date, reverse=True)
    return DisclosureSummary(
        ticker=ticker, market=market, as_of=as_of, items=recent[:MAX_DISCLOSURES_PER_QUERY]
    )


def analyze(
    client: DisclosureReader,
    disclosures: list[DisclosureRecord],
    ticker: str,
    market: str,
    as_of: date,
) -> DisclosureAnalysis | None:
    if not disclosures:
        return None

    summary = _build_summary(disclosures, ticker, market, as_of)
    user_prompt = build_user_prompt(summary)
    read = client.read_disclosures(SYSTEM_PROMPT, user_prompt)

    return DisclosureAnalysis(
        ticker=summary.ticker,
        market=summary.market,
        as_of=summary.as_of,
        market_impact=read.market_impact,
        confidence=read.confidence,
        key_event=read.key_event,
        rationale=read.rationale,
        caveat=read.caveat,
        model=client.model,
    )


def _stance(analysis: DisclosureAnalysis, action: str) -> str:
    label = CONFIDENCE_LABELS.get(analysis.confidence, analysis.confidence)
    impact_label = IMPACT_LABELS.get(analysis.market_impact, analysis.market_impact)
    agrees = (action == "BUY" and analysis.market_impact == "POSITIVE") or (
        action == "SELL" and analysis.market_impact == "NEGATIVE"
    )
    conflicts = (action == "BUY" and analysis.market_impact == "NEGATIVE") or (
        action == "SELL" and analysis.market_impact == "POSITIVE"
    )

    if agrees:
        return f"공시분석: {analysis.key_event} ({impact_label}) — {action} 신호에 동의 (신뢰도 {label})"
    if conflicts:
        return f"공시분석: {analysis.key_event} ({impact_label}) — {action} 신호와 상충됩니다 — 주의 (신뢰도 {label})"
    return f"공시분석: {analysis.key_event} (영향 중립, 신뢰도 {label})"


def to_reasons(analysis: DisclosureAnalysis | None, action: str) -> list[str]:
    if analysis is None:
        return []

    reasons = [_stance(analysis, action), analysis.rationale]
    if analysis.caveat:
        reasons.append(f"단서: {analysis.caveat}")
    reasons.append(NEWS_DISCLAIMER)
    return reasons
