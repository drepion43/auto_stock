"""추론 경로: 스냅샷 -> LLM 판독 -> 규칙엔진 후보에 대한 보조 문구.

`analyze`는 `action`을 받지 않고 독립적으로 방향성을 판단하며(동조 방어), `ticker`/`market`/
`date`는 LLM 응답이 아니라 `records`에서 채운다(환각 방어). `to_reasons`가 코드에서 동의/
상충/중립을 비교한다 — `ml_predictor.predictor._stance`와 정확히 같은 구조다. 데이터 부족
시 `build_snapshot`이 `None`을 반환하므로 `analyze`도 `None`을 반환하고 `client.read_pattern`을
호출하지 않는다 — API 호출 자체가 발생하지 않는 1층 방어(에러 처리 계획 문서 참고)."""

from auto_stock.data.models import OHLCVRecord
from auto_stock.llm_chart_analyst.models import ChartAnalysis, ChartPatternReader
from auto_stock.llm_chart_analyst.prompt import SYSTEM_PROMPT, build_user_prompt
from auto_stock.llm_chart_analyst.snapshot import build_snapshot

LLM_DISCLAIMER = "(LLM 차트해석은 백테스트 미검증 정성 신호입니다)"
CONFIDENCE_LABELS = {"LOW": "낮음", "MEDIUM": "보통", "HIGH": "높음"}


def analyze(client: ChartPatternReader, records: list[OHLCVRecord]) -> ChartAnalysis | None:
    snapshot = build_snapshot(records)
    if snapshot is None:
        return None

    user_prompt = build_user_prompt(snapshot)
    read = client.read_pattern(SYSTEM_PROMPT, user_prompt)

    return ChartAnalysis(
        ticker=snapshot.ticker,
        market=snapshot.market,
        date=snapshot.date,
        direction=read.direction,
        confidence=read.confidence,
        pattern_name=read.pattern_name,
        rationale=read.rationale,
        caveat=read.caveat,
        model=client.model,
    )


def _stance(analysis: ChartAnalysis, action: str) -> str:
    label = CONFIDENCE_LABELS.get(analysis.confidence, analysis.confidence)
    agrees = (action == "BUY" and analysis.direction == "UP") or (
        action == "SELL" and analysis.direction == "DOWN"
    )
    conflicts = (action == "BUY" and analysis.direction == "DOWN") or (
        action == "SELL" and analysis.direction == "UP"
    )

    if agrees:
        return f"LLM 차트분석: {analysis.pattern_name} 감지 — {action} 신호에 동의 (신뢰도 {label})"
    if conflicts:
        return f"LLM 차트분석: {analysis.pattern_name} 감지 — {action} 신호와 상충됩니다 — 주의 (신뢰도 {label})"
    return f"LLM 차트분석: {analysis.pattern_name} (방향성 중립, 신뢰도 {label})"


def to_reasons(analysis: ChartAnalysis | None, action: str) -> list[str]:
    if analysis is None:
        return []

    reasons = [_stance(analysis, action), analysis.rationale]
    if analysis.caveat:
        reasons.append(f"단서: {analysis.caveat}")
    reasons.append(LLM_DISCLAIMER)
    return reasons
