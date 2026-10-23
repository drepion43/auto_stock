"""M7 핵심 경고 추출(결정론) — 계획서 §5.2(재검토 §15 #9).

충실도는 "말한 것"만 본다. 유리한 신호만 골라 전하는 답변을 잡으려면 도구 결과에 있던 경고가
답변에 반영됐는지 봐야 한다. 경고 목록은 여기서 코드로 뽑고, 반영 여부는 G-Eval이 판정한다.
(계획서의 `warnings.py`는 표준 라이브러리 `warnings`와 이름이 겹쳐 `tool_warnings.py`로 둔다.)"""

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

from auto_stock.evaluation.answer_capture import ToolCallRecord

STALE_WARNING_DAYS = 5  # 주말·휴일을 감안한 달력일 — 이보다 오래된 시세·지수 데이터는 경고
_CAVEAT_TOOLS = {"analyze_chart_pattern": "차트 분석", "analyze_disclosures": "공시 해석", "analyze_news_sentiment": "뉴스 해석"}
_ERROR_PREVIEW = 80


def _get(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _stale(value: Any, today: date) -> bool:
    try:
        return date.fromisoformat(str(value)[:10]) < today - timedelta(days=STALE_WARNING_DAYS)
    except ValueError:
        return False


def _signal_warnings(name: str, result: dict) -> list[str]:
    found: list[str] = []
    if _get(result, "candidate", "action") == "SELL":
        found.append("규칙엔진 매도(SELL) 신호")
    predicted = _get(result, "prediction", "predicted_return")
    if isinstance(predicted, (int, float)) and predicted < 0:
        # 초과 수익률 음수는 "시장 평균보다 못하다"이지 하락 예측이 아니다 — 경고 문구도 그 의미로 남긴다.
        if _get(result, "prediction", "return_basis") == "excess_vs_region_mean":
            found.append(f"ML 예측 초과 수익률 음수 — 지역 평균 대비 열위({predicted})")
        else:
            found.append(f"ML 예측 수익률 음수({predicted})")
    if name == "analyze_chart_pattern" and _get(result, "analysis", "direction") == "DOWN":
        found.append("차트 패턴 하락(DOWN)")
    if _get(result, "analysis", "market_impact") == "NEGATIVE":
        found.append(f"부정적 공시: {_get(result, 'analysis', 'key_event')}")
    if _get(result, "analysis", "sentiment") == "NEGATIVE":
        found.append(f"부정적 뉴스 논조: {_get(result, 'analysis', 'key_headline')}")
    caveat = _get(result, "analysis", "caveat")
    if name in _CAVEAT_TOOLS and caveat:
        found.append(f"{_CAVEAT_TOOLS[name]} 단서: {caveat}")
    if _get(result, "judgment", "outlook") == "DOWN":
        found.append("서브에이전트 판단 하락(DOWN)")
    return found


def _status_warnings(name: str, result: dict, today: date) -> list[str]:
    found: list[str] = []
    limit_check = _get(result, "suggestion", "limit_check")
    if limit_check not in (None, "PASS", "NOT_APPLICABLE"):
        found.append(f"포지션 한도 점검: {limit_check}")
    stop_loss = _get(result, "suggestion", "stop_loss_price")
    if stop_loss is not None:
        found.append(f"손절가 {stop_loss}")
    if result.get("errors"):
        found.append(f"{name}: 일부 데이터 조회 실패")
    if result.get("data_as_of_note"):
        found.append(str(result["data_as_of_note"]))
    if name in ("get_market_scan_recommendations", "get_sector_recommendations"):
        if result.get("refreshing"):
            found.append("추천 결과 갱신 중(이전 스냅샷)")
        if "scanned_at" in result and result.get("scanned_at") is None:
            found.append("추천 결과가 아직 없음(분석 전)")
    for data_date in (_get(result, "latest", "date"), _get(result, "quote", "date")):
        if data_date and _stale(data_date, today):
            found.append(f"{name} 데이터 기준일 {data_date}(오래됨)")
    return found


def _warnings_for(call: ToolCallRecord, today: date) -> list[str]:
    result = call.result
    if not isinstance(result, dict):
        return []
    if result.get("available") is False:
        return [f"{call.name}: 데이터 조회 실패({str(result.get('error', ''))[:_ERROR_PREVIEW]})"]
    return _signal_warnings(call.name, result) + _status_warnings(call.name, result, today)


def extract_warnings(calls: Sequence[ToolCallRecord], today: date) -> tuple[str, ...]:
    return tuple(dict.fromkeys(w for call in calls for w in _warnings_for(call, today)))
