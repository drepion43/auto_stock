"""평가 턴 캡처와 채점 문맥 만들기 — 계획서 §4.2, §4.3, §7.4.

메인 루프의 `on_event`가 내는 `tool_done`(이름·인자·결과)을 그대로 모은다(에이전트 코드 수정 없음).
충실도 채점 문맥이 상한을 넘으면 앞부분을 자르지 않고, 답변에 등장한 종목명·티커·숫자가 들어 있는
결과 원소를 먼저 고른다(§7.4 선택적 문맥) — 단순 절단은 정상 주장을 "근거 없음"으로 만든다."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from auto_stock.chat_agent.agent_middleware import parse_delegation_target
from auto_stock.chat_agent.tool_registry import TOOL_SPECS
from auto_stock.evaluation.golden import LLM_GENERATED_TOOLS

DEFAULT_CONTEXT_LIMIT = 12_000  # 케이스당 문맥 상한(문자) — §4.2, §7.4
_DESCRIPTIONS = {spec.name: spec.description for spec in TOOL_SPECS}

_ANSWER_TOKEN = re.compile(r"[가-힣A-Za-z0-9.]{2,}")


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    arguments: Mapping[str, Any]
    result: Any
    # 서브에이전트 내부 호출이면 그 서브에이전트 이름(subagent 모드의 stock_analyst 신호 도구) — 메인 답변이 직접
    # 본 결과가 아니므로 채점 문맥·정직성 규칙에서는 뺀다(`main_visible`). 메인 호출은 None.
    subagent: str | None = None


def main_visible(records: Sequence[ToolCallRecord]) -> tuple[ToolCallRecord, ...]:
    """메인 에이전트가 결과를 직접 받은 호출만 — 답변 근거(M2·M3·M5·M6·M7·M8, 정책 규칙)는 이것으로 판단한다."""
    return tuple(record for record in records if record.subagent is None)


@dataclass(frozen=True, slots=True)
class ContextBundle:
    text: str
    truncated: bool
    llm_generated_tools: tuple[str, ...]


def to_jsonable(value: Any) -> Any:
    """날짜·집합 등 JSON에 없는 타입을 문자열로 바꾼 사본."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class ToolCallSink:
    """한 턴의 `tool_done` 이벤트 수집기."""

    def __init__(self) -> None:
        self._records: list[ToolCallRecord] = []

    @property
    def records(self) -> tuple[ToolCallRecord, ...]:
        return tuple(self._records)

    def on_event(self, event: dict) -> None:
        if event.get("type") != "tool_done":
            return
        name, arguments = _normalize_delegation(event.get("name") or "", event.get("arguments"))
        self._records.append(
            ToolCallRecord(
                name=name,
                arguments=to_jsonable(arguments) if isinstance(arguments, dict) else {"raw": str(arguments)},
                result=to_jsonable(event.get("result")),
                subagent=event.get("subagent"),
            )
        )


def _normalize_delegation(name: str, arguments: Any) -> tuple[str, Any]:
    """subagent 모드(docs/design/stock-analyst-subagent-migration.md)의 `task(subagent_type="stock_analyst")` 위임을
    tool 모드와 같은 `stock_analyst(ticker, market)` 호출로 기록한다 — 골든 기대 호출·LLM 생성 도구 분류·도구 설명이
    두 모드에서 같게 적용되도록. 서브에이전트 내부 신호 도구 호출은 원래 이름 그대로 남는다."""
    if name != "task" or not isinstance(arguments, dict) or arguments.get("subagent_type") != "stock_analyst":
        return name, arguments
    description = str(arguments.get("description", ""))
    target = parse_delegation_target(description)
    if target is None:
        return "stock_analyst", {"raw": description}
    request = "\n".join(line for line in description.splitlines()[1:] if line.strip())
    return "stock_analyst", {"ticker": target[0], "market": target[1], "request": request}


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _chunks(record: ToolCallRecord) -> list[str]:
    """도구 결과 1건을 문맥 조각으로 — 최상위 리스트 필드는 원소마다 따로 쪼갠다."""
    head = f"[{record.name}] args={_dump(record.arguments)}"
    result = record.result
    if not isinstance(result, dict):
        return [f"{head} result={_dump(result)}"]
    scalars = {key: value for key, value in result.items() if not isinstance(value, list)}
    chunks = [f"{head} result={_dump(scalars)}"]
    for key, value in result.items():
        if isinstance(value, list):
            chunks.extend(f"[{record.name}].{key}[{i}]={_dump(item)}" for i, item in enumerate(value))
    return chunks


def _description_chunks(tool_calls: Sequence[ToolCallRecord]) -> list[str]:
    """호출된 도구의 설명(모델이 받은 그대로) — 정책 고지("실시간이 아니라 최근 종가 기준")와 필드 의미
    ("scanned_at이 null이면 한 번도 실행된 적 없다")는 결과가 아니라 설명에만 있어, 빼면 정확한 답변이
    "근거 없는 주장"으로 감점된다(Phase 0 실측, 계획서 §16.3.1)."""
    names = list(dict.fromkeys(call.name for call in tool_calls))
    return [f"[도구 설명: {name}] {_DESCRIPTIONS[name]}" for name in names if name in _DESCRIPTIONS]


def build_context(tool_calls: Sequence[ToolCallRecord], answer: str, limit: int = DEFAULT_CONTEXT_LIMIT) -> ContextBundle:
    llm_tools = tuple(sorted({call.name for call in tool_calls if call.name in LLM_GENERATED_TOOLS}))
    descriptions = _description_chunks(tool_calls)
    chunks = descriptions + [chunk for call in tool_calls for chunk in _chunks(call)]
    full = "\n".join(chunks)
    if len(full) <= limit:
        return ContextBundle(full, False, llm_tools)

    # 도구 설명을 먼저, 그다음 답변 토큰을 많이 포함한 조각부터(동률이면 원래 순서). "추천" 같은 흔한
    # 단어 하나만 겹치는 조각보다 종목명·티커·숫자까지 겹치는 조각이 앞선다.
    answer_tokens = set(_ANSWER_TOKEN.findall(answer))
    hits = [sum(token in chunk for token in answer_tokens) for chunk in chunks]
    order = sorted(range(len(chunks)), key=lambda i: (i >= len(descriptions), -hits[i], i))
    chosen: set[int] = set()
    used = 0
    for i in order:
        cost = len(chunks[i]) + (1 if chosen else 0)
        if used + cost > limit:
            continue
        chosen.add(i)
        used += cost
    text = "\n".join(chunks[i] for i in sorted(chosen))
    return ContextBundle(text, True, llm_tools)


def _records_from(raw: Sequence[Mapping[str, Any]]) -> tuple[ToolCallRecord, ...]:
    return tuple(ToolCallRecord(r["name"], r["arguments"], r["result"], r.get("subagent")) for r in raw)


@dataclass(frozen=True)
class CapturedTurn:
    """`data/eval/answers_*.json`의 케이스 1건(§4.2 스키마)."""

    case_id: str
    query_id: str
    turn_index: int
    repeat: int
    question: str
    answer: str
    tool_calls: tuple[ToolCallRecord, ...]
    history_tool_calls: tuple[ToolCallRecord, ...]
    fallback: bool
    injected_fault: Mapping[str, Any] | None
    replay_fixture: str | None
    injection_marker: str | None
    usage: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["llm_generated_tools"] = sorted({c.name for c in self.tool_calls if c.name in LLM_GENERATED_TOOLS})
        return to_jsonable(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapturedTurn":
        return cls(
            case_id=data["case_id"],
            query_id=data["query_id"],
            turn_index=data["turn_index"],
            repeat=data["repeat"],
            question=data["question"],
            answer=data["answer"],
            tool_calls=_records_from(data["tool_calls"]),
            history_tool_calls=_records_from(data["history_tool_calls"]),
            fallback=data["fallback"],
            injected_fault=data["injected_fault"],
            replay_fixture=data["replay_fixture"],
            injection_marker=data["injection_marker"],
            usage=data.get("usage") or {},
        )
