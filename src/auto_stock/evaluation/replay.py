"""도구 결과 녹화·재생 — 계획서 §4.5.

라이브 실행은 데이터가 날마다 바뀌어, 점수 변화가 에이전트 때문인지 데이터 때문인지 구분할 수
없다. `--record`로 (도구, 인자) → 결과를 저장하고 `--replay`로 같은 결과를 다시 먹인다. 회귀 비교는
재생 실행끼리만 한다(§8.3).

- 서브에이전트(stock_analyst 등)는 메인이 받은 **최종 출력**이 고정된다(내부 LLM 호출 0회).
- 픽스처에 없는 조합은 `available=false, reason=replay_miss`로 돌려주고 집계한다 — 에이전트가
  녹화 때와 다른 인자를 쓴 것 자체가 신호다."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from auto_stock.chat_agent.tool_registry import TOOL_SPECS
from auto_stock.evaluation.answer_capture import to_jsonable
from auto_stock.evaluation.tool_patching import patch_tool

REPLAY_MISS_REASON = "replay_miss"
_REGISTRY_NAMES = frozenset(spec.name for spec in TOOL_SPECS)


def call_key(name: str, arguments: Mapping[str, Any]) -> str:
    return f"{name}|{json.dumps(to_jsonable(dict(arguments)), sort_keys=True, ensure_ascii=False)}"


class FixtureRecorder:
    """`on_event`로 메인 루프의 `tool_done` 이벤트를 받아 결과를 쌓는다."""

    def __init__(self) -> None:
        self._entries: dict[str, list[Any]] = {}

    @property
    def entries(self) -> dict[str, list[Any]]:
        return {key: list(results) for key, results in self._entries.items()}

    def on_event(self, event: dict) -> None:
        if event.get("type") != "tool_done" or event.get("name") not in _REGISTRY_NAMES:
            return
        arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
        key = call_key(event["name"], arguments)
        self._entries.setdefault(key, []).append(to_jsonable(event.get("result")))

    def save(self, path: Path, meta: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": dict(meta), "entries": self.entries}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ReplayStore:
    def __init__(self, entries: Mapping[str, Sequence[Any]], meta: Mapping[str, Any] | None = None) -> None:
        self._entries = {key: list(results) for key, results in entries.items()}
        self._cursor: dict[str, int] = {}
        self.misses: list[str] = []
        self.meta = dict(meta or {})

    @classmethod
    def load(cls, path: Path) -> "ReplayStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["entries"], payload.get("meta"))

    def reset_cursor(self) -> None:
        """대화마다 처음 결과부터 다시 재생한다 — 앞 대화·반복 회차의 소비 위치가 새지 않게."""
        self._cursor = {}

    def lookup(self, name: str, arguments: Mapping[str, Any]) -> Any:
        key = call_key(name, arguments)
        results = self._entries.get(key)
        if not results:
            self.misses.append(key)
            return {
                "available": False,
                "reason": REPLAY_MISS_REASON,
                "error": "녹화 픽스처에 없는 도구 호출입니다(평가 재생 모드)",
            }
        index = self._cursor.get(key, 0)
        self._cursor[key] = index + 1
        return results[min(index, len(results) - 1)]


def replay_patchers(store: ReplayStore) -> list:
    def _make(tool: str):
        def _replayed(context: Any, **kwargs: Any) -> Any:
            return store.lookup(tool, kwargs)

        return _replayed

    return [patch_tool(spec.name, _make(spec.name)) for spec in TOOL_SPECS]
