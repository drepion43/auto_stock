"""평가용 도구 교체(녹화 재생·실패 주입·인젝션 스텁)의 공통 바닥.

`tool_registry`는 도구 함수를 호출 시점에 `getattr(chat_tools, fn_name)`으로 찾는다 — 그래서
`chat_tools`의 `tool_*` 속성만 바꾸면 에이전트 코드 수정 없이 결과를 갈아끼울 수 있다
(기존 `scripts/eval_main_agent.py`의 Q10 스텁과 같은 방식)."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from unittest import mock

from auto_stock.chat_agent import tools as chat_tools
from auto_stock.chat_agent.tool_registry import TOOL_SPECS

_FN_NAMES = {spec.name: spec.fn_name for spec in TOOL_SPECS}


def tool_fn_name(tool: str) -> str:
    try:
        return _FN_NAMES[tool]
    except KeyError:
        raise ValueError(f"unknown registry tool: {tool!r}") from None


def current_tool(tool: str) -> Callable[..., dict]:
    """지금 `chat_tools`에 걸려 있는 구현(이미 다른 패치가 걸려 있으면 그 패치)을 돌려준다."""
    return getattr(chat_tools, tool_fn_name(tool))


def patch_tool(tool: str, replacement: Callable[..., dict]):
    return mock.patch.object(chat_tools, tool_fn_name(tool), replacement)


@contextmanager
def applied(patchers: Sequence) -> Iterator[None]:
    """패치를 순서대로 걸고, 블록이 끝나면 역순으로 푼다(예외가 나도)."""
    with ExitStack() as stack:
        for patcher in patchers:
            stack.enter_context(patcher)
        yield
