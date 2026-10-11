"""메인 대화 에이전트에게 노출하는 도구들의 OpenAI Responses API 함수콜링 스키마.

스키마는 더 이상 여기서 손으로 쓰지 않는다 — `tool_registry.py`의 `TOOL_SPECS`(도구 단일
정의)에서 파생해 재export만 한다(docs/design/main-agent-migration-plan.md §4.0). 구 루프
(`loop.py`)가 이 모듈 이름으로 import하므로 하위 호환을 위해 남겨 둔다.

형태는 `chat.completions`의 `{"function": {...}}` 중첩이 아니라 Responses API 고유의 평평한
형태(`type`/`name`/`description`/`parameters`/`strict`가 최상위 키)다(Context7
`/openai/openai-python`로 확인). 레지스트리 도입 전과 바이트 단위로 같다는 사실은
`tests/chat_agent/fixtures/tool_schemas_golden.json` 골든 테스트가 고정한다.
"""

from auto_stock.chat_agent.tool_registry import TOOL_SCHEMAS

__all__ = ["TOOL_SCHEMAS"]
