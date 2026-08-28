"""`notifier/credentials.py`와 동일 패턴: 진입점이 필요한 시점에 `load_dotenv()`를 직접
호출하고, 필수 크리덴셜이 없으면 `KeyError`로 즉시 실패한다 — 4중 방어의 마지막 층."""

import os

from dotenv import load_dotenv

from auto_stock.llm_chart_analyst.models import LLMConfig

DEFAULT_MODEL = "gpt-5.6-luna"  # 사용자 확정 — OPENAI_MODEL 환경변수로 교체 가능
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2  # SDK 내장 백오프 재시도
DEFAULT_MAX_CALLS_PER_RUN = 20  # 비용 안전밸브


def load_llm_config() -> LLMConfig:
    load_dotenv()
    return LLMConfig(
        api_key=os.environ["OPENAI_API_KEY"],  # 없으면 KeyError — 진입점에서 즉시 실패
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        max_calls_per_run=int(os.environ.get("LLM_MAX_CALLS_PER_RUN", DEFAULT_MAX_CALLS_PER_RUN)),
    )
