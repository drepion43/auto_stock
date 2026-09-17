"""notifier/credentials.py, llm_chart_analyst/credentials.py와 동일 패턴: 진입점이 필요한
시점에 `load_dotenv()`를 직접 호출하고, 필수 크리덴셜이 없으면 `KeyError`로 즉시 실패한다.

`OPENAI_API_KEY`는 llm_chart_analyst(#3)와 동일 값을 재사용한다 — 신규 발급이 필요한 것은
`DART_API_KEY`뿐이다(data/sources/dart_source.py 소관, 이 파일과 무관). 모델 등급만
`NEWS_OPENAI_MODEL`로 독립 재정의 가능하게 해 #3과 다른 등급이 필요해져도 이 파일의 변경이
없게 한다.
"""

import os

from dotenv import load_dotenv

from auto_stock.news_disclosure.models import LLMConfig

DEFAULT_MODEL = "gpt-5.6-luna"  # #3과 동일 등급 — 구조화 3분류+짧은 요약 과제
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2  # SDK 내장 백오프 재시도
DEFAULT_MAX_CALLS_PER_RUN = 20  # 비용 안전밸브
DISCLOSURE_LOOKBACK_DAYS = 90  # 최근 3개월 공시만 조회
MAX_DISCLOSURES_PER_QUERY = 10  # 프롬프트에 넣을 최대 건수


def load_llm_config() -> LLMConfig:
    load_dotenv()
    return LLMConfig(
        api_key=os.environ["OPENAI_API_KEY"],  # 없으면 KeyError — 진입점에서 즉시 실패
        model=os.environ.get("NEWS_OPENAI_MODEL", DEFAULT_MODEL),
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        max_calls_per_run=DEFAULT_MAX_CALLS_PER_RUN,
    )
