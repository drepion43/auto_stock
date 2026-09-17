"""notifier/credentials.py, news_disclosure/credentials.py와 동일 패턴: 진입점이 필요한
시점에 `load_dotenv()`를 직접 호출하고, 필수 크리덴셜이 없으면 `KeyError`로 즉시 실패한다.

`OPENAI_API_KEY`는 #3/#4-공시와 동일 값을 재사용한다 — 신규 발급이 필요한 것은
`NAVER_CLIENT_ID`/`NAVER_CLIENT_SECRET`뿐이다(data/sources/naver_news_source.py 소관,
이 파일과 무관. GDELT는 크리덴셜 자체가 없음). 모델 등급만 `NEWS_SENTIMENT_OPENAI_MODEL`로
독립 재정의 가능하게 해 공시 해석(#4-공시)과 다른 등급이 필요해져도 이 파일의 변경이
없게 한다.
"""

import os

from dotenv import load_dotenv

from auto_stock.news_sentiment.models import LLMConfig

DEFAULT_MODEL = "gpt-5.6-luna"  # #3/#4-공시와 동일 등급 — 구조화 3분류+짧은 요약 과제
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2  # SDK 내장 백오프 재시도
DEFAULT_MAX_CALLS_PER_RUN = 20  # 비용 안전밸브
# 뉴스 기사는 공시(DISCLOSURE_LOOKBACK_DAYS=90)보다 훨씬 빠르게 관련성을 잃는다 —
# 최근 2주 정도가 "지금 주가와 관련 있는 뉴스"로 합리적인 창.
NEWS_LOOKBACK_DAYS = 14
MAX_ARTICLES_PER_QUERY = 10  # 프롬프트에 넣을 최대 건수


def load_llm_config() -> LLMConfig:
    load_dotenv()
    return LLMConfig(
        api_key=os.environ["OPENAI_API_KEY"],  # 없으면 KeyError — 진입점에서 즉시 실패
        model=os.environ.get("NEWS_SENTIMENT_OPENAI_MODEL", DEFAULT_MODEL),
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        max_calls_per_run=DEFAULT_MAX_CALLS_PER_RUN,
    )
