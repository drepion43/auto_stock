"""notifier/credentials.py, news_disclosure/credentials.py, news_sentiment/credentials.py와
동일 패턴: 진입점이 필요한 시점에 `load_dotenv()`를 직접 호출하고, 필수 크리덴셜이 없으면
`KeyError`로 즉시 실패한다.

`OPENAI_API_KEY`는 #3/#4-공시/#4-뉴스와 동일 값을 재사용한다 — 새 크리덴셜 없음. 모델
등급만 `CHAT_AGENT_OPENAI_MODEL`로 독립 재정의 가능하게 해 대화 에이전트가 다른 등급이
필요해져도 다른 신호원 모듈의 변경이 없게 한다.

`MAX_TICKERS_PER_QUERY`/`MAX_LLM_CALLS_PER_QUERY`는 `DEFAULT_MAX_CALLS_PER_RUN`(클라이언트
인스턴스 수명 예산, 즉 `chat_cli.py` 대화 세션 전체)과는 다른 층위의 개념이다 — 이건 대화
"한 턴"마다 리셋되는 예산이다(설계문서 확정 사항, `chat_agent/models.py`의 `QueryBudget`
참고). `DEFAULT_MAX_CALLS_PER_RUN`은 세션 전체의 폭주만 막는 넉넉한 안전밸브이므로 턴당
예산보다 훨씬 커야 한다 — 반대로 두면(예: 배치 모듈들의 기본값 20을 그대로 재사용하면)
몇 턴 만에 세션 전체가 영구 잠긴다(코드 리뷰에서 발견한 실제 버그, Stage E 리뷰 수정).
"""

import os

from dotenv import load_dotenv

from auto_stock.chat_agent.models import LLMConfig

DEFAULT_MODEL = "gpt-5.6-luna"  # 다른 신호원 모듈과 동일 등급
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2  # SDK 내장 백오프 재시도
# `OpenAIChatAgentClient`는 `chat_cli.py`에서 대화 세션 전체(여러 턴)에 걸쳐 재사용되는
# 단일 인스턴스이므로, 이 값은 배치 파이프라인처럼 "한 번의 스크립트 실행"이 아니라
# "대화 세션 전체"를 스코프로 하는 안전밸브다. 이전에는 다른 배치용 클라이언트와 같은
# 기본값(20)을 그대로 썼는데, 한 턴이 최대 MAX_TOOL_ITERATIONS(5)회의 실제 API 호출을
# 소모하는 챗봇에서는 4턴 만에 클라이언트가 영구 잠기는 버그였다(코드 리뷰에서 발견,
# Stage E 리뷰 수정) — 실제 비용 상한은 턴마다 리셋되는 QueryBudget이 담당하므로, 이건
# 세션 전체의 폭주만 막는 넉넉한 상한이다. 초기값, 실사용 관찰 후 조정 대상.
DEFAULT_MAX_CALLS_PER_RUN = 500

MAX_TICKERS_PER_QUERY = 5  # 한 턴에 해석할 수 있는 최대 티커 수(관련기업 딥다이브 개수 상한에도 재사용)

# LangGraph가 실제로 강제하는 재귀 상한(단순 프롬프트 권고가 아님, agent.invoke(...,
# config={"recursion_limit": N})) — 초기값, 실사용 관찰 후 조정 대상(이 프로젝트의 ATR
# 승수 등 다른 "TBD, 초기값" 상수들과 동일한 성격). 아래 *_LLM_CALL_ESTIMATE가 이 값을
# 그대로 참조하므로(과소 사전차감 방지, 아래 설명) 먼저 정의한다.
FIND_RELATED_RECURSION_LIMIT = 6
STOCK_ANALYST_RECURSION_LIMIT = 8

# find_related_companies/stock_analyst는 deepagents 서브에이전트라 내부 실제 LLM 호출
# 횟수를 QueryBudget이 정확히 셀 수 없다(deepagents 내부 루프가 우리 QueryBudget 객체를
# 모름) — 호출 전에 이 추정치만큼 한 번에 선차감하는 근사 방식이다. **반드시 각각의
# recursion_limit 이상이어야 한다**(보안 리뷰 발견사항: 이전에는 3/4로 각 재귀상한
# 6/8보다 작아서, 서브에이전트가 재귀상한까지 실제로 LLM을 호출하면 QueryBudget이 그
# 실제 지출을 과소 계상했다 — 예산상으로는 남아 있어도 실제 OpenAI 비용은 초과할 수
# 있었다). 재귀상한과 동일하게 맞춰 "과소 사전차감"을 구조적으로 불가능하게 한다.
FIND_RELATED_LLM_CALL_ESTIMATE = FIND_RELATED_RECURSION_LIMIT
STOCK_ANALYST_LLM_CALL_ESTIMATE = STOCK_ANALYST_RECURSION_LIMIT

# Stage E(관련기업 자율 딥다이브) 반영 — 자기 자신 포함 최대 5개사를 stock_analyst로
# 딥다이브 + find_related_companies 1회를 감당해야 하는 최악의 경우를 커버해야 한다.
# 위 두 ESTIMATE가 각각의 recursion_limit과 같아지도록 재조정됐으므로 이 값도 상향했다.
# 초과분은 예산 부족으로 우아하게 저하(일부 종목 미분석)한다.
MAX_LLM_CALLS_PER_QUERY = 50
MAX_TOOL_ITERATIONS = 5  # 한 턴 안에서 모델-도구 왕복을 허용하는 최대 라운드 수


def load_llm_config() -> LLMConfig:
    load_dotenv()
    return LLMConfig(
        api_key=os.environ["OPENAI_API_KEY"],  # 없으면 KeyError — 진입점에서 즉시 실패
        model=os.environ.get("CHAT_AGENT_OPENAI_MODEL", DEFAULT_MODEL),
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        max_calls_per_run=DEFAULT_MAX_CALLS_PER_RUN,
    )
