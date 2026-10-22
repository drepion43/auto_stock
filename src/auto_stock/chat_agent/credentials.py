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
# recommendation-synthesis.md §5 2차(LLM 제안+뉴스검증) 섹터/테마 종목 탐색 —
# find_related_companies와 동일한 "제안 → verify_companies_co_mentioned_in_news로
# 검증" 루프 복잡도라 같은 값을 쓴다.
SECTOR_THEME_RECURSION_LIMIT = 6
# docs/design/etf-composition-analysis.md §C — "여러 종목 신호를 비교·종합"하는 복잡도가
# recommendation_synthesis와 비슷하다고 보고 같은 값(6)을 쓴다. 재조사 tool은 없다.
ETF_COMPOSITION_RECURSION_LIMIT = 6

# find_related_companies/stock_analyst는 deepagents 서브에이전트라 내부 실제 LLM 호출
# 횟수를 QueryBudget이 정확히 셀 수 없다(deepagents 내부 루프가 우리 QueryBudget 객체를
# 모름) — 호출 전에 이 추정치만큼 한 번에 선차감하는 근사 방식이다. **반드시 각각의
# recursion_limit 이상이어야 한다**(보안 리뷰 발견사항: 이전에는 3/4로 각 재귀상한
# 6/8보다 작아서, 서브에이전트가 재귀상한까지 실제로 LLM을 호출하면 QueryBudget이 그
# 실제 지출을 과소 계상했다 — 예산상으로는 남아 있어도 실제 OpenAI 비용은 초과할 수
# 있었다). 재귀상한과 동일하게 맞춰 "과소 사전차감"을 구조적으로 불가능하게 한다.
FIND_RELATED_LLM_CALL_ESTIMATE = FIND_RELATED_RECURSION_LIMIT
STOCK_ANALYST_LLM_CALL_ESTIMATE = STOCK_ANALYST_RECURSION_LIMIT
SECTOR_THEME_LLM_CALL_ESTIMATE = SECTOR_THEME_RECURSION_LIMIT
ETF_COMPOSITION_LLM_CALL_ESTIMATE = ETF_COMPOSITION_RECURSION_LIMIT

# analyze_etf_composition이 구성종목별 공시/뉴스감성 조회를 병렬로 돌릴 때의 동시성 —
# orchestrator/deep_scan.py와 같은 ThreadPoolExecutor 패턴. 구성종목 1개당 LLM 호출은
# 최대 2건(공시 1 + 뉴스감성 1)이라 예산 상한 계산에도 쓴다.
ETF_HOLDING_MAX_WORKERS = 4
ETF_HOLDING_LLM_CALLS_PER_HOLDING = 2

# 섹터/테마 탐색이 한 번에 제안할 수 있는 종목 수 상한 — 숏리스트 캡(20, orchestrator/
# recommendation_coordinator.py)과 같은 자릿수로 맞춰 비용을 통제한다.
MAX_SECTOR_THEME_TICKERS = 20

# Stage E(관련기업 자율 딥다이브) 반영 — 자기 자신 포함 최대 5개사를 stock_analyst로
# 딥다이브 + find_related_companies 1회를 감당해야 하는 최악의 경우를 커버해야 한다.
# 위 두 ESTIMATE가 각각의 recursion_limit과 같아지도록 재조정됐으므로 이 값도 상향했다.
# 초과분은 예산 부족으로 우아하게 저하(일부 종목 미분석)한다.
MAX_LLM_CALLS_PER_QUERY = 50
MAX_TOOL_ITERATIONS = 5  # 한 턴 안에서 모델-도구 왕복을 허용하는 최대 라운드 수

# --- 1단계: deepagents 메인 루프(docs/design/main-agent-migration-plan.md §4.1, §5) ---
# 메인 모델 호출의 턴당 상한 — 구 루프의 MAX_TOOL_ITERATIONS와 같은 값(ModelCallLimitMiddleware run_limit).
MAIN_AGENT_RUN_LIMIT = MAX_TOOL_ITERATIONS
# 세션 전체 메인 모델 호출 상한(결정 D5) — 실패해서 롤백된 턴의 호출도 센다(체크포인트 밖 카운터).
MAIN_AGENT_SESSION_CALL_LIMIT = DEFAULT_MAX_CALLS_PER_RUN
# 마지막 응답의 실제 입력 토큰이 이 값을 넘으면 다음 턴 전에 히스토리를 압축하고 서버 체인을
# 새로 시작한다(결정 D5) — previous_response_id 체인은 미들웨어로 줄일 수 없기 때문.
MAIN_AGENT_INPUT_TOKEN_LIMIT = 200_000
MAIN_AGENT_COMPACT_KEEP_TURNS = 2  # 압축 시 남기는 최근 (질문, 답변) 쌍 수
# 한 턴이 새로 띄울 수 있는 백그라운드 추천 배치 수(결정 D17) — 배치 1회가 LLM을 최대
# 약 268회 호출하므로, 모델이 질의 표현만 바꿔 여러 배치를 띄우는 비용 폭주를 막는다.
MAX_BATCH_TRIGGERS_PER_TURN = 1

LOOP_MODES = ("legacy", "deep")
# stock_analyst 연결 방식(docs/design/stock-analyst-subagent-migration.md): tool = 기존 agent-as-tool(기본),
# subagent = 메인 에이전트의 subagents=에 등록하고 신호 도구 8개를 서브에이전트로 옮긴다(결정 D1 안 A).
STOCK_ANALYST_MODES = ("tool", "subagent")


def load_stock_analyst_mode() -> str:
    """`CHAT_AGENT_STOCK_ANALYST_MODE` 기능 플래그 — 평가(설계 §8) 통과 전까지 기본은 `tool`. 알 수 없는 값은 실패."""
    load_dotenv()
    mode = os.environ.get("CHAT_AGENT_STOCK_ANALYST_MODE", "tool").strip().lower()
    if mode not in STOCK_ANALYST_MODES:
        raise ValueError(f"CHAT_AGENT_STOCK_ANALYST_MODE must be one of {STOCK_ANALYST_MODES}, got {mode!r}")
    return mode


def load_loop_mode() -> str:
    """`CHAT_AGENT_LOOP` 기능 플래그. 2026-10-12 eval(결정 D11) 통과 후 기본값을 deep으로 바꿨다 —
    문제가 생기면 `CHAT_AGENT_LOOP=legacy`로 즉시 구 루프로 되돌릴 수 있다(2주 실사용 후 구 루프 제거,
    결정 D12). 알 수 없는 값은 조용히 넘기지 않고 실패한다."""
    load_dotenv()
    mode = os.environ.get("CHAT_AGENT_LOOP", "deep").strip().lower()
    if mode not in LOOP_MODES:
        raise ValueError(f"CHAT_AGENT_LOOP must be one of {LOOP_MODES}, got {mode!r}")
    return mode


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
