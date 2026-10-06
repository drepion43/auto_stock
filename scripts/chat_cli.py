"""대화형 챗봇 에이전트 CLI 진입점(Stage C) — `chat_agent.loop.run_turn`을 stdin 루프로
감싼다. `run_recommendations_with_signals.py`와 동일하게 ML 아티팩트/LLM 크리덴셜이
없어도 즉시 종료하지 않는다 — 이 스크립트는 배치가 아니라 대화형이라, 일부 신호원이
꺼져 있어도 나머지 도구로 계속 대화할 수 있어야 한다(`ChatToolContext`의 각 필드가
`None`이면 해당 도구가 `available=False`로 스스로 저하되는 기존 계약을 그대로 재사용).

`QueryBudget`은 대화 "한 턴"마다 새로 만든다(`chat_agent/models.py`의 `QueryBudget`
docstring, `loop.py`의 `run_turn` docstring 참고) — 세션 전체가 아니라 매 사용자
입력마다 `MAX_LLM_CALLS_PER_QUERY`가 리셋된다.

사용자 요구사항(2026-09-06, 2차): 결과가 나온 "뒤" 요약이 아니라 "지금 tool 호출
중입니다"/"reasoning 중입니다" 같은 활동 상태를 작업이 시작되는 시점에 실시간으로
보고 싶다는 요청 — `run_turn`의 `on_event` 콜백(thinking/reasoning/tool_start/
tool_done 4종 이벤트, 각각 해당 블로킹 작업 시작 직전 또는 직후에 emit됨)을 받아
`_ActivityPrinter`가 즉시 출력한다(진짜 토큰 스트리밍이 아니라 활동 단위 진행 표시,
사용자가 명시적으로 선택한 방식). reasoning 요약은 API 자체는 거부하지 않지만
실제로는 사용 모델에 따라 항상 빈 문자열일 수 있다(`client.py` 참고 — 2026-09-06
실제 API로 확인) — 그 경우 reasoning 줄은 그냥 안 뜬다.

이 스크립트는 실제 OpenAI API를 호출한다(비용 발생) — OPENAI_API_KEY 발급 전에는
실행하지 말 것.

Run from the project root so `.env`/`data/` resolve correctly:
    .venv/Scripts/python scripts/chat_cli.py   (Windows)
    .venv/bin/python scripts/chat_cli.py        (macOS/Linux)

AccountState는 `run_recommendations.py`/`run_recommendations_with_signals.py`와 동일한
이유로 placeholder다(브로커 연동 전, MVP-1(#8)에서 교체 예정).

Prints only counts/messages — never prints OPENAI_API_KEY, DART_API_KEY,
SEC_EDGAR_USER_AGENT, NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, or ACCOUNT_EQUITY.
"""

import itertools
import json
import os
import sys
from datetime import timedelta

# Windows 콘솔 기본 코드페이지(cp949)는 이모지·em-dash 등 한글 텍스트에 흔한 문자를
# 인코딩하지 못해 UnicodeEncodeError로 죽는다(실제 실행에서 발견) — stdout/stderr를
# UTF-8로 강제 재설정해 플랫폼 로캘과 무관하게 항상 출력되게 한다. stdin도 마찬가지로
# 재설정한다 — 안 하면 cp949로 사용자 입력을 잘못 디코딩해(한글 질문이 깨짐) 그 깨진
# 문자열이 API 요청 직렬화 단계에서 예외를 일으키고, 그게 run_turn의 넓은 except에
# 잡혀 "일시적인 오류"로만 보여 원인을 알 수 없게 되는 버그를 실제로 겪었다.
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from auto_stock.chat_agent.client import OpenAIChatAgentClient
from auto_stock.chat_agent.credentials import MAX_LLM_CALLS_PER_QUERY, load_llm_config
from auto_stock.chat_agent.loop import run_turn
from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.tools import ChatToolContext
from auto_stock.data.cache import OHLCVCache
from auto_stock.data.scan_cache import MarketScanCache
from auto_stock.llm_chart_analyst.client import OpenAIChartClient
from auto_stock.llm_chart_analyst.credentials import (
    load_llm_config as load_chart_llm_config,
)
from auto_stock.ml_predictor.artifact import ModelArtifactError, load_model
from auto_stock.news_disclosure.client import OpenAIDisclosureClient
from auto_stock.news_disclosure.credentials import (
    load_llm_config as load_disclosure_llm_config,
)
from auto_stock.news_sentiment.client import OpenAINewsSentimentClient
from auto_stock.news_sentiment.credentials import (
    load_llm_config as load_sentiment_llm_config,
)
from auto_stock.orchestrator.recommendation_coordinator import RecommendationCoordinator
from auto_stock.risk_sizing.models import AccountState

DEFAULT_ACCOUNT_EQUITY = 10_000_000.0
_MARKETS = ["KRX", "NASDAQ"]
_EXIT_COMMANDS = {"exit", "quit", "종료"}
_RESULT_PREVIEW_MAX_CHARS = 200  # tool 결과가 길면(공시/뉴스 해석 등) 터미널을 뒤덮지 않도록 절단


class _ActivityPrinter:
    """`run_turn`의 `on_event` 콜백 구현체 — 사용자 요구사항(2026-09-06, 2차): "현재
    tool 호출 중입니다", "reasoning 중입니다" 같은 활동 상태를 결과가 나온 뒤가 아니라
    작업이 시작되는 시점에 명시적으로 남겨달라는 요청. `tool_start`에서 매긴 번호를
    바로 뒤따라오는 `tool_done`에 그대로 재사용하기 위해 인스턴스 상태(순번)를 갖는다
    — 한 턴 동안 tool 실행은 항상 순차적이라(비동기 병행 없음) start/done이 교차될
    일이 없다."""

    def __init__(self) -> None:
        self._call_counter = itertools.count(1)
        self._current_index = 0

    def __call__(self, event: dict) -> None:
        event_type = event["type"]
        if event_type == "thinking":
            print("  🤔 생각 중입니다...")
        elif event_type == "reasoning":
            print(f"  💭 reasoning: {event['text']}")
        elif event_type == "tool_start":
            self._current_index = next(self._call_counter)
            print(f"  [{self._current_index}] 🔧 {event['name']}({event['arguments']}) 호출 중입니다...")
        elif event_type == "tool_done":
            result_preview = json.dumps(event["result"], ensure_ascii=False)
            if len(result_preview) > _RESULT_PREVIEW_MAX_CHARS:
                result_preview = result_preview[:_RESULT_PREVIEW_MAX_CHARS] + "..."
            print(f"  [{self._current_index}]     -> {result_preview}")


def _load_ml_models() -> dict:
    models = {}
    for market in _MARKETS:
        try:
            models[market] = load_model(market)
        except ModelArtifactError:
            models[market] = None  # 대화형이라 없어도 계속 진행 — 해당 도구만 available=False
    return models


def _load_chart_client() -> OpenAIChartClient | None:
    try:
        return OpenAIChartClient(load_chart_llm_config())
    except KeyError:
        return None


def _load_news_client() -> OpenAIDisclosureClient | None:
    try:
        return OpenAIDisclosureClient(load_disclosure_llm_config())
    except KeyError:
        return None


def _load_sentiment_client() -> OpenAINewsSentimentClient | None:
    try:
        return OpenAINewsSentimentClient(load_sentiment_llm_config())
    except KeyError:
        return None


def main() -> None:
    try:
        agent_config = load_llm_config()
    except KeyError:
        print("FAILED: OPENAI_API_KEY가 .env에 설정되어 있지 않습니다 — 챗봇을 쓰려면 먼저 발급/설정하세요.")
        sys.exit(1)

    reader = OpenAIChatAgentClient(agent_config)
    cache = OHLCVCache("data/ohlcv.duckdb")
    scan_cache = MarketScanCache("data/market_scan.duckdb")
    ml_models = _load_ml_models()
    llm_client = _load_chart_client()
    news_client = _load_news_client()
    sentiment_client = _load_sentiment_client()
    account = AccountState(
        equity=float(os.environ.get("ACCOUNT_EQUITY", DEFAULT_ACCOUNT_EQUITY)),
        held_tickers=frozenset(),
        total_exposure_pct=0.0,
    )
    recommendation_coordinator = RecommendationCoordinator(
        cache=cache,
        scan_cache=scan_cache,
        ml_models=ml_models,
        llm_client=llm_client,
        news_client=news_client,
        sentiment_client=sentiment_client,
        account=account,
        agent_model=agent_config.model,
        stale_after=timedelta(hours=int(os.environ.get("RECOMMENDATION_STALE_HOURS", 24))),
        universe_size=int(os.environ.get("MARKET_SCAN_UNIVERSE_SIZE", 200)),
    )
    for market in _MARKETS:
        recommendation_coordinator.ensure_fresh(market)

    print("auto_stock 대화형 리서치 챗봇 — 종료하려면 'exit' 입력")
    previous_response_id: str | None = None
    while True:
        try:
            user_message = input("you> ").strip()
        except EOFError:
            break
        if not user_message:
            continue
        if user_message.lower() in _EXIT_COMMANDS:
            break

        context = ChatToolContext(
            cache=cache,
            ml_models=ml_models,
            llm_client=llm_client,
            news_client=news_client,
            sentiment_client=sentiment_client,
            budget=QueryBudget(max_llm_calls=MAX_LLM_CALLS_PER_QUERY),
            account=account,
            agent_model=agent_config.model,
            scan_cache=scan_cache,
            recommendation_coordinator=recommendation_coordinator,
        )
        text, previous_response_id = run_turn(
            reader,
            context,
            user_message,
            previous_response_id,
            on_event=_ActivityPrinter(),
        )
        print(f"bot> {text}")


if __name__ == "__main__":
    main()
