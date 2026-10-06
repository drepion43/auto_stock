"""심층분석(②단계) — 숏리스트 종목마다 기존 `stock_analyst`(deepagents, 자율 판단)를
재사용한다(recommendation-synthesis-plan.md §3). 신규 agent를 만들지 않는다 — 고정
fan-out(모든 종목에 3신호 균일 호출)보다 agent 자율성을 우선하기로 사용자가 확정했다
(2026-10-02). 트레이드오프로 종목마다 `signals_used`가 달라질 수 있다는 점은 ③단계
종합 프롬프트가 흡수한다.

`ThreadPoolExecutor.map()`을 쓰는 이유: 내부적으로 입력 순서를 그대로 보존해서 결과를
반환한다는 표준 라이브러리의 보장된 계약이라, 종목 간 동시 실행을 하면서도 "결과 순서가
입력 종목 순서와 일치해야 한다"(③단계의 비교·순위 매기기에 필요)는 요구를 추가 로직
없이 만족시킨다. 이 프로젝트에 asyncio 전례가 없고 threading만 쓰는 기존 관행과도
일치한다(ScanCoordinator 등).

종목별 실패격리는 `run_stock_analyst` 자신이 이미 보장하지만(절대 raise 안 함), 이
함수에 전달되는 `bind_tools` 콜백이 호출부 실수로 예외를 낼 수 있으므로 한 번 더
감싼다 — scan_market/build_shortlist와 동일한 원칙."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.stock_analyst import run_stock_analyst
from auto_stock.ml_predictor.models import MLPrediction
from auto_stock.orchestrator.shortlist import ShortlistEntry
from auto_stock.rule_engine.models import Candidate

DEFAULT_MAX_WORKERS = 8  # recommendation-synthesis-plan.md §3, 2026-10-05 확정


@dataclass(frozen=True, slots=True)
class TickerJudgment:
    ticker: str
    market: str
    rule_candidate: Candidate | None
    ml_prediction: MLPrediction | None
    stock_analyst_result: dict


def run_deep_scan(
    shortlist: list[ShortlistEntry],
    budget: QueryBudget,
    agent_model: str,
    bind_tools: Callable[[str, str], list],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[TickerJudgment]:
    if not shortlist:
        return []

    def _judge(entry: ShortlistEntry) -> TickerJudgment:
        try:
            tools = bind_tools(entry.ticker, entry.market)
            result = run_stock_analyst(budget, agent_model, tools, entry.ticker, entry.market)
        except Exception as exc:  # bind_tools 실수 격리 — run_stock_analyst 자신은 이미 격리됨
            result = {"available": False, "error": str(exc)}
        return TickerJudgment(
            ticker=entry.ticker, market=entry.market,
            rule_candidate=entry.rule_candidate, ml_prediction=entry.ml_prediction,
            stock_analyst_result=result,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_judge, shortlist))
