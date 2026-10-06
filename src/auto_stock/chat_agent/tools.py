"""메인 대화 에이전트에게 노출하는 5개 도구의 실제 구현. `orchestrator/pipeline.py`의
`_ml_reasons`/`_llm_reasons`/`_news_reasons`/`_sentiment_reasons`와 동일한 실패격리
원칙(절대 raise 안 함, market별 소스 분기, 실패 단계별 별도 try/except)을 재사용하지만,
반환 형태는 `(reasons, error)` 튜플이 아니라 모델에게 그대로 JSON으로 보여줄
`{"available": bool, ...}` dict다.

`analyze_rule_engine`/`analyze_ml_prediction`은 LLM을 쓰지 않으므로 `QueryBudget`을
소비하지 않는다. `analyze_disclosures`/`analyze_news_sentiment`는 `_news_reasons`/
`_sentiment_reasons`와 동일하게 이름 해석 실패와 소스조회 실패, LLM 해석 실패를 각각
별도 try/except로 분리한다 — 하나로 묶으면 DART/EDGAR 실패가 "네이버뉴스/GDELT 조회
실패"로 잘못 라벨링되는 회귀가 있었다(이 세션에서 실제로 잡은 버그, docs/design/
news-sentiment.md 참고).
"""

from datetime import date, timedelta

from langchain_core.tools import tool as langchain_tool

from auto_stock.chat_agent.models import QueryBudget
from auto_stock.chat_agent.related_companies import run_find_related_companies
from auto_stock.chat_agent.stock_analyst import run_stock_analyst
from typing import TYPE_CHECKING

from auto_stock.chat_agent.ticker_resolution import resolve_ticker
from auto_stock.data.cache import OHLCVCache
from auto_stock.data.scan_cache import MarketScanCache
from auto_stock.data.service import get_ohlcv

if TYPE_CHECKING:
    # recommendation_coordinator.py가 이미 이 모듈의 ChatToolContext/_bind_stock_analyst_tools를
    # import하므로(재사용 목적), 여기서 되돌아가는 top-level import는 순환이 된다
    # (stock_analyst.py가 이미 겪은 것과 동일한 문제, 그 모듈 docstring 참고) — 타입힌트
    # 전용으로만 쓰고 런타임에는 로드하지 않는다.
    from auto_stock.orchestrator.recommendation_coordinator import RecommendationCoordinator
from auto_stock.data.sources.dart_source import fetch_disclosures, resolve_corp_code, resolve_corp_name
from auto_stock.data.sources.edgar_source import fetch_filings, resolve_cik, resolve_company_title
from auto_stock.data.sources.gdelt_source import search_articles
from auto_stock.data.sources.naver_news_source import search_news
from auto_stock.llm_chart_analyst.analyst import analyze as analyze_chart
from auto_stock.llm_chart_analyst.models import ChartPatternReader
from auto_stock.ml_predictor.models import ModelBundle
from auto_stock.ml_predictor.predictor import predict
from auto_stock.news_disclosure.analyst import analyze as analyze_disclosures
from auto_stock.news_disclosure.credentials import DISCLOSURE_LOOKBACK_DAYS
from auto_stock.news_disclosure.models import DisclosureReader
from auto_stock.news_sentiment.analyst import analyze as analyze_sentiment
from auto_stock.news_sentiment.credentials import NEWS_LOOKBACK_DAYS
from auto_stock.news_sentiment.models import NewsSentimentReader
from auto_stock.orchestrator.pipeline import DEFAULT_LOOKBACK_DAYS
from auto_stock.risk_sizing.models import AccountState
from auto_stock.risk_sizing.sizing import suggest_position
from auto_stock.rule_engine.engine import generate_candidates

_NEWS_SOURCE_LABELS = {"KRX": "DART", "NASDAQ": "EDGAR"}
_SENTIMENT_SOURCE_LABELS = {"KRX": "네이버뉴스", "NASDAQ": "GDELT"}


class ChatToolContext:
    """6개 도구가 공유하는 의존성 묶음. dataclass가 아닌 일반 클래스인 이유는 `budget`
    필드가 매 도구 호출마다 mutable하게 갱신되는 `QueryBudget` 참조이기 때문이다 —
    dataclass여도 동작은 같지만, 이 묶음 자체가 "여러 도구가 공유하는 가변 상태로의
    핸들"이라는 점을 명확히 하기 위해 일반 클래스로 둔다."""

    def __init__(
        self,
        cache: OHLCVCache,
        ml_models: dict[str, ModelBundle | None],
        llm_client: ChartPatternReader | None,
        news_client: DisclosureReader | None,
        sentiment_client: NewsSentimentReader | None,
        budget: QueryBudget,
        account: AccountState,
        agent_model: str,
        scan_cache: MarketScanCache,
        recommendation_coordinator: "RecommendationCoordinator",
    ) -> None:
        self.cache = cache
        self.ml_models = ml_models
        self.llm_client = llm_client
        self.news_client = news_client
        self.sentiment_client = sentiment_client
        self.budget = budget
        self.account = account
        self.scan_cache = scan_cache
        self.recommendation_coordinator = recommendation_coordinator
        # find_related_companies/stock_analyst가 deepagents create_deep_agent(model=...)에
        # 넘길 메인 대화 에이전트 자신의 모델 문자열. llm_client 등 다른 신호원 클라이언트의
        # 모델과는 독립적이다(각자 다른 등급일 수 있음) — chat_agent 자신의 LLMConfig.model.
        self.agent_model = agent_model


def _fetch_records(context: ChatToolContext, ticker: str, market: str):
    end = date.today()
    start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    return get_ohlcv(context.cache, ticker, start, end, market)


def tool_analyze_rule_engine(context: ChatToolContext, ticker: str, market: str) -> dict:
    try:
        records = _fetch_records(context, ticker, market)
        candidates = generate_candidates(records)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if not candidates:
        return {"available": True, "candidate": None}
    candidate = candidates[0]
    return {"available": True, "candidate": {"action": candidate.action, "reasons": candidate.reasons}}


def tool_analyze_ml_prediction(context: ChatToolContext, ticker: str, market: str) -> dict:
    model = context.ml_models.get(market)
    if model is None:
        return {"available": False, "error": f"{market} 시장에 학습된 ML 모델이 없습니다"}
    try:
        records = _fetch_records(context, ticker, market)
        prediction = predict(model, records)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if prediction is None:
        return {"available": True, "prediction": None}
    return {
        "available": True,
        "prediction": {
            "probability_up": prediction.probability_up,
            "top_features": prediction.top_features,
        },
    }


def tool_analyze_chart_pattern(context: ChatToolContext, ticker: str, market: str) -> dict:
    if context.llm_client is None:
        return {"available": False, "error": "LLM 차트분석 클라이언트가 설정되지 않았습니다"}
    try:
        records = _fetch_records(context, ticker, market)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if not context.budget.try_consume_llm_call():
        return {"available": False, "error": "이번 대화 턴의 LLM 호출 예산을 모두 사용했습니다"}
    try:
        analysis = analyze_chart(context.llm_client, records)
    except Exception as exc:
        return {"available": False, "error": f"LLM 차트분석 실패: {exc}"}
    if analysis is None:
        return {"available": True, "analysis": None}
    return {
        "available": True,
        "analysis": {
            "direction": analysis.direction,
            "confidence": analysis.confidence,
            "pattern_name": analysis.pattern_name,
            "rationale": analysis.rationale,
            "caveat": analysis.caveat,
        },
    }


def tool_analyze_disclosures(context: ChatToolContext, ticker: str, market: str) -> dict:
    if context.news_client is None:
        return {"available": False, "error": "공시해석 클라이언트가 설정되지 않았습니다"}
    source_label = _NEWS_SOURCE_LABELS.get(market)
    if source_label is None:
        return {"available": False, "error": f"지원하지 않는 시장입니다: {market}"}

    try:
        corp_id = resolve_corp_code(ticker) if market == "KRX" else resolve_cik(ticker)
        if corp_id is None:
            return {"available": True, "analysis": None}  # 미등록 종목 — 공시 없음과 동일 취급
        end = date.today()
        start = end - timedelta(days=DISCLOSURE_LOOKBACK_DAYS)
        disclosures = (
            fetch_disclosures(corp_id, start, end)
            if market == "KRX"
            else fetch_filings(corp_id, start, end)
        )
    except Exception as exc:
        return {"available": False, "error": f"{source_label} 조회 실패: {exc}"}

    if not context.budget.try_consume_llm_call():
        return {"available": False, "error": "이번 대화 턴의 LLM 호출 예산을 모두 사용했습니다"}
    try:
        analysis = analyze_disclosures(context.news_client, disclosures, ticker, market, end)
    except Exception as exc:
        return {"available": False, "error": f"LLM 공시해석 실패: {exc}"}
    if analysis is None:
        return {"available": True, "analysis": None}
    return {
        "available": True,
        "analysis": {
            "market_impact": analysis.market_impact,
            "confidence": analysis.confidence,
            "key_event": analysis.key_event,
            "rationale": analysis.rationale,
            "caveat": analysis.caveat,
        },
    }


def tool_analyze_news_sentiment(context: ChatToolContext, ticker: str, market: str) -> dict:
    if context.sentiment_client is None:
        return {"available": False, "error": "뉴스감성 클라이언트가 설정되지 않았습니다"}
    source_label = _SENTIMENT_SOURCE_LABELS.get(market)
    if source_label is None:
        return {"available": False, "error": f"지원하지 않는 시장입니다: {market}"}

    try:
        name = resolve_corp_name(ticker) if market == "KRX" else resolve_company_title(ticker)
    except Exception as exc:
        name_source_label = _NEWS_SOURCE_LABELS.get(market, source_label)
        return {"available": False, "error": f"{name_source_label} 조회 실패: {exc}"}
    if name is None:
        return {"available": True, "analysis": None}

    try:
        end = date.today()
        start = end - timedelta(days=NEWS_LOOKBACK_DAYS)
        if market == "KRX":
            articles = search_news(name, ticker, market, start, end)
        else:
            safe_name = name.replace('"', "")
            articles = search_articles(f'"{safe_name}"', ticker, market, start, end)
    except Exception as exc:
        return {"available": False, "error": f"{source_label} 조회 실패: {exc}"}

    if not context.budget.try_consume_llm_call():
        return {"available": False, "error": "이번 대화 턴의 LLM 호출 예산을 모두 사용했습니다"}
    try:
        analysis = analyze_sentiment(context.sentiment_client, articles, ticker, market, end)
    except Exception as exc:
        return {"available": False, "error": f"LLM 뉴스감성 해석 실패: {exc}"}
    if analysis is None:
        return {"available": True, "analysis": None}
    return {
        "available": True,
        "analysis": {
            "sentiment": analysis.sentiment,
            "confidence": analysis.confidence,
            "key_headline": analysis.key_headline,
            "rationale": analysis.rationale,
            "caveat": analysis.caveat,
        },
    }


def tool_analyze_position_sizing(context: ChatToolContext, ticker: str, market: str) -> dict:
    """`risk_sizing.suggest_position`은 절대 raise하지 않으므로(모든 실패 경로가
    `limit_check="NOT_APPLICABLE"`로 귀결) OHLCV 조회 실패만 격리하면 된다. LLM을
    쓰지 않으므로 `QueryBudget`을 소비하지 않는다. PRD §6 한도는 참고용 `limit_check`
    로만 반영되고 집행/차단은 하지 않는다 — 답변 문구에 "참고용, 투자 조언 아님"을
    붙이도록 지시하는 건 `prompt.py`의 시스템 프롬프트 몫이다."""
    try:
        records = _fetch_records(context, ticker, market)
        candidates = generate_candidates(records)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if not candidates:
        return {"available": True, "suggestion": None}

    suggestion = suggest_position(candidates[0], records, context.account)
    return {
        "available": True,
        "suggestion": {
            "action": suggestion.action,
            "suggested_quantity": suggestion.suggested_quantity,
            "suggested_allocation_pct": suggestion.suggested_allocation_pct,
            "stop_loss_price": suggestion.stop_loss_price,
            "take_profit_price": suggestion.take_profit_price,
            "reference_price": suggestion.reference_price,
            "limit_check": suggestion.limit_check,
            "notes": suggestion.notes,
        },
    }


def tool_get_price_data(context: ChatToolContext, ticker: str, market: str) -> dict:
    """실사용 중 발견된 버그(2026-09-26) — 규칙엔진 후보가 없는 종목(예: 당시 활성 매매
    신호가 없던 SK하이닉스)을 물으면 analyze_rule_engine/analyze_position_sizing 둘 다
    candidate/suggestion이 null이 되어 가격 자체를 아예 반환하지 못했다(OHLCV 조회 자체는
    정상이었는데도). 이 도구는 규칙엔진 신호 유무와 무관하게 캐시된 최신 OHLCV 한 건을
    그대로 반환한다 — chat-agent-plan.md Stage E에서 미리 남겨뒀던 `get_price_data` TBD
    항목."""
    try:
        records = _fetch_records(context, ticker, market)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if not records:
        return {"available": True, "latest": None}
    latest = records[-1]
    return {
        "available": True,
        "latest": {
            "date": latest.date.isoformat(),
            "open": latest.open,
            "high": latest.high,
            "low": latest.low,
            "close": latest.close,
            "volume": latest.volume,
        },
    }


def tool_resolve_ticker(context: ChatToolContext, query: str) -> dict:
    """모델이 대화 중 스스로 호출해 종목명/코드를 (ticker, market)으로 확정하는 9번째
    도구. LLM에게 티커/시장을 직접 판단하게 하지 않고(환각 방지) 반드시 이 도구를
    거치게 한다 — 모호(2건 이상)/미발견(0건) 처리는 모델이 시스템 프롬프트 지침에
    따라 답변 문구를 스스로 생성한다(사용자 확정 결정, 캔드 메시지 함수 대신 프롬프트
    가이드). LLM을 쓰지 않으므로 QueryBudget을 소비하지 않는다(analyze_rule_engine/
    ml_prediction과 동일)."""
    try:
        resolution = resolve_ticker(query)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "matches": [
            {"ticker": m.ticker, "market": m.market, "name": m.name} for m in resolution.matches
        ],
        "is_resolved": resolution.is_resolved,
        "is_ambiguous": resolution.is_ambiguous,
        "is_not_found": resolution.is_not_found,
    }


def tool_find_related_companies(context: ChatToolContext, ticker: str, market: str) -> dict:
    """관련기업 딥다이브의 진입점. deepagents 서브에이전트(`related_companies.py`)는
    회사"명"으로 리서치하므로, 먼저 티커를 이름으로 해석한 뒤 위임한다 — 이름 해석
    실패와 서브에이전트 실행 실패를 별도 단계로 분리해 다른 5개 도구와 동일하게
    오귀속을 피한다(예: DART 이름해석 실패를 "서브에이전트 실패"로 잘못 라벨링하지
    않음)."""
    try:
        name = resolve_corp_name(ticker) if market == "KRX" else resolve_company_title(ticker)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    if name is None:
        return {"available": False, "error": "종목명을 확인할 수 없어 관련기업을 조사할 수 없습니다"}

    return run_find_related_companies(context.budget, context.agent_model, ticker, market, name)


_STOCK_ANALYST_TOOL_SPECS = [
    ("analyze_rule_engine", "규칙 기반 기술적 매매 후보(RSI/SMA 크로스 등)를 조회한다."),
    ("analyze_ml_prediction", "ML 분류 모델의 상승확률 예측을 조회한다."),
    ("analyze_chart_pattern", "LLM 차트 패턴 해석(방향성/신뢰도/근거)을 조회한다."),
    ("analyze_disclosures", "최근 공식 공시(DART/EDGAR)에 대한 LLM 해석을 조회한다."),
    ("analyze_news_sentiment", "최근 뉴스 논조에 대한 LLM 해석을 조회한다."),
    ("analyze_position_sizing", "ATR 기반 손절/익절가·참고용 매수 수량을 조회한다."),
]


def _bind_stock_analyst_tools(context: ChatToolContext, ticker: str, market: str) -> list:
    """`stock_analyst.py`는 순환 임포트를 피하려고 `tools.py`를 import하지 않는다
    (stock_analyst.py 모듈 docstring 참고) — 대신 이 함수가 여기 정의된 6개
    `tool_analyze_*`를 종목/시장에 클로저로 고정한, 인자 없는 langchain `@tool`로
    감싸 `run_stock_analyst`에 전달한다. 서브에이전트 인스턴스 1개는 종목 1개만
    다루므로 인자가 필요 없다 — 모델이 판단할 것은 "어떤 도구를 부를지"뿐이다."""
    dispatch = {
        "analyze_rule_engine": tool_analyze_rule_engine,
        "analyze_ml_prediction": tool_analyze_ml_prediction,
        "analyze_chart_pattern": tool_analyze_chart_pattern,
        "analyze_disclosures": tool_analyze_disclosures,
        "analyze_news_sentiment": tool_analyze_news_sentiment,
        "analyze_position_sizing": tool_analyze_position_sizing,
    }
    def _make_bound_tool(fn):
        def _bound_tool() -> dict:
            return fn(context, ticker, market)

        return _bound_tool

    bound = []
    for name, description in _STOCK_ANALYST_TOOL_SPECS:
        bound.append(langchain_tool(name, description=description)(_make_bound_tool(dispatch[name])))
    return bound


def tool_stock_analyst(context: ChatToolContext, ticker: str, market: str) -> dict:
    bound_tools = _bind_stock_analyst_tools(context, ticker, market)
    return run_stock_analyst(context.budget, context.agent_model, bound_tools, ticker, market)


def tool_get_market_scan_recommendations(context: ChatToolContext, market: str) -> dict:
    """전체 스캔형 질의("추천해줄만한 주식 있어?") 전용 10번째 도구 — 종목을 지정하지
    않으므로 ticker는 없지만, `orchestrator.recommendation_coordinator.RecommendationCoordinator`가
    온디맨드로 적재하는 market별 추천 캐시(`data/scan_cache.py`의 `recommendation_cache`)를
    읽으므로 market은 받는다. 규칙엔진/ML(①숏리스트) + 차트/공시/뉴스감성(②심층분석,
    `stock_analyst` 재사용) + LLM 종합(③, `recommendation_synthesis`)까지 전부 거친
    순위(rank)·근거(summary) 포함 결과다(recommendation-synthesis-plan.md) — 예전엔
    규칙엔진 신호만 쓰는 무료 경로(`scan_market`/`ScanCoordinator`)였으나 2026-10-05
    교체됨. `scan_market` 자체는 폐기되지 않았고 `scripts/run_market_scan.py`의
    수동/오프라인 경로로 남아 있다. LLM 미사용, QueryBudget 소비 없음(이 도구 자신은
    — 백그라운드로 트리거되는 파이프라인은 별도의 배치 전용 QueryBudget을 쓴다).
    `ensure_fresh`를 먼저 호출해 캐시가 stale하면 백그라운드 파이프라인을 자가치유
    트리거하고, `is_in_progress`로 그 트리거가 지금 이 호출부터 진행 중인지를
    `refreshing`에 반영한다. 파이프라인이 한 번도 실행된 적 없어도 에러가 아니라
    정상 상태다(recommendations가 빈 배열, scanned_at이 None) — 모델이 그 상태를
    그대로 안내하면 된다."""
    context.recommendation_coordinator.ensure_fresh(market)
    scanned_at, recommendations = context.scan_cache.get_latest_recommendations(market)
    return {
        "available": True,
        "scanned_at": scanned_at.isoformat() if scanned_at is not None else None,
        "refreshing": context.recommendation_coordinator.is_in_progress(market),
        "recommendations": recommendations,
    }


TOOL_DISPATCH = {
    "resolve_ticker": tool_resolve_ticker,
    "analyze_rule_engine": tool_analyze_rule_engine,
    "analyze_ml_prediction": tool_analyze_ml_prediction,
    "analyze_chart_pattern": tool_analyze_chart_pattern,
    "analyze_disclosures": tool_analyze_disclosures,
    "analyze_news_sentiment": tool_analyze_news_sentiment,
    "analyze_position_sizing": tool_analyze_position_sizing,
    "get_price_data": tool_get_price_data,
    "find_related_companies": tool_find_related_companies,
    "stock_analyst": tool_stock_analyst,
    "get_market_scan_recommendations": tool_get_market_scan_recommendations,
}
