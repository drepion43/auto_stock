"""End-to-end MVP-0 wiring: data -> rule engine -> risk sizing -> explanation -> notification.

Pure composition of already-built, already-tested modules (docs/design/orchestrator.md)
— no new calculation logic here. Ticker list and AccountState are supplied by the
caller (see design doc for why the orchestrator doesn't decide those itself).

`ml_model`/`llm_client`/`news_client`/`sentiment_client` are optional boost-only signal
sources (ML #2, LLM #3, 공시 #4, 뉴스감성 #4-뉴스 — docs/design/ml-predictor-plan.md 핵심
설계 결정 2, docs/design/llm-chart-analyst-plan.md "오케스트레이터 통합", docs/design/
news-disclosure-phase3-plan.md, docs/design/news-sentiment-plan.md): when all four are
omitted (default None), this function is byte-for-byte identical to the pre-extension
behavior — no extra_reasons kwarg is even passed to generate_explanation. The branch
condition below is deliberately a single `and`-chained check (not separate `if`s) so the
existing regression tests locking the 2-positional-argument call are not broken by adding
new signal sources.

All four failures never block notification delivery — they're isolated in
`_ml_reasons`/`_llm_reasons`/`_news_reasons`/`_sentiment_reasons` and only surface via
`PipelineResult.errors`, independently of each other (one failing must not suppress the
others' reasons). `_news_reasons`/`_sentiment_reasons` uniquely chain two external lookups
(공시/뉴스 소스 then OpenAI) and prefix their error message themselves before returning,
unlike `_ml_reasons`/`_llm_reasons` whose prefix is added at the call site. `extra_reasons`
merge order is ML → LLM차트 → 공시 → 뉴스감성(항상 마지막).
"""

from datetime import date, timedelta

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.models import OHLCVRecord
from auto_stock.data.service import get_ohlcv
from auto_stock.data.sources.dart_source import fetch_disclosures, resolve_corp_code, resolve_corp_name
from auto_stock.data.sources.edgar_source import fetch_filings, resolve_cik, resolve_company_title
from auto_stock.data.sources.gdelt_source import search_articles
from auto_stock.data.sources.naver_news_source import search_news
from auto_stock.explainer.generator import generate_explanation
from auto_stock.llm_chart_analyst.analyst import analyze as analyze_chart
from auto_stock.llm_chart_analyst.analyst import to_reasons as to_chart_reasons
from auto_stock.llm_chart_analyst.models import ChartPatternReader
from auto_stock.ml_predictor.models import ModelBundle
from auto_stock.ml_predictor.predictor import predict, to_reasons
from auto_stock.news_disclosure.analyst import analyze as analyze_disclosures
from auto_stock.news_disclosure.analyst import to_reasons as to_disclosure_reasons
from auto_stock.news_disclosure.credentials import DISCLOSURE_LOOKBACK_DAYS
from auto_stock.news_disclosure.models import DisclosureReader
from auto_stock.news_sentiment.analyst import analyze as analyze_sentiment
from auto_stock.news_sentiment.analyst import to_reasons as to_sentiment_reasons
from auto_stock.news_sentiment.credentials import NEWS_LOOKBACK_DAYS
from auto_stock.news_sentiment.models import NewsSentimentReader
from auto_stock.notifier.models import TelegramCredentials
from auto_stock.notifier.telegram_bot import send_notification
from auto_stock.orchestrator.models import PipelineResult
from auto_stock.risk_sizing.models import AccountState
from auto_stock.risk_sizing.sizing import suggest_position
from auto_stock.rule_engine.engine import generate_candidates

DEFAULT_LOOKBACK_DAYS = 120  # SMA60 warmup + ATR(14), padded for weekends/holidays


def _ml_reasons(
    ml_model: ModelBundle | None, records: list[OHLCVRecord], action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — ML 실패가 추천 발송을 막아서는 안 된다(설계 결정 5)."""
    if ml_model is None:
        return [], None
    try:
        prediction = predict(ml_model, records)
        return to_reasons(prediction, action), None
    except Exception as exc:  # ML failure is isolated the same way per-ticker errors are
        return [], str(exc)


def _llm_reasons(
    llm_client: ChartPatternReader | None, records: list[OHLCVRecord], action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — LLM 실패가 추천 발송을 막아서는 안 된다(설계 결정 6)."""
    if llm_client is None:
        return [], None
    try:
        analysis = analyze_chart(llm_client, records)
        return to_chart_reasons(analysis, action), None
    except Exception as exc:  # LLM failure is isolated the same way per-ticker errors are
        return [], str(exc)


_NEWS_SOURCE_LABELS = {"KRX": "DART", "NASDAQ": "EDGAR"}


def _news_reasons(
    news_client: DisclosureReader | None, ticker: str, market: str, action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — 공시 조회든 LLM 해석이든 실패가 추천 발송을 막아서는 안 된다.

    시장별로 다른 공시 소스를 쓴다(KRX -> DART, NASDAQ -> EDGAR) — 해석 계층(analyze_disclosures/
    to_disclosure_reasons)은 시장과 무관하게 완전히 동일하게 재사용된다(docs/design/
    news-disclosure-nasdaq-plan.md 핵심 설계 결정 2/4). 별도 헬퍼로 쪼개지 않고 이 함수 안에서
    분기하는 이유도 같은 문서에 있다 — 서로 다른 신호원이 아니라 같은 신호원의 시장별 소스
    차이일 뿐이기 때문(_ml_reasons/_llm_reasons/_news_reasons를 분리한 것과는 다른 판단).

    공시 조회와 LLM 해석은 서로 다른 실패 지점이라 별도 try/except로 감싸고, 에러 메시지에
    "DART 조회 실패"/"EDGAR 조회 실패"/"LLM 공시해석 실패" 접두사를 붙여 반환한다 — `_ml_reasons`/
    `_llm_reasons`가 호출부에서 신호원별 접두사를 붙이는 것과 달리, 이 함수는 두 단계가 있어
    스스로 접두사를 결정한다."""
    if news_client is None:
        return [], None

    source_label = _NEWS_SOURCE_LABELS.get(market)
    if source_label is None:
        return [], None  # 지원하지 않는 시장 — 조용히 스킵(향후 시장 확장 대비)

    try:
        if market == "KRX":
            corp_id = resolve_corp_code(ticker)
        else:  # NASDAQ
            corp_id = resolve_cik(ticker)
        if corp_id is None:
            return [], None  # 미등록 종목 — 에러 아님, 조용히 스킵

        end = date.today()
        start = end - timedelta(days=DISCLOSURE_LOOKBACK_DAYS)
        disclosures = (
            fetch_disclosures(corp_id, start, end)
            if market == "KRX"
            else fetch_filings(corp_id, start, end)
        )
    except Exception as exc:
        return [], f"{source_label} 조회 실패: {exc}"

    try:
        analysis = analyze_disclosures(news_client, disclosures, ticker, market, end)
        return to_disclosure_reasons(analysis, action), None
    except Exception as exc:
        return [], f"LLM 공시해석 실패: {exc}"


_SENTIMENT_SOURCE_LABELS = {"KRX": "네이버뉴스", "NASDAQ": "GDELT"}


def _sentiment_reasons(
    sentiment_client: NewsSentimentReader | None, ticker: str, market: str, action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — 이름 해석이든 뉴스 조회든 LLM 해석이든 실패가 추천 발송을
    막아서는 안 된다.

    `_news_reasons`(공시)와 동일한 market 분기 구조다(KRX -> 네이버뉴스, NASDAQ -> GDELT)
    — 다만 티커→회사명 해석부터 필요하다(뉴스 API는 티커가 아니라 회사명으로 검색해야
    한다, docs/design/news-sentiment-plan.md 핵심 설계 결정 2). GDELT 쪽은 정확 구문
    검색을 위해 회사명을 따옴표로 감싸 질의를 만든다(핵심 설계 결정 3) — naver_news_source.py/
    gdelt_source.py는 이 질의 구성 로직을 모르는 순수 데이터 접근 계층이라 여기서 만든다.

    이름 해석(DART/EDGAR 호출)과 뉴스 조회(네이버/GDELT 호출)는 서로 다른 외부 시스템을
    부르므로 별도 try/except로 감싼다 — 하나로 묶으면 DART/EDGAR 실패가 "네이버뉴스/GDELT
    조회 실패"로 잘못 라벨링된다(코드 리뷰 MEDIUM). `_news_reasons`가 이미 쓰는
    `_NEWS_SOURCE_LABELS`(DART/EDGAR)를 이름 해석 실패 라벨로 재사용한다."""
    if sentiment_client is None:
        return [], None

    source_label = _SENTIMENT_SOURCE_LABELS.get(market)
    if source_label is None:
        return [], None  # 지원하지 않는 시장 — 조용히 스킵(향후 시장 확장 대비)

    try:
        name = resolve_corp_name(ticker) if market == "KRX" else resolve_company_title(ticker)
    except Exception as exc:
        name_source_label = _NEWS_SOURCE_LABELS.get(market, source_label)
        return [], f"{name_source_label} 조회 실패: {exc}"
    if name is None:
        return [], None  # 회사명 해석 실패 — 에러 아님, 조용히 스킵

    try:
        end = date.today()
        start = end - timedelta(days=NEWS_LOOKBACK_DAYS)
        if market == "KRX":
            articles = search_news(name, ticker, market, start, end)
        else:
            # 회사명에 따옴표가 들어있으면 GDELT 정확구문 검색 문법이 깨진다 — 제거해서
            # 방어한다(코드 리뷰 LOW, 실사용에서 발생 가능성은 낮지만 비용이 거의 없음).
            safe_name = name.replace('"', "")
            articles = search_articles(f'"{safe_name}"', ticker, market, start, end)
    except Exception as exc:
        return [], f"{source_label} 조회 실패: {exc}"

    try:
        analysis = analyze_sentiment(sentiment_client, articles, ticker, market, end)
        return to_sentiment_reasons(analysis, action), None
    except Exception as exc:
        return [], f"LLM 뉴스감성 해석 실패: {exc}"


def run_recommendation_pipeline(
    cache: OHLCVCache,
    tickers: list[str],
    market: str,
    account: AccountState,
    credentials: TelegramCredentials,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ml_model: ModelBundle | None = None,
    llm_client: ChartPatternReader | None = None,
    news_client: DisclosureReader | None = None,
    sentiment_client: NewsSentimentReader | None = None,
) -> PipelineResult:
    end = date.today()
    start = end - timedelta(days=lookback_days)

    sent = []
    errors = []
    for ticker in tickers:
        try:
            records = get_ohlcv(cache, ticker, start, end, market)
            for candidate in generate_candidates(records):
                sizing = suggest_position(candidate, records, account)
                if (
                    ml_model is None
                    and llm_client is None
                    and news_client is None
                    and sentiment_client is None
                ):
                    # 회귀 잠금: 보조 신호가 하나도 없으면 2-위치인자 호출을 그대로 보존한다
                    # (tests/orchestrator/test_pipeline.py의 assert_called_once_with(ANY, ANY))
                    explanation = generate_explanation(candidate, sizing)
                else:
                    extra_reasons: list[str] = []
                    ml_r, ml_error = _ml_reasons(ml_model, records, candidate.action)
                    extra_reasons.extend(ml_r)
                    llm_r, llm_error = _llm_reasons(llm_client, records, candidate.action)
                    extra_reasons.extend(llm_r)
                    news_r, news_error = _news_reasons(
                        news_client, candidate.ticker, candidate.market, candidate.action
                    )
                    extra_reasons.extend(news_r)
                    sentiment_r, sentiment_error = _sentiment_reasons(
                        sentiment_client, candidate.ticker, candidate.market, candidate.action
                    )
                    extra_reasons.extend(sentiment_r)

                    explanation = generate_explanation(candidate, sizing, extra_reasons=extra_reasons)

                    if ml_error is not None:
                        errors.append((ticker, f"ML 예측 실패: {ml_error}"))
                    if llm_error is not None:
                        errors.append((ticker, f"LLM 차트분석 실패: {llm_error}"))
                    if news_error is not None:
                        # news_error는 "DART 조회 실패: "/"EDGAR 조회 실패: "/"LLM 공시해석 실패: "
                        # 접두사를 이미 포함
                        errors.append((ticker, news_error))
                    if sentiment_error is not None:
                        # sentiment_error는 "네이버뉴스 조회 실패: "/"GDELT 조회 실패: "/
                        # "LLM 뉴스감성 해석 실패: " 접두사를 이미 포함
                        errors.append((ticker, sentiment_error))
                send_notification(explanation, credentials)
                sent.append(explanation)
        except Exception as exc:  # per-ticker isolation is deliberate — see design doc
            errors.append((ticker, str(exc)))

    return PipelineResult(sent=sent, errors=errors)
