"""추론 경로: 뉴스 기사 목록 -> LLM 판독 -> 규칙엔진 후보에 대한 보조 문구.

`analyze`는 `action`을 받지 않고 독립적으로 감성을 판단하며(동조 방어), `ticker`/`market`는
호출자가 넘긴 값을 그대로 라벨링에만 쓰고 프롬프트에는 넣지 않는다(환각 방어, prompt.py
참고). 기사 목록이 비어 있으면 `None`을 반환하고 `client.read_sentiment`를 호출하지
않는다 — API 호출 자체가 발생하지 않는 1층 방어(news_disclosure/analyst.py와 동일 구조).
`to_reasons`가 코드에서 동의/상충/중립을 비교한다.
"""

from datetime import date

from auto_stock.data.models import NewsArticle
from auto_stock.news_sentiment.credentials import MAX_ARTICLES_PER_QUERY
from auto_stock.news_sentiment.models import NewsSentimentAnalysis, NewsSentimentReader, NewsSummary
from auto_stock.news_sentiment.prompt import SYSTEM_PROMPT, build_user_prompt

NEWS_SENTIMENT_DISCLAIMER = "(뉴스 감성분석은 백테스트 미검증 정성 신호입니다)"
CONFIDENCE_LABELS = {"LOW": "낮음", "MEDIUM": "보통", "HIGH": "높음"}
SENTIMENT_LABELS = {"POSITIVE": "긍정적", "NEGATIVE": "부정적", "NEUTRAL": "중립"}


def _build_summary(
    articles: list[NewsArticle], ticker: str, market: str, as_of: date
) -> NewsSummary:
    recent = sorted(articles, key=lambda item: item.published_at, reverse=True)
    return NewsSummary(
        ticker=ticker, market=market, as_of=as_of, items=recent[:MAX_ARTICLES_PER_QUERY]
    )


def analyze(
    client: NewsSentimentReader,
    articles: list[NewsArticle],
    ticker: str,
    market: str,
    as_of: date,
) -> NewsSentimentAnalysis | None:
    if not articles:
        return None

    summary = _build_summary(articles, ticker, market, as_of)
    user_prompt = build_user_prompt(summary)
    read = client.read_sentiment(SYSTEM_PROMPT, user_prompt)

    return NewsSentimentAnalysis(
        ticker=summary.ticker,
        market=summary.market,
        as_of=summary.as_of,
        sentiment=read.sentiment,
        confidence=read.confidence,
        key_headline=read.key_headline,
        rationale=read.rationale,
        caveat=read.caveat,
        model=client.model,
    )


def _stance(analysis: NewsSentimentAnalysis, action: str) -> str:
    label = CONFIDENCE_LABELS.get(analysis.confidence, analysis.confidence)
    sentiment_label = SENTIMENT_LABELS.get(analysis.sentiment, analysis.sentiment)
    agrees = (action == "BUY" and analysis.sentiment == "POSITIVE") or (
        action == "SELL" and analysis.sentiment == "NEGATIVE"
    )
    conflicts = (action == "BUY" and analysis.sentiment == "NEGATIVE") or (
        action == "SELL" and analysis.sentiment == "POSITIVE"
    )

    if agrees:
        return f"뉴스감성: {analysis.key_headline} ({sentiment_label}) — {action} 신호에 동의 (신뢰도 {label})"
    if conflicts:
        return f"뉴스감성: {analysis.key_headline} ({sentiment_label}) — {action} 신호와 상충됩니다 — 주의 (신뢰도 {label})"
    return f"뉴스감성: {analysis.key_headline} (논조 중립, 신뢰도 {label})"


def to_reasons(analysis: NewsSentimentAnalysis | None, action: str) -> list[str]:
    if analysis is None:
        return []

    reasons = [_stance(analysis, action), analysis.rationale]
    if analysis.caveat:
        reasons.append(f"단서: {analysis.caveat}")
    reasons.append(NEWS_SENTIMENT_DISCLAIMER)
    return reasons
