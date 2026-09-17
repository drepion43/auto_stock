"""news_sentiment 테스트 전용 fixture.

`OPENAI_API_KEY`를 더미 값으로 monkeypatch하는 autouse 픽스처는 실수로도 실제 API 호출이
발생하지 않도록 하는 안전장치다(모든 테스트가 이 conftest를 통해 이 픽스처를 자동 적용받는다).
"""

from datetime import date, timedelta

import pytest

from auto_stock.data.models import NewsArticle
from auto_stock.news_sentiment.models import NewsSentimentAnalysis
from auto_stock.news_sentiment.schema import NewsSentimentRead


@pytest.fixture(autouse=True)
def _dummy_openai_api_key(monkeypatch, mocker):
    # load_dotenv 자체도 모킹한다 — 그렇지 않으면 로컬 .env에 OPENAI_API_KEY=(빈 값이라도)가
    # 있을 때 monkeypatch.delenv 직후 load_dotenv()가 그 빈 값을 다시 주입해 "키 없으면
    # KeyError" 테스트가 거짓으로 실패한다(코드 리뷰 MEDIUM).
    mocker.patch("auto_stock.news_sentiment.credentials.load_dotenv")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-not-real")


def make_articles(
    n: int,
    ticker: str = "005930",
    market: str = "KRX",
    start: date = date(2024, 6, 1),
    title: str = "삼성전자 관련 기사",
    source: str = "example.com",
) -> list[NewsArticle]:
    return [
        NewsArticle(
            ticker=ticker,
            market=market,
            title=f"{title} {i}",
            url=f"https://{source}/article/{i}",
            published_at=start + timedelta(days=i),
            source=source,
        )
        for i in range(n)
    ]


class FakeNewsSentimentReader:
    """`NewsSentimentReader` Protocol 구현체 — 진짜 SDK 없이 analyst.py를 테스트하기 위함."""

    def __init__(
        self,
        response: NewsSentimentRead | None = None,
        error: Exception | None = None,
        model: str = "gpt-5.6-luna",
    ) -> None:
        self.model = model
        self._response = response or NewsSentimentRead(
            sentiment="POSITIVE",
            confidence="MEDIUM",
            key_headline="신규 투자 발표",
            rationale="시장이 긍정적으로 반응할 것으로 예상되는 투자 소식입니다.",
            caveat=None,
        )
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def read_sentiment(self, system_prompt: str, user_prompt: str) -> NewsSentimentRead:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture
def fake_reader() -> FakeNewsSentimentReader:
    return FakeNewsSentimentReader()


def make_news_sentiment_analysis(**overrides) -> NewsSentimentAnalysis:
    values = dict(
        ticker="005930",
        market="KRX",
        as_of=date(2024, 6, 10),
        sentiment="POSITIVE",
        confidence="MEDIUM",
        key_headline="신규 투자 발표",
        rationale="시장이 긍정적으로 반응할 것으로 예상되는 투자 소식입니다.",
        caveat=None,
        model="gpt-5.6-luna",
    )
    values.update(overrides)
    return NewsSentimentAnalysis(**values)
