"""naver_news_source.py 테스트 — requests를 모킹해 실제 네이버 API를 호출하지 않는다."""

from datetime import date

import pytest
import requests

from auto_stock.data.models import NewsArticle
from auto_stock.data.sources.naver_news_source import NaverNewsApiError, search_news


@pytest.fixture(autouse=True)
def _dummy_naver_credentials(monkeypatch, mocker):
    mocker.patch("auto_stock.data.sources.naver_news_source.load_dotenv")
    monkeypatch.setenv("NAVER_CLIENT_ID", "dummy-client-id")
    monkeypatch.setenv("NAVER_CLIENT_SECRET", "dummy-client-secret-not-real")


def _naver_payload(items: list[dict]) -> dict:
    return {"lastBuildDate": "Thu, 15 Jan 2026 09:30:00 +0900", "total": len(items), "items": items}


def _news_item(
    title: str = "삼성전자, 신규 반도체 라인 <b>투자</b> 발표",
    originallink: str = "https://www.example-press.co.kr/article/1",
    link: str = "https://n.news.naver.com/article/001/1",
    description: str = "삼성전자가 &quot;신규 투자&quot;를 발표했다.",
    pub_date: str = "Thu, 15 Jan 2026 09:30:00 +0900",
) -> dict:
    return {
        "title": title,
        "originallink": originallink,
        "link": link,
        "description": description,
        "pubDate": pub_date,
    }


class _FakeResponse:
    def __init__(self, json_data: dict | None = None):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_search_news_returns_articles_within_date_range(mocker):
    payload = _naver_payload([_news_item(pub_date="Thu, 15 Jan 2026 09:30:00 +0900")])
    mocker.patch(
        "auto_stock.data.sources.naver_news_source.requests.get", return_value=_FakeResponse(json_data=payload)
    )

    articles = search_news("삼성전자", "005930", "KRX", date(2026, 1, 1), date(2026, 1, 31))

    assert articles == [
        NewsArticle(
            ticker="005930",
            market="KRX",
            title="삼성전자, 신규 반도체 라인 투자 발표",
            url="https://n.news.naver.com/article/001/1",
            published_at=date(2026, 1, 15),
            source="www.example-press.co.kr",
        )
    ]


def test_search_news_strips_html_tags_and_unescapes_entities():
    from auto_stock.data.sources.naver_news_source import _clean_html

    assert _clean_html("삼성전자, 신규 반도체 라인 <b>투자</b> 발표") == "삼성전자, 신규 반도체 라인 투자 발표"
    assert _clean_html("삼성전자가 &quot;신규 투자&quot;를 발표했다.") == '삼성전자가 "신규 투자"를 발표했다.'


def test_clean_html_does_not_reconstitute_entity_escaped_tags():
    """코드 리뷰 MEDIUM: 태그 제거 -> 엔티티 언이스케이프 순서였을 때, &lt;script&gt;처럼
    엔티티로 이스케이프된 가짜 태그는 제거 단계를 통과한 뒤 언이스케이프되면서 진짜 태그로
    되살아났다. 언이스케이프를 먼저 하면 되살아난 태그도 제거 단계에서 함께 걸러진다."""
    from auto_stock.data.sources.naver_news_source import _clean_html

    result = _clean_html("삼성전자 &lt;script&gt;alert(1)&lt;/script&gt; 투자")

    assert "<script>" not in result
    assert "</script>" not in result


def test_search_news_excludes_articles_outside_date_range(mocker):
    payload = _naver_payload(
        [
            _news_item(pub_date="Thu, 15 Jan 2026 09:30:00 +0900"),
            _news_item(pub_date="Mon, 01 Dec 2025 09:30:00 +0900"),  # 조회 기간 밖
        ]
    )
    mocker.patch(
        "auto_stock.data.sources.naver_news_source.requests.get", return_value=_FakeResponse(json_data=payload)
    )

    articles = search_news("삼성전자", "005930", "KRX", date(2026, 1, 1), date(2026, 1, 31))

    assert len(articles) == 1


def test_search_news_returns_empty_list_when_no_items(mocker):
    payload = _naver_payload([])
    mocker.patch(
        "auto_stock.data.sources.naver_news_source.requests.get", return_value=_FakeResponse(json_data=payload)
    )

    articles = search_news("존재하지않는회사이름", "999999", "KRX", date(2026, 1, 1), date(2026, 1, 31))

    assert articles == []


def test_search_news_requires_naver_credentials(monkeypatch):
    monkeypatch.delenv("NAVER_CLIENT_ID", raising=False)

    with pytest.raises(KeyError, match="NAVER_CLIENT_ID"):
        search_news("삼성전자", "005930", "KRX", date(2026, 1, 1), date(2026, 1, 31))


def test_search_news_requires_naver_client_secret(monkeypatch):
    monkeypatch.delenv("NAVER_CLIENT_SECRET", raising=False)

    with pytest.raises(KeyError, match="NAVER_CLIENT_SECRET"):
        search_news("삼성전자", "005930", "KRX", date(2026, 1, 1), date(2026, 1, 31))


def test_search_news_wraps_connection_error_without_leaking_credentials(mocker):
    mocker.patch(
        "auto_stock.data.sources.naver_news_source.requests.get",
        side_effect=requests.exceptions.ConnectionError(
            "Max retries exceeded — headers={'X-Naver-Client-Secret': 'dummy-client-secret-not-real'}"
        ),
    )

    with pytest.raises(NaverNewsApiError) as exc_info:
        search_news("삼성전자", "005930", "KRX", date(2026, 1, 1), date(2026, 1, 31))

    assert "dummy-client-secret-not-real" not in str(exc_info.value)


def test_search_news_wraps_malformed_pub_date_as_naver_news_api_error(mocker):
    payload = _naver_payload([_news_item(pub_date="not-a-valid-date")])
    mocker.patch(
        "auto_stock.data.sources.naver_news_source.requests.get", return_value=_FakeResponse(json_data=payload)
    )

    with pytest.raises(NaverNewsApiError):
        search_news("삼성전자", "005930", "KRX", date(2026, 1, 1), date(2026, 1, 31))
