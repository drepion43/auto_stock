"""gdelt_source.py 테스트 — requests를 모킹해 실제 GDELT API를 호출하지 않는다."""

from datetime import date

import pytest
import requests

from auto_stock.data.models import NewsArticle
from auto_stock.data.sources import gdelt_source
from auto_stock.data.sources.gdelt_source import (
    GdeltApiError,
    company_phrase,
    search_articles,
)


@pytest.fixture(autouse=True)
def _no_real_throttle_sleep(monkeypatch):
    """요청 간격 조절(5초 1회 제한 대응)이 테스트를 실제로 재우지 않도록 대기만 기록한다."""
    sleeps: list[float] = []
    monkeypatch.setattr(gdelt_source, "_sleep", sleeps.append)
    monkeypatch.setattr(gdelt_source, "_next_allowed_at", None)
    return sleeps


def _gdelt_payload(articles: list[dict]) -> dict:
    return {"articles": articles}


def _article(
    title: str = "Apple announces new chip investment",
    url: str = "https://example-press.com/article/1",
    seendate: str = "20260115T093000Z",
    domain: str = "example-press.com",
) -> dict:
    return {
        "url": url,
        "url_mobile": url,
        "title": title,
        "seendate": seendate,
        "socialimage": "https://example-press.com/img.jpg",
        "domain": domain,
        "language": "English",
        "sourcecountry": "United States",
    }


class _FakeResponse:
    def __init__(self, json_data: dict | None = None):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_search_articles_returns_articles_within_date_range(mocker):
    payload = _gdelt_payload([_article(seendate="20260115T093000Z")])
    mocker.patch("auto_stock.data.sources.gdelt_source.requests.get", return_value=_FakeResponse(json_data=payload))

    articles = search_articles('"Apple Inc"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))

    assert articles == [
        NewsArticle(
            ticker="AAPL",
            market="NASDAQ",
            title="Apple announces new chip investment",
            url="https://example-press.com/article/1",
            published_at=date(2026, 1, 15),
            source="example-press.com",
        )
    ]


def test_search_articles_excludes_articles_outside_date_range(mocker):
    payload = _gdelt_payload(
        [
            _article(seendate="20260115T093000Z"),
            _article(seendate="20251201T093000Z"),  # 조회 기간 밖
        ]
    )
    mocker.patch("auto_stock.data.sources.gdelt_source.requests.get", return_value=_FakeResponse(json_data=payload))

    articles = search_articles('"Apple Inc"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))

    assert len(articles) == 1


def test_search_articles_returns_empty_list_when_no_articles(mocker):
    payload = _gdelt_payload([])
    mocker.patch("auto_stock.data.sources.gdelt_source.requests.get", return_value=_FakeResponse(json_data=payload))

    articles = search_articles('"Nonexistent Corp"', "XYZ", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))

    assert articles == []


def test_search_articles_wraps_connection_error_as_gdelt_api_error(mocker):
    mocker.patch(
        "auto_stock.data.sources.gdelt_source.requests.get",
        side_effect=requests.exceptions.ConnectionError("Max retries exceeded"),
    )

    with pytest.raises(GdeltApiError) as exc_info:
        search_articles('"Apple Inc"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))

    assert "Max retries exceeded" not in str(exc_info.value)


def test_search_articles_wraps_malformed_seendate_as_gdelt_api_error(mocker):
    payload = _gdelt_payload([_article(seendate="not-a-valid-date")])
    mocker.patch("auto_stock.data.sources.gdelt_source.requests.get", return_value=_FakeResponse(json_data=payload))

    with pytest.raises(GdeltApiError):
        search_articles('"Apple Inc"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))


def test_search_articles_wraps_none_seendate_as_gdelt_api_error(mocker):
    """코드 리뷰 MEDIUM: seendate가 null(None)이면 datetime.strptime이 ValueError가 아니라
    TypeError를 던지는데, 기존 except (KeyError, ValueError)는 이를 잡지 못해 정제되지 않은
    예외가 그대로 새어나갔다."""
    payload = _gdelt_payload([_article(seendate=None)])
    mocker.patch("auto_stock.data.sources.gdelt_source.requests.get", return_value=_FakeResponse(json_data=payload))

    with pytest.raises(GdeltApiError):
        search_articles('"Apple Inc"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))


def test_search_articles_requires_no_credentials(mocker, monkeypatch):
    """GDELT는 API 키가 없다 — 크리덴셜 환경변수 부재로 실패하지 않아야 한다."""
    monkeypatch.delenv("SOME_UNRELATED_VAR", raising=False)
    payload = _gdelt_payload([])
    mocker.patch("auto_stock.data.sources.gdelt_source.requests.get", return_value=_FakeResponse(json_data=payload))

    articles = search_articles('"Apple Inc"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))

    assert articles == []


# --- GDELT 요청 제한(HTTP 429 "Please limit requests to one every 5 seconds", 2026-10-12 실측) ---


class _RateLimitedResponse:
    status_code = 429

    def raise_for_status(self):
        raise requests.exceptions.HTTPError("429 Too Many Requests", response=self)

    def json(self):
        return {}


def test_consecutive_requests_are_spaced_by_the_minimum_interval(mocker, _no_real_throttle_sleep):
    mocker.patch("auto_stock.data.sources.gdelt_source.requests.get", return_value=_FakeResponse(json_data=_gdelt_payload([])))
    mocker.patch.object(gdelt_source, "_now", side_effect=[100.0, 101.0])

    search_articles('"Apple"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))
    search_articles('"Nvidia"', "NVDA", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))

    assert _no_real_throttle_sleep == [pytest.approx(gdelt_source._MIN_INTERVAL_SECONDS - 1.0)]


def test_rate_limited_request_is_retried_once_after_waiting(mocker, _no_real_throttle_sleep):
    payload = _gdelt_payload([_article(seendate="20260115T093000Z")])
    mock_get = mocker.patch(
        "auto_stock.data.sources.gdelt_source.requests.get",
        side_effect=[_RateLimitedResponse(), _FakeResponse(json_data=payload)],
    )

    articles = search_articles('"Apple"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))

    assert len(articles) == 1
    assert mock_get.call_count == 2
    assert _no_real_throttle_sleep  # 재시도 전에 간격만큼 기다렸다


def test_rate_limited_twice_raises_gdelt_api_error(mocker):
    mocker.patch(
        "auto_stock.data.sources.gdelt_source.requests.get",
        side_effect=[_RateLimitedResponse(), _RateLimitedResponse()],
    )

    with pytest.raises(GdeltApiError):
        search_articles('"Apple"', "AAPL", "NASDAQ", date(2026, 1, 1), date(2026, 1, 31))



# --- 검색어: SEC 법인명 → 뉴스 표기(2026-10-12 실측: "NVIDIA CORP" 27건 vs "NVIDIA" 50건 이상) ---


@pytest.mark.parametrize(
    ("legal_name", "phrase"),
    [
        ("NVIDIA CORP", '"NVIDIA"'),
        ("Apple Inc.", '"Apple"'),
        ("GOLDMAN SACHS GROUP INC", '"GOLDMAN SACHS"'),
        ("AMAZON COM INC", '"AMAZON"'),
        ("META PLATFORMS, INC.", '"META PLATFORMS"'),
        ("WALT DISNEY CO", '"WALT DISNEY"'),
        ("BERKSHIRE HATHAWAY INC /DE/", '"BERKSHIRE HATHAWAY"'),
        ("Taiwan Semiconductor Manufacturing Co Ltd", '"Taiwan Semiconductor Manufacturing"'),
        ("TSMC", '"TSMC"'),
    ],
)
def test_company_phrase_strips_legal_suffixes(legal_name, phrase):
    assert company_phrase(legal_name) == phrase


def test_company_phrase_removes_embedded_quotes():
    """회사명 안의 따옴표는 GDELT 정확구문 문법을 깨뜨린다(기존 방어 유지)."""
    assert company_phrase('Apple "Inc."') == '"Apple"'


def test_company_phrase_keeps_name_when_everything_would_be_stripped():
    assert company_phrase("Holdings Inc") == '"Holdings Inc"'
