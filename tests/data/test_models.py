from datetime import date

import pytest

from auto_stock.data.models import DisclosureRecord, NewsArticle, OHLCVRecord


def test_creates_valid_krx_record():
    # Arrange / Act
    record = OHLCVRecord(
        ticker="005930",
        market="KRX",
        date=date(2026, 1, 2),
        open=70000.0,
        high=71000.0,
        low=69500.0,
        close=70500.0,
        volume=12_345_678,
    )

    # Assert
    assert record.ticker == "005930"
    assert record.market == "KRX"
    assert record.close == 70500.0


def test_creates_valid_nasdaq_record():
    record = OHLCVRecord(
        ticker="AAPL",
        market="NASDAQ",
        date=date(2026, 1, 2),
        open=190.0,
        high=192.5,
        low=189.0,
        close=191.2,
        volume=54_000_000,
    )

    assert record.market == "NASDAQ"


def test_rejects_unknown_market():
    with pytest.raises(ValueError, match="market"):
        OHLCVRecord(
            ticker="005930",
            market="NYSE",
            date=date(2026, 1, 2),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=1,
        )


def test_rejects_high_below_low():
    with pytest.raises(ValueError, match="high"):
        OHLCVRecord(
            ticker="005930",
            market="KRX",
            date=date(2026, 1, 2),
            open=100.0,
            high=90.0,
            low=95.0,
            close=92.0,
            volume=1,
        )


def test_creates_disclosure_record():
    record = DisclosureRecord(
        corp_code="00126380",
        ticker="005930",
        report_name="주요사항보고서(유상증자결정)",
        filed_date=date(2026, 1, 15),
        remark="",
    )

    assert record.corp_code == "00126380"
    assert record.ticker == "005930"
    assert record.report_name == "주요사항보고서(유상증자결정)"
    assert record.filed_date == date(2026, 1, 15)
    assert record.remark == ""


def test_creates_news_article():
    article = NewsArticle(
        ticker="005930",
        market="KRX",
        title="삼성전자, 신규 반도체 라인 투자 발표",
        url="https://example.com/news/1",
        published_at=date(2026, 1, 15),
        source="example.com",
    )

    assert article.ticker == "005930"
    assert article.market == "KRX"
    assert article.title == "삼성전자, 신규 반도체 라인 투자 발표"
    assert article.url == "https://example.com/news/1"
    assert article.published_at == date(2026, 1, 15)
    assert article.source == "example.com"
