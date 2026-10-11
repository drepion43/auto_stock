"""index_constituents.py — S&P500/다우존스 구성종목 조회. fdr.StockListing/requests를
모킹해 실제 Wikipedia/FDR 호출 없이 테스트한다."""

import pandas as pd
import pytest

from auto_stock.data.sources import index_constituents
from auto_stock.data.sources.index_constituents import (
    IndexConstituentsError,
    fetch_dow_constituents,
    fetch_sp500_constituents,
)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


_DOW_TABLE_HTML = """
<table>
<tr><th>Company</th><th>Symbol</th></tr>
<tr><td>Apple</td><td>AAPL</td></tr>
<tr><td>Boeing</td><td>BA</td></tr>
</table>
""" + "\n".join(f"<tr><td>Company{i}</td><td>SYM{i}</td></tr>" for i in range(28))
# 30개(헤더 2 + 28 생성)로 20~40 범위 내에 들어오도록 패딩.


def test_fetch_sp500_constituents_downloads_and_caches(mocker, tmp_path):
    cache_path = tmp_path / "sp500.json"
    mocker.patch("auto_stock.data.sources.index_constituents.SP500_CACHE_PATH", cache_path)
    listing_df = pd.DataFrame({"Symbol": ["AAPL", "MSFT"]})
    mock_listing = mocker.patch(
        "auto_stock.data.sources.index_constituents.fdr.StockListing", return_value=listing_df
    )

    result = fetch_sp500_constituents()

    assert result == {"AAPL", "MSFT"}
    mock_listing.assert_called_once_with("S&P500")
    assert cache_path.exists()


def test_fetch_sp500_constituents_uses_cache_without_refetching(mocker, tmp_path):
    import json

    cache_path = tmp_path / "sp500.json"
    cache_path.write_text(json.dumps(["AAPL", "MSFT"]), encoding="utf-8")
    mocker.patch("auto_stock.data.sources.index_constituents.SP500_CACHE_PATH", cache_path)
    mock_listing = mocker.patch("auto_stock.data.sources.index_constituents.fdr.StockListing")

    result = fetch_sp500_constituents()

    assert result == {"AAPL", "MSFT"}
    mock_listing.assert_not_called()


def test_fetch_dow_constituents_downloads_and_caches(mocker, tmp_path):
    cache_path = tmp_path / "dow.json"
    mocker.patch("auto_stock.data.sources.index_constituents.DOW_CACHE_PATH", cache_path)
    mocker.patch(
        "auto_stock.data.sources.index_constituents.requests.get",
        return_value=_FakeResponse(_DOW_TABLE_HTML),
    )

    result = fetch_dow_constituents()

    assert "AAPL" in result
    assert "BA" in result
    assert cache_path.exists()


def test_fetch_dow_constituents_wraps_request_failure(mocker, tmp_path):
    import requests

    cache_path = tmp_path / "dow.json"
    mocker.patch("auto_stock.data.sources.index_constituents.DOW_CACHE_PATH", cache_path)
    mocker.patch(
        "auto_stock.data.sources.index_constituents.requests.get",
        side_effect=requests.exceptions.ConnectionError("boom"),
    )

    with pytest.raises(IndexConstituentsError):
        fetch_dow_constituents()


def test_fetch_dow_constituents_raises_when_no_matching_table_found(mocker, tmp_path):
    cache_path = tmp_path / "dow.json"
    mocker.patch("auto_stock.data.sources.index_constituents.DOW_CACHE_PATH", cache_path)
    mocker.patch(
        "auto_stock.data.sources.index_constituents.requests.get",
        return_value=_FakeResponse("<table><tr><th>Nothing</th></tr><tr><td>x</td></tr></table>"),
    )

    with pytest.raises(IndexConstituentsError):
        fetch_dow_constituents()
