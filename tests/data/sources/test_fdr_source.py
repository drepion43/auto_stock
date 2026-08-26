from datetime import date

import pandas as pd

from auto_stock.data.sources import fdr_source


def _fake_fdr_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [70000.0, 70500.0],
            "High": [71000.0, 71200.0],
            "Low": [69500.0, 70100.0],
            "Close": [70500.0, 70900.0],
            "Volume": [12_345_678, 9_876_543],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )


def test_fetch_ohlcv_maps_fdr_dataframe_to_records(mocker):
    # Arrange
    mocker.patch.object(fdr_source.fdr, "DataReader", return_value=_fake_fdr_dataframe())

    # Act
    records = fdr_source.fetch_ohlcv("005930", date(2026, 1, 1), date(2026, 1, 6), market="KRX")

    # Assert
    assert len(records) == 2
    first = records[0]
    assert first.ticker == "005930"
    assert first.market == "KRX"
    assert first.date == date(2026, 1, 2)
    assert first.open == 70000.0
    assert first.volume == 12_345_678


def test_fetch_ohlcv_supports_nasdaq_market(mocker):
    mocker.patch.object(fdr_source.fdr, "DataReader", return_value=_fake_fdr_dataframe())

    records = fdr_source.fetch_ohlcv("AAPL", date(2026, 1, 1), date(2026, 1, 6), market="NASDAQ")

    assert all(r.market == "NASDAQ" for r in records)


def test_fetch_ohlcv_returns_empty_list_for_empty_response(mocker):
    mocker.patch.object(fdr_source.fdr, "DataReader", return_value=pd.DataFrame())

    records = fdr_source.fetch_ohlcv("000000", date(2026, 1, 1), date(2026, 1, 6), market="KRX")

    assert records == []
