from datetime import date

import FinanceDataReader as fdr
import pandas as pd

from auto_stock.data.models import OHLCVRecord

_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(ticker: str, start: date, end: date, market: str) -> list[OHLCVRecord]:
    """Fetch OHLCV history for a KRX or NASDAQ ticker via FinanceDataReader.

    Rows with a missing price/volume are skipped: Yahoo returns the still-open session's bar with
    Close=NaN (seen 2026-10-23 for every US ticker), which would otherwise be cached and poison
    labels and inference features."""
    df = fdr.DataReader(ticker, start, end)

    records: list[OHLCVRecord] = []
    for index, row in df.iterrows():
        if pd.isna(row[_PRICE_COLUMNS]).any():
            continue
        records.append(
            OHLCVRecord(
                ticker=ticker,
                market=market,
                date=index.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
        )
    return records
