from datetime import date

import FinanceDataReader as fdr

from auto_stock.data.models import OHLCVRecord


def fetch_ohlcv(ticker: str, start: date, end: date, market: str) -> list[OHLCVRecord]:
    """Fetch OHLCV history for a KRX or NASDAQ ticker via FinanceDataReader."""
    df = fdr.DataReader(ticker, start, end)

    records: list[OHLCVRecord] = []
    for index, row in df.iterrows():
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
