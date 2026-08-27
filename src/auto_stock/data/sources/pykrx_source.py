from datetime import date
from dotenv import load_dotenv

from pykrx import stock


def get_ticker_list(as_of: date, market: str = "ALL") -> list[str]:
    """전체 상장종목 코드 리스트 (KRX)."""
    load_dotenv()
    return stock.get_market_ticker_list(as_of.strftime("%Y%m%d"), market=market)


def get_market_cap(ticker: str, as_of: date) -> int | None:
    """특정 종목의 특정일자 시가총액. 데이터가 없으면 None."""
    load_dotenv()
    date_str = as_of.strftime("%Y%m%d")
    df = stock.get_market_cap(date_str, date_str, ticker)
    if df.empty:
        return None
    return int(df.iloc[-1]["시가총액"])
