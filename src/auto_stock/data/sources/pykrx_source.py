import contextlib
import io
from datetime import date

from dotenv import load_dotenv

# pykrx는 import되는 순간(모듈 최상위 `website/comm/webio.py:12`의
# `_session = build_krx_session()`) KRX 로그인을 시도하고, KRX_ID/KRX_PW가 없으면
# "KRX 로그인 실패..." 노이즈를 stdout에 출력한다 — 이 프로젝트는 그 로그인이 필요한
# 프리미엄 엔드포인트를 쓰지 않고 공개 OHLCV/티커 목록만 쓰므로 실제 동작에는 영향이
# 없다(사용자가 실제 챗봇 실행 중 이 메시지를 보고 문의해 확인함, 2026-09-06). import
# 시점의 출력만 억제하고, 이후 실제 함수 호출 시의 출력(있다면)은 그대로 둔다.
with contextlib.redirect_stdout(io.StringIO()):
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
