"""Manual verification: confirms pykrx can log in with .env credentials and fetch real data.

Run from the project root so `.env` is found:
    .venv/Scripts/python scripts/verify_pykrx.py   (Windows)
    .venv/bin/python scripts/verify_pykrx.py        (macOS/Linux)

Checks two things pykrx_source.py wraps: the ticker universe listing
(get_ticker_list) and a single ticker's market cap (get_market_cap). Both
require a successful KRX login (KRX_ID/KRX_PW in .env) — see
docs/design/ml-predictor.md "모델 아티팩트" section for the load_dotenv() fix
this script is meant to confirm.

Prints only counts/values — never prints KRX_ID or KRX_PW. (pykrx itself
prints the login ID, not the password, to stdout during login — that's
library behavior, not this script's.)
"""

from datetime import date

from auto_stock.data.sources.pykrx_source import get_market_cap, get_ticker_list

SAMPLE_TICKER = "005930"  # 삼성전자 — always listed, good smoke-test target
today = date.today()

try:
    tickers = get_ticker_list(today, market="ALL")
    if not tickers:
        print("FAILED: get_ticker_list이 빈 리스트를 반환했습니다 (로그인 실패 또는 휴장일일 수 있음)")
    else:
        print(f"SUCCESS: 종목 유니버스 {len(tickers)}개 수신 (예: {tickers[:3]})")

    market_cap = get_market_cap(SAMPLE_TICKER, today)
    if market_cap is None:
        print(f"FAILED: {SAMPLE_TICKER} 시가총액 조회 결과 없음 (휴장일일 수 있음)")
    else:
        print(market_cap)
        print(f"SUCCESS: {SAMPLE_TICKER} 시가총액 = {market_cap:,}원")
except Exception as e:
    print(f"FAILED: {e}")
