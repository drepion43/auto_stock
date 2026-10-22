"""학습 유니버스 선정 — 2026-10-22 진단(docs/design/ml-predictor-mamba-plan.md "2026-10-22 진단")에서
찾은 문제를 고친다.

이전: `universe_for("KRX")[:200]` — KOSPI 목록 뒤에 KOSDAQ을 붙인 리스트의 앞 200개라 **KOSDAQ이
통째로 빠졌고**, pykrx가 종목명 가나다순으로 주므로 KOSPI 200개도 우선주가 섞인 임의 집합이었다.

지금:
- 원자 시장(KOSPI/KOSDAQ/NASDAQ/NYSE)마다 따로 `size`개를 고른다.
- 국내: **학습 구간 시작일 기준** 시가총액 상위 보통주(우선주 제외, 그날 거래대금 0인 정지 종목 제외).
  시작일 시점으로 고르므로 이후 상장폐지된 종목도 후보에 들어가 생존 편향이 줄고, 이후 상장된
  종목은 들어오지 않아 미래 정보로 고르지 않는다. 시가총액 상위라 소형 스팩은 자연히 빠진다.
- 미국: 과거 시점 상장 목록을 주는 무료 소스가 없어 **현재 상장 목록**(FDR, 시가총액 순)을 그대로
  쓴다 — 미국 쪽 생존 편향은 남는다(한계로 기록)."""

from collections.abc import Callable
from datetime import date

from auto_stock.data.market_routing import KRX_MARKETS
from auto_stock.data.service import get_universe
from auto_stock.data.sources import pykrx_source

CapTable = Callable[[date, str], dict[str, dict]]

_COMMON_STOCK_SUFFIX = "0"  # KRX 단축코드: 보통주는 끝자리 0, 우선주는 5·7·9·K 등
_KRX_CODE_LENGTH = 6


def is_common_stock(ticker: str) -> bool:
    return len(ticker) == _KRX_CODE_LENGTH and ticker.endswith(_COMMON_STOCK_SUFFIX)


def select_krx_universe(
    market: str, as_of: date, size: int, cap_table: CapTable = pykrx_source.get_market_cap_table
) -> list[str]:
    table = cap_table(as_of, market)
    if not table:
        # 빈 표는 정상 결과일 수 없다(KRX 로그인 실패·서버 장애) — 조용히 빈 유니버스로 학습하지 않는다.
        raise RuntimeError(f"{market} market-cap table for {as_of} is empty — cannot select a universe")
    candidates = [
        (ticker, row["market_cap"])
        for ticker, row in table.items()
        if is_common_stock(ticker) and row["trading_value"] > 0
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [ticker for ticker, _cap in candidates[:size]]


def select_universe(market: str, as_of: date, size: int) -> list[str]:
    if market in KRX_MARKETS:
        return select_krx_universe(market, as_of, size)
    return get_universe(market)[:size]
