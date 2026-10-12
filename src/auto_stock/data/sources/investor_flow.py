"""pykrx 기반 수급 데이터(외국인/기관 순매수, 공매도 비중) — 데이터 수집 계층(#0)
확장. `pykrx_source.py`와 같은 위치: 함수 호출 시점에 `load_dotenv()`를 호출해
KRX_ID/KRX_PW 로그인을 재시도한다(모듈 import 시점에는 아직 `.env`가 로드되기
전이라 pykrx 내부 최초 로그인이 늘 실패하기 때문 — `pykrx_source.py` docstring과
동일 이유).

**국내(KOSPI/KOSDAQ) 전용** — 미국은 기관 보유현황이 13F(분기 공시)뿐이라 실시간성과
파싱 복잡도가 완전히 다르다(SEC EDGAR 공시 목록과는 별개 포맷). `market`이 그 외
값이면 조용히 빈 결과/`None`을 반환한다(`sector_classification.py`의 "pykrx 미지원
시장은 에러 아님" 관례와 동일).

**캐싱하지 않는다** — DART/EDGAR corp_code 캐시나 `pykrx_source.resolve_market`
캐시는 "거의 안 바뀌는 참조 매핑"이라 7일 TTL이 맞지만, 외국인/기관 순매수와 공매도
잔고는 매일(심지어 일중) 바뀌는 시계열 데이터다 — 그 패턴을 그대로 가져오면 "오늘
수급"을 묻는 질의에 최대 6일 묵은 값을 돌려주는 버그가 된다. 매 호출마다 pykrx를
직접 조회한다.

2026-10-05 `sector_classification.py`에서 실측된 선례(로그인 없이 pykrx 호출하면
KRX 서버가 파싱 불가능한 응답을 줘 "Expecting value"로 실패)가 있다 — 이 모듈의
함수들도 동일하게 로그인 세션을 요구할 가능성이 높다(라이브 미검증, 사용자가 직접
`scripts/verify_sources_individually.py`로 확인 필요)."""

import contextlib
import io
from datetime import date, timedelta

from dotenv import load_dotenv

with contextlib.redirect_stdout(io.StringIO()):
    from pykrx import stock

_SUPPORTED_MARKETS = frozenset({"KOSPI", "KOSDAQ"})

# pykrx의 get_market_net_purchases_of_equities_by_ticker는 1회 호출당 1개
# investor만 조회한다(동시조회 옵션 없음) — "외국인"/"기관합계" 각각 별도 호출.
_FOREIGN_INVESTOR = "외국인"
_INSTITUTION_INVESTOR = "기관합계"

# KRX는 공매도 잔고를 며칠 늦게 공개한다(2026-10-13 실측: 10/8~10/13은 아직 없고 10/6이 최신) —
# 연휴·주말을 넘길 만큼 거슬러 올라가 "가장 최근에 공개된 날"을 찾는다.
_SHORTING_LOOKBACK_DAYS = 10


def fetch_net_purchases(ticker: str, market: str, start: date, end: date) -> dict[str, int] | None:
    """외국인/기관 순매수 거래량·거래대금. market이 KOSPI/KOSDAQ이 아니면 None
    (미지원 시장, 에러 아님). 데이터가 없는 티커도 None."""
    if market not in _SUPPORTED_MARKETS:
        return None

    load_dotenv()
    from_date, to_date = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    foreign_df = stock.get_market_net_purchases_of_equities_by_ticker(
        from_date, to_date, market=market, investor=_FOREIGN_INVESTOR
    )
    institution_df = stock.get_market_net_purchases_of_equities_by_ticker(
        from_date, to_date, market=market, investor=_INSTITUTION_INVESTOR
    )

    if ticker not in foreign_df.index or ticker not in institution_df.index:
        return None

    foreign_row = foreign_df.loc[ticker]
    institution_row = institution_df.loc[ticker]
    return {
        "foreign_net_volume": int(foreign_row["순매수거래량"]),
        "foreign_net_value": int(foreign_row["순매수거래대금"]),
        "institution_net_volume": int(institution_row["순매수거래량"]),
        "institution_net_value": int(institution_row["순매수거래대금"]),
    }


def fetch_shorting_balance(ticker: str, market: str, as_of: date) -> dict | None:
    """공매도 잔고/비중(시가총액 대비 %). market이 KOSPI/KOSDAQ이 아니면 None.
    데이터가 없는 티커·휴장일도 None(에러 아님)."""
    if market not in _SUPPORTED_MARKETS:
        return None

    load_dotenv()
    date_str = as_of.strftime("%Y%m%d")
    df = stock.get_shorting_balance_by_ticker(date_str, market=market)

    if ticker not in df.index:
        return None

    row = df.loc[ticker]
    return {
        "balance_shares": int(row["공매도잔고"]),
        "balance_value": int(row["공매도금액"]),
        "market_cap_ratio_pct": float(row["비중"]),
    }


def fetch_latest_shorting_balance(
    ticker: str, market: str, as_of: date, max_lookback_days: int = _SHORTING_LOOKBACK_DAYS
) -> tuple[date, dict] | None:
    """`as_of`부터 하루씩 거슬러 올라가 가장 최근에 공개된 공매도 잔고를 `(기준일, 값)`으로
    반환한다. 공시 지연으로 아직 공개되지 않은 날은 pykrx가 빈 결과를 주거나 내부 컬럼 오류를
    낸다(출력도 찍는다) — 둘 다 "그날은 데이터 없음"으로 보고 다음 날로 넘어간다. 끝까지 없거나
    미지원 시장이면 None."""
    if market not in _SUPPORTED_MARKETS:
        return None
    for offset in range(max_lookback_days + 1):
        day = as_of - timedelta(days=offset)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                balance = fetch_shorting_balance(ticker, market, day)
        except Exception:
            balance = None
        if balance is not None:
            return day, balance
    return None
