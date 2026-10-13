"""국내 종목의 날짜별 펀더멘털(PER/PBR/DIV/시가총액/거래량/상장주식수/매출액) 조립.

`features.py`의 중립값 폴백과 짝을 이루는 모듈 — 미국 종목은 애초에 호출하지 않고
`None`을 넘기면 된다(펀더멘털 피처 전부가 중립값+available=0으로 깔린다). 국내
종목은 pykrx(일별 종가 기준, lookahead 문제 없음)와 DART 재무제표(연간, 사업연도
종료 후 공시 유예 적용 — `dart_financials_source.fetch_latest_revenue`가 날짜별로
"그 시점에 실제 공시돼 있던" 매출액만 반환하도록 이미 처리)를 날짜별로 합친다."""

from datetime import date

from auto_stock.data.market_routing import KRX_MARKETS
from auto_stock.data.sources import dart_financials_source, dart_source, pykrx_source


def build_fundamentals_by_date(ticker: str, market: str, start: date, end: date) -> dict[date, dict] | None:
    """`market`이 국내(KOSPI/KOSDAQ)가 아니면 즉시 None(해외 종목, 호출 자체가 불필요).
    국내 종목이어도 DART corp_code를 못 찾거나(비상장 등) DART 조회 자체가 실패하면
    (예: DART_API_KEY 미설정) 매출액만 전부 None으로 빠지고 PER/PBR/DIV/회전율은
    정상 포함된다 — 두 데이터소스(pykrx/DART)는 서로 독립이라 한쪽 실패가 다른 쪽까지
    끌고 내려가면 안 된다(부분 결측은 에러가 아니다)."""
    if market not in KRX_MARKETS:
        return None

    fundamentals = pykrx_source.get_market_fundamental_range(ticker, start, end)
    cap_details = pykrx_source.get_market_cap_detail_range(ticker, start, end)

    try:
        corp_code = dart_source.resolve_corp_code(ticker)
    except Exception:
        corp_code = None

    # 사업연도(bsns_year) 단위로만 메모이즈한다 — fetch_latest_revenue 자신도 디스크
    # 캐시가 있지만, 같은 호출 안에서 수백 개 날짜가 같은 연도로 뭉치는 걸 파이썬
    # 레벨에서도 한 번만 계산하기 위함.
    revenue_by_year: dict[tuple[int, bool], int | None] = {}

    def _revenue_for(as_of: date) -> int | None:
        if corp_code is None:
            return None
        year_bucket = (as_of.year, as_of.month >= 4)
        if year_bucket not in revenue_by_year:
            try:
                revenue_by_year[year_bucket] = dart_financials_source.fetch_latest_revenue(corp_code, as_of)
            except Exception:
                revenue_by_year[year_bucket] = None
        return revenue_by_year[year_bucket]

    all_dates = set(fundamentals.keys()) | set(cap_details.keys())
    result: dict[date, dict] = {}
    for d in all_dates:
        result[d] = {**fundamentals.get(d, {}), **cap_details.get(d, {}), "revenue": _revenue_for(d)}
    return result
