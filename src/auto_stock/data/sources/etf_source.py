"""국내 ETF 구성종목(holdings) + ETF 유니버스 조회 — 데이터 수집 계층(#0) 확장.

ETF 가격은 구성종목들의 가중평균 움직임을 그대로 반영한다 — "ETF 자체의 공시/뉴스"를
찾는 대신 "구성종목들의 공시/뉴스를 집계"하면 "이 ETF가 왜 움직였는지"에 답할 수
있다는 설계 전제(`chat_agent/etf_composition.py`가 이 전제를 바탕으로 종합한다).

`get_etf_portfolio_deposit_file(etf_ticker)`는 `sector_classification.py`가 쓰는
`get_index_portfolio_deposit_file(index_code)`와 이름이 비슷하지만 **완전히 다른
API 패밀리**다 — 전자는 ETF 티커를 직접 받아 구성종목+비중을 한 번에 반환하므로
(지수 역조회 같은 중간 단계 불필요), `fetch_sector_names`→`fetch_constituents`의
2단계 패턴을 쓸 필요가 없다.

**국내(KOSPI/KOSDAQ) 전용** — 미국 ETF(SPY/QQQ 등) holdings는 SEC EDGAR의 Form
N-PORT(분기 공시, raw XML 직접 파싱 필요)로만 무료 조회 가능하고 신선도도 분기
단위라 국내와 전혀 다른 작업이 된다 — 사용자 확인 후 범위에서 명시적으로 제외했다.

2026-10-05 `sector_classification.py`에서 실측된 선례(로그인 없이 pykrx 지수 API를
호출하면 "Expecting value"로 실패)가 있다 — `get_etf_portfolio_deposit_file`도
동일하게 로그인 세션을 요구할 가능성이 있어 함수 호출 시점에 `load_dotenv()`를
호출한다(라이브 미검증, 사용자가 `scripts/verify_sources_individually.py`로 확인
필요)."""

import contextlib
import io
from datetime import date

import FinanceDataReader as fdr
from dotenv import load_dotenv

with contextlib.redirect_stdout(io.StringIO()):
    from pykrx import stock

DEFAULT_TOP_N = 10  # recommendation_coordinator.DEFAULT_SHORTLIST_CAP(20)보다 작게—
# ETF 분석은 "참고용 요약"이라는 성격을 반영, 종목 수만큼 선형으로 커지는 LLM 비용 관리.


def fetch_etf_holdings(
    etf_ticker: str, top_n: int = DEFAULT_TOP_N, as_of: date | None = None
) -> list[tuple[str, float]]:
    """ETF 구성종목을 비중(%) 내림차순으로 상위 `top_n`개 반환한다. `(종목코드, 비중)`
    튜플 리스트. ETF가 아니거나 holdings 데이터가 없으면 빈 리스트(에러 아님) —
    `sector_classification.fetch_sector_names`의 "지원 안 되는 입력은 조용히 스킵"
    관례와 동일.

    2026-10-11 라이브 실측으로 확인한 두 가지 함정:
    - **날짜를 반드시 명시한다** — 생략하면 pykrx가 `get_index_ohlcv_by_date`로 최근
      영업일을 찾는데, 이 환경에서 그 지수 API가 "Expecting value"로 실패한다. 명시
      날짜(주말 포함)를 넘기면 KRX가 그 시점의 최신 PDF를 반환한다.
    - **pykrx의 `비중` 컬럼을 쓰지 않는다** — `시가총액` 기준으로 계산되는데 주식 행의
      시가총액이 전부 0이라 현금(원화예금) 행만 100, 나머지는 0으로 나온다. 실제 비중은
      1CU당 평가금액인 `금액` 컬럼으로 직접 계산한다(KODEX 200에서 삼성전자 34.5%,
      SK하이닉스 27.0%로 상식과 일치 확인)."""
    load_dotenv()
    as_of_str = (as_of or date.today()).strftime("%Y%m%d")
    df = stock.get_etf_portfolio_deposit_file(etf_ticker, as_of_str)
    if df is None or df.empty:
        return []

    total_amount = df["금액"].sum()
    if total_amount <= 0:
        return []
    weights = (df["금액"] / total_amount * 100).sort_values(ascending=False).head(top_n)
    return [(ticker, round(float(weight), 2)) for ticker, weight in weights.items()]


def fetch_etf_universe() -> list[str]:
    """국내 ETF 전체 종목코드 리스트. `fdr.StockListing("ETF/KR")`는 holdings를
    제공하지 않지만(FDR 전체를 확인한 결과 ETF 구성종목 조회 기능 자체가 없음) 유니버스
    조회에는 쓸 수 있다 — holdings는 이 모듈에서 pykrx로 별도 조회한다."""
    listing = fdr.StockListing("ETF/KR")
    return listing["Symbol"].tolist()


def fetch_etf_listing() -> list[tuple[str, str]]:
    """국내 ETF `(종목코드, ETF명)` 전체 리스트 — `ticker_resolution.py`가 ETF 이름/코드를
    해석할 때 쓴다. DART의 `corpCode.xml`은 상장"법인"(기업) 전용이라 ETF(운용사가
    발행하는 펀드, 법인이 아님)를 포함하지 않을 가능성이 높다(라이브 미검증, 사용자가
    `scripts/verify_sources_individually.py`로 확인 필요) — DART 역조회가 실패했을 때의
    보조 경로로 이 함수를 쓴다."""
    listing = fdr.StockListing("ETF/KR")
    return list(zip(listing["Symbol"].tolist(), listing["Name"].tolist()))
