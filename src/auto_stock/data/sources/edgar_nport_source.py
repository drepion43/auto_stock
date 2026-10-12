"""SEC EDGAR Form N-PORT(미국 ETF/펀드 구성종목) 연동 — 데이터 수집 계층(#0)의 확장,
`etf_source.py`(국내 ETF)의 미국 대응. `edgar_source.py`와 같은 위치지만 별도 모듈로
둔 이유: N-PORT는 공시 목록(`fetch_filings`)이 아니라 **구성종목 holdings**를 다루고,
응답 스키마(XML, `invstOrSecs`)가 공시 목록 API와 완전히 다르다.

**미국 ETF 티커가 두 개의 서로 다른 SEC 매핑 체계에 흩어져 있다**(2026-10 라이브
호출로 확인) — SPY/DIA/GLD처럼 UIT(Unit Investment Trust)형 ETF는 일반 상장기업과
같은 `edgar_source.COMPANY_TICKERS_URL`(equity 맵)에 있지만, QQQ/VOO/IVV/VTI 등은
그 맵에 전혀 없고 완전히 별도인 `company_tickers_mf.json`(뮤추얼펀드 series/class
체계, 필드: cik/seriesId/classId/symbol)에만 있다. `resolve_etf_cik`가 두 맵을 순서대로
시도해 이 분기를 감춘다.

**N-PORT holding에는 종목코드(ticker) 필드가 없다** — CUSIP/ISIN과 자유형식 `name`
(거래소 표기와 다를 수 있음, 예: 라이브 샘플에서 "AES Corp/The")만 있다. 그래서 이
모듈은 holding을 `(name, pct)`로만 반환하고, 이름→티커 역변환(완전일치만 허용, 실패한
holding은 스킵)은 `chat_agent/tools.py`가 기존 `edgar_source.resolve_ticker_by_name`을
재사용해 책임진다 — 이 모듈 자신은 종목 식별을 하지 않는다(`naver_news_source.py`가
market/ticker를 호출자가 채우게 하는 것과 동일한 책임 분리).

N-PORT는 분기 공시(최대 2개월 지연)라 국내 pykrx(거의 일별)보다 신선도가 훨씬 낮다 —
호출부가 이 사실을 답변에 반영해야 한다(`docs/design/etf-composition-analysis.md` 참고).
"""

import json
import os
import time
from pathlib import Path

import requests
from defusedxml import ElementTree
from dotenv import load_dotenv

from auto_stock.data.sources.edgar_source import SUBMISSIONS_URL_TEMPLATE, resolve_cik

MF_TICKERS_URL = "https://www.sec.gov/files/company_tickers_mf.json"
MF_TICKER_MAP_CACHE_PATH = Path("data/edgar_mf_ticker_map.json")
MF_TICKER_MAP_REFRESH_DAYS = 7

_ARCHIVES_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/primary_doc.xml"
_NPORT_FORM_TYPE = "NPORT-P"
_REQUEST_TIMEOUT_SECONDS = 30
_NPORT_NAMESPACE = {"n": "http://www.sec.gov/edgar/nport"}


class EdgarNportError(Exception):
    pass


def _load_edgar_user_agent() -> str:
    load_dotenv()
    return os.environ["SEC_EDGAR_USER_AGENT"]


def _download_mf_ticker_map() -> dict[str, str]:
    """`company_tickers_mf.json`을 다운로드해 {symbol: cik(10자리 0-패딩)} 매핑으로
    파싱한다 — 뮤추얼펀드/ETF series·class 체계(QQQ/VOO 등은 equity 맵에 없고 여기만
    있다). 응답은 `{"fields": ["cik","seriesId","classId","symbol"], "data": [[...]]}`
    형태(company_tickers.json과 다른 스키마)."""
    user_agent = _load_edgar_user_agent()
    try:
        response = requests.get(
            MF_TICKERS_URL, headers={"User-Agent": user_agent}, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # dart_source.py/edgar_source.py와 동일 원칙: 원본 예외를 문자열화하지 않고
        # 타입명만 남긴다.
        raise EdgarNportError(f"EDGAR 뮤추얼펀드 티커맵 요청 실패: {type(exc).__name__}") from exc

    payload = response.json()
    fields = payload.get("fields", [])
    try:
        cik_idx = fields.index("cik")
        symbol_idx = fields.index("symbol")
    except ValueError:
        return {}

    mapping: dict[str, str] = {}
    for row in payload.get("data", []):
        symbol = row[symbol_idx]
        cik = row[cik_idx]
        if symbol and cik is not None:
            mapping[symbol] = f"{int(cik):010d}"
    return mapping


def _is_cache_fresh(cache_path: Path, refresh_days: int) -> bool:
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < refresh_days * 86400


def _load_or_refresh_mf_ticker_map(cache_path: Path, refresh_days: int) -> dict[str, str]:
    if _is_cache_fresh(cache_path, refresh_days):
        return json.loads(cache_path.read_text(encoding="utf-8"))
    mapping = _download_mf_ticker_map()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # DART/EDGAR 캐시와 동일한 원자적 쓰기 패턴(임시파일 + os.replace).
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(mapping), encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return mapping


def resolve_etf_cik(
    ticker: str,
    mf_cache_path: Path | str = MF_TICKER_MAP_CACHE_PATH,
    mf_refresh_days: int = MF_TICKER_MAP_REFRESH_DAYS,
) -> str | None:
    """미국 ETF 티커로 CIK를 역조회한다. SPY/DIA/GLD처럼 equity 맵(`edgar_source.
    resolve_cik`)에 있는 UIT형 ETF를 먼저 시도하고, 거기 없으면 QQQ/VOO 등이 속한
    뮤추얼펀드 series/class 맵을 시도한다. 둘 다 없으면 None(ETF가 아님, 에러 아님)."""
    cik = resolve_cik(ticker)
    if cik is not None:
        return cik
    mapping = _load_or_refresh_mf_ticker_map(Path(mf_cache_path), mf_refresh_days)
    return mapping.get(ticker)


def _latest_nport_accession(cik: str, user_agent: str) -> str | None:
    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    try:
        response = requests.get(url, headers={"User-Agent": user_agent}, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise EdgarNportError(f"EDGAR submissions 요청 실패: {type(exc).__name__}") from exc

    payload = response.json()
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    for form, accession in zip(forms, accessions):
        if form == _NPORT_FORM_TYPE:
            return accession  # submissions API는 최신순으로 반환한다(날짜 정렬 불필요)
    return None


def _fetch_nport_xml(cik: str, accession: str, user_agent: str) -> bytes:
    cik_int = int(cik)
    accession_no_dashes = accession.replace("-", "")
    url = _ARCHIVES_URL_TEMPLATE.format(cik_int=cik_int, accession_no_dashes=accession_no_dashes)
    try:
        response = requests.get(url, headers={"User-Agent": user_agent}, timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise EdgarNportError(f"EDGAR N-PORT 문서 요청 실패: {type(exc).__name__}") from exc
    return response.content


def _parse_nport_holdings(xml_bytes: bytes) -> list[tuple[str, float]]:
    """N-PORT primary_doc.xml에서 `invstOrSecs/invstOrSec`마다 (name, pctVal)을
    추출한다. `defusedxml`로 파싱한다(dart_source.py와 동일한 XXE 방지 원칙)."""
    try:
        root = ElementTree.fromstring(xml_bytes)
    except Exception as exc:
        raise EdgarNportError(f"N-PORT XML 파싱 실패: {type(exc).__name__}") from exc

    holdings: list[tuple[str, float]] = []
    for sec in root.iter("{http://www.sec.gov/edgar/nport}invstOrSec"):
        name = sec.findtext("{http://www.sec.gov/edgar/nport}name")
        pct_val = sec.findtext("{http://www.sec.gov/edgar/nport}pctVal")
        if not name or pct_val is None:
            continue
        try:
            holdings.append((name, float(pct_val)))
        except ValueError:
            continue
    return holdings


def fetch_latest_nport_holdings(cik: str, top_n: int = 10) -> list[tuple[str, float]] | None:
    """해당 CIK의 가장 최근 NPORT-P 공시에서 비중(순자산 대비 %) 내림차순 상위
    `top_n`개 구성종목을 `(name, pct)`로 반환한다. NPORT-P 공시 자체가 없으면(UIT가
    아닌 일반 상장기업 등) None(에러 아님). `pct`는 순자산 대비 비율(0~100대 값이
    아니라 N-PORT 원본 그대로 — 호출부가 필요시 백분율로 변환)."""
    user_agent = _load_edgar_user_agent()
    accession = _latest_nport_accession(cik, user_agent)
    if accession is None:
        return None

    xml_bytes = _fetch_nport_xml(cik, accession, user_agent)
    holdings = _parse_nport_holdings(xml_bytes)
    holdings.sort(key=lambda h: h[1], reverse=True)
    return holdings[:top_n]
