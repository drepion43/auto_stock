"""DART(전자공시시스템) OpenAPI 연동 — 데이터 수집 계층(#0)의 확장.

`pykrx_source.py`와 같은 위치: FDR/pykrx가 다루지 않는 국내 공시 데이터를 보조한다.
DART API는 KRX 종목코드가 아니라 자체 `corp_code`로 종목을 식별하므로, 전체
매핑(`corpCode.xml`, zip)을 다운로드해 로컬에 캐싱하고 종목코드로 역조회한다
(docs/design/news-disclosure-plan.md 핵심 설계 결정 3). `DART_API_KEY`는
`notifier/credentials.py`와 동일하게 필요 시점에 직접 `load_dotenv()`를 호출해
로드한다.
"""

import json
import os
import time
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path

import requests
from defusedxml import ElementTree
from dotenv import load_dotenv

from auto_stock.data.models import DisclosureRecord

CORP_CODE_CACHE_PATH = Path("data/dart_corp_codes.json")
CORP_CODE_REFRESH_DAYS = 7
CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
DISCLOSURE_LIST_URL = "https://opendart.fss.or.kr/api/list.json"

_STATUS_OK = "000"
_STATUS_NO_DATA = "013"

_REQUEST_TIMEOUT_SECONDS = 30


class DartApiError(Exception):
    pass


def _load_dart_api_key() -> str:
    load_dotenv()
    return os.environ["DART_API_KEY"]


def _download_corp_code_map() -> dict[str, dict[str, str]]:
    """corpCode.xml(zip)을 다운로드해 {stock_code: {"corp_code":..., "corp_name":...}}
    매핑으로 파싱한다. `corp_name`은 뉴스 감성분석(#4-뉴스) 확장에서 뉴스 검색 쿼리를
    만들 때 쓰인다(docs/design/news-sentiment-plan.md 핵심 설계 결정 2) — 이전에는
    파싱만 하고 버렸다.

    비상장 등 `stock_code`가 없는 항목은 종목코드로 역조회할 일이 없어 제외한다."""
    api_key = _load_dart_api_key()
    try:
        response = requests.get(
            CORP_CODE_URL, params={"crtfc_key": api_key}, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # requests/urllib3는 예외 문자열에 요청 URL(크리덴셜이 담긴 쿼리스트링 포함)을
        # 그대로 포함시킨다 — 원본 예외를 절대 노출하지 않고 타입명만 남긴다(보안 리뷰 CRITICAL).
        raise DartApiError(f"DART API 요청 실패: {type(exc).__name__}") from exc

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])

    root = ElementTree.fromstring(xml_bytes)
    status = root.findtext("status")
    if status is not None and status != _STATUS_OK:
        # 잘못된/폐기된 키를 넣으면 DART는 HTTP 오류가 아니라 <list> 항목 없이
        # <status>/<message>만 있는 "정상" zip을 돌려준다 — 이걸 빈 매핑으로 오인해
        # 캐싱하면 뉴스 신호가 CORP_CODE_REFRESH_DAYS 동안 에러 기록 없이 조용히
        # 꺼진다(코드 리뷰 HIGH). 캐싱 전에 반드시 에러로 승격시킨다.
        message = root.findtext("message") or ""
        raise DartApiError(f"DART API 오류(status={status}): {message}")

    mapping: dict[str, dict[str, str]] = {}
    for item in root.findall("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        corp_name = (item.findtext("corp_name") or "").strip()
        if stock_code:
            mapping[stock_code] = {"corp_code": corp_code, "corp_name": corp_name}
    return mapping


def _is_cache_fresh(cache_path: Path, refresh_days: int) -> bool:
    if not cache_path.exists():
        return False
    age_seconds = time.time() - cache_path.stat().st_mtime
    return age_seconds < refresh_days * 86400


def _load_or_refresh_corp_map(cache_path: Path, refresh_days: int) -> dict:
    if _is_cache_fresh(cache_path, refresh_days):
        return json.loads(cache_path.read_text(encoding="utf-8"))
    mapping = _download_corp_code_map()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 임시파일 + os.replace()로 원자적으로 쓴다 — write_text() 도중 프로세스가
    # 죽으면(OOM, 컨테이너 재시작 등) 손상된 캐시가 남아 refresh_days 동안
    # 자연 복구되지 않는 문제를 막는다(코드 리뷰 MEDIUM #1).
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    tmp_path.write_text(json.dumps(mapping), encoding="utf-8")
    os.replace(tmp_path, cache_path)
    return mapping


def _corp_code_and_name(entry) -> tuple[str | None, str | None]:
    """캐시 항목에서 (corp_code, corp_name)을 뽑는다. `entry`가 문자열이면 뉴스
    감성분석 확장 전(corp_name이 없던 시절)에 저장된 구 스키마 캐시다 — corp_code만
    반환하고 corp_name은 None(하위호환, docs/design/news-sentiment-plan.md 핵심
    설계 결정 2). refresh_days가 지나면 새 스키마로 자연 갱신된다."""
    if entry is None:
        return None, None
    if isinstance(entry, str):
        return entry, None
    return entry.get("corp_code"), entry.get("corp_name")


def resolve_corp_code(
    ticker: str,
    cache_path: Path | str = CORP_CODE_CACHE_PATH,
    refresh_days: int = CORP_CODE_REFRESH_DAYS,
) -> str | None:
    """`corpCode.xml` 캐시에서 티커로 corp_code를 역조회한다.

    캐시가 없거나 `refresh_days`보다 오래되면 재다운로드 후 캐시를 갱신한다.
    나스닥 등 DART에 등록되지 않은 티커는 None을 반환한다(에러 아님)."""
    mapping = _load_or_refresh_corp_map(Path(cache_path), refresh_days)
    corp_code, _ = _corp_code_and_name(mapping.get(ticker))
    return corp_code


def resolve_corp_name(
    ticker: str,
    cache_path: Path | str = CORP_CODE_CACHE_PATH,
    refresh_days: int = CORP_CODE_REFRESH_DAYS,
) -> str | None:
    """`corpCode.xml` 캐시에서 티커로 회사명(corp_name)을 역조회한다. 뉴스 검색
    API는 티커가 아니라 회사명 문자열로 질의해야 하므로(docs/design/
    news-sentiment-plan.md 핵심 설계 결정 2), 그 쿼리 입력을 만드는 데 쓰인다.
    `resolve_corp_code`와 캐시(다운로드/저장)를 공유한다 — 시장이 같은 티커에 대해
    두 함수를 순서대로 호출해도 DART API를 두 번 부르지 않는다."""
    mapping = _load_or_refresh_corp_map(Path(cache_path), refresh_days)
    _, corp_name = _corp_code_and_name(mapping.get(ticker))
    return corp_name


def resolve_ticker_by_name(
    name_query: str,
    cache_path: Path | str = CORP_CODE_CACHE_PATH,
    refresh_days: int = CORP_CODE_REFRESH_DAYS,
) -> list[tuple[str, str]]:
    """`corpCode.xml` 캐시에서 회사명 문자열로 종목코드를 역조회한다(`resolve_corp_name`의
    반대 방향) — 대화형 챗봇(#Stage B)이 사용자가 입력한 회사명을 종목코드로 바꿀 때 쓴다.
    새 DART API 호출 없이 기존 캐시를 재사용한다.

    대소문자 무시 완전일치가 하나라도 있으면 그것만 반환한다(부분일치 후보를 섞지 않음 —
    "삼성전자" 질의에 "삼성전자우"까지 끼워 넣지 않기 위함). 완전일치가 없으면 부분문자열
    포함 여부로 폴백한다. 이름이 없는 구 스키마(문자열) 캐시 항목은 매칭 대상에서 제외한다."""
    mapping = _load_or_refresh_corp_map(Path(cache_path), refresh_days)
    query_casefold = name_query.casefold()

    named_entries = [
        (ticker, corp_name)
        for ticker, entry in mapping.items()
        for _, corp_name in [_corp_code_and_name(entry)]
        if corp_name
    ]

    exact_matches = [(t, n) for t, n in named_entries if n.casefold() == query_casefold]
    if exact_matches:
        return exact_matches
    return [(t, n) for t, n in named_entries if query_casefold in n.casefold()]


def _parse_dart_date(value: str) -> date:
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def fetch_disclosures(corp_code: str, start: date, end: date) -> list[DisclosureRecord]:
    """DART 공시검색(`list.json`) 호출. 결과 없으면 빈 리스트(예외 아님)."""
    api_key = _load_dart_api_key()
    try:
        response = requests.get(
            DISCLOSURE_LIST_URL,
            params={
                "crtfc_key": api_key,
                "corp_code": corp_code,
                "bgn_de": start.strftime("%Y%m%d"),
                "end_de": end.strftime("%Y%m%d"),
                "page_no": 1,
                "page_count": 100,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        # requests/urllib3는 예외 문자열에 요청 URL(크리덴셜이 담긴 쿼리스트링 포함)을
        # 그대로 포함시킨다 — 원본 예외를 절대 노출하지 않고 타입명만 남긴다(보안 리뷰 CRITICAL).
        raise DartApiError(f"DART API 요청 실패: {type(exc).__name__}") from exc
    payload = response.json()

    status = payload.get("status")
    if status == _STATUS_NO_DATA:
        return []
    if status != _STATUS_OK:
        raise DartApiError(f"DART API 오류(status={status}): {payload.get('message')}")

    return [
        DisclosureRecord(
            corp_code=corp_code,
            ticker=item.get("stock_code", ""),
            # .get(key, "")은 키가 아예 없을 때만 기본값을 쓰고, DART가 명시적으로
            # JSON null을 보내면 None을 그대로 반환해 DisclosureRecord의 str 타입
            # 계약을 조용히 깬다 — `or ""`로 두 경우 모두 방어한다(코드 리뷰 MEDIUM #2).
            report_name=item.get("report_nm") or "",
            filed_date=_parse_dart_date(item["rcept_dt"]),
            remark=item.get("rm") or "",
        )
        for item in payload.get("list", [])
    ]
