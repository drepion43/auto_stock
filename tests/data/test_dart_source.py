"""dart_source.py 테스트 — requests를 모킹해 실제 DART API를 호출하지 않는다."""

import json
import os
import time
import zipfile
from datetime import date
from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, tostring

import pytest
import requests

from auto_stock.data.models import DisclosureRecord
from auto_stock.data.sources.dart_source import (
    DartApiError,
    fetch_disclosures,
    resolve_corp_code,
    resolve_corp_name,
    resolve_ticker_by_name,
)


@pytest.fixture(autouse=True)
def _dummy_dart_api_key(monkeypatch, mocker):
    # 실제 .env에 DART_API_KEY가 생기더라도 테스트가 영향받지 않도록 load_dotenv 자체를 모킹
    mocker.patch("auto_stock.data.sources.dart_source.load_dotenv")
    monkeypatch.setenv("DART_API_KEY", "dummy-dart-key-not-real")


def _corp_code_zip_bytes(entries: list[tuple[str, str]]) -> bytes:
    """entries: (stock_code, corp_code) 목록으로 corpCode.xml(zip) 바이트를 만든다."""
    root = Element("result")
    for stock_code, corp_code in entries:
        item = SubElement(root, "list")
        SubElement(item, "corp_code").text = corp_code
        SubElement(item, "corp_name").text = "테스트기업"
        SubElement(item, "stock_code").text = stock_code
        SubElement(item, "modify_date").text = "20260101"
    xml_bytes = tostring(root, encoding="utf-8")

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("CORPCODE.xml", xml_bytes)
    return buffer.getvalue()


def _corp_code_error_zip_bytes(status: str, message: str) -> bytes:
    """DART가 유효하지 않은 키 등으로 오류를 낼 때 실제로 반환하는 형식
    (<list> 항목 없이 <status>/<message>만 있는 zip)을 흉내낸다."""
    root = Element("result")
    SubElement(root, "status").text = status
    SubElement(root, "message").text = message
    xml_bytes = tostring(root, encoding="utf-8")

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("CORPCODE.xml", xml_bytes)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes = b"", json_data: dict | None = None):
        self.content = content
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_resolve_corp_code_downloads_and_caches_when_cache_missing(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    zip_bytes = _corp_code_zip_bytes([("005930", "00126380"), (" ", "00999999")])
    mock_get = mocker.patch(
        "auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(content=zip_bytes)
    )

    result = resolve_corp_code("005930", cache_path=cache_path)

    assert result == "00126380"
    assert cache_path.exists()
    mock_get.assert_called_once()


def test_resolve_corp_code_skips_entries_without_stock_code(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    zip_bytes = _corp_code_zip_bytes([("005930", "00126380"), (" ", "00999999")])
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(content=zip_bytes))

    resolve_corp_code("005930", cache_path=cache_path)
    mapping = json.loads(cache_path.read_text(encoding="utf-8"))

    assert "00999999" not in {entry["corp_code"] for entry in mapping.values()}


def test_resolve_corp_code_returns_none_for_ticker_not_in_dart(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    zip_bytes = _corp_code_zip_bytes([("005930", "00126380")])
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(content=zip_bytes))

    result = resolve_corp_code("AAPL", cache_path=cache_path)  # 나스닥 종목 — DART 미등록

    assert result is None


def test_resolve_corp_code_uses_cache_without_calling_api_when_fresh(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(json.dumps({"005930": "00126380"}), encoding="utf-8")
    mock_get = mocker.patch("auto_stock.data.sources.dart_source.requests.get")

    result = resolve_corp_code("005930", cache_path=cache_path)

    assert result == "00126380"
    mock_get.assert_not_called()


def test_resolve_corp_code_redownloads_when_cache_is_stale(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(json.dumps({"005930": "OLD_CODE"}), encoding="utf-8")
    stale_time = time.time() - 8 * 86400  # 기본 갱신주기(7일) 초과
    os.utime(cache_path, (stale_time, stale_time))

    zip_bytes = _corp_code_zip_bytes([("005930", "00126380")])
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(content=zip_bytes))

    result = resolve_corp_code("005930", cache_path=cache_path)

    assert result == "00126380"


def test_fetch_disclosures_returns_records_on_success(mocker):
    payload = {
        "status": "000",
        "message": "정상",
        "list": [
            {
                "corp_code": "00126380",
                "stock_code": "005930",
                "report_nm": "주요사항보고서(유상증자결정)",
                "rcept_dt": "20260115",
                "rm": "",
            }
        ],
    }
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(json_data=payload))

    records = fetch_disclosures("00126380", date(2026, 1, 1), date(2026, 1, 31))

    assert records == [
        DisclosureRecord(
            corp_code="00126380",
            ticker="005930",
            report_name="주요사항보고서(유상증자결정)",
            filed_date=date(2026, 1, 15),
            remark="",
        )
    ]


def test_fetch_disclosures_returns_empty_list_when_status_no_data(mocker):
    payload = {"status": "013", "message": "조회된 데이타가 없습니다."}
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(json_data=payload))

    records = fetch_disclosures("00126380", date(2026, 1, 1), date(2026, 1, 31))

    assert records == []


def test_fetch_disclosures_raises_on_error_status(mocker):
    payload = {"status": "020", "message": "요청 제한을 초과하였습니다."}
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(json_data=payload))

    with pytest.raises(DartApiError, match="020"):
        fetch_disclosures("00126380", date(2026, 1, 1), date(2026, 1, 31))


def test_fetch_disclosures_requires_dart_api_key(monkeypatch):
    monkeypatch.delenv("DART_API_KEY", raising=False)

    with pytest.raises(KeyError, match="DART_API_KEY"):
        fetch_disclosures("00126380", date(2026, 1, 1), date(2026, 1, 31))


def test_resolve_corp_code_requires_dart_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("DART_API_KEY", raising=False)

    with pytest.raises(KeyError, match="DART_API_KEY"):
        resolve_corp_code("005930", cache_path=tmp_path / "corp_codes.json")


# --- 보안 리뷰 CRITICAL 수정 회귀: requests 예외 문자열에 DART_API_KEY가 담긴 요청 URL이
# 그대로 포함될 수 있으므로(requests/urllib3의 표준 동작), 원본 예외를 절대 그대로
# 노출하지 않고 DartApiError로 정제해서 감싼다. ---


def test_resolve_corp_code_wraps_connection_error_without_leaking_api_key(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    mocker.patch(
        "auto_stock.data.sources.dart_source.requests.get",
        side_effect=requests.exceptions.ConnectionError(
            "Max retries exceeded with url: /api/corpCode.xml?crtfc_key=dummy-dart-key-not-real"
        ),
    )

    with pytest.raises(DartApiError) as exc_info:
        resolve_corp_code("005930", cache_path=cache_path)

    assert "dummy-dart-key-not-real" not in str(exc_info.value)


def test_fetch_disclosures_wraps_connection_error_without_leaking_api_key(mocker):
    mocker.patch(
        "auto_stock.data.sources.dart_source.requests.get",
        side_effect=requests.exceptions.ConnectionError(
            "Max retries exceeded with url: /api/list.json?crtfc_key=dummy-dart-key-not-real"
        ),
    )

    with pytest.raises(DartApiError) as exc_info:
        fetch_disclosures("00126380", date(2026, 1, 1), date(2026, 1, 31))

    assert "dummy-dart-key-not-real" not in str(exc_info.value)


def test_fetch_disclosures_wraps_http_error_from_raise_for_status_without_leaking_api_key(mocker):
    class _FailingResponse:
        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "401 Client Error: Unauthorized for url: "
                "https://opendart.fss.or.kr/api/list.json?crtfc_key=dummy-dart-key-not-real"
            )

    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FailingResponse())

    with pytest.raises(DartApiError) as exc_info:
        fetch_disclosures("00126380", date(2026, 1, 1), date(2026, 1, 31))

    assert "dummy-dart-key-not-real" not in str(exc_info.value)


def test_resolve_corp_code_wraps_http_error_from_raise_for_status_without_leaking_api_key(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"

    class _FailingResponse:
        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "401 Client Error: Unauthorized for url: "
                "https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key=dummy-dart-key-not-real"
            )

    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FailingResponse())

    with pytest.raises(DartApiError) as exc_info:
        resolve_corp_code("005930", cache_path=cache_path)

    assert "dummy-dart-key-not-real" not in str(exc_info.value)


# --- 코드 리뷰 HIGH 수정 회귀: DART는 유효하지 않은 키에 대해 HTTP 오류가 아니라
# <list> 항목 없이 <status>/<message>만 담긴 "정상" zip을 반환한다. 이를 빈 매핑으로
# 오인해 캐싱하면 뉴스 신호가 7일간 조용히 꺼진 채 에러 기록도 남지 않는다. ---


def test_resolve_corp_code_raises_on_error_status_instead_of_caching_empty_mapping(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    zip_bytes = _corp_code_error_zip_bytes("013", "등록되지 않은 키입니다.")
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(content=zip_bytes))

    with pytest.raises(DartApiError, match="013"):
        resolve_corp_code("005930", cache_path=cache_path)

    assert not cache_path.exists()  # 빈 매핑이 캐시로 저장되면 안 됨


# --- 코드 리뷰 MEDIUM #1 수정 회귀: 캐시 쓰기는 임시파일 + os.replace()로 원자적이어야
# 한다(중간에 프로세스가 죽어도 손상된 캐시가 남지 않도록). ---


def test_resolve_corp_code_leaves_no_leftover_temp_file_after_cache_write(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    zip_bytes = _corp_code_zip_bytes([("005930", "00126380")])
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(content=zip_bytes))

    resolve_corp_code("005930", cache_path=cache_path)

    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "005930": {"corp_code": "00126380", "corp_name": "테스트기업"}
    }
    leftover_files = set(os.listdir(tmp_path)) - {cache_path.name}
    assert leftover_files == set()


# --- 코드 리뷰 MEDIUM #2 수정 회귀: DART가 report_nm/rm에 명시적으로 JSON null을
# 보내면 dict.get(key, "")은 기본값이 아니라 None을 그대로 반환한다 — DisclosureRecord의
# str 타입 계약을 조용히 깬다. ---


def test_fetch_disclosures_treats_explicit_null_remark_and_report_name_as_empty_string(mocker):
    payload = {
        "status": "000",
        "message": "정상",
        "list": [
            {
                "corp_code": "00126380",
                "stock_code": "005930",
                "report_nm": None,
                "rcept_dt": "20260115",
                "rm": None,
            }
        ],
    }
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(json_data=payload))

    records = fetch_disclosures("00126380", date(2026, 1, 1), date(2026, 1, 31))

    assert records[0].report_name == ""
    assert records[0].remark == ""


# --- 뉴스 감성분석(#4-뉴스) 확장: resolve_corp_name — 뉴스 검색 쿼리를 만들기 위한
# 회사명 조회. resolve_corp_code와 캐시를 공유한다(docs/design/news-sentiment-plan.md
# 핵심 설계 결정 2). ---


def test_resolve_corp_name_downloads_and_caches_when_cache_missing(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    zip_bytes = _corp_code_zip_bytes([("005930", "00126380")])
    mock_get = mocker.patch(
        "auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(content=zip_bytes)
    )

    result = resolve_corp_name("005930", cache_path=cache_path)

    assert result == "테스트기업"
    mock_get.assert_called_once()


def test_resolve_corp_name_returns_none_for_ticker_not_in_dart(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    zip_bytes = _corp_code_zip_bytes([("005930", "00126380")])
    mocker.patch("auto_stock.data.sources.dart_source.requests.get", return_value=_FakeResponse(content=zip_bytes))

    result = resolve_corp_name("AAPL", cache_path=cache_path)

    assert result is None


def test_resolve_corp_name_uses_cache_without_calling_api_when_fresh(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(
        json.dumps({"005930": {"corp_code": "00126380", "corp_name": "삼성전자"}}), encoding="utf-8"
    )
    mock_get = mocker.patch("auto_stock.data.sources.dart_source.requests.get")

    result = resolve_corp_name("005930", cache_path=cache_path)

    assert result == "삼성전자"
    mock_get.assert_not_called()


def test_resolve_corp_name_returns_none_gracefully_for_legacy_flat_cache_schema(mocker, tmp_path):
    """뉴스 감성분석 확장 전(corp_name이 없던 시절)에 저장된 평면 캐시를 만나도
    크래시하지 않고 None을 반환해야 한다 — refresh_days가 지나면 자연히 새 스키마로
    갱신된다."""
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(json.dumps({"005930": "00126380"}), encoding="utf-8")  # 구 스키마
    mock_get = mocker.patch("auto_stock.data.sources.dart_source.requests.get")

    name_result = resolve_corp_name("005930", cache_path=cache_path)
    code_result = resolve_corp_code("005930", cache_path=cache_path)

    assert name_result is None
    assert code_result == "00126380"  # corp_code 조회는 구 스키마에서도 여전히 동작
    mock_get.assert_not_called()


# --- 대화형 챗봇(Stage B) 확장: resolve_ticker_by_name — 사용자가 채팅에 입력한
# 회사명 문자열로 종목코드를 역조회한다(resolve_corp_name의 반대 방향). 기존 캐시를
# 그대로 재사용하며 새 DART API 호출은 없다. ---


def test_resolve_ticker_by_name_exact_match_is_case_insensitive(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(
        json.dumps(
            {
                "005930": {"corp_code": "00126380", "corp_name": "삼성전자"},
                "000660": {"corp_code": "00164779", "corp_name": "SK하이닉스"},
            }
        ),
        encoding="utf-8",
    )
    mock_get = mocker.patch("auto_stock.data.sources.dart_source.requests.get")

    result = resolve_ticker_by_name("sk하이닉스", cache_path=cache_path)

    assert result == [("000660", "SK하이닉스")]
    mock_get.assert_not_called()


def test_resolve_ticker_by_name_falls_back_to_substring_match(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(
        json.dumps(
            {
                "005930": {"corp_code": "00126380", "corp_name": "삼성전자"},
                "006400": {"corp_code": "00126186", "corp_name": "삼성SDI"},
            }
        ),
        encoding="utf-8",
    )
    mocker.patch("auto_stock.data.sources.dart_source.requests.get")

    result = resolve_ticker_by_name("삼성", cache_path=cache_path)

    assert sorted(result) == [("005930", "삼성전자"), ("006400", "삼성SDI")]


def test_resolve_ticker_by_name_exact_match_excludes_broader_substring_hits(mocker, tmp_path):
    """완전일치가 하나라도 있으면 부분일치 후보는 섞지 않는다."""
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(
        json.dumps(
            {
                "005930": {"corp_code": "00126380", "corp_name": "삼성전자"},
                "006400": {"corp_code": "00126186", "corp_name": "삼성전자우"},
            }
        ),
        encoding="utf-8",
    )
    mocker.patch("auto_stock.data.sources.dart_source.requests.get")

    result = resolve_ticker_by_name("삼성전자", cache_path=cache_path)

    assert result == [("005930", "삼성전자")]


def test_resolve_ticker_by_name_returns_empty_list_when_no_match(mocker, tmp_path):
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(
        json.dumps({"005930": {"corp_code": "00126380", "corp_name": "삼성전자"}}), encoding="utf-8"
    )
    mocker.patch("auto_stock.data.sources.dart_source.requests.get")

    result = resolve_ticker_by_name("존재하지않는회사아무거나", cache_path=cache_path)

    assert result == []


def test_resolve_ticker_by_name_excludes_legacy_flat_cache_entries(mocker, tmp_path):
    """이름이 없는 구 스키마(문자열) 캐시 항목은 이름 매칭 대상에서 제외한다."""
    cache_path = tmp_path / "corp_codes.json"
    cache_path.write_text(
        json.dumps(
            {
                "005930": "00126380",  # 구 스키마 — corp_name 없음
                "000660": {"corp_code": "00164779", "corp_name": "SK하이닉스"},
            }
        ),
        encoding="utf-8",
    )
    mocker.patch("auto_stock.data.sources.dart_source.requests.get")

    result = resolve_ticker_by_name("SK하이닉스", cache_path=cache_path)

    assert result == [("000660", "SK하이닉스")]
