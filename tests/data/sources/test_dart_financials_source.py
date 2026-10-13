"""dart_financials_source.py 테스트 — requests를 모킹해 실제 DART API를 호출하지 않는다."""

from datetime import date

import pytest

from auto_stock.data.sources.dart_financials_source import (
    DartFinancialsError,
    fetch_latest_revenue,
)


@pytest.fixture(autouse=True)
def _dummy_dart_api_key(monkeypatch, mocker):
    mocker.patch("auto_stock.data.sources.dart_financials_source.load_dotenv")
    monkeypatch.setenv("DART_API_KEY", "dummy-dart-key-not-real")


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def _ok_payload(items: list[dict]) -> dict:
    return {"status": "000", "message": "정상", "list": items}


def _no_data_payload() -> dict:
    return {"status": "013", "message": "조회된 데이타가 없습니다."}


def _account(name: str, amount: str) -> dict:
    return {"account_nm": name, "thstrm_amount": amount}


def test_fetch_latest_revenue_matches_primary_account_name(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_ok_payload([_account("매출액", "279,604,799,000,000")])),
    )

    revenue = fetch_latest_revenue("00126380", date(2026, 6, 1), cache_path=cache_path)

    assert revenue == 279_604_799_000_000


def test_fetch_latest_revenue_falls_back_to_operating_revenue(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_ok_payload([_account("영업수익", "1,000,000,000")])),
    )

    revenue = fetch_latest_revenue("00164779", date(2026, 6, 1), cache_path=cache_path)

    assert revenue == 1_000_000_000


def test_fetch_latest_revenue_falls_back_to_interest_income(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_ok_payload([_account("이자수익", "500,000,000")])),
    )

    revenue = fetch_latest_revenue("00111111", date(2026, 6, 1), cache_path=cache_path)

    assert revenue == 500_000_000


def test_fetch_latest_revenue_returns_none_when_no_account_matches(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mock_get = mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_ok_payload([_account("자산총계", "1,000,000")])),
    )

    revenue = fetch_latest_revenue("00999999", date(2026, 6, 1), cache_path=cache_path)

    assert revenue is None
    # CFS에서 못 찾았으니 OFS로 1회 재시도해야 한다.
    assert mock_get.call_count == 2


def test_fetch_latest_revenue_returns_none_on_status_no_data(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_no_data_payload()),
    )

    revenue = fetch_latest_revenue("00888888", date(2026, 6, 1), cache_path=cache_path)

    assert revenue is None


def test_fetch_latest_revenue_raises_on_other_error_status(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse({"status": "020", "message": "요청 제한을 초과하였습니다."}),
    )

    with pytest.raises(DartFinancialsError, match="020"):
        fetch_latest_revenue("00888888", date(2026, 6, 1), cache_path=cache_path)


def test_fetch_latest_revenue_tries_ofs_when_cfs_has_no_match(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    responses = [
        _FakeResponse(_ok_payload([_account("자산총계", "1,000,000")])),  # CFS: no match
        _FakeResponse(_ok_payload([_account("매출액", "2,000,000")])),  # OFS: matches
    ]
    mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        side_effect=responses,
    )

    revenue = fetch_latest_revenue("00777777", date(2026, 6, 1), cache_path=cache_path)

    assert revenue == 2_000_000


def test_fetch_latest_revenue_caches_result(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mock_get = mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_ok_payload([_account("매출액", "100")])),
    )

    first = fetch_latest_revenue("00126380", date(2026, 6, 1), cache_path=cache_path)
    second = fetch_latest_revenue("00126380", date(2026, 6, 1), cache_path=cache_path)

    assert first == second == 100
    assert mock_get.call_count == 1  # 두 번째 호출은 캐시에서 읽어 API를 다시 안 부른다


def test_fetch_latest_revenue_does_not_cache_none_result(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mock_get = mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_no_data_payload()),
    )

    fetch_latest_revenue("00999999", date(2026, 6, 1), cache_path=cache_path)
    fetch_latest_revenue("00999999", date(2026, 6, 1), cache_path=cache_path)

    assert mock_get.call_count == 4  # CFS+OFS 두 번 호출을 두 번 다 — 캐시 안 됐으므로


def test_fetch_latest_revenue_uses_prior_year_before_april(mocker, tmp_path):
    """4월 이전이면 전년도 사업보고서도 아직 공시 전일 수 있어 전전년도를 쓴다(lookahead 방어)."""
    cache_path = tmp_path / "dart_revenue_cache.json"
    mock_get = mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_ok_payload([_account("매출액", "100")])),
    )

    fetch_latest_revenue("00126380", date(2026, 2, 1), cache_path=cache_path)

    called_params = mock_get.call_args.kwargs["params"]
    assert called_params["bsns_year"] == "2024"


def test_fetch_latest_revenue_uses_prior_year_after_april(mocker, tmp_path):
    cache_path = tmp_path / "dart_revenue_cache.json"
    mock_get = mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        return_value=_FakeResponse(_ok_payload([_account("매출액", "100")])),
    )

    fetch_latest_revenue("00126380", date(2026, 6, 1), cache_path=cache_path)

    called_params = mock_get.call_args.kwargs["params"]
    assert called_params["bsns_year"] == "2025"


def test_fetch_latest_revenue_wraps_request_exception_without_leaking_url(mocker, tmp_path):
    import requests

    cache_path = tmp_path / "dart_revenue_cache.json"
    mocker.patch(
        "auto_stock.data.sources.dart_financials_source.requests.get",
        side_effect=requests.exceptions.ConnectionError("http://leaked?crtfc_key=secret"),
    )

    with pytest.raises(DartFinancialsError) as exc_info:
        fetch_latest_revenue("00126380", date(2026, 6, 1), cache_path=cache_path)

    assert "secret" not in str(exc_info.value)
    assert "ConnectionError" in str(exc_info.value)
