"""majestic-waddling-breeze.md "온디맨드 배치 스캔 트리거" 계획 — `ScanCoordinator`는
`chat_cli.py` 시작 시 선제 트리거와 `tool_get_market_scan_recommendations`의 자가치유
트리거 양쪽에서 재사용되는 idempotent 진입점이다. sleep 기반 타이밍은 플레이키해서 쓰지
않는다 — `threading.Event`로 백그라운드 스레드가 정확히 어느 시점에 있는지 결정적으로
제어한다.

market 세분화(2026-10) 이후: `ensure_fresh`가 받는 값은 원자 market(KOSPI/KOSDAQ/
NASDAQ/NYSE)이거나 합성 버킷(KRX/US/SP500/DOW)일 수 있다. 원자 market은 그대로
한 번만 스캔하고(기존 동작), 합성 버킷은 원자 market별로 쪼개 각각 스캔한 뒤 결과를
합쳐 버킷 자신의 키로 캐싱한다."""

from datetime import timedelta
from threading import Event

import pytest

from auto_stock.data.cache import OHLCVCache
from auto_stock.data.scan_cache import MarketScanCache
from auto_stock.explainer.models import Explanation
from auto_stock.orchestrator.scan_coordinator import ScanCoordinator
from auto_stock.risk_sizing.models import AccountState


@pytest.fixture
def cache(tmp_path):
    return OHLCVCache(tmp_path / "cache.duckdb")


@pytest.fixture
def scan_cache(tmp_path):
    return MarketScanCache(tmp_path / "scan.duckdb")


@pytest.fixture
def account():
    return AccountState(equity=10_000_000.0, held_tickers=frozenset(), total_exposure_pct=0.0)


def _coordinator(cache, scan_cache, account, **overrides):
    defaults = dict(stale_after=timedelta(hours=12), universe_size=200)
    defaults.update(overrides)
    return ScanCoordinator(cache=cache, scan_cache=scan_cache, account=account, **defaults)


def test_ensure_fresh_noop_when_not_stale(mocker, cache, scan_cache, account):
    scan_cache.put_scan("KOSPI", [])  # 방금 스캔됨 -> 신선
    coordinator = _coordinator(cache, scan_cache, account, stale_after=timedelta(hours=999))
    mock_scan_market = mocker.patch("auto_stock.orchestrator.scan_coordinator.scan_market")

    thread = coordinator.ensure_fresh("KOSPI")

    assert thread is None
    mock_scan_market.assert_not_called()


def test_ensure_fresh_triggers_scan_when_never_scanned_for_atomic_market(mocker, cache, scan_cache, account):
    """원자 market(KOSPI)은 그대로 한 번만 스캔한다 — 합성 버킷 쪼개기 로직이 생겨도
    원자 market의 기존 동작은 바뀌지 않는다."""
    coordinator = _coordinator(cache, scan_cache, account)
    mocker.patch("auto_stock.orchestrator.scan_coordinator.get_universe", return_value=["005930"])
    explanation = Explanation(ticker="005930", market="KOSPI", action="BUY", summary="RSI 과매도")
    mock_scan_market = mocker.patch(
        "auto_stock.orchestrator.scan_coordinator.scan_market", return_value=([explanation], [])
    )

    thread = coordinator.ensure_fresh("KOSPI")
    assert thread is not None
    thread.join(timeout=5)

    scanned_at, explanations = scan_cache.get_latest("KOSPI")
    assert scanned_at is not None
    assert [e.ticker for e in explanations] == ["005930"]
    mock_scan_market.assert_called_once()
    called_cache = mock_scan_market.call_args.args[0]
    assert isinstance(called_cache, OHLCVCache)
    assert mock_scan_market.call_args.args[1] == ["005930"]
    assert mock_scan_market.call_args.args[2] == "KOSPI"


def test_ensure_fresh_splits_composite_bucket_into_atomic_markets(mocker, cache, scan_cache, account):
    """합성 버킷("KRX"=코스피+코스닥)은 원자 market별로 쪼개 각각 scan_market을
    호출하고, 결과를 합쳐 버킷 자신의 키("KRX")로 캐싱한다."""
    coordinator = _coordinator(cache, scan_cache, account)

    def fake_get_universe(market, as_of):
        return {"KOSPI": ["005930"], "KOSDAQ": ["091990"]}[market]

    mocker.patch("auto_stock.orchestrator.scan_coordinator.get_universe", side_effect=fake_get_universe)

    def fake_scan_market(cache_arg, tickers, market, *args, **kwargs):
        explanation = Explanation(ticker=tickers[0], market=market, action="BUY", summary=f"{market} 신호")
        return [explanation], []

    mock_scan_market = mocker.patch(
        "auto_stock.orchestrator.scan_coordinator.scan_market", side_effect=fake_scan_market
    )

    thread = coordinator.ensure_fresh("KRX")
    assert thread is not None
    thread.join(timeout=5)

    assert mock_scan_market.call_count == 2
    scanned_markets = {c.args[2] for c in mock_scan_market.call_args_list}
    assert scanned_markets == {"KOSPI", "KOSDAQ"}

    scanned_at, explanations = scan_cache.get_latest("KRX")
    assert scanned_at is not None
    assert {e.ticker for e in explanations} == {"005930", "091990"}


def test_ensure_fresh_does_not_spawn_second_thread_while_in_progress(mocker, cache, scan_cache, account):
    started, release = Event(), Event()

    def fake_scan_market(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        return [], []

    mocker.patch("auto_stock.orchestrator.scan_coordinator.get_universe", return_value=[])
    mocker.patch("auto_stock.orchestrator.scan_coordinator.scan_market", side_effect=fake_scan_market)
    coordinator = _coordinator(cache, scan_cache, account)

    t1 = coordinator.ensure_fresh("KOSPI")
    assert started.wait(timeout=5)
    assert coordinator.is_in_progress("KOSPI") is True

    t2 = coordinator.ensure_fresh("KOSPI")
    assert t2 is None  # 이미 in-flight -> 두 번째 스레드 스폰 안 함

    release.set()
    t1.join(timeout=5)
    assert coordinator.is_in_progress("KOSPI") is False


def test_two_different_markets_scan_independently_without_blocking_each_other(mocker, cache, scan_cache, account):
    kospi_started, nasdaq_started = Event(), Event()
    release = Event()

    def fake_scan_market(cache_arg, tickers, market, *args, **kwargs):
        (kospi_started if market == "KOSPI" else nasdaq_started).set()
        release.wait(timeout=5)
        return [], []

    mocker.patch("auto_stock.orchestrator.scan_coordinator.get_universe", return_value=[])
    mocker.patch("auto_stock.orchestrator.scan_coordinator.scan_market", side_effect=fake_scan_market)
    coordinator = _coordinator(cache, scan_cache, account)

    t1 = coordinator.ensure_fresh("KOSPI")
    t2 = coordinator.ensure_fresh("NASDAQ")

    assert kospi_started.wait(timeout=5)
    assert nasdaq_started.wait(timeout=5)  # 서로 기다리지 않고 둘 다 시작됨

    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)


def test_background_scan_failure_is_swallowed_and_leaves_market_retriable(mocker, cache, scan_cache, account, capsys):
    mocker.patch("auto_stock.orchestrator.scan_coordinator.get_universe", return_value=["005930"])
    mocker.patch(
        "auto_stock.orchestrator.scan_coordinator.scan_market", side_effect=RuntimeError("네트워크 오류")
    )
    coordinator = _coordinator(cache, scan_cache, account)

    thread = coordinator.ensure_fresh("KOSPI")
    assert thread is not None
    thread.join(timeout=5)

    assert coordinator.is_in_progress("KOSPI") is False
    assert "KOSPI" in capsys.readouterr().err

    # 실패로 캐시가 여전히 stale이므로 재트리거 가능해야 한다
    mocker.patch("auto_stock.orchestrator.scan_coordinator.scan_market", return_value=([], []))
    retry_thread = coordinator.ensure_fresh("KOSPI")
    assert retry_thread is not None
    retry_thread.join(timeout=5)


def test_run_uses_thread_local_cursor_connections(mocker, cache, scan_cache, account):
    mocker.patch("auto_stock.orchestrator.scan_coordinator.get_universe", return_value=[])
    mocker.patch("auto_stock.orchestrator.scan_coordinator.scan_market", return_value=([], []))
    cache_cursor_spy = mocker.spy(cache, "cursor")
    scan_cache_cursor_spy = mocker.spy(scan_cache, "cursor")
    coordinator = _coordinator(cache, scan_cache, account)

    thread = coordinator.ensure_fresh("KOSPI")
    assert thread is not None
    thread.join(timeout=5)

    cache_cursor_spy.assert_called_once()
    scan_cache_cursor_spy.assert_called_once()
