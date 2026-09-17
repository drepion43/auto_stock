from datetime import date, timedelta

from auto_stock.data.models import OHLCVRecord
from auto_stock.explainer.models import Explanation
from auto_stock.orchestrator.market_scan import scan_market
from auto_stock.risk_sizing.models import AccountState, SizingSuggestion
from auto_stock.rule_engine.models import Candidate


def _records(ticker="005930", market="KRX", n=2):
    start = date(2026, 1, 1)
    return [
        OHLCVRecord(
            ticker=ticker, market=market, date=start + timedelta(days=i),
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000,
        )
        for i in range(n)
    ]


def _account():
    return AccountState(equity=10_000_000.0, held_tickers=frozenset(), total_exposure_pct=0.0)


def _candidate(ticker="005930", market="KRX"):
    return Candidate(ticker=ticker, market=market, action="BUY", reasons=["RSI 과매도"])


def _sizing(ticker="005930", market="KRX"):
    return SizingSuggestion(
        ticker=ticker, market=market, action="BUY",
        suggested_quantity=10, suggested_allocation_pct=0.05,
        stop_loss_price=95.0, take_profit_price=110.0, reference_price=100.0,
        limit_check="PASS", notes=[],
    )


def _explanation(ticker="005930", market="KRX"):
    return Explanation(ticker=ticker, market=market, action="BUY", summary=f"{ticker}: RSI 과매도.")


def test_buy_candidate_produces_explanation_without_sending_notification(mocker):
    mocker.patch("auto_stock.orchestrator.market_scan.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.market_scan.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.market_scan.suggest_position", return_value=_sizing())
    mocker.patch("auto_stock.orchestrator.market_scan.generate_explanation", return_value=_explanation())

    explanations, errors = scan_market(cache=mocker.Mock(), tickers=["005930"], market="KRX", account=_account())

    assert explanations == [_explanation()]
    assert errors == []


def test_ticker_with_no_candidates_is_skipped(mocker):
    mocker.patch("auto_stock.orchestrator.market_scan.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.market_scan.generate_candidates", return_value=[])
    mock_sizing = mocker.patch("auto_stock.orchestrator.market_scan.suggest_position")

    explanations, errors = scan_market(cache=mocker.Mock(), tickers=["005930"], market="KRX", account=_account())

    assert explanations == []
    assert errors == []
    mock_sizing.assert_not_called()


def test_one_ticker_failure_does_not_block_the_rest(mocker):
    def fake_get_ohlcv(cache, ticker, start, end, market):
        if ticker == "000660":
            raise RuntimeError("데이터 소스 오류")
        return _records(ticker=ticker)

    mocker.patch("auto_stock.orchestrator.market_scan.get_ohlcv", side_effect=fake_get_ohlcv)
    mocker.patch("auto_stock.orchestrator.market_scan.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.market_scan.suggest_position", return_value=_sizing())
    mocker.patch("auto_stock.orchestrator.market_scan.generate_explanation", return_value=_explanation())

    explanations, errors = scan_market(
        cache=mocker.Mock(), tickers=["005930", "000660"], market="KRX", account=_account()
    )

    assert [e.ticker for e in explanations] == ["005930"]
    assert len(errors) == 1
    assert errors[0][0] == "000660"


def test_scan_market_never_touches_telegram(mocker):
    """텔레그램 발송은 orchestrator.pipeline의 몫이다 — 이 배치 스캔은 캐시 적재 전용이라
    market_scan.py 안에 send_notification을 참조하는 코드가 전혀 없어야 한다."""
    mocker.patch("auto_stock.orchestrator.market_scan.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.market_scan.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.market_scan.suggest_position", return_value=_sizing())
    mocker.patch("auto_stock.orchestrator.market_scan.generate_explanation", return_value=_explanation())

    import auto_stock.orchestrator.market_scan as market_scan_module

    assert not hasattr(market_scan_module, "send_notification")

    scan_market(cache=mocker.Mock(), tickers=["005930"], market="KRX", account=_account())
