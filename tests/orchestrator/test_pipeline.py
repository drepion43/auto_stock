from datetime import date, timedelta

from auto_stock.explainer.models import Explanation
from auto_stock.notifier.models import TelegramCredentials
from auto_stock.notifier.telegram_bot import TelegramNotificationError
from auto_stock.orchestrator.pipeline import run_recommendation_pipeline
from auto_stock.risk_sizing.models import AccountState, SizingSuggestion
from auto_stock.rule_engine.models import Candidate
from auto_stock.data.models import OHLCVRecord


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


def _credentials():
    return TelegramCredentials(bot_token="123:ABC", chat_id="999")


def _candidate(ticker="005930", market="KRX"):
    return Candidate(ticker=ticker, market=market, action="BUY", reasons=["RSI 과매도"])


def _sizing(ticker="005930", market="KRX"):
    return SizingSuggestion(
        ticker=ticker, market=market, action="BUY",
        suggested_quantity=10, suggested_allocation_pct=0.05,
        stop_loss_price=95.0, take_profit_price=110.0,
        limit_check="PASS", notes=[],
    )


def _explanation(ticker="005930", market="KRX"):
    return Explanation(ticker=ticker, market=market, action="BUY", summary=f"{ticker}: RSI 과매도.")


def test_buy_candidate_flows_through_all_stages_and_lands_in_sent(mocker):
    mocker.patch("auto_stock.orchestrator.pipeline.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.pipeline.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.pipeline.suggest_position", return_value=_sizing())
    mocker.patch("auto_stock.orchestrator.pipeline.generate_explanation", return_value=_explanation())
    mock_send = mocker.patch("auto_stock.orchestrator.pipeline.send_notification")

    result = run_recommendation_pipeline(
        cache=mocker.Mock(), tickers=["005930"], market="KRX",
        account=_account(), credentials=_credentials(),
    )

    assert len(result.sent) == 1
    assert result.sent[0].ticker == "005930"
    assert result.errors == []
    mock_send.assert_called_once()


def test_ticker_with_no_candidates_skips_sizing_explanation_and_notification(mocker):
    mocker.patch("auto_stock.orchestrator.pipeline.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.pipeline.generate_candidates", return_value=[])
    mock_sizing = mocker.patch("auto_stock.orchestrator.pipeline.suggest_position")
    mock_explain = mocker.patch("auto_stock.orchestrator.pipeline.generate_explanation")
    mock_send = mocker.patch("auto_stock.orchestrator.pipeline.send_notification")

    result = run_recommendation_pipeline(
        cache=mocker.Mock(), tickers=["005930"], market="KRX",
        account=_account(), credentials=_credentials(),
    )

    assert result.sent == []
    assert result.errors == []
    mock_sizing.assert_not_called()
    mock_explain.assert_not_called()
    mock_send.assert_not_called()


def test_data_fetch_failure_is_isolated_and_other_tickers_still_processed(mocker):
    def fake_get_ohlcv(cache, ticker, start, end, market):
        if ticker == "BAD":
            raise RuntimeError("fetch failed")
        return _records(ticker=ticker)

    mocker.patch("auto_stock.orchestrator.pipeline.get_ohlcv", side_effect=fake_get_ohlcv)
    mocker.patch(
        "auto_stock.orchestrator.pipeline.generate_candidates",
        side_effect=lambda records: [_candidate(ticker=records[0].ticker)],
    )
    mocker.patch(
        "auto_stock.orchestrator.pipeline.suggest_position",
        side_effect=lambda candidate, records, account: _sizing(ticker=candidate.ticker),
    )
    mocker.patch(
        "auto_stock.orchestrator.pipeline.generate_explanation",
        side_effect=lambda candidate, sizing: _explanation(ticker=candidate.ticker),
    )
    mocker.patch("auto_stock.orchestrator.pipeline.send_notification")

    result = run_recommendation_pipeline(
        cache=mocker.Mock(), tickers=["BAD", "005930"], market="KRX",
        account=_account(), credentials=_credentials(),
    )

    assert len(result.sent) == 1
    assert result.sent[0].ticker == "005930"
    assert len(result.errors) == 1
    assert result.errors[0][0] == "BAD"
    assert "fetch failed" in result.errors[0][1]


def test_notification_failure_is_isolated_and_recorded_as_error(mocker):
    mocker.patch("auto_stock.orchestrator.pipeline.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.pipeline.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.pipeline.suggest_position", return_value=_sizing())
    mocker.patch("auto_stock.orchestrator.pipeline.generate_explanation", return_value=_explanation())
    mocker.patch(
        "auto_stock.orchestrator.pipeline.send_notification",
        side_effect=TelegramNotificationError("텔레그램 요청 실패"),
    )

    result = run_recommendation_pipeline(
        cache=mocker.Mock(), tickers=["005930"], market="KRX",
        account=_account(), credentials=_credentials(),
    )

    assert result.sent == []
    assert len(result.errors) == 1
    assert result.errors[0][0] == "005930"
    assert "텔레그램 요청 실패" in result.errors[0][1]


def test_pipeline_without_ml_model_behaves_identically_to_before(mocker):
    # Regression lock: ml_model defaults to None and must not change call signatures
    # for the existing (unmocked-for-ml) collaborators — this mirrors
    # test_buy_candidate_flows_through_all_stages_and_lands_in_sent exactly, minus
    # explicitly passing ml_model (i.e. relying on the default).
    mocker.patch("auto_stock.orchestrator.pipeline.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.pipeline.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.pipeline.suggest_position", return_value=_sizing())
    mock_explain = mocker.patch(
        "auto_stock.orchestrator.pipeline.generate_explanation",
        side_effect=lambda candidate, sizing: _explanation(ticker=candidate.ticker),
    )
    mock_send = mocker.patch("auto_stock.orchestrator.pipeline.send_notification")
    mock_predict = mocker.patch("auto_stock.orchestrator.pipeline.predict")

    result = run_recommendation_pipeline(
        cache=mocker.Mock(), tickers=["005930"], market="KRX",
        account=_account(), credentials=_credentials(),
    )

    assert len(result.sent) == 1
    assert result.sent[0].ticker == "005930"
    assert result.errors == []
    mock_send.assert_called_once()
    mock_explain.assert_called_once_with(mocker.ANY, mocker.ANY)  # no extra_reasons kwarg at all
    mock_predict.assert_not_called()


def test_pipeline_with_ml_model_passes_extra_reasons_to_explainer(mocker):
    mocker.patch("auto_stock.orchestrator.pipeline.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.pipeline.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.pipeline.suggest_position", return_value=_sizing())
    mock_explain = mocker.patch("auto_stock.orchestrator.pipeline.generate_explanation", return_value=_explanation())
    mock_send = mocker.patch("auto_stock.orchestrator.pipeline.send_notification")
    fake_prediction = mocker.Mock()
    mocker.patch("auto_stock.orchestrator.pipeline.predict", return_value=fake_prediction)
    mocker.patch(
        "auto_stock.orchestrator.pipeline.to_reasons",
        return_value=["ML 모델도 BUY 신호에 동의합니다", "(ML 신호는 백테스트 검증 전 참고용 보조 지표입니다)"],
    )

    result = run_recommendation_pipeline(
        cache=mocker.Mock(), tickers=["005930"], market="KRX",
        account=_account(), credentials=_credentials(), ml_model=mocker.Mock(),
    )

    assert len(result.sent) == 1
    assert result.errors == []
    mock_explain.assert_called_once()
    _, kwargs = mock_explain.call_args
    assert kwargs["extra_reasons"] == [
        "ML 모델도 BUY 신호에 동의합니다",
        "(ML 신호는 백테스트 검증 전 참고용 보조 지표입니다)",
    ]
    mock_send.assert_called_once()


def test_ml_prediction_failure_still_sends_notification_and_records_error(mocker):
    mocker.patch("auto_stock.orchestrator.pipeline.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.pipeline.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.pipeline.suggest_position", return_value=_sizing())
    mock_explain = mocker.patch("auto_stock.orchestrator.pipeline.generate_explanation", return_value=_explanation())
    mock_send = mocker.patch("auto_stock.orchestrator.pipeline.send_notification")
    mocker.patch("auto_stock.orchestrator.pipeline.predict", side_effect=RuntimeError("모델 추론 실패"))

    result = run_recommendation_pipeline(
        cache=mocker.Mock(), tickers=["005930"], market="KRX",
        account=_account(), credentials=_credentials(), ml_model=mocker.Mock(),
    )

    assert len(result.sent) == 1  # notification still sent despite ML failure
    mock_send.assert_called_once()
    assert len(result.errors) == 1
    assert result.errors[0][0] == "005930"
    assert "모델 추론 실패" in result.errors[0][1]
    _, kwargs = mock_explain.call_args
    assert kwargs["extra_reasons"] == []
