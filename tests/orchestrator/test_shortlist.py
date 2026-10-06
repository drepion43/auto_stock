"""shortlist.py TDD — build_shortlist는 scan_market과 동일한 티커별 실패격리 원칙을
쓰지만, 규칙엔진 OR ML(둘 다 공짜 계산) 중 하나라도 방향성 있는 의견을 내면 숏리스트에
넣는다(recommendation-synthesis-plan.md §2) — 규칙엔진 AND 게이트가 되어 ML만 잡는
종목을 놓치는 걸 막기 위함."""

from datetime import date, timedelta

from auto_stock.data.models import OHLCVRecord
from auto_stock.ml_predictor.models import MLPrediction
from auto_stock.orchestrator.shortlist import ShortlistEntry, build_shortlist
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


def _candidate(ticker="005930", market="KRX"):
    return Candidate(ticker=ticker, market=market, action="BUY", reasons=["RSI 과매도"])


def _prediction(ticker="005930", market="KRX", probability_up=0.5):
    return MLPrediction(
        ticker=ticker, market=market, date=date(2026, 1, 2),
        probability_up=probability_up, top_features=[],
    )


def test_rule_signal_only_is_included(mocker):
    mocker.patch("auto_stock.orchestrator.shortlist.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.shortlist.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.shortlist.predict", return_value=_prediction(probability_up=0.5))

    shortlist, errors = build_shortlist(
        cache=mocker.Mock(), tickers=["005930"], market="KRX", ml_model=mocker.Mock()
    )

    assert errors == []
    assert shortlist == [
        ShortlistEntry(
            ticker="005930", market="KRX",
            rule_candidate=_candidate(), ml_prediction=_prediction(probability_up=0.5),
        )
    ]


def test_ml_signal_only_is_included_even_without_rule_candidate(mocker):
    """규칙엔진이 후보를 못 냈어도 ML이 방향성 있는 의견(상승확률 >= 0.55)을 내면
    숏리스트에 들어가야 한다 — AND 게이트 방지가 이 모듈의 핵심 목적."""
    mocker.patch("auto_stock.orchestrator.shortlist.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.shortlist.generate_candidates", return_value=[])
    mocker.patch("auto_stock.orchestrator.shortlist.predict", return_value=_prediction(probability_up=0.7))

    shortlist, errors = build_shortlist(
        cache=mocker.Mock(), tickers=["005930"], market="KRX", ml_model=mocker.Mock()
    )

    assert errors == []
    assert len(shortlist) == 1
    assert shortlist[0].rule_candidate is None
    assert shortlist[0].ml_prediction.probability_up == 0.7


def test_both_signals_present_included_once(mocker):
    mocker.patch("auto_stock.orchestrator.shortlist.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.shortlist.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.shortlist.predict", return_value=_prediction(probability_up=0.1))

    shortlist, errors = build_shortlist(
        cache=mocker.Mock(), tickers=["005930"], market="KRX", ml_model=mocker.Mock()
    )

    assert errors == []
    assert len(shortlist) == 1


def test_neither_signal_is_excluded(mocker):
    """규칙엔진 후보 없음 + ML이 중립 구간(0.45~0.55)이면 숏리스트에서 빠져야 비용이
    통제된다."""
    mocker.patch("auto_stock.orchestrator.shortlist.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.shortlist.generate_candidates", return_value=[])
    mocker.patch("auto_stock.orchestrator.shortlist.predict", return_value=_prediction(probability_up=0.5))

    shortlist, errors = build_shortlist(
        cache=mocker.Mock(), tickers=["005930"], market="KRX", ml_model=mocker.Mock()
    )

    assert shortlist == []
    assert errors == []


def test_ml_prediction_none_is_treated_as_no_opinion_not_error(mocker):
    """predict()가 데이터 부족으로 None을 반환하는 건(ml_predictor 기존 관례) 에러가
    아니라 "ML 의견 없음"으로 취급해야 한다."""
    mocker.patch("auto_stock.orchestrator.shortlist.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.shortlist.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.shortlist.predict", return_value=None)

    shortlist, errors = build_shortlist(
        cache=mocker.Mock(), tickers=["005930"], market="KRX", ml_model=mocker.Mock()
    )

    assert errors == []
    assert shortlist[0].ml_prediction is None
    assert shortlist[0].rule_candidate == _candidate()


def test_ml_model_none_falls_back_to_rule_engine_only(mocker):
    """market에 학습된 모델이 없으면(ml_model=None) predict를 호출하지 않고 규칙엔진
    신호만으로 숏리스트를 구성해야 한다 — tool_analyze_ml_prediction과 달리 여기선
    에러가 아니라 조용한 저하다(배치 스크리닝이라 사용자에게 보여줄 응답이 없음)."""
    mocker.patch("auto_stock.orchestrator.shortlist.get_ohlcv", return_value=_records())
    mocker.patch("auto_stock.orchestrator.shortlist.generate_candidates", return_value=[_candidate()])
    mock_predict = mocker.patch("auto_stock.orchestrator.shortlist.predict")

    shortlist, errors = build_shortlist(
        cache=mocker.Mock(), tickers=["005930"], market="KRX", ml_model=None
    )

    assert errors == []
    assert shortlist[0].ml_prediction is None
    mock_predict.assert_not_called()


def test_one_ticker_failure_does_not_block_the_rest(mocker):
    def fake_get_ohlcv(cache, ticker, start, end, market):
        if ticker == "000660":
            raise RuntimeError("데이터 소스 오류")
        return _records(ticker=ticker)

    mocker.patch("auto_stock.orchestrator.shortlist.get_ohlcv", side_effect=fake_get_ohlcv)
    mocker.patch("auto_stock.orchestrator.shortlist.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.shortlist.predict", return_value=_prediction(probability_up=0.5))

    shortlist, errors = build_shortlist(
        cache=mocker.Mock(), tickers=["005930", "000660"], market="KRX", ml_model=mocker.Mock()
    )

    assert [e.ticker for e in shortlist] == ["005930"]
    assert len(errors) == 1
    assert errors[0][0] == "000660"


def test_fetches_ohlcv_only_once_per_ticker(mocker):
    """규칙엔진·ML 둘 다 같은 records를 쓰므로 get_ohlcv를 종목당 2번 부르면 낭비다."""
    mock_get_ohlcv = mocker.patch(
        "auto_stock.orchestrator.shortlist.get_ohlcv", return_value=_records()
    )
    mocker.patch("auto_stock.orchestrator.shortlist.generate_candidates", return_value=[_candidate()])
    mocker.patch("auto_stock.orchestrator.shortlist.predict", return_value=_prediction(probability_up=0.5))

    build_shortlist(cache=mocker.Mock(), tickers=["005930"], market="KRX", ml_model=mocker.Mock())

    assert mock_get_ohlcv.call_count == 1
