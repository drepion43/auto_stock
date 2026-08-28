import inspect

from auto_stock.llm_chart_analyst.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    render_bar_table,
    render_indicators,
)
from auto_stock.llm_chart_analyst.snapshot import build_snapshot
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS

from .conftest import make_records, random_walk_prices


def _snapshot(ticker="005930", market="KRX"):
    records = make_records(random_walk_prices(80), ticker=ticker, market=market)
    return build_snapshot(records)


def test_build_user_prompt_does_not_leak_ticker():
    snapshot = _snapshot(ticker="005930")

    prompt = build_user_prompt(snapshot)

    assert "005930" not in prompt


def test_build_user_prompt_does_not_leak_market():
    snapshot = _snapshot(market="KRX")

    prompt = build_user_prompt(snapshot)

    assert "KRX" not in prompt
    assert "NASDAQ" not in prompt


def test_build_user_prompt_does_not_leak_real_date():
    snapshot = _snapshot()

    prompt = build_user_prompt(snapshot)

    assert snapshot.date.isoformat() not in prompt
    assert str(snapshot.date.year) not in prompt


def test_build_user_prompt_signature_has_no_action_parameter():
    """동조(sycophancy) 방어 잠금: 함수 시그니처 자체가 action을 받지 않는다."""
    signature = inspect.signature(build_user_prompt)

    assert "action" not in signature.parameters


def test_build_user_prompt_renders_indicator_values():
    snapshot = _snapshot()

    prompt = build_user_prompt(snapshot)
    expected_rsi = f"{snapshot.indicators['rsi_14'] * 100:.1f}"

    assert "RSI" in prompt
    assert expected_rsi in prompt


def test_build_user_prompt_renders_all_bars():
    snapshot = _snapshot()

    prompt = build_user_prompt(snapshot)

    for bar in snapshot.bars:
        assert f"t-{bar.offset}" in prompt


def test_build_user_prompt_horizon_wording_is_linked_to_label_horizon_days():
    snapshot = _snapshot()

    prompt = build_user_prompt(snapshot)

    assert f"약 {LABEL_HORIZON_DAYS}거래일" in prompt


def test_system_prompt_mentions_neutral_and_auxiliary_keywords():
    assert "NEUTRAL" in SYSTEM_PROMPT
    assert "보조" in SYSTEM_PROMPT


def test_render_bar_table_includes_offsets_and_values():
    snapshot = _snapshot()

    table = render_bar_table(snapshot)

    assert "t-0" in table
    assert f"t-{snapshot.bars[0].offset}" in table


def test_render_indicators_includes_rsi_label():
    snapshot = _snapshot()

    text = render_indicators(snapshot)

    assert "RSI" in text
