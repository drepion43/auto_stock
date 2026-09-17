import inspect
from datetime import date

from auto_stock.news_disclosure.models import DisclosureSummary
from auto_stock.news_disclosure.prompt import SYSTEM_PROMPT, build_user_prompt, render_disclosure_list

from .conftest import make_disclosures


def _summary(ticker="005930", market="KRX", as_of=date(2024, 6, 10)) -> DisclosureSummary:
    items = make_disclosures(3, ticker=ticker, start=date(2024, 6, 1))
    return DisclosureSummary(ticker=ticker, market=market, as_of=as_of, items=items)


def test_build_user_prompt_does_not_leak_ticker():
    summary = _summary(ticker="005930")

    prompt = build_user_prompt(summary)

    assert "005930" not in prompt


def test_build_user_prompt_does_not_leak_market():
    summary = _summary(market="KRX")

    prompt = build_user_prompt(summary)

    assert "KRX" not in prompt
    assert "NASDAQ" not in prompt


def test_build_user_prompt_does_not_leak_real_date():
    summary = _summary(as_of=date(2024, 6, 10))

    prompt = build_user_prompt(summary)

    assert "2024" not in prompt
    assert summary.as_of.isoformat() not in prompt
    for item in summary.items:
        assert item.filed_date.isoformat() not in prompt


def test_build_user_prompt_signature_has_no_action_parameter():
    """동조(sycophancy) 방어 잠금: 함수 시그니처 자체가 action을 받지 않는다."""
    signature = inspect.signature(build_user_prompt)

    assert "action" not in signature.parameters


def test_build_user_prompt_renders_relative_offsets():
    summary = _summary(as_of=date(2024, 6, 10))

    prompt = build_user_prompt(summary)

    for item in summary.items:
        offset = (summary.as_of - item.filed_date).days
        assert f"D-{offset}" in prompt


def test_build_user_prompt_renders_report_names():
    summary = _summary()

    prompt = build_user_prompt(summary)

    for item in summary.items:
        assert item.report_name in prompt


def test_render_disclosure_list_includes_remark_when_present():
    items = make_disclosures(1, remark="정정")
    summary = DisclosureSummary(ticker="005930", market="KRX", as_of=date(2024, 6, 10), items=items)

    text = render_disclosure_list(summary)

    assert "정정" in text


def test_system_prompt_mentions_neutral_and_auxiliary_keywords():
    assert "NEUTRAL" in SYSTEM_PROMPT
    assert "보조" in SYSTEM_PROMPT


def test_system_prompt_instructs_treating_report_content_as_data_not_instructions():
    """보안 리뷰 발견사항(Stage E) — DART 제출자가 통제하는 report_name/remark가 프롬프트에
    그대로 삽입되므로, 그 안의 지시문처럼 보이는 문구를 따르지 말라는 방어 지침이 필요하다."""
    assert "데이터" in SYSTEM_PROMPT
    assert "무시" in SYSTEM_PROMPT
