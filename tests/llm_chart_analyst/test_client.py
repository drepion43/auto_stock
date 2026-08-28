"""client.py는 SDK를 완전히 모킹한다 — 실제 OpenAI API를 호출하지 않는다(conftest의
autouse `OPENAI_API_KEY` monkeypatch가 실수 방지 안전망)."""

import httpx2 as httpx
import openai
import pydantic
import pytest

from auto_stock.llm_chart_analyst.client import LLMChartAnalystError, OpenAIChartClient
from auto_stock.llm_chart_analyst.models import LLMConfig
from auto_stock.llm_chart_analyst.schema import ChartPatternRead

SECRET_API_KEY = "sk-should-never-leak-0123456789"


def _config(api_key: str = SECRET_API_KEY, max_calls_per_run: int = 20) -> LLMConfig:
    return LLMConfig(
        api_key=api_key,
        model="gpt-5.6-luna",
        max_tokens=1024,
        timeout_seconds=30.0,
        max_retries=2,
        max_calls_per_run=max_calls_per_run,
    )


def _fake_response(status_code: int = 503):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return httpx.Response(status_code=status_code, request=request)


def _parsed_response(parsed) -> object:
    class _Response:
        output_parsed = parsed

    return _Response()


def test_read_pattern_calls_responses_parse_with_expected_arguments(mocker):
    mock_openai_cls = mocker.patch("auto_stock.llm_chart_analyst.client.openai.OpenAI")
    parsed = ChartPatternRead(
        direction="UP", confidence="MEDIUM", pattern_name="패턴", rationale="근거", caveat=None
    )
    mock_openai_cls.return_value.responses.parse.return_value = _parsed_response(parsed)

    client = OpenAIChartClient(_config())
    result = client.read_pattern("system prompt text", "user prompt text")

    assert result is parsed
    mock_openai_cls.return_value.responses.parse.assert_called_once_with(
        model="gpt-5.6-luna",
        input=[
            {"role": "system", "content": "system prompt text"},
            {"role": "user", "content": "user prompt text"},
        ],
        text_format=ChartPatternRead,
        max_output_tokens=1024,
    )


def test_openai_client_is_constructed_with_config_values(mocker):
    mock_openai_cls = mocker.patch("auto_stock.llm_chart_analyst.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.parse.return_value = _parsed_response(
        ChartPatternRead(direction="NEUTRAL", confidence="LOW", pattern_name="x", rationale="y", caveat=None)
    )

    OpenAIChartClient(_config(max_calls_per_run=5))

    mock_openai_cls.assert_called_once_with(api_key=SECRET_API_KEY, timeout=30.0, max_retries=2)


@pytest.mark.parametrize(
    ("exc_factory", "expected_fragment"),
    [
        (lambda: openai.APITimeoutError(httpx.Request("POST", "https://x")), "시간 초과"),
        (lambda: openai.APIConnectionError(request=httpx.Request("POST", "https://x")), "연결 실패"),
        (lambda: openai.RateLimitError("rate", response=_fake_response(429), body=None), "레이트리밋"),
        (
            lambda: openai.AuthenticationError("auth", response=_fake_response(401), body=None),
            "인증 실패",
        ),
        (
            lambda: openai.InternalServerError("boom", response=_fake_response(500), body=None),
            "status=500",
        ),
        (lambda: pydantic.ValidationError.from_exception_data("ChartPatternRead", []), "스키마 검증"),
    ],
)
def test_sdk_exceptions_are_mapped_to_llm_chart_analyst_error(mocker, exc_factory, expected_fragment):
    mock_openai_cls = mocker.patch("auto_stock.llm_chart_analyst.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.parse.side_effect = exc_factory()

    client = OpenAIChartClient(_config())

    with pytest.raises(LLMChartAnalystError) as exc_info:
        client.read_pattern("system", "user")

    assert expected_fragment in str(exc_info.value)


def test_none_output_parsed_raises_llm_chart_analyst_error(mocker):
    mock_openai_cls = mocker.patch("auto_stock.llm_chart_analyst.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.parse.return_value = _parsed_response(None)

    client = OpenAIChartClient(_config())

    with pytest.raises(LLMChartAnalystError, match="비어"):
        client.read_pattern("system", "user")


def test_call_budget_blocks_calls_beyond_max_calls_per_run_without_touching_sdk(mocker):
    mock_openai_cls = mocker.patch("auto_stock.llm_chart_analyst.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.parse.return_value = _parsed_response(
        ChartPatternRead(direction="NEUTRAL", confidence="LOW", pattern_name="x", rationale="y", caveat=None)
    )

    client = OpenAIChartClient(_config(max_calls_per_run=2))

    client.read_pattern("s", "u")
    client.read_pattern("s", "u")

    with pytest.raises(LLMChartAnalystError, match="예산"):
        client.read_pattern("s", "u")

    assert mock_openai_cls.return_value.responses.parse.call_count == 2


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: openai.APITimeoutError(httpx.Request("POST", "https://x")),
        lambda: openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
        lambda: openai.RateLimitError("rate", response=_fake_response(429), body=None),
        lambda: openai.AuthenticationError("auth", response=_fake_response(401), body=None),
        lambda: openai.InternalServerError("boom", response=_fake_response(500), body=None),
    ],
)
def test_api_key_never_appears_in_error_messages(mocker, exc_factory):
    mock_openai_cls = mocker.patch("auto_stock.llm_chart_analyst.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.parse.side_effect = exc_factory()

    client = OpenAIChartClient(_config())

    with pytest.raises(LLMChartAnalystError) as exc_info:
        client.read_pattern("system", "user")

    assert SECRET_API_KEY not in str(exc_info.value)


def test_budget_exceeded_error_does_not_mention_api_key(mocker):
    mock_openai_cls = mocker.patch("auto_stock.llm_chart_analyst.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.parse.return_value = _parsed_response(
        ChartPatternRead(direction="NEUTRAL", confidence="LOW", pattern_name="x", rationale="y", caveat=None)
    )
    client = OpenAIChartClient(_config(max_calls_per_run=0))

    with pytest.raises(LLMChartAnalystError) as exc_info:
        client.read_pattern("s", "u")

    assert SECRET_API_KEY not in str(exc_info.value)
    mock_openai_cls.return_value.responses.parse.assert_not_called()
