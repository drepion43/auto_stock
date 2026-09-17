"""client.py는 SDK를 완전히 모킹한다 — 실제 OpenAI API를 호출하지 않는다(conftest의
autouse `OPENAI_API_KEY` monkeypatch가 실수 방지 안전망).

news_sentiment/test_client.py와 동일 구조이지만, 이 클라이언트는 `responses.parse(
text_format=...)`가 아니라 `responses.create(tools=...)`를 쓰므로 pydantic 구조화출력
경로가 없다 — `pydantic.ValidationError` 매핑은 의도적으로 두지 않는다(도달 불가능한
분기를 만들지 않기 위함, 다른 3개 클라이언트와의 유일한 차이)."""

import httpx2 as httpx
import openai
import pytest

from auto_stock.chat_agent.client import ChatAgentError, OpenAIChatAgentClient
from auto_stock.chat_agent.models import LLMConfig

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


class _FunctionCallItem:
    def __init__(self, name: str, arguments: str, call_id: str):
        self.type = "function_call"
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class _MessageItem:
    def __init__(self):
        self.type = "message"


class _SummaryPart:
    def __init__(self, text: str):
        self.text = text


class _ReasoningItem:
    def __init__(self, summary_texts: list[str]):
        self.type = "reasoning"
        self.summary = [_SummaryPart(text) for text in summary_texts]


def _sdk_response(
    output_text: str = "",
    function_calls=None,
    response_id: str = "resp_1",
    reasoning_summary_parts: list[str] | None = None,
):
    function_calls = function_calls or []

    class _Response:
        pass

    resp = _Response()
    resp.output = [_MessageItem()]
    if reasoning_summary_parts is not None:
        resp.output.append(_ReasoningItem(reasoning_summary_parts))
    resp.output += [
        _FunctionCallItem(fc["name"], fc["arguments"], fc["call_id"]) for fc in function_calls
    ]
    resp.output_text = output_text
    resp.id = response_id
    return resp


def test_create_response_calls_responses_create_with_expected_arguments(mocker):
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.return_value = _sdk_response(output_text="안녕하세요")

    client = OpenAIChatAgentClient(_config())
    result = client.create_response(
        input=[{"role": "user", "content": "삼성전자 지금 어때?"}],
        tools=[{"type": "function", "name": "analyze_rule_engine"}],
        tool_choice="auto",
        previous_response_id=None,
    )

    assert result.output_text == "안녕하세요"
    assert result.function_calls == []
    assert result.response_id == "resp_1"
    assert result.reasoning_summary == ""
    mock_openai_cls.return_value.responses.create.assert_called_once_with(
        model="gpt-5.6-luna",
        input=[{"role": "user", "content": "삼성전자 지금 어때?"}],
        tools=[{"type": "function", "name": "analyze_rule_engine"}],
        tool_choice="auto",
        previous_response_id=None,
        max_output_tokens=1024,
        reasoning={"summary": "auto"},
    )


def test_create_response_extracts_function_calls_and_ignores_message_items(mocker):
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.return_value = _sdk_response(
        function_calls=[
            {"name": "analyze_rule_engine", "arguments": '{"ticker": "005930", "market": "KRX"}', "call_id": "call_1"},
            {"name": "analyze_ml_prediction", "arguments": '{"ticker": "005930", "market": "KRX"}', "call_id": "call_2"},
        ]
    )

    client = OpenAIChatAgentClient(_config())
    result = client.create_response(input="x", tools=[], tool_choice="auto", previous_response_id=None)

    assert [fc.name for fc in result.function_calls] == ["analyze_rule_engine", "analyze_ml_prediction"]
    assert result.function_calls[0].call_id == "call_1"
    assert result.function_calls[0].arguments == '{"ticker": "005930", "market": "KRX"}'


def test_previous_response_id_is_passed_through_for_turn_internal_chaining(mocker):
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.return_value = _sdk_response(response_id="resp_2")

    client = OpenAIChatAgentClient(_config())
    client.create_response(input="x", tools=[], tool_choice="none", previous_response_id="resp_1")

    mock_openai_cls.return_value.responses.create.assert_called_once_with(
        model="gpt-5.6-luna",
        input="x",
        tools=[],
        tool_choice="none",
        previous_response_id="resp_1",
        max_output_tokens=1024,
        reasoning={"summary": "auto"},
    )


def test_create_response_extracts_reasoning_summary_text(mocker):
    """실제 API로 확인함(2026-09-06): gpt-5.6-luna는 reasoning={"summary": "auto"}를
    거부하지 않지만 summary 텍스트가 항상 비어 있는 reasoning 아이템만 돌려준다 —
    그래도 SDK가 언젠가/다른 모델에서 실제 텍스트를 채워 반환할 경우를 위해 추출
    로직 자체는 정확해야 한다."""
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.return_value = _sdk_response(
        reasoning_summary_parts=["티커부터 확정해야 한다", "그다음 규칙엔진을 확인한다"]
    )

    client = OpenAIChatAgentClient(_config())
    result = client.create_response(input="x", tools=[], tool_choice="auto", previous_response_id=None)

    assert result.reasoning_summary == "티커부터 확정해야 한다\n그다음 규칙엔진을 확인한다"


def test_openai_client_is_constructed_with_config_values(mocker):
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.return_value = _sdk_response()

    OpenAIChatAgentClient(_config(max_calls_per_run=5))

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
        (
            lambda: openai.APIResponseValidationError(response=_fake_response(200), body=None),
            "API 오류",
        ),
    ],
)
def test_sdk_exceptions_are_mapped_to_chat_agent_error(mocker, exc_factory, expected_fragment):
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.side_effect = exc_factory()

    client = OpenAIChatAgentClient(_config())

    with pytest.raises(ChatAgentError) as exc_info:
        client.create_response(input="x", tools=[], tool_choice="auto", previous_response_id=None)

    assert expected_fragment in str(exc_info.value)


def test_call_budget_blocks_calls_beyond_max_calls_per_run_without_touching_sdk(mocker):
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.return_value = _sdk_response()

    client = OpenAIChatAgentClient(_config(max_calls_per_run=2))

    client.create_response(input="x", tools=[], tool_choice="auto", previous_response_id=None)
    client.create_response(input="x", tools=[], tool_choice="auto", previous_response_id=None)

    with pytest.raises(ChatAgentError, match="예산"):
        client.create_response(input="x", tools=[], tool_choice="auto", previous_response_id=None)

    assert mock_openai_cls.return_value.responses.create.call_count == 2


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: openai.APITimeoutError(httpx.Request("POST", "https://x")),
        lambda: openai.APIConnectionError(request=httpx.Request("POST", "https://x")),
        lambda: openai.RateLimitError("rate", response=_fake_response(429), body=None),
        lambda: openai.AuthenticationError("auth", response=_fake_response(401), body=None),
        lambda: openai.InternalServerError("boom", response=_fake_response(500), body=None),
        lambda: openai.APIResponseValidationError(response=_fake_response(200), body=None),
    ],
)
def test_api_key_never_appears_in_error_messages(mocker, exc_factory):
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.side_effect = exc_factory()

    client = OpenAIChatAgentClient(_config())

    with pytest.raises(ChatAgentError) as exc_info:
        client.create_response(input="x", tools=[], tool_choice="auto", previous_response_id=None)

    assert SECRET_API_KEY not in str(exc_info.value)


def test_llm_config_repr_does_not_expose_api_key():
    config = _config()

    assert SECRET_API_KEY not in repr(config)
    assert SECRET_API_KEY not in str(config)


def test_budget_exceeded_error_does_not_mention_api_key(mocker):
    mock_openai_cls = mocker.patch("auto_stock.chat_agent.client.openai.OpenAI")
    mock_openai_cls.return_value.responses.create.return_value = _sdk_response()
    client = OpenAIChatAgentClient(_config(max_calls_per_run=0))

    with pytest.raises(ChatAgentError) as exc_info:
        client.create_response(input="x", tools=[], tool_choice="auto", previous_response_id=None)

    assert SECRET_API_KEY not in str(exc_info.value)
    mock_openai_cls.return_value.responses.create.assert_not_called()
