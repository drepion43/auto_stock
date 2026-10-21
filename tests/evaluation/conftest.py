"""evaluation 테스트 전용 — 유료 호출·외부 전송이 없도록 막는다."""

import os
from types import SimpleNamespace
from typing import Any, Union, get_args, get_origin

import pytest

# deepeval은 import 시점에 텔레메트리를 초기화할 수 있어 모듈 로드 전에 끈다(계획서 §9).
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"


def _fill_value(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (list, tuple, set):
        return []
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _fill_value(args[0]) if args else None
    if annotation is str:
        return "ok"
    if annotation in (int, float):
        return 10
    if annotation is bool:
        return True
    return None


def fill_schema(schema: Any) -> Any:
    """deepeval 내부 pydantic 스키마의 최소 인스턴스 — 가짜 judge 응답용."""
    return schema.model_construct(**{name: _fill_value(f.annotation) for name, f in schema.model_fields.items()})


class FakeResponses:
    def __init__(self, input_tokens: int = 1000, output_tokens: int = 100, fail_parse: bool = False) -> None:
        self.calls: list[dict] = []
        self._usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
        self._fail_parse = fail_parse

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        parsed = None if self._fail_parse else fill_schema(kwargs["text_format"])
        return SimpleNamespace(output_parsed=parsed, usage=self._usage)

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="ok", usage=self._usage)


@pytest.fixture
def fake_responses_cls():
    return FakeResponses


@pytest.fixture
def fake_client():
    return SimpleNamespace(responses=FakeResponses())
