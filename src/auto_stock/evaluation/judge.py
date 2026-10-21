"""비용 상한을 강제하는 채점 모델 래퍼 `BudgetedOpenAIJudge` — 계획서 §6, §7.3.

DeepEval 지표 객체에서 비용을 읽는 공식 방법을 확인하지 못해(2026-10-20) 직접 센다. 매 호출 **전에**
"누적 + 이번 호출 최대 예상 비용"으로 상한을 검사하고(넘으면 `JudgeBudgetExceeded`, API 미호출),
호출 뒤에는 응답의 실제 `usage` 토큰으로 비용을 더한다.

- 최대 예상 입력 토큰은 UTF-8 바이트 수 / 2로 잡는다(한글 1.5·영문 0.5 토큰/글자 상당 — 보수적이지만
  보장은 아니다). 실제 사용량은 호출 뒤 더하므로 최악의 경우 상한을 호출 1회분 넘을 수 있다.
- API 호출이 예외로 끝나도(파싱 실패 등) 과금됐을 수 있어 최대 예상 비용을 기록한다.
- 지표는 `async_mode=False`로 돌려 누적이 순서대로 되게 한다(`judged_metrics`). `a_generate`도
  동기 호출로 위임한다 — 병렬로 상한을 넘겨 끝나는 일을 막는다."""

import math
import os
from collections.abc import Mapping
from typing import Any

from deepeval.models import DeepEvalBaseLLM

from auto_stock.evaluation.budget import CostLedger
from auto_stock.evaluation.pricing import cost_usd, price_for

# 계획서 §6 — 사용자 결정(2026-10-22)으로 평가 대상과 같은 luna를 쓴다. 같은 모델이 자기 답을 채점하면
# 관대해질 수 있어(self-preference) 사람 보정(§8.2)·변이 탐지율(§8.4)로 편향을 측정한다.
DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"
DEFAULT_MAX_OUTPUT_TOKENS = 4096  # 추론 토큰 포함 — 사전 비용 검사에서 그대로 상한으로 쓴다
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2
_PROMPT_OVERHEAD_TOKENS = 200  # 메시지 포맷 등 프롬프트 밖 입력 토큰 여유
_BYTES_PER_TOKEN_LOWER = 2  # 토큰 1개가 최소 이만큼의 UTF-8 바이트를 덮는다고 보는 보수적 가정


class JudgeOutputError(RuntimeError):
    """구조화 출력 파싱 결과가 비어 있음."""


class BudgetedOpenAIJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        ledger: CostLedger,
        model_name: str = DEFAULT_JUDGE_MODEL,
        client: Any = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        env: Mapping[str, str] | None = None,
    ) -> None:
        price_for(model_name, env)  # 단가를 모르면 상한을 강제할 수 없다 — 생성 시점에 실패
        self._ledger = ledger
        self._model_name = model_name
        self._client = client
        self._max_output_tokens = max_output_tokens
        self._env = env
        super().__init__(model_name)

    def load_model(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"], timeout=DEFAULT_TIMEOUT_SECONDS, max_retries=DEFAULT_MAX_RETRIES
            )
        return self._client

    def get_model_name(self) -> str:
        return self._model_name

    def estimate_max_cost(self, prompt: str) -> float:
        input_upper = math.ceil(len(prompt.encode("utf-8")) / _BYTES_PER_TOKEN_LOWER) + _PROMPT_OVERHEAD_TOKENS
        return cost_usd(self._model_name, input_upper, self._max_output_tokens, self._env)

    def generate(self, prompt: str, schema: Any = None) -> Any:
        max_cost = self.estimate_max_cost(prompt)
        self._ledger.check_judge(max_cost)
        common = {"model": self._model_name, "input": prompt, "max_output_tokens": self._max_output_tokens}
        try:
            if schema is not None:
                response = self._client.responses.parse(text_format=schema, **common)
            else:
                response = self._client.responses.create(**common)
        except Exception:
            # 실패한 호출도 과금됐을 수 있다(예: 출력 길이 초과 뒤 파싱 실패) — 최대 예상 비용으로 기록한다.
            self._ledger.add_judge(max_cost)
            raise
        usage = getattr(response, "usage", None)
        cost = cost_usd(self._model_name, usage.input_tokens, usage.output_tokens, self._env) if usage else max_cost
        self._ledger.add_judge(cost)
        if schema is None:
            return response.output_text
        if response.output_parsed is None:
            raise JudgeOutputError("judge returned no parsed output")
        return response.output_parsed

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        return self.generate(prompt, schema)
