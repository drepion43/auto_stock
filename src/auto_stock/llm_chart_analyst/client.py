"""`openai` SDK를 import하는 유일한 파일. 모든 SDK 예외를 `LLMChartAnalystError`로 통일한다
(`notifier/telegram_bot.py`가 `TelegramNotificationError`로 통일한 것과 동일한 관례).

포착 순서가 중요하다: `APITimeoutError` ⊂ `APIConnectionError`, `RateLimitError`/
`AuthenticationError` ⊂ `APIStatusError`이므로 반드시 좁은 예외부터 잡는다. 메시지에는
API 키·전체 응답 본문·프롬프트 원문을 절대 포함하지 않는다.
"""

import openai
import pydantic

from auto_stock.llm_chart_analyst.models import LLMConfig
from auto_stock.llm_chart_analyst.schema import ChartPatternRead


class LLMChartAnalystError(Exception):
    pass


class OpenAIChartClient:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self.model = config.model  # ChartPatternReader Protocol 계약 — 감사 추적용
        self._calls_made = 0
        self._client = openai.OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def read_pattern(self, system_prompt: str, user_prompt: str) -> ChartPatternRead:
        if self._calls_made >= self._config.max_calls_per_run:
            raise LLMChartAnalystError(
                f"LLM 호출 예산 초과 (max_calls_per_run={self._config.max_calls_per_run})"
            )
        self._calls_made += 1

        try:
            response = self._client.responses.parse(
                model=self._config.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=ChartPatternRead,
                max_output_tokens=self._config.max_tokens,
            )
        except openai.APITimeoutError as exc:
            raise LLMChartAnalystError(f"LLM 응답 시간 초과({self._config.timeout_seconds}s)") from exc
        except openai.APIConnectionError as exc:
            raise LLMChartAnalystError("LLM 연결 실패") from exc
        except openai.RateLimitError as exc:
            raise LLMChartAnalystError("LLM 레이트리밋 초과") from exc
        except openai.AuthenticationError as exc:
            raise LLMChartAnalystError("LLM 인증 실패 — OPENAI_API_KEY 확인 필요") from exc
        except openai.APIStatusError as exc:
            raise LLMChartAnalystError(f"LLM API 오류(status={exc.status_code})") from exc
        except pydantic.ValidationError as exc:
            raise LLMChartAnalystError("LLM 응답 스키마 검증 실패") from exc

        if response.output_parsed is None:
            raise LLMChartAnalystError("LLM 응답 파싱 결과가 비어 있음")

        return response.output_parsed
