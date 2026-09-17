"""`openai` SDK를 import하는 유일한 파일. news_disclosure/client.py·llm_chart_analyst/client.py와
동일한 예외 매핑 표를 그대로 복제한다(코드는 공유하지 않음 — news-disclosure-plan.md
핵심 설계 결정 7). 모든 SDK 예외를 `NewsSentimentError`로 통일한다.

포착 순서가 중요하다: `APITimeoutError` ⊂ `APIConnectionError`, `RateLimitError`/
`AuthenticationError` ⊂ `APIStatusError`이므로 반드시 좁은 예외부터 잡는다. 메시지에는
API 키·전체 응답 본문·프롬프트 원문을 절대 포함하지 않는다.
"""

import openai
import pydantic

from auto_stock.news_sentiment.models import LLMConfig
from auto_stock.news_sentiment.schema import NewsSentimentRead


class NewsSentimentError(Exception):
    pass


class OpenAINewsSentimentClient:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self.model = config.model  # NewsSentimentReader Protocol 계약 — 감사 추적용
        self._calls_made = 0
        self._client = openai.OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def read_sentiment(self, system_prompt: str, user_prompt: str) -> NewsSentimentRead:
        if self._calls_made >= self._config.max_calls_per_run:
            raise NewsSentimentError(
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
                text_format=NewsSentimentRead,
                max_output_tokens=self._config.max_tokens,
            )
        except openai.APITimeoutError as exc:
            raise NewsSentimentError(f"LLM 응답 시간 초과({self._config.timeout_seconds}s)") from exc
        except openai.APIConnectionError as exc:
            raise NewsSentimentError("LLM 연결 실패") from exc
        except openai.RateLimitError as exc:
            raise NewsSentimentError("LLM 레이트리밋 초과") from exc
        except openai.AuthenticationError as exc:
            raise NewsSentimentError("LLM 인증 실패 — OPENAI_API_KEY 확인 필요") from exc
        except openai.APIStatusError as exc:
            raise NewsSentimentError(f"LLM API 오류(status={exc.status_code})") from exc
        except openai.APIError as exc:
            # Catch-all for openai.APIError siblings not explicitly named above
            # (e.g. APIResponseValidationError) — keeps the "every SDK exception
            # becomes NewsSentimentError" contract even if the SDK adds new
            # exception subclasses.
            raise NewsSentimentError(f"LLM API 오류: {type(exc).__name__}") from exc
        except pydantic.ValidationError as exc:
            raise NewsSentimentError("LLM 응답 스키마 검증 실패") from exc

        if response.output_parsed is None:
            raise NewsSentimentError("LLM 응답 파싱 결과가 비어 있음")

        return response.output_parsed
