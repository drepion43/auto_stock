"""`openai` SDK를 import하는 유일한 파일. news_disclosure/client.py·news_sentiment/client.py와
동일한 예외 매핑 표를 그대로 복제한다(코드는 공유하지 않음 — news-disclosure-plan.md
핵심 설계 결정 7). 모든 SDK 예외를 `ChatAgentError`로 통일한다.

다른 3개 클라이언트와의 유일한 차이: 이 클라이언트는 `responses.parse(text_format=...)`가
아니라 `responses.create(tools=...)`를 쓴다 — 구조화 출력(pydantic 파싱) 경로가 아예 없으므로
`pydantic.ValidationError` 매핑은 의도적으로 두지 않는다(도달 불가능한 except 분기를 만들지
않기 위함).

포착 순서가 중요하다: `APITimeoutError` ⊂ `APIConnectionError`, `RateLimitError`/
`AuthenticationError` ⊂ `APIStatusError`이므로 반드시 좁은 예외부터 잡는다. 메시지에는
API 키·전체 응답 본문·프롬프트 원문을 절대 포함하지 않는다.
"""

import openai

from auto_stock.chat_agent.models import ChatModelResponse, FunctionCall, LLMConfig


class ChatAgentError(Exception):
    pass


class OpenAIChatAgentClient:
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self.model = config.model  # ChatAgentReader Protocol 계약 — 감사 추적용
        self._calls_made = 0
        self._client = openai.OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def create_response(
        self,
        input: list[dict] | str,
        tools: list[dict],
        tool_choice: str,
        previous_response_id: str | None,
    ) -> ChatModelResponse:
        if self._calls_made >= self._config.max_calls_per_run:
            raise ChatAgentError(
                f"LLM 호출 예산 초과 (max_calls_per_run={self._config.max_calls_per_run})"
            )
        self._calls_made += 1

        try:
            response = self._client.responses.create(
                model=self._config.model,
                input=input,
                tools=tools,
                tool_choice=tool_choice,
                previous_response_id=previous_response_id,
                max_output_tokens=self._config.max_tokens,
                # 사용자 CLI에 tool 호출 사유를 보여주기 위한 요청 — 실제 API로 확인함
                # (2026-09-06): 지원 안 하는 모델이어도 이 파라미터 자체 때문에 에러가
                # 나지는 않는다(gpt-5.6-luna는 summary가 항상 빈 reasoning 아이템만
                # 돌려줌 — 아래 reasoning_summary 추출 로직 참고).
                reasoning={"summary": "auto"},
            )
        except openai.APITimeoutError as exc:
            raise ChatAgentError(f"LLM 응답 시간 초과({self._config.timeout_seconds}s)") from exc
        except openai.APIConnectionError as exc:
            raise ChatAgentError("LLM 연결 실패") from exc
        except openai.RateLimitError as exc:
            raise ChatAgentError("LLM 레이트리밋 초과") from exc
        except openai.AuthenticationError as exc:
            raise ChatAgentError("LLM 인증 실패 — OPENAI_API_KEY 확인 필요") from exc
        except openai.APIStatusError as exc:
            raise ChatAgentError(f"LLM API 오류(status={exc.status_code})") from exc
        except openai.APIError as exc:
            # Catch-all for openai.APIError siblings not explicitly named above
            # (e.g. APIResponseValidationError) — keeps the "every SDK exception
            # becomes ChatAgentError" contract even if the SDK adds new
            # exception subclasses.
            raise ChatAgentError(f"LLM API 오류: {type(exc).__name__}") from exc

        function_calls = [
            FunctionCall(name=item.name, arguments=item.arguments, call_id=item.call_id)
            for item in response.output
            if item.type == "function_call"
        ]
        reasoning_summary = "\n".join(
            part.text
            for item in response.output
            if item.type == "reasoning"
            for part in item.summary
        )
        return ChatModelResponse(
            output_text=response.output_text,
            function_calls=function_calls,
            response_id=response.id,
            reasoning_summary=reasoning_summary,
        )
