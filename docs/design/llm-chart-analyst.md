# LLM 차트분석 에이전트 (#3) — 구현 결과

> `docs/design/llm-chart-analyst.md`(사전 계획)의 실제 구현 결과 기록. Phase 0~4 전체 완료.

## 목적·입출력

OHLCV + 기술적 지표의 수치 요약을 OpenAI GPT(`gpt-5.6-luna`)에 입력해 차트 패턴을 구조화된 형태로 해석하고, 규칙엔진 후보에 대한 **보조 근거**로만 추가한다. ML예측(#2)과 동일하게 자체 후보를 만들지도, 알림을 필터링하지도 않는다.

## 모듈 구조 (계획 그대로 구현)

| 파일 | 역할 |
|---|---|
| `models.py` | `BarSummary`/`ChartSnapshot`/`ChartAnalysis`/`LLMConfig`/`ChartPatternReader`(Protocol) |
| `schema.py` | `ChartPatternRead` — `responses.parse`의 `text_format` |
| `snapshot.py` | `build_snapshot(records, recent_bars=30)` — 정규화 가격·거래량비 봉 + 지표 |
| `prompt.py` | `SYSTEM_PROMPT`, `build_user_prompt(snapshot)` |
| `credentials.py` | `load_llm_config()` |
| `client.py` | `OpenAIChartClient` — `openai`를 import하는 유일한 파일 |
| `analyst.py` | `analyze(client, records)`, `to_reasons(analysis, action)` |

계획서의 8개 핵심 설계 결정(수치 입력, 환각 방어, 동조 방어, `extra_reasons` 경유, 구조화 출력, 실패 격리, SDK 직접 호출, OpenAI 프로바이더)은 코드로 그대로 구현됐다 — 상세 근거는 `llm-chart-analyst.md` 참고, 여기서는 중복 서술하지 않는다.

## 오케스트레이터 통합

`orchestrator/pipeline.py`에 `llm_client: ChartPatternReader | None = None` 파라미터 추가. `ml_model is None and llm_client is None`일 때만 기존 2-위치인자 `generate_explanation` 호출을 보존(회귀 잠금). `_llm_reasons` 헬퍼는 절대 raise하지 않고 `([], str(exc))`를 반환 — ML 실패와 독립적으로 격리된다.

## code-reviewer + security-reviewer 리뷰 결과 반영 (Phase 4)

CRITICAL/HIGH 없음. MEDIUM 3건 + LOW 2건 전부 TDD로 수정, 191개 테스트로 회귀 확인.

**MEDIUM**
1. **중복 로직 통합** — `ml_predictor/features.py::_volume_ratio`와 `llm_chart_analyst/snapshot.py::_volume_ratios`가 동일한 거래량비 계산을 각자 구현하고 있었다. `rule_engine/indicators.py`에 공용 `volume_ratio()` 함수로 추출해 두 모듈이 재사용하도록 통합 — 두 보조 신호가 같은 날짜에 같은 거래량비 수치를 보게 됐다.
2. **dataclass repr의 시크릿 노출 가능성** — `LLMConfig.api_key`, `TelegramCredentials.bot_token`이 기본 dataclass `__repr__`로 평문 노출될 수 있었다(현재는 실제로 로그/print되는 곳이 없어 미착취 상태였으나, 향후 `logger.debug(f"{config}")`류의 실수 시 노출될 잠재 위험). 두 필드 모두 `field(repr=False)`로 수정.
3. **`openai.APIResponseValidationError` 예외 매핑 누락** — `client.py`의 예외 매핑이 `APIStatusError`/`APIConnectionError` 등은 잡지만 형제 클래스인 `APIResponseValidationError`(SDK 응답이 스키마와 안 맞는 경우)는 누락돼 있었다. `except openai.APIError` catch-all을 `APIStatusError` 다음, `pydantic.ValidationError` 이전에 추가해 "모든 SDK 예외는 `LLMChartAnalystError`로 통일한다"는 계약을 실제로 완성했다.

**LOW**
4. `pyproject.toml`의 `openai` 의존성이 버전 미고정 상태였음 — `openai>=3.5,<4`로 고정(설치된 3.5.0의 예외 계층 가정이 메이저 버전 변경으로 깨지는 것을 방지).
5. `docs/CODEBASE_REVIEW.md`가 "#3 Phase 1~2 완료"/"Phase 3 미착수"라고 쓰면서 바로 아래 문단에서는 Phase 3 완료를 서술하는 자기모순이 있었음 — 헤더와 서술을 실제 상태(Phase 0~3 완료)로 정정.

**추가로 확인만 하고 손대지 않은 것**: 예외 계층(`APITimeoutError`/`RateLimitError`/`AuthenticationError`가 `APIStatusError`의 정확한 서브클래스 관계)과 `responses.parse`/`output_parsed` 실존 여부를 설치된 `openai==3.5.0` 소스 코드에서 직접 재확인 — 계획서의 가정과 일치함을 확인했다(추측이 아닌 검증).

## 테스트 현황

191개 통과(리뷰 전 188개 + 이번 리뷰 반영 3개 신규 테스트). 커버리지 방침 무변경 — `openai.OpenAI`는 전부 모킹, 실제 API 호출 없음.

## 남은 작업 (사용자 액션 필요)

`OPENAI_API_KEY` 발급 후:
1. `scripts/verify_llm_chart_analyst.py` 실호출 모드로 1회 검증
2. `scripts/run_recommendations_with_signals.py`로 ML+LLM 두 블록이 텔레그램에 함께 뜨는지 육안 확인

이 두 가지는 유료 API 호출을 수반해 이번 리뷰 범위에 포함하지 않았다.
