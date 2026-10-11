# 뉴스/공시 분석 (#4) — Phase 4 계획: 문서화 + 리뷰

> `docs/design/news-disclosure-plan.md`의 Phase 4를 실행 직전 수준으로 구체화한 계획 문서. Phase 0~3(API 조사, 데이터 수집 계층, 해석 계층, 오케스트레이터 배선)이 이미 구현·테스트 완료된 상태다(`docs/design/news-disclosure-phase1.md`, `docs/design/news-disclosure-phase3-plan.md` 참고, 전체 스위트 258개 통과). 이 문서는 아직 실행되지 않았다 — 계획만 남기고 실행은 별도 착수. `docs/design/llm-chart-analyst-plan.md`/`llm-chart-analyst.md`(#3의 동일 Phase)와 같은 패턴을 그대로 따른다.

## 1. 배경

Phase 0~3이 끝나 있다 — `dart_source.py`(Phase 1), `news_disclosure/` 패키지(Phase 2), `orchestrator/pipeline.py`의 `_news_reasons` 배선 + `scripts/verify_news_disclosure.py`(Phase 3). 아직:

- **리뷰가 안 됐다** — #3(LLM차트분석)이 Phase 4에서 거친 `code-reviewer`+`security-reviewer` 병렬 리뷰를 #4는 아직 받지 않았다.
- **실제 API로 검증된 적이 없다** — `DART_API_KEY` 미발급으로 `scripts/verify_news_disclosure.py`를 한 번도 실행하지 못했다.
- **관련 문서(PRD/IMPLEMENTATION_PLAN/CODEBASE_REVIEW)가 "구현 전" 상태로 남아 있다** — 실제로는 Phase 1~3 코드가 이미 존재하므로 실제 상태와 문서가 어긋나 있다.

## 2. 리뷰 대상 파일

`code-reviewer`+`security-reviewer`에게 넘길 범위 — Phase 1~3에서 새로 만들거나 수정한 파일 전부:

- `src/auto_stock/data/models.py`(`DisclosureRecord` 추가분)
- `src/auto_stock/data/sources/dart_source.py`(신규)
- `src/auto_stock/news_disclosure/`(신규 패키지 전체: `models.py`/`schema.py`/`prompt.py`/`credentials.py`/`client.py`/`analyst.py`)
- `src/auto_stock/orchestrator/pipeline.py`(`_news_reasons` 추가분 + `news_client` 배선)
- `scripts/verify_news_disclosure.py`(신규), `scripts/run_recommendations_with_signals.py`(변경분)
- 대응 테스트: `tests/data/test_dart_source.py`, `tests/news_disclosure/`, `tests/orchestrator/test_pipeline.py`(뉴스 관련 케이스)

## 3. `code-reviewer` + `security-reviewer` 병렬 실행

#3 Phase 4와 동일 패턴 — 두 agent를 **병렬**로 실행한다. `security-reviewer`가 특히 확인해야 할 이 기능 고유의 포인트(#3와 겹치지 않는 부분):

- **두 번째 크리덴셜**: 이 모듈은 `OPENAI_API_KEY` 외에 `DART_API_KEY`도 다룬다 — `data/sources/dart_source.py::_load_dart_api_key()`가 `notifier/credentials.py`/`llm_chart_analyst/credentials.py`와 동일하게 `os.environ["DART_API_KEY"]`(없으면 `KeyError`) 패턴을 따르는지, 에러 메시지·로그에 키 값이 노출되지 않는지.
- **XML 파싱 보안**: `dart_source.py`가 표준 `xml.etree.ElementTree`가 아니라 `defusedxml.ElementTree`를 쓰는지(Phase 1에서 세션 보안 훅 경고로 이미 반영됨 — 리뷰에서 재확인 지점).
- **로컬 캐시 파일**: `data/dart_corp_codes.json`이 `.gitignore`에 실제로 걸려 있는지, 캐시 파일 자체에 민감정보가 들어가지 않는지(공개 기업 매핑 정보이므로 원래 민감하지 않음 — 리뷰에서 확인만).
- **`LLMConfig.api_key`의 `repr=False`**: #3 Phase 4에서 발견된 "dataclass repr이 시크릿을 노출할 수 있다"는 MEDIUM 이슈가 `news_disclosure/models.py::LLMConfig`에도 이미 반영돼 있는지(Phase 2 구현 시점에 처음부터 `field(repr=False)`로 작성됨 — 리뷰에서 재확인).
- **`client.py`의 SDK 예외 매핑**: `llm_chart_analyst/client.py`가 Phase 4에서 놓쳤던 `openai.APIResponseValidationError` catch-all이 `news_disclosure/client.py`에도 처음부터 포함돼 있는지(Phase 2 구현 시점에 이미 반영됨 — 회귀 확인).

`code-reviewer`는 일반 품질 관점(중복 로직, 명명, 테스트 커버리지)을 본다 — 특히 `_ml_reasons`/`_llm_reasons`/`_news_reasons` 3개 헬퍼 간 구조적 유사성이 추출할 만한 공통 로직인지, 아니면 (계획 문서가 이미 명시한) "구조는 같지만 코드는 공유하지 않는다"는 프로젝트 관례를 의도적으로 지킨 것인지 판단.

## 4. 리뷰 후 수정 절차

#3와 동일 — CRITICAL/HIGH는 반드시 수정, MEDIUM은 TDD로 수정, LOW는 비용 대비 판단. 각 수정 후 `pytest -q --cov=src --cov-report=term-missing`로 전체 회귀 확인(현재 258개 통과, 99% 커버리지가 기준선).

## 5. `docs/design/news-disclosure.md` 작성 (구현 결과 기록)

`docs/design/llm-chart-analyst.md`와 동일한 구조로 작성한다:

1. **목적·입출력**: DART 공시 목록 → OpenAI 해석 → 규칙엔진 후보에 대한 보조 근거. ML/LLM차트와 동일하게 자체 후보를 만들지도, 알림을 필터링하지도 않음.
2. **모듈 구조 표**: `dart_source.py`(데이터 수집 계층 확장) + `news_disclosure/`의 6개 파일과 역할 요약, `news-disclosure-plan.md`의 7개 핵심 설계 결정이 코드로 그대로 구현됐음을 확인(중복 서술 없이 참조만).
3. **오케스트레이터 통합**: `_news_reasons`가 DART/LLM 2단계 실패를 독립적으로 격리하는 방식, 3항 None 분기.
4. **`code-reviewer` + `security-reviewer` 리뷰 결과 반영** 섹션: §3~4의 실제 실행 결과(발견 건수·심각도·수정 내역)를 여기 기록.
5. **테스트 현황**: 리뷰 반영 후 최종 테스트 수·커버리지.
6. **남은 작업(사용자 액션 필요)**: `DART_API_KEY` 발급 후 `scripts/verify_news_disclosure.py` dry-run/실호출 검증, `scripts/run_recommendations_with_signals.py`로 3개 신호(ML+LLM차트+뉴스공시) 블록이 텔레그램에 함께 뜨는지 육안 확인.

## 6. 문서 갱신 대상

| 파일 | 변경 내용 |
|---|---|
| `docs/PRD.md` | §5 "뉴스/공시 계층" 문단, §7.2 "뉴스/공시 데이터" 확정 표기 뒤에 구현 완료 사실 반영(다른 계층들이 이미 §4 아키텍처 구체화 블록쿼트로 기록된 것과 같은 패턴) |
| `docs/IMPLEMENTATION_PLAN.md` | 107번째 줄 "3. 뉴스/공시 분석 에이전트 (#4): 계획 완료(구현 전)" → "구현 완료(Phase 0~4 전체)"로 갱신, `docs/design/news-disclosure.md` 참조 추가(#3 항목이 이미 이 패턴으로 갱신돼 있음) |
| `docs/CODEBASE_REVIEW.md` | 177번째 줄 "뉴스/공시 분석(#4) 신호원 — 계획 완료(...), 구현 전"을 "완료" 섹션으로 이동, `src/auto_stock/news_disclosure/` 모듈 표 신규 추가(`llm_chart_analyst/` 표와 같은 형식) |

## 7. 검증 절차

1. `code-reviewer`+`security-reviewer` 병렬 실행 → 발견사항 정리
2. CRITICAL/HIGH 즉시 수정, MEDIUM TDD 수정 → `pytest -q --cov=src --cov-report=term-missing` 회귀 확인
3. `docs/design/news-disclosure.md` 작성
4. PRD/IMPLEMENTATION_PLAN/CODEBASE_REVIEW 갱신
5. (사용자 액션, 이 Phase의 필수 조건 아님) `DART_API_KEY` 발급 후 `scripts/verify_news_disclosure.py` 실행 — 발급 전이라면 이 항목은 "남은 작업"으로 문서에 남기고 Phase 4를 완료 처리해도 무방(#3도 `OPENAI_API_KEY` 실호출 검증을 Phase 4 완료 조건으로 삼지 않고 별도 남은 작업으로 분리한 선례를 따름)

## 8. 이번 문서가 아닌 것

- 실제 `code-reviewer`/`security-reviewer` 실행 및 수정 — 별도 요청 시 착수
- `DART_API_KEY` 실호출 검증 — 사용자의 키 발급이 전제 조건이며, 위 §7처럼 Phase 4 완료와 독립적으로 처리
- Phase 5(빅카인즈 확장) — `news-disclosure-plan.md`가 이미 별도 스코프로 명시
