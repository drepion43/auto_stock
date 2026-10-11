# 뉴스/공시 분석 (#4) — Phase 3 계획: 오케스트레이터 배선 + 실제 API 최초 호출

> `docs/design/news-disclosure-plan.md`의 Phase 3을 실행 직전 수준으로 구체화한 계획 문서. Phase 0~2(API 조사, 데이터 수집 계층 `dart_source.py`, 해석 계층 `news_disclosure/`)는 이미 구현·테스트 완료된 상태(`docs/design/news-disclosure-phase1.md` 참고, 전체 스위트 252개 통과)이며, 이 문서는 아직 구현되지 않았다 — 계획만 남기고 실행은 별도 착수.

## 1. 배경

Phase 0~2가 끝나 있다:

- Phase 1: `data/models.py::DisclosureRecord`, `data/sources/dart_source.py::resolve_corp_code`/`fetch_disclosures`
- Phase 2: `news_disclosure/` 패키지(`models.py`/`schema.py`/`prompt.py`/`credentials.py`/`client.py`/`analyst.py`) — `analyze()`/`to_reasons()`가 `llm_chart_analyst`(#3)와 동일한 계약으로 완성됨

아직 이 둘은 `orchestrator/pipeline.py`에 배선되지 않았고, DART API도 실제로 호출된 적이 없다(`DART_API_KEY` 미발급). Phase 3은 이 배선과 최초 실호출을 완료하는 단계다. 이 문서는 그 실행에 들어가기 전 설계를 확정해 둔다.

## 2. `_news_reasons` 헬퍼 설계

`pipeline.py`의 `_ml_reasons`/`_llm_reasons`와 동일한 계약(절대 raise하지 않고 `(reasons, error)` 튜플 반환)을 따르되, 이 신호원만 유일하게 **두 단계 외부 조회**(DART → OpenAI)가 순차로 들어간다:

```python
from datetime import date, timedelta

from auto_stock.data.sources.dart_source import fetch_disclosures, resolve_corp_code
from auto_stock.news_disclosure.analyst import analyze as analyze_disclosures
from auto_stock.news_disclosure.analyst import to_reasons as to_disclosure_reasons
from auto_stock.news_disclosure.credentials import DISCLOSURE_LOOKBACK_DAYS
from auto_stock.news_disclosure.models import DisclosureReader


def _news_reasons(
    news_client: DisclosureReader | None, ticker: str, market: str, action: str
) -> tuple[list[str], str | None]:
    """절대 raise하지 않는다 — DART 조회든 LLM 해석이든 실패가 추천 발송을 막아서는 안 된다."""
    if news_client is None:
        return [], None
    try:
        corp_code = resolve_corp_code(ticker)
        if corp_code is None:
            return [], None  # DART 미등록 종목(나스닥 등) — 에러 아님, 조용히 스킵
        end = date.today()
        start = end - timedelta(days=DISCLOSURE_LOOKBACK_DAYS)
        disclosures = fetch_disclosures(corp_code, start, end)
    except Exception as exc:
        return [], f"DART 조회 실패: {exc}"

    try:
        analysis = analyze_disclosures(news_client, disclosures, ticker, market, end)
        return to_disclosure_reasons(analysis, action), None
    except Exception as exc:
        return [], f"LLM 공시해석 실패: {exc}"
```

**DART 실패와 LLM 실패를 구분 표기하기로 결정** (`news-disclosure-plan.md`가 미확정으로 남겨뒀던 항목) — `_ml_reasons`/`_llm_reasons`가 이미 `f"ML 예측 실패: {exc}"`/`f"LLM 차트분석 실패: {exc}"` 형태로 신호원별 접두사를 붙이고 있어(`pipeline.py` 100-103번째 줄), 뉴스/공시만 단일 `except Exception`으로 뭉뚱그리면 에러 로그에서 원인 파악이 더 어려워진다. 두 개의 `try/except`로 분리하는 비용은 작고, 신호원 3개가 됐을 때 어느 단계가 실패했는지 즉시 구분되는 이점이 크다.

## 3. `run_recommendation_pipeline` 시그니처 변경

```python
def run_recommendation_pipeline(
    cache: OHLCVCache,
    tickers: list[str],
    market: str,
    account: AccountState,
    credentials: TelegramCredentials,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ml_model: ModelBundle | None = None,
    llm_client: ChartPatternReader | None = None,
    news_client: DisclosureReader | None = None,   # 신규
) -> PipelineResult:
    ...
    if ml_model is None and llm_client is None and news_client is None:
        # 회귀 잠금 그대로 — 셋 다 None일 때만 2-위치인자 호출 보존
        explanation = generate_explanation(candidate, sizing)
    else:
        extra_reasons: list[str] = []
        ml_r, ml_error = _ml_reasons(ml_model, records, candidate.action)
        extra_reasons.extend(ml_r)
        llm_r, llm_error = _llm_reasons(llm_client, records, candidate.action)
        extra_reasons.extend(llm_r)
        news_r, news_error = _news_reasons(news_client, candidate.ticker, candidate.market, candidate.action)
        extra_reasons.extend(news_r)

        explanation = generate_explanation(candidate, sizing, extra_reasons=extra_reasons)

        if ml_error is not None:
            errors.append((ticker, f"ML 예측 실패: {ml_error}"))
        if llm_error is not None:
            errors.append((ticker, f"LLM 차트분석 실패: {llm_error}"))
        if news_error is not None:
            errors.append((ticker, news_error))   # 이미 "DART 조회 실패: "/"LLM 공시해석 실패: " 접두사 포함
```

**변경 지점 요약**:
- 분기 조건에 `and news_client is None` 추가(2항 → 3항).
- `extra_reasons` 병합 순서는 ML → LLM차트 → 뉴스/공시 — `news-disclosure-plan.md` 핵심 설계 결정 4가 이미 못박은 순서 그대로.
- `_news_reasons`가 반환하는 에러 메시지는 이미 접두사가 붙어 있으므로 `_ml_reasons`/`_llm_reasons`처럼 호출부에서 다시 `f"... 실패: {error}"`로 감싸지 않는다(중복 접두사 방지).

## 4. `tests/orchestrator/test_pipeline.py` 신규 케이스

기존 24개 테스트는 전부 무수정 통과가 필수 조건이다. 추가할 케이스(ML/LLM 배선 테스트의 뉴스판 대응):

1. `test_pipeline_without_any_auxiliary_signal_behaves_identically_to_before` **확장** — 기존 테스트에 `mock_analyze_disclosures = mocker.patch(".../analyze_disclosures")`를 추가하고 `mock_analyze_disclosures.assert_not_called()` 단언을 더한다(3-None 회귀 잠금).
2. `test_pipeline_with_news_client_only_passes_news_reasons` — `news_client`만 전달, `resolve_corp_code`/`fetch_disclosures`/`analyze_disclosures`를 모킹해 `extra_reasons`에 뉴스 근거만 담기는지.
3. `test_pipeline_with_ml_llm_and_news_merges_reasons_in_order` — 셋 다 전달, `extra_reasons` 순서가 ML → LLM차트 → 뉴스/공시인지.
4. `test_nasdaq_ticker_skips_dart_lookup_entirely` — `resolve_corp_code`가 `None`을 반환하도록 모킹 → `fetch_disclosures`가 `assert_not_called()`인지(나스닥 스캔에 `news_client`를 켜도 안전함을 증명).
5. `test_dart_failure_does_not_suppress_ml_or_llm_reasons` — `fetch_disclosures`가 예외를 던지도록 모킹, `errors`에 `"DART 조회 실패"` 포함 + `extra_reasons`에는 ML/LLM 근거가 그대로 남는지.
6. `test_news_llm_failure_does_not_suppress_ml_or_llm_reasons` — `analyze_disclosures`가 예외를 던지도록 모킹, `errors`에 `"LLM 공시해석 실패"` 포함 + ML/LLM차트 근거는 유지되는지.
7. `test_all_three_signal_failures_record_three_errors_and_still_send` — ML/LLM차트/뉴스 셋 다 실패해도 알림은 발송되고 `errors`에 3건이 담기는지(`test_both_failures_record_two_errors_and_still_send`의 3-신호 확장판).

## 5. `scripts/verify_news_disclosure.py` (신규)

`scripts/verify_llm_chart_analyst.py`와 동일한 dry-run/실호출 2-모드 패턴을 복제한다:

- **dry-run** (`NEWS_VERIFY_DRY_RUN=1`, 비용 0): 샘플 티커(`005930`)로 `resolve_corp_code` → `fetch_disclosures`(**DART는 무료이므로 dry-run에서도 실제로 호출**, `llm_chart_analyst`의 dry-run이 OHLCV는 실제로 가져오되 OpenAI만 건너뛰는 것과 동일한 비대칭) → 결과 건수와 `build_user_prompt` 렌더링 결과만 출력, `OpenAIDisclosureClient`는 생성하지 않음.
- **실호출**: `DART_API_KEY`/`OPENAI_API_KEY` 둘 다 `.env`에 있어야 함(하나라도 없으면 안내 후 종료) → `load_llm_config()` → `max_calls_per_run=1`로 강제(`dataclasses.replace`) → `analyze()` 실행 → `market_impact`/`confidence`/`key_event`/`rationale`/`caveat` 출력.
- 공시가 0건이면(`analyze`가 `None` 반환) "공시 없음 — 정상"으로 안내하고 실패 취급하지 않는다(`llm_chart_analyst`의 워밍업-부족 케이스와 달리, 이건 정상 흐름이라는 차이를 스크립트 출력 문구에 명시).
- 크리덴셜 값은 절대 출력하지 않는다(기존 스크립트들과 동일한 안전 규칙).

## 6. `scripts/run_recommendations_with_signals.py` 변경

기존 스크립트(ML+LLM 동시 실행)에 `news_client` 로딩을 추가한다:

```python
from auto_stock.news_disclosure.client import OpenAIDisclosureClient
from auto_stock.news_disclosure.credentials import load_llm_config as load_news_llm_config

try:
    news_llm_config = load_news_llm_config()
except KeyError:
    print("FAILED: OPENAI_API_KEY가 .env에 설정되어 있지 않습니다 — 뉴스/공시 보조 신호를 켜려면 먼저 설정하세요.")
    sys.exit(1)

news_client = OpenAIDisclosureClient(news_llm_config)
```

`DART_API_KEY`는 `dart_source.py`가 호출 시점에 직접 읽으므로(`_load_dart_api_key()`) 이 스크립트에서 사전 검증하지 않는다 — 없으면 파이프라인 실행 중 `_news_reasons`가 잡아서 `errors`에 담는다(설계상 의도된 실패 격리 경로이지, 이 스크립트가 별도로 막을 필요는 없음). `run_recommendation_pipeline(...)` 호출에 `news_client=news_client` 인자를 추가한다.

## 7. 검증 절차

1. `pytest tests/orchestrator/test_pipeline.py -v` — 신규 7개 + 기존 24개 전부 통과
2. `pytest -q --cov=src --cov-report=term-missing` — 전체 회귀, 커버리지 80%+ 유지(Phase 2 완료 시점 기준 99%)
3. `DART_API_KEY` 발급 확인 후 `scripts/verify_news_disclosure.py` dry-run(비용 0) → 실호출(OpenAI 소액 비용, DART 무료) 순서로 수동 실행
4. `scripts/run_recommendations_with_signals.py` 실행 → 텔레그램 메시지에 공시 근거 블록이 포함되는지(또는 공시 없으면 정상적으로 생략되는지) 육안 확인

## 8. API 키 전제조건

`DART_API_KEY` 발급이 이 Phase 실행의 선행 조건이다(Phase 0에서 이미 안내됨, `.env.example`에 항목은 이미 추가돼 있음 — `docs/design/news-disclosure-phase1.md` 참고). 아직 미발급 상태라면 Phase 3의 코드 배선(2~4절)까지는 `DART_API_KEY` 없이도(모킹 기반 TDD로) 진행 가능하지만, 5~7절(실호출 검증)은 발급 후에만 가능하다.

## 9. 이번 문서가 아닌 것

- 실제 코드 구현 — 이 문서는 계획만 담는다. 구현은 별도 요청 시 착수한다.
- Phase 4(문서화 + `code-reviewer`/`security-reviewer` 리뷰) — Phase 3 구현 완료 후 착수.
- 상충·중재 정책 재검토 — `news-disclosure-plan.md`가 이미 "병렬 노출 유지, 신호원 3개가 된 시점이 재검토 시점"이라고 못박아 뒀고, 이 Phase에서 바로 그 시점(ML+LLM+뉴스 3개)이 되지만, 재검토 자체는 여전히 별도 스코프로 남겨둔다.
