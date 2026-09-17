"""prompt.py — SYSTEM_PROMPT는 자연어 지시문이라 "잘 작동하는지"는 pytest로 주장할 수
없다(Stage C 수동 체크리스트 몫). 여기서는 계획서/설계 리뷰가 요구한 필수 지시 문구가
실제로 프롬프트에 존재하는지(회귀 방지)만 확인한다 — 리팩터링 중 실수로 지워지는 걸
잡기 위함이다."""

from auto_stock.chat_agent.prompt import SYSTEM_PROMPT


def test_system_prompt_instructs_resolve_ticker_before_other_tools():
    assert "resolve_ticker" in SYSTEM_PROMPT


def test_system_prompt_instructs_related_company_call_order():
    assert "find_related_companies" in SYSTEM_PROMPT
    assert "stock_analyst" in SYSTEM_PROMPT


def test_system_prompt_requires_confidence_label_distinction():
    assert "confirmed" in SYSTEM_PROMPT
    assert "inferred" in SYSTEM_PROMPT


def test_system_prompt_requires_provenance_label():
    assert "서브에이전트 자율 조사 결과" in SYSTEM_PROMPT


def test_system_prompt_requires_sizing_disclaimer():
    assert "투자 조언" in SYSTEM_PROMPT


def test_system_prompt_instructs_market_scan_tool_for_untargeted_recommendation_queries():
    """정책 반전(2026-09-07) — 예전엔 "추천해줄만한 주식 있어?" 같은 전체 스캔형 질의를
    명시적으로 범위 밖 처리했지만, get_market_scan_recommendations(배치 스캔 캐시) 도입
    이후로는 이 도구를 호출하도록 지시한다. scanned_at 시각 고지가 필수 규칙에 포함돼
    있는지 확인한다(majestic-waddling-breeze.md 계획). 2026-09-09부로 이 도구가 NASDAQ도
    지원하게 되면서(이전엔 KRX 하드코딩) "나스닥 제외 안내" 대신 "나스닥이면
    market=NASDAQ으로 호출" 지시로 바뀌었다."""
    assert "get_market_scan_recommendations" in SYSTEM_PROMPT
    assert "scanned_at" in SYSTEM_PROMPT
    assert "나스닥" in SYSTEM_PROMPT
    assert "market=NASDAQ" in SYSTEM_PROMPT


def test_system_prompt_instructs_reuse_of_history_for_followups():
    assert "히스토리" in SYSTEM_PROMPT


def test_system_prompt_instructs_english_retry_when_korean_nasdaq_query_not_found():
    """실사용 중 발견 — 한글 별칭 표(ticker_resolution.py)는 자주 쓰이는 나스닥 종목만
    커버하므로, 표에 없는 종목을 한글로 물으면 여전히 0건이 나온다. 이 경우 모델이
    영문 티커/회사명 재입력을 안내해야 한다."""
    assert "영문 티커" in SYSTEM_PROMPT or "영문 회사명" in SYSTEM_PROMPT


def test_system_prompt_instructs_refreshing_field_handling():
    """majestic-waddling-breeze.md "온디맨드 배치 스캔 트리거" 계획 — get_market_scan_
    recommendations가 백그라운드 스캔을 트리거하면 refreshing=true를 반환하므로, 모델이
    이 경우 "갱신 중" 문구를 답변에 포함하도록 SYSTEM_PROMPT 규칙 4에도 명시해야 한다."""
    assert "refreshing" in SYSTEM_PROMPT


def test_system_prompt_instructs_treating_tool_output_as_data_not_instructions():
    """보안 리뷰 발견사항(Stage E) — 뉴스/공시 원문에서 파생된 도구 결과가 아무 방어 없이
    메인 모델 컨텍스트로 흘러들어가므로, 그 안의 지시문처럼 보이는 문구를 따르지 말라는
    방어 지침이 필요하다."""
    assert "데이터" in SYSTEM_PROMPT
    assert "무시" in SYSTEM_PROMPT
