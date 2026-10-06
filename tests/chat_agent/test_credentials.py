import pytest

from auto_stock.chat_agent.credentials import (
    DEFAULT_MODEL,
    FIND_RELATED_LLM_CALL_ESTIMATE,
    FIND_RELATED_RECURSION_LIMIT,
    MAX_LLM_CALLS_PER_QUERY,
    MAX_TICKERS_PER_QUERY,
    MAX_TOOL_ITERATIONS,
    SECTOR_THEME_LLM_CALL_ESTIMATE,
    SECTOR_THEME_RECURSION_LIMIT,
    STOCK_ANALYST_LLM_CALL_ESTIMATE,
    STOCK_ANALYST_RECURSION_LIMIT,
    load_llm_config,
)


def test_load_llm_config_uses_env_api_key_and_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    monkeypatch.delenv("CHAT_AGENT_OPENAI_MODEL", raising=False)

    config = load_llm_config()

    assert config.api_key == "sk-example"
    assert config.model == DEFAULT_MODEL


def test_load_llm_config_reads_model_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    monkeypatch.setenv("CHAT_AGENT_OPENAI_MODEL", "gpt-5.6-terra")

    config = load_llm_config()

    assert config.model == "gpt-5.6-terra"


def test_load_llm_config_raises_key_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(KeyError):
        load_llm_config()


def test_per_query_budget_constants_are_internally_consistent(monkeypatch):
    """MAX_TICKERS_PER_QUERY/MAX_TOOL_ITERATIONS/MAX_LLM_CALLS_PER_QUERY는 서로 다른
    개념이지만 모두 "대화 한 턴" 스코프다 — 상호 정합성만 확인한다."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    load_llm_config()

    assert MAX_TICKERS_PER_QUERY < MAX_LLM_CALLS_PER_QUERY
    assert MAX_TOOL_ITERATIONS > 0


def test_client_max_calls_per_run_is_a_whole_session_safety_valve_not_a_per_turn_budget(monkeypatch):
    """코드 리뷰 발견사항(Stage E) — `OpenAIChatAgentClient`는 `chat_cli.py`에서 세션
    전체(여러 턴)에 걸쳐 재사용되는 단일 인스턴스이므로, `max_calls_per_run`(내부
    `_calls_made` 카운터)은 배치 파이프라인처럼 "한 번의 스크립트 실행"이 아니라
    "대화 세션 전체"를 스코프로 해야 한다. 배치용 기본값(20)을 그대로 쓰면 한 턴이
    최대 MAX_TOOL_ITERATIONS(5)회를 소모하는 챗봇에서 4턴 만에 클라이언트가 영구
    잠긴다 — 이번 리뷰에서 발견해 기본값을 상향했다. 최소 여러 턴(20턴)을 감당할 만큼
    넉넉해야 한다는 것만 확인한다(정확한 상한은 실사용 관찰 후 조정 대상)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-example")
    config = load_llm_config()

    assert config.max_calls_per_run >= MAX_TOOL_ITERATIONS * 20


def test_max_llm_calls_per_query_covers_worst_case_related_company_deep_dive():
    """MAX_TICKERS_PER_QUERY(자기 포함 5개)를 전부 stock_analyst로 딥다이브하면서
    find_related_companies까지 한 번 돌리는 최악의 경우를 예산이 감당해야 한다."""
    worst_case = FIND_RELATED_LLM_CALL_ESTIMATE + MAX_TICKERS_PER_QUERY * STOCK_ANALYST_LLM_CALL_ESTIMATE

    assert MAX_LLM_CALLS_PER_QUERY >= worst_case


def test_llm_call_estimates_cover_real_recursion_limit_worst_case():
    """보안 리뷰 발견사항(Stage E) — 사전차감 추정치가 실제 재귀상한(LangGraph가 실제로
    강제하는 하드 상한)보다 작으면, 서브에이전트가 재귀상한까지 실제로 LLM을 호출했을 때
    QueryBudget이 그 실제 지출을 과소 계상하게 된다. 추정치는 반드시 재귀상한 이상이어야
    한다(과소 사전차감 금지) — 그래야 예산이 실제 최악의 지출을 항상 커버한다."""
    assert FIND_RELATED_LLM_CALL_ESTIMATE >= FIND_RELATED_RECURSION_LIMIT
    assert STOCK_ANALYST_LLM_CALL_ESTIMATE >= STOCK_ANALYST_RECURSION_LIMIT
    assert SECTOR_THEME_LLM_CALL_ESTIMATE >= SECTOR_THEME_RECURSION_LIMIT


def test_subagent_recursion_limits_are_positive_and_bounded():
    assert 0 < FIND_RELATED_RECURSION_LIMIT <= 20
    assert 0 < STOCK_ANALYST_RECURSION_LIMIT <= 20
    assert 0 < SECTOR_THEME_RECURSION_LIMIT <= 20
