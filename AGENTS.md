# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project status

**Code exists (updated 2026-10-11)**: Python 3.12 package under `src/auto_stock/` (setuptools src layout, `pyproject.toml`, `uv.lock`, git repo on `master`), ~650 pytest tests under `tests/`. Run fast tests with `.venv/Scripts/python -m pytest -q -m "not slow"` (`slow` = model-fitting smoke tests). LLM provider is OpenAI (`openai`, `langchain-openai`, `deepagents`). Chat entry point: `scripts/chat_cli.py`. Note: `docs/` is gitignored (`.gitignore` `docs/*`), so design docs are local-only.

`docs/PRD.md` defines the product: a Python system that recommends and semi-automatically executes stock trades (코스피/코스닥 + 나스닥), gated by a confirmed risk policy (§6) and a paper-trading-first broker strategy (§7). Read `docs/PRD.md` before writing any code — it is the source of truth for scope, architecture, risk limits, and broker roles. Remaining open items are tracked in its §11.

**Architecture update (2026-08-29, subagent count updated 2026-10-11)**: PRD §4's original all-`deepagents` proposal has been refined — see `docs/_review_parts/06_chat_agent.md` and `docs/design/main-agent-migration-plan.md`. Most modules (rule engine #1, risk sizing #5, ML predictor #2, LLM chart analyst #3, news/disclosure #4, and the main conversational agent) are plain functions / native tool-calling, not `deepagents` subagents. `deepagents` is scoped to four subagents, all implemented:
- `find_related_companies` and `stock_analyst(ticker)` — exposed to the main agent as agent-as-tool.
- `recommendation_synthesis` — invoked by the background batch coordinator (`orchestrator/recommendation_coordinator.py`), not the main agent; see `docs/design/recommendation-synthesis.md`.
- `etf_composition` (`chat_agent/etf_composition.py`, 2026-10-11) — synthesis-only, no tools; called from the plain tool `analyze_etf_composition` after it collects disclosure/news signals for an ETF's top holdings (domestic ETFs only); see `docs/design/etf-composition-analysis.md`.

Markets in code are still `KRX`/`NASDAQ` (`data/models.py::VALID_MARKETS`); a KOSPI/KOSDAQ/NASDAQ/NYSE split is drafted in untracked modules but not wired in.

**Main loop migration (2026-10-11, branch `feat/main-agent-registry`)**: tools are defined once in `chat_agent/tool_registry.py` (`TOOL_SPECS`) — adding a tool means one `ToolSpec` + its `tool_*` function in `tools.py`; schemas, dispatch and stock_analyst bindings are derived, and golden fixtures in `tests/chat_agent/fixtures/` pin what the model receives. The deepagents-based main loop (`chat_agent/deep_loop.py` + `agent_middleware.py`) is the default since 2026-10-12 (eval passed); `CHAT_AGENT_LOOP=legacy` rolls back to `loop.py`, which is removed after a 2-week trial (decision D12). Plan and decisions D1–D19: `docs/design/main-agent-migration-plan.md`.

**Current priority (2026-10-05)**: MVP-1 (semi-automated order execution) and everything after it are excluded from current work — the near-term goal is finishing the stock-recommendation chatbot only (see `docs/PRD.md` §9). Do not start broker integration, order execution, or execution-side risk enforcement unless the user explicitly reverses this.

Key facts from the PRD that code MUST respect:
- **Risk policy (§6, confirmed)**: 종목당 5% 이내, 최대 동시 10종목, 일일 손실 2%/월간 10% 중단, ATR 기반 손절익절·포지션 사이징.
- **Broker roles (§7, confirmed)**: 토스증권 Open API = 실전계좌 전용 (모의투자 미지원, 공식 확인됨). 한국투자증권/키움증권 = 모의투자 검증의 필수 경로. Never use 토스 for paper-trading validation.
- Credentials: environment variables or a secret manager only, never hardcoded.

## Project-specific tooling

- **`risk-policy-guardian`** subagent — invoke (or let it auto-trigger) when writing/modifying risk management, position sizing, or order execution code, to check it against PRD §6.
- **`broker-safety-reviewer`** subagent — invoke (or let it auto-trigger) when writing/modifying broker API integration, credential handling, or trading-mode switching code, to check it against PRD §7.
- **`/risk-check`** skill — validate a proposed strategy/order plan against PRD §6 risk limits, usable even before implementation code exists.

## Next steps

- Unfinished, uncommitted work (as of 2026-10-11): KOSPI/KOSDAQ/NASDAQ/NYSE market split (`data/market_routing.py`, `orchestrator/market_buckets.py` drafted; `VALID_MARKETS` not yet changed), `rule_engine/technical_context.py` (needs `adx` etc. in `indicators.py`), `data/sources/index_constituents.py` (Dow test failing), `index_quote.py`/`investor_flow.py` (not wired as tools).
- Main-loop trial: the deepagents loop is the default since 2026-10-12; remove legacy `loop.py`, its tests and the `CHAT_AGENT_LOOP` flag after ~2026-10-26 (decision D12). Still open from the plan: global concurrent-batch cap (second half of D17).
- US news (GDELT) is heavily rate-limited (HTTP 429) and slow; throttling/retry and legal-suffix-free queries were added but it remains unreliable — options and pending decisions in `docs/design/us-news-source-reliability.md`.
- Rotate the Naver Open API client secret: a real client ID was committed in `.env.example` in history (commit f788c18, now blanked).
- `docs/_review_parts/*.md` partly describe planned (not implemented) state — each file has a dated correction note at the top of the affected section.
