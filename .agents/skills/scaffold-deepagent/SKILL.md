---
name: scaffold-deepagent
description: Scaffolds a new LangChain deepagents subagent module for auto_stock, following the project's consistent structure. Use when creating one of the PRD §4 subagents (차트분석/예측 에이전트, 규칙엔진, ML 예측 모듈, 뉴스/공시 분석 에이전트, 리스크·포지션 사이징 에이전트, 추천 설명 생성기, 주문 실행 에이전트, 알림/승인 봇) or any new subagent for the trading pipeline. Takes the subagent name/role as $ARGUMENTS.
---

Scaffold a new LangChain `deepagents` subagent module for auto_stock, for the subagent named in `$ARGUMENTS`.

1. **Read `docs/PRD.md` §4** to confirm this subagent's exact role and how it fits the pipeline. The PRD's subagent table is the source of truth — do not invent scope beyond what it defines. If `$ARGUMENTS` doesn't match one of the 8 PRD §4 subagents, confirm with the user whether this is an intentional new addition to the pipeline before proceeding.

2. **Verify the current `deepagents` API before writing any code.** This library evolves quickly — do not rely on memorized constructor signatures or subagent config shapes from training data. Look it up now (context7, or fetch the official docs) for: the current subagent definition shape, how tools are attached, and how middleware (memory/filesystem/summarization) is configured.

3. **Follow the project's module layout** (establish it if this is the first subagent scaffolded):
   - `src/auto_stock/subagents/<name>/agent.py` — subagent definition
   - `src/auto_stock/subagents/<name>/prompts.py` — system prompt
   - `src/auto_stock/subagents/<name>/tools.py` — tool functions
   - `tests/subagents/test_<name>.py` — write the test first (TDD), per this org's testing conventions

4. **Route to the right safety reviewer after scaffolding**: if this subagent touches risk management, position sizing, or order execution, tell the user to run the `risk-policy-guardian` subagent on it. If it touches broker API integration or credentials, tell them to run `broker-safety-reviewer`.

5. **Keep the system prompt scoped exactly to the PRD §4 role description** for this subagent — no speculative extra responsibilities.
