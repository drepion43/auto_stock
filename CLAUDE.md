# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is still **unbootstrapped** at the code level: no source code, no manifest (package.json/requirements.txt/etc.), no README, and no git repository exist yet. (The previously-vendored `.claude/skills/gstack/` third-party tool is no longer present on disk.)

**Planning is complete**, however. `docs/PRD.md` defines the product: a Python + LangChain `deepagents` system that recommends and semi-automatically executes stock trades (코스피/코스닥 + 나스닥), gated by a confirmed risk policy (§6) and a paper-trading-first broker strategy (§7). Read `docs/PRD.md` before writing any code — it is the source of truth for scope, architecture, risk limits, and broker roles. Remaining open items are tracked in its §11.

Key facts from the PRD that code MUST respect:
- **Risk policy (§6, confirmed)**: 종목당 5% 이내, 최대 동시 10종목, 일일 손실 2%/월간 10% 중단, ATR 기반 손절익절·포지션 사이징.
- **Broker roles (§7, confirmed)**: 토스증권 Open API = 실전계좌 전용 (모의투자 미지원, 공식 확인됨). 한국투자증권/키움증권 = 모의투자 검증의 필수 경로. Never use 토스 for paper-trading validation.
- Credentials: environment variables or a secret manager only, never hardcoded.

## Project-specific tooling

- **`risk-policy-guardian`** subagent — invoke (or let it auto-trigger) when writing/modifying risk management, position sizing, or order execution code, to check it against PRD §6.
- **`broker-safety-reviewer`** subagent — invoke (or let it auto-trigger) when writing/modifying broker API integration, credential handling, or trading-mode switching code, to check it against PRD §7.
- **`/risk-check`** skill — validate a proposed strategy/order plan against PRD §6 risk limits, usable even before implementation code exists.

## Next steps

- Confirm the language/stack setup and initialize a proper manifest before writing code (PRD already fixes Python + LangChain `deepagents`).
- Run `git init` if no `.git` directory exists yet.
- Re-run `/init` once real code exists so this file can be filled in with actual build/test/lint commands and conventions.
