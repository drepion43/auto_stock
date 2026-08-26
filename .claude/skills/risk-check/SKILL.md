---
name: risk-check
description: Validates a proposed trading strategy, order plan, or position-sizing scheme against the auto_stock PRD's confirmed risk policy (docs/PRD.md §6). Use when the user proposes a strategy or order plan and wants to check it against the risk limits, or when reviewing risk-related code/config.
---

Read `docs/PRD.md` §6 (리스크 관리 정책) to get the current confirmed risk policy — it is the source of truth, not this file. As of the last PRD update it is:

- 종목당 최대 투자 비중: 계좌 자산의 5% 이내
- 최대 동시 보유 종목 수: 10종목 (총 익스포저 50%, 나머지는 현금)
- 일일 최대 손실 한도: 계좌 자산의 2% (도달 시 당일 자동매매 중단)
- 월간 최대 손실 한도: 계좌 자산의 10% (도달 시 당월 자동매매 중단)
- 손절매/익절매 기준: ATR(변동성) 기반 동적 조정
- 포지션 사이징: ATR(변동성) 기반 동적 조절

Given a proposed strategy, order plan, or position-sizing scheme (described in chat or provided as a file), check each item against the policy above and report:

1. **위반 여부**: 각 한도별로 통과/위반/판단불가(정보부족) 표시
2. **구체적 근거**: 어떤 수치가 어떤 한도를 얼마나 초과하는지
3. **수정 제안**: 위반 시 한도 내로 맞추는 구체적 조정안

If the proposal doesn't specify enough detail to check a given limit (e.g., no ATR value given), say so explicitly rather than guessing. Do not invent numbers not present in the PRD or the user's proposal.
