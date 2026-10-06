"""chat_agent/credentials.py의 FIND_RELATED_*/STOCK_ANALYST_* 상수 쌍과 동일한 패턴 —
LangGraph가 실제로 강제하는 재귀 상한(RECURSION_LIMIT)과, QueryBudget 사전차감용
추정치(LLM_CALL_ESTIMATE)를 분리해서 두되 둘을 항상 같게 고정한다. 재귀상한보다 추정치가
작으면 agent가 실제로 재귀상한까지 reinvestigate_ticker를 호출했을 때 QueryBudget이
그 실제 지출을 과소 계상하는 보안 버그가 된다(Stage E 코드리뷰에서 발견된 패턴과 동일,
chat_agent/credentials.py 참고) — 구조적으로 재발을 막기 위해 아예 같은 상수를 참조한다.

`find_related_companies`와 복잡도가 비슷하다고 판단해(규칙엔진 후보/ML확률 비교 +
애매한 소수 종목만 선택적으로 재조사) `STOCK_ANALYST_RECURSION_LIMIT`(8)이 아니라
`FIND_RELATED_RECURSION_LIMIT`(6)과 같은 값을 쓴다(recommendation-synthesis-plan.md §7,
2026-10-05 확정)."""

RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT = 6
RECOMMENDATION_SYNTHESIS_LLM_CALL_ESTIMATE = RECOMMENDATION_SYNTHESIS_RECURSION_LIMIT
