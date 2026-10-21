"""메인 대화 에이전트 최종 답변의 오프라인 평가 — docs/design/llm-answer-evaluation-plan.md.

평가 코드 전용 패키지다. 챗봇 실행 경로(`chat_agent/`)는 이 패키지를 import하지 않는다.

deepeval은 import 시점에 텔레메트리를 초기화할 수 있다. 이 패키지의 어떤 모듈이든 먼저 로드되는
이 자리에서 opt-out을 강제해, 어느 진입점(러너·테스트·스크립트)으로 들어와도 deepeval보다 앞서
적용되게 한다(계획서 §9 — 수집 항목이 자료마다 달라 끄는 쪽으로 고정)."""

import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
