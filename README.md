## 자동 테스트 (pytest)

# 빠른 테스트 전체 (약 40초, 현재 800개 통과)
.venv/Scripts/python -m pytest -q -m "not slow" tests/

# 챗봇 관련 테스트만
.venv/Scripts/python -m pytest -q tests/chat_agent

# 모델 학습이 들어간 느린 테스트까지 전부
.venv/Scripts/python -m pytest -q tests/

# 전체 테스트
.venv/Scripts/python scripts/eval_main_agent.py

# chat Test
.venv/Scripts/python scripts/chat_cli.py