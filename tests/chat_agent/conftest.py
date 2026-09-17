"""chat_agent 테스트 전용 fixture. news_sentiment/conftest.py와 동일 패턴."""

import pytest


@pytest.fixture(autouse=True)
def _dummy_openai_api_key(monkeypatch, mocker):
    # load_dotenv 자체도 모킹한다 — 로컬 .env에 OPENAI_API_KEY=(빈 값이라도)가 있으면
    # monkeypatch.delenv 직후 load_dotenv()가 그 빈 값을 다시 주입해 "키 없으면 KeyError"
    # 테스트가 거짓으로 실패한다(다른 3개 모듈에서 이미 겪은 코드 리뷰 MEDIUM과 동일 원인).
    mocker.patch("auto_stock.chat_agent.credentials.load_dotenv")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-not-real")
