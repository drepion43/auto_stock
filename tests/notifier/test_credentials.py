import pytest

from auto_stock.notifier.credentials import load_telegram_credentials


@pytest.fixture(autouse=True)
def _no_dotenv_file(mocker):
    # A real .env may exist locally (see docs/design/telegram-notifier.md).
    # These tests must be isolated from it, or monkeypatch.delenv below would
    # get silently refilled by load_dotenv() reading that real file.
    mocker.patch("auto_stock.notifier.credentials.load_dotenv")


def test_loads_credentials_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")

    credentials = load_telegram_credentials()

    assert credentials.bot_token == "123:ABC"
    assert credentials.chat_id == "999"


def test_raises_when_bot_token_missing(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")

    with pytest.raises(KeyError, match="TELEGRAM_BOT_TOKEN"):
        load_telegram_credentials()


def test_raises_when_chat_id_missing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(KeyError, match="TELEGRAM_CHAT_ID"):
        load_telegram_credentials()
