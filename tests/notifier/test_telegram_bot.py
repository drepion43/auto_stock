import pytest

from auto_stock.explainer.models import Explanation
from auto_stock.notifier.models import TelegramCredentials
from auto_stock.notifier.telegram_bot import TelegramNotificationError, send_notification


def _explanation(action="BUY", summary="005930: RSI(14)=25.0 과매도 구간 진입."):
    return Explanation(ticker="005930", market="KRX", action=action, summary=summary)


def _credentials():
    return TelegramCredentials(bot_token="123:ABC", chat_id="999")


def test_sends_message_to_correct_url_and_payload(mocker):
    mock_post = mocker.patch("auto_stock.notifier.telegram_bot.requests.post")
    mock_post.return_value.ok = True
    mock_post.return_value.json.return_value = {"ok": True}

    send_notification(_explanation("BUY"), _credentials())

    mock_post.assert_called_once()
    url, kwargs = mock_post.call_args[0][0], mock_post.call_args[1]
    assert url == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert kwargs["json"]["chat_id"] == "999"
    assert kwargs["timeout"] > 0


def test_message_is_prefixed_with_korean_action_label(mocker):
    mock_post = mocker.patch("auto_stock.notifier.telegram_bot.requests.post")
    mock_post.return_value.ok = True
    mock_post.return_value.json.return_value = {"ok": True}

    send_notification(_explanation("BUY", "005930: RSI 과매도."), _credentials())
    buy_text = mock_post.call_args[1]["json"]["text"]

    send_notification(_explanation("SELL", "005930: RSI 과매수."), _credentials())
    sell_text = mock_post.call_args[1]["json"]["text"]

    assert buy_text.startswith("[매수]")
    assert sell_text.startswith("[매도]")


def test_raises_when_response_not_ok(mocker):
    mock_post = mocker.patch("auto_stock.notifier.telegram_bot.requests.post")
    mock_post.return_value.ok = True
    mock_post.return_value.json.return_value = {"ok": False, "description": "chat not found"}

    with pytest.raises(TelegramNotificationError):
        send_notification(_explanation(), _credentials())


def test_raises_when_http_error_response(mocker):
    mock_post = mocker.patch("auto_stock.notifier.telegram_bot.requests.post")
    mock_post.return_value.ok = False
    mock_post.return_value.status_code = 401

    with pytest.raises(TelegramNotificationError):
        send_notification(_explanation(), _credentials())


def test_raises_telegram_notification_error_on_network_failure(mocker):
    import requests

    mocker.patch("auto_stock.notifier.telegram_bot.requests.post", side_effect=requests.Timeout("timed out"))

    with pytest.raises(TelegramNotificationError):
        send_notification(_explanation(), _credentials())
