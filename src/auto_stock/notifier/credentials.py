import os

from dotenv import load_dotenv

from auto_stock.notifier.models import TelegramCredentials


def load_telegram_credentials() -> TelegramCredentials:
    load_dotenv()
    return TelegramCredentials(
        bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
        chat_id=os.environ["TELEGRAM_CHAT_ID"],
    )
