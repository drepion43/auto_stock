"""Manual verification: sends one real Telegram message using .env credentials.

Run from the project root so `.env` is found:
    .venv/Scripts/python scripts/verify_telegram.py   (Windows)
    .venv/bin/python scripts/verify_telegram.py        (macOS/Linux)

Prints only SUCCESS/FAILED — never prints TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.
See docs/design/telegram-notifier.md for troubleshooting.
"""

from auto_stock.explainer.models import Explanation
from auto_stock.notifier.credentials import load_telegram_credentials
from auto_stock.notifier.telegram_bot import TelegramNotificationError, send_notification

explanation = Explanation(
    ticker="TEST", market="KRX", action="BUY",
    summary="auto_stock 알림봇 연동 확인용 테스트 메시지입니다.",
)

try:
    send_notification(explanation, load_telegram_credentials())
    print("SUCCESS: 텔레그램 메시지 전송 성공")
except TelegramNotificationError as e:
    print(f"FAILED: {e}")
except KeyError as e:
    print(f"FAILED: 환경변수 누락 {e}")
