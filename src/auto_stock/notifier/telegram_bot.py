"""Notification-only Telegram sender for MVP-0 (IMPLEMENTATION_PLAN.md §7).

Approval/rejection callback handling is out of scope here — that's added in
MVP-1 alongside the order execution agent (#8). This module only pushes the
Explanation (#6) summary as a one-way message.
"""

import requests

from auto_stock.explainer.models import Explanation
from auto_stock.notifier.models import TelegramCredentials

_ACTION_LABELS = {"BUY": "매수", "SELL": "매도"}
_REQUEST_TIMEOUT_SECONDS = 10


class TelegramNotificationError(Exception):
    pass


def _format_message(explanation: Explanation) -> str:
    label = _ACTION_LABELS.get(explanation.action, explanation.action)
    return f"[{label}] {explanation.summary}"


def send_notification(explanation: Explanation, credentials: TelegramCredentials) -> None:
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{credentials.bot_token}/sendMessage",
            json={"chat_id": credentials.chat_id, "text": _format_message(explanation)},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise TelegramNotificationError(f"텔레그램 요청 중 네트워크 오류: {exc}") from exc

    if not response.ok:
        raise TelegramNotificationError(f"텔레그램 요청 실패 (status={getattr(response, 'status_code', '?')})")

    payload = response.json()
    if not payload.get("ok"):
        raise TelegramNotificationError(f"텔레그램 API 오류: {payload.get('description', 'unknown')}")
