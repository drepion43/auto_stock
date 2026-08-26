from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TelegramCredentials:
    bot_token: str
    chat_id: str
