from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TelegramCredentials:
    bot_token: str = field(repr=False)  # never let dataclass repr/str echo the secret
    chat_id: str
