"""모델별 토큰 단가와 비용 계산 — 계획서 §6, §7.3.

가격표에 없는 모델은 환경변수 재정의가 없으면 실패한다(fail-closed) — 단가를 모르면 비용 상한을
강제할 수 없기 때문이다."""

import os
import re
from collections.abc import Mapping

# USD per 1M tokens (input, output), Standard tier, ≤272K input.
# Source: https://developers.openai.com/api/docs/pricing (checked 2026-10-20). 가격이 바뀌면 여기만 고친다.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
}

_TOKENS_PER_UNIT = 1_000_000


def _env_prefix(model: str) -> str:
    return "EVAL_PRICE_" + re.sub(r"[^A-Z0-9]", "_", model.upper())


def price_for(model: str, env: Mapping[str, str] | None = None) -> tuple[float, float]:
    env = os.environ if env is None else env
    prefix = _env_prefix(model)
    override_in, override_out = env.get(f"{prefix}_IN"), env.get(f"{prefix}_OUT")
    if override_in is not None and override_out is not None:
        return float(override_in), float(override_out)
    if model not in PRICES_PER_MILLION:
        raise KeyError(f"no price for model {model!r}; set {prefix}_IN and {prefix}_OUT")
    return PRICES_PER_MILLION[model]


def cost_usd(model: str, input_tokens: int, output_tokens: int, env: Mapping[str, str] | None = None) -> float:
    price_in, price_out = price_for(model, env)
    return (input_tokens * price_in + output_tokens * price_out) / _TOKENS_PER_UNIT
