"""N거래일 후 상승(1)/하락(0) 이진 레이블 산출.

레이블은 반드시 close[t+H] 하나의 값만 참조해야 한다 — 인덱스 경계를 한 칸이라도
잘못 잡으면 조용한 lookahead가 되므로 test_labeling.py에서 경계를 못박아 검증한다.
"""

LABEL_HORIZON_DAYS = 5
LABEL_THRESHOLD = 0.0


def forward_returns(closes: list[float], horizon: int = LABEL_HORIZON_DAYS) -> list[float | None]:
    n = len(closes)
    out: list[float | None] = [None] * n
    for i in range(n - horizon):
        base = closes[i]
        out[i] = None if base == 0 else closes[i + horizon] / base - 1
    return out


def to_label(forward_return: float | None, threshold: float = LABEL_THRESHOLD) -> int | None:
    if forward_return is None:
        return None
    return 1 if forward_return > threshold else 0
