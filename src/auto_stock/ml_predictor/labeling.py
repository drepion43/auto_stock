"""다음 거래일 수익률(regression 타깃) 산출 — MambaStock 원본과 동일한 레이블 구조.

레이블은 반드시 close[t+1] 하나의 값만 참조해야 한다 — 인덱스 경계를 한 칸이라도
잘못 잡으면 조용한 lookahead가 되므로 test_labeling.py에서 경계를 못박아 검증한다.
"""

LABEL_HORIZON_DAYS = 1


def forward_returns(closes: list[float], horizon: int = LABEL_HORIZON_DAYS) -> list[float | None]:
    n = len(closes)
    out: list[float | None] = [None] * n
    for i in range(n - horizon):
        base = closes[i]
        out[i] = None if base == 0 else closes[i + horizon] / base - 1
    return out
