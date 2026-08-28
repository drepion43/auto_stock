"""`ChartSnapshot` -> LLM 프롬프트 렌더링. API 호출 전혀 없는 순수 계층.

`build_user_prompt`는 의도적으로 `ticker`/`market`/실제 `date`/`action`을 받지 않는다 —
전자 3개는 환각 방어(모델의 사전지식으로 인한 편향 차단, PRD §10), `action` 부재는 동조
방어(모델이 규칙엔진 신호에 맞춰 답을 짜맞추지 못하게 함)를 함수 시그니처 수준에서 못박는다.
예측 지평 문구는 ML #2와 같은 지평을 말하도록 `LABEL_HORIZON_DAYS`를 재사용한다 — 하드코딩하면
ML 쪽 지평이 바뀔 때 이 프롬프트만 조용히 어긋난다.
"""

from auto_stock.llm_chart_analyst.models import ChartSnapshot
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS

SYSTEM_PROMPT = """당신은 주가 차트의 기술적 패턴을 해석하는 분석가다.

입력으로 최근 거래일의 정규화된 OHLCV 표와 기술적 지표 값만 주어진다.
종목명·종목코드·시장·실제 날짜·뉴스·재무·거시 정보는 의도적으로 제공하지 않는다.

규칙:
1. 주어진 수치만으로 판단한다. 제공되지 않은 정보(기업명, 업종, 뉴스, 실적)를
   추측하거나 지어내지 않는다.
2. 뚜렷한 패턴이 없으면 반드시 direction="NEUTRAL"을 선택한다.
   억지로 패턴을 만들어내지 않는다. NEUTRAL은 실패가 아니라 정상적인 답이다.
3. confidence는 수치가 판단을 얼마나 명확히 지지하는지를 LOW/MEDIUM/HIGH로만
   표현한다. 확률이나 퍼센트를 쓰지 않는다.
4. 목표가·손절가·매매 수량은 제시하지 않는다. 별도의 리스크 관리 모듈이 결정한다.
5. rationale은 한국어 1~2문장으로, 어떤 수치가 근거인지 구체적으로 언급한다.
6. 이 분석은 매매 결정 자체가 아니라, 다른 신호와 함께 사람에게 제시되는
   보조 참고 자료다. 단정적 확언을 피한다.
"""


def render_bar_table(snapshot: ChartSnapshot) -> str:
    header = "오프셋  시가     고가     저가     종가    거래량비"
    rows = [header]
    for bar in snapshot.bars:
        rows.append(
            f"t-{bar.offset:<4}{bar.open:>7.1f}{bar.high:>8.1f}{bar.low:>8.1f}"
            f"{bar.close:>8.1f}{bar.volume_ratio:>9.2f}"
        )
    return "\n".join(rows)


def render_indicators(snapshot: ChartSnapshot) -> str:
    values = snapshot.indicators
    lines = [
        f"RSI(14): {values['rsi_14'] * 100:.1f}",
        f"MACD 히스토그램/종가: {values['macd_hist_norm']:.4f}",
        f"종가/SMA20 - 1: {values['close_to_sma20']:+.1%}",
        f"종가/SMA60 - 1: {values['close_to_sma60']:+.1%}",
        f"SMA20/SMA60 - 1: {values['sma20_to_sma60']:+.1%}",
        f"ATR(14)/종가: {values['atr_14_pct']:.1%}",
        f"20일 거래량비: {values['volume_ratio_20']:.2f}",
        f"20일 채널 내 위치(0=저점, 1=고점): {values['channel_position_20']:.2f}",
        f"수익률 1일/5일/20일: {values['return_1d']:+.1%} / {values['return_5d']:+.1%} / {values['return_20d']:+.1%}",
    ]
    return "\n".join(lines)


def build_user_prompt(snapshot: ChartSnapshot) -> str:
    bar_count = len(snapshot.bars)
    return (
        f"아래는 어떤 상장 종목의 최근 {bar_count}거래일 차트 요약이다.\n\n"
        f"[정규화 가격] 구간 첫 종가 = 100 기준\n"
        f"{render_bar_table(snapshot)}\n\n"
        f"[기술적 지표] 최신 거래일 기준\n"
        f"{render_indicators(snapshot)}\n\n"
        f"이 차트에서 관찰되는 패턴과 향후 약 {LABEL_HORIZON_DAYS}거래일 방향성을 판단하라."
    )
