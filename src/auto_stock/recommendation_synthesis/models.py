"""recommendation_synthesis 도메인 모델. `chat_agent/stock_analyst.py`의
`StockAnalystResult`와 동일하게 deepagents의 구조화 출력(`response_format`)으로 쓰인다."""

from pydantic import BaseModel


class RecommendationItem(BaseModel):
    ticker: str
    market: str
    action: str  # "BUY" | "SELL" | "HOLD"
    rank: int
    summary: str


class RecommendationSynthesisResult(BaseModel):
    recommendations: list[RecommendationItem]
