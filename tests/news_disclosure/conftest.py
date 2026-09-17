"""news_disclosure 테스트 전용 fixture.

`OPENAI_API_KEY`를 더미 값으로 monkeypatch하는 autouse 픽스처는 실수로도 실제 API 호출이
발생하지 않도록 하는 안전장치다(모든 테스트가 이 conftest를 통해 이 픽스처를 자동 적용받는다).
"""

from datetime import date, timedelta

import pytest

from auto_stock.data.models import DisclosureRecord
from auto_stock.news_disclosure.models import DisclosureAnalysis
from auto_stock.news_disclosure.schema import DisclosureRead


@pytest.fixture(autouse=True)
def _dummy_openai_api_key(monkeypatch, mocker):
    # load_dotenv 자체도 모킹한다 — 그렇지 않으면 로컬 .env에 OPENAI_API_KEY=(빈 값이라도)가
    # 있을 때 monkeypatch.delenv 직후 load_dotenv()가 그 빈 값을 다시 주입해 "키 없으면
    # KeyError" 테스트가 거짓으로 실패한다(코드 리뷰 MEDIUM, news_sentiment 리뷰에서 발견).
    mocker.patch("auto_stock.news_disclosure.credentials.load_dotenv")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key-not-real")


def make_disclosures(
    n: int,
    ticker: str = "005930",
    corp_code: str = "00126380",
    start: date = date(2024, 6, 1),
    report_name: str = "분기보고서 제출",
    remark: str = "",
) -> list[DisclosureRecord]:
    return [
        DisclosureRecord(
            corp_code=corp_code,
            ticker=ticker,
            report_name=f"{report_name} {i}",
            filed_date=start + timedelta(days=i),
            remark=remark,
        )
        for i in range(n)
    ]


class FakeDisclosureReader:
    """`DisclosureReader` Protocol 구현체 — 진짜 SDK 없이 analyst.py를 테스트하기 위함."""

    def __init__(
        self,
        response: DisclosureRead | None = None,
        error: Exception | None = None,
        model: str = "gpt-5.6-luna",
    ) -> None:
        self.model = model
        self._response = response or DisclosureRead(
            market_impact="POSITIVE",
            confidence="MEDIUM",
            key_event="유상증자 결정",
            rationale="자금조달을 통한 사업 확장 계획이 공시되었습니다.",
            caveat=None,
        )
        self._error = error
        self.calls: list[tuple[str, str]] = []

    def read_disclosures(self, system_prompt: str, user_prompt: str) -> DisclosureRead:
        self.calls.append((system_prompt, user_prompt))
        if self._error is not None:
            raise self._error
        return self._response


@pytest.fixture
def fake_reader() -> FakeDisclosureReader:
    return FakeDisclosureReader()


def make_disclosure_analysis(**overrides) -> DisclosureAnalysis:
    values = dict(
        ticker="005930",
        market="KRX",
        as_of=date(2024, 6, 10),
        market_impact="POSITIVE",
        confidence="MEDIUM",
        key_event="유상증자 결정",
        rationale="자금조달을 통한 사업 확장 계획이 공시되었습니다.",
        caveat=None,
        model="gpt-5.6-luna",
    )
    values.update(overrides)
    return DisclosureAnalysis(**values)
