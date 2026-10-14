"""피처+레이블 결합, 종목별 시퀀스 구성, 시간순 분할(embargo 포함).

`train_test_split`/`KFold`/`shuffle=True`는 이 모듈에서 절대 사용하지 않는다 — 랜덤
분할은 미래 정보가 학습 세트로 새는 것을 감출 수 있어 금융 시계열에서 금지된다.
Mamba 교체(종목 1개 = 시퀀스 1개, 배치 축 = 종목)로 "표본 1개 = 피처벡터 1개"의 flat
pooled 구조에서 "표본 1개 = 종목의 (가변 길이) 시퀀스" 구조로 바뀌었지만, 분할 자체는
여전히 **전역 날짜 기준**으로 한다 — 종목별로 각자 다른 날짜에서 자르면 한 종목의
미래가 다른 종목의 과거와 뒤섞여도 감지되지 않는 시간적 정보 누출을 낳는다. 구체적으로:
전역 날짜 컷오프를 구한 뒤, 각 종목의 시퀀스를 그 컷오프(+embargo)로 "같은 지점에서"
자른다 — 종목마다 길이가 달라도 자르는 **날짜**는 전부 동일하다."""

from datetime import date

from auto_stock.data.models import OHLCVRecord
from auto_stock.ml_predictor.features import FEATURE_NAMES, build_feature_vectors
from auto_stock.ml_predictor.labeling import LABEL_HORIZON_DAYS, forward_returns
from auto_stock.ml_predictor.models import LabeledSample, SequenceDataset, SequenceDatasetSplit, TickerSequence


def build_ticker_sequence(
    records: list[OHLCVRecord], fundamentals_by_date: dict[date, dict] | None = None
) -> TickerSequence | None:
    """단일 종목 레코드 리스트 -> 워밍업 구간과 레이블 없는 마지막 1개 행을 제거한
    `TickerSequence`. 표본이 하나도 없으면 None(빈 시퀀스는 만들지 않음)."""
    if not records:
        return None

    vectors = build_feature_vectors(records, fundamentals_by_date)
    closes = [r.close for r in records]
    fwd_returns = forward_returns(closes, LABEL_HORIZON_DAYS)

    samples: list[LabeledSample] = []
    for vector, fwd_return in zip(vectors, fwd_returns):
        if vector is None or fwd_return is None:
            continue
        samples.append(LabeledSample(feature=vector, label=fwd_return))

    if not samples:
        return None

    record = records[0]
    return TickerSequence(ticker=record.ticker, market=record.market, samples=samples)


def build_sequence_dataset(
    records_by_ticker: list[list[OHLCVRecord]],
    fundamentals_by_date_by_ticker: list[dict[date, dict] | None] | None = None,
) -> SequenceDataset:
    """여러 종목의 레코드를 종목별 `TickerSequence`로 묶는다. `fundamentals_by_date_by_ticker`가
    주어지면 `records_by_ticker`와 같은 순서로 1:1 대응한다(국내 종목만 값을 넘기고
    해외 종목은 None으로 채우면 됨)."""
    fundamentals = fundamentals_by_date_by_ticker or [None] * len(records_by_ticker)
    sequences: list[TickerSequence] = []
    for records, fundamentals_by_date in zip(records_by_ticker, fundamentals):
        sequence = build_ticker_sequence(records, fundamentals_by_date)
        if sequence is not None:
            sequences.append(sequence)
    return SequenceDataset(sequences=sequences, feature_names=FEATURE_NAMES)


def _all_dates(dataset: SequenceDataset) -> list[date]:
    dates = {s.feature.date for seq in dataset.sequences for s in seq.samples}
    return sorted(dates)


def _slice_sequence(sequence: TickerSequence, allowed_dates: set[date]) -> TickerSequence | None:
    samples = [s for s in sequence.samples if s.feature.date in allowed_dates]
    if not samples:
        return None
    return TickerSequence(ticker=sequence.ticker, market=sequence.market, samples=samples)


def chronological_split(
    dataset: SequenceDataset, test_ratio: float = 0.2, embargo_days: int | None = None
) -> SequenceDatasetSplit:
    """전역 날짜 기준 시간순 분할 + embargo 갭. 각 종목 시퀀스를 같은 날짜 컷오프로
    자른다 — 종목마다 길이가 달라도 자르는 지점(날짜)은 전부 동일하다. 기본
    embargo=LABEL_HORIZON_DAYS."""
    if embargo_days is None:
        embargo_days = LABEL_HORIZON_DAYS

    unique_dates = _all_dates(dataset)
    n_dates = len(unique_dates)
    split_idx = int(n_dates * (1 - test_ratio))
    split_idx = max(1, min(split_idx, n_dates - 1))

    embargo_start_idx = max(0, split_idx - embargo_days)
    train_dates = set(unique_dates[:embargo_start_idx])
    test_dates = set(unique_dates[split_idx:])

    train_sequences = [s for s in (_slice_sequence(seq, train_dates) for seq in dataset.sequences) if s is not None]
    test_sequences = [s for s in (_slice_sequence(seq, test_dates) for seq in dataset.sequences) if s is not None]

    return SequenceDatasetSplit(
        train=SequenceDataset(sequences=train_sequences, feature_names=dataset.feature_names),
        test=SequenceDataset(sequences=test_sequences, feature_names=dataset.feature_names),
    )


def train_val_test_split(
    dataset: SequenceDataset, val_ratio: float = 0.1, test_ratio: float = 0.2, embargo_days: int | None = None
) -> SequenceDatasetSplit:
    """시간순 3분할: [train | embargo | validation | embargo | test]. test 구간은
    `chronological_split(dataset, test_ratio)`와 정확히 같다(validation 추가 전후 test 결과
    비교 가능). validation은 test 앞쪽 구간에서 다시 같은 규칙으로 잘라낸다 —
    `val_ratio`는 전체 날짜 대비 비율."""
    outer = chronological_split(dataset, test_ratio=test_ratio, embargo_days=embargo_days)
    inner = chronological_split(outer.train, test_ratio=val_ratio / (1 - test_ratio), embargo_days=embargo_days)
    return SequenceDatasetSplit(train=inner.train, test=outer.test, validation=inner.test)


def collate_sequences(
    sequences: list[TickerSequence], feature_names: list[str]
) -> tuple[list[list[list[float]]], list[list[float]], list[list[bool]]]:
    """가변 길이 종목 시퀀스 배치를 패딩한다 — 배치 축 = 종목(§B, 설계 리스크 B).

    종목마다 상장일/보유 레코드 수가 달라 길이가 다르다. 배치 내 최대 길이로
    0-패딩하고, `mask[i][t] = True`인 시점만 실제 데이터다(패딩된 타임스텝은
    loss 계산에서 마스크로 제외해야 한다 — `training.py::_masked_mse_loss` 참고).
    반환 shape: X=(batch, max_len, feature_dim), y=(batch, max_len), mask=(batch, max_len)."""
    if not sequences:
        return [], [], []

    max_len = max(len(seq.samples) for seq in sequences)
    X: list[list[list[float]]] = []
    y: list[list[float]] = []
    mask: list[list[bool]] = []
    for seq in sequences:
        rows = [[s.feature.values[name] for name in feature_names] for s in seq.samples]
        labels = [s.label for s in seq.samples]
        pad_len = max_len - len(rows)
        X.append(rows + [[0.0] * len(feature_names)] * pad_len)
        y.append(labels + [0.0] * pad_len)
        mask.append([True] * len(rows) + [False] * pad_len)
    return X, y, mask


def to_xy_flat(dataset: SequenceDataset) -> tuple[list[list[float]], list[float]]:
    """평가 지표(MSE/MAE/R2) 계산용 — 모든 종목의 모든 시점을 하나의 flat 리스트로
    펼친다(배치 구조와 무관, 순수 평가 전용)."""
    X: list[list[float]] = []
    y: list[float] = []
    for seq in dataset.sequences:
        for s in seq.samples:
            X.append([s.feature.values[name] for name in dataset.feature_names])
            y.append(s.label)
    return X, y
