import csv

from auto_stock.ml_predictor.models import EpochRecord
from auto_stock.ml_predictor.reporting import best_val_epoch, plot_training_history, save_history_csv

HISTORY = [
    EpochRecord(epoch=1, train_loss=0.30, val_loss=0.32, learning_rate=0.01, elapsed_seconds=1.0),
    EpochRecord(epoch=2, train_loss=0.20, val_loss=0.25, learning_rate=0.01, elapsed_seconds=2.0),
    EpochRecord(epoch=3, train_loss=0.15, val_loss=0.27, learning_rate=0.01, elapsed_seconds=3.0),
]


def test_best_val_epoch_picks_lowest_val_loss():
    assert best_val_epoch(HISTORY) == 2


def test_best_val_epoch_is_none_without_validation():
    history = [EpochRecord(epoch=1, train_loss=0.3, val_loss=None, learning_rate=0.01, elapsed_seconds=1.0)]

    assert best_val_epoch(history) is None


def test_save_history_csv_writes_one_row_per_epoch(tmp_path):
    path = save_history_csv(HISTORY, tmp_path / "history.csv")

    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["1", "2", "3"]
    assert set(rows[0]) == {"epoch", "train_loss", "val_loss", "learning_rate", "elapsed_seconds"}


def test_save_history_csv_leaves_val_loss_blank_without_validation(tmp_path):
    history = [EpochRecord(epoch=1, train_loss=0.3, val_loss=None, learning_rate=0.01, elapsed_seconds=1.0)]

    path = save_history_csv(history, tmp_path / "history.csv")

    with path.open(encoding="utf-8") as f:
        assert next(csv.DictReader(f))["val_loss"] == ""


def test_plot_training_history_writes_png(tmp_path):
    path = plot_training_history(HISTORY, tmp_path / "history.png")

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_training_history_works_without_validation(tmp_path):
    history = [EpochRecord(epoch=i, train_loss=1 / i, val_loss=None, learning_rate=0.01, elapsed_seconds=i) for i in (1, 2)]

    path = plot_training_history(history, tmp_path / "history.png")

    assert path.exists()
