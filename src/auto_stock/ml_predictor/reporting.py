"""학습 기록(`EpochRecord` 목록) -> history CSV + 학습 곡선 PNG.

그래프는 두 패널이다: 위는 train/val loss(같은 masked MSE 척도라 한 축에 그림),
아래는 학습률. loss와 학습률은 단위가 달라 한 그림에 y축 두 개로 겹치지 않는다.
색은 dataviz 기본 팔레트 categorical 1·2번(train=파랑, val=주황, CVD 검증 통과)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: training runs in a terminal, never opens a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from auto_stock.ml_predictor.models import EpochRecord  # noqa: E402

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
LINE_WIDTH = 1.5

CSV_FIELDS = ["epoch", "train_loss", "val_loss", "val_ic", "learning_rate", "elapsed_seconds"]


def best_val_epoch(history: list[EpochRecord] | tuple[EpochRecord, ...]) -> int | None:
    """val loss가 가장 낮은 epoch(동률이면 앞선 epoch). validation이 없으면 None."""
    scored = [(h.val_loss, h.epoch) for h in history if h.val_loss is not None]
    return min(scored)[1] if scored else None


def best_val_ic_epoch(history: list[EpochRecord] | tuple[EpochRecord, ...]) -> int | None:
    """val IC가 가장 높은 epoch(동률이면 앞선 epoch) — `training.train`이 저장하는 가중치의 epoch."""
    scored = [(-h.val_ic, h.epoch) for h in history if h.val_ic is not None and h.val_ic == h.val_ic]
    return min(scored)[1] if scored else None


def save_history_csv(history: list[EpochRecord] | tuple[EpochRecord, ...], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for h in history:
            writer.writerow(
                {
                    "epoch": h.epoch,
                    "train_loss": h.train_loss,
                    "val_loss": "" if h.val_loss is None else h.val_loss,
                    "val_ic": "" if h.val_ic is None else h.val_ic,
                    "learning_rate": h.learning_rate,
                    "elapsed_seconds": round(h.elapsed_seconds, 3),
                }
            )
    return path


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)


def _direct_label(ax: plt.Axes, x: float, y: float, text: str, color: str, y_offset: float) -> None:
    # Identity is carried by a colored marker; the text itself stays in ink color.
    # train/val labels get opposite vertical offsets so they never collide when the curves converge.
    ax.plot([x], [y], marker="o", markersize=5, color=color, zorder=3)
    ax.annotate(text, (x, y), xytext=(6, y_offset), textcoords="offset points", va="center",
                fontsize=9, color=TEXT_SECONDARY)


def _mark_best_val(ax: plt.Axes, best: int, best_loss: float, last_epoch: int) -> None:
    ax.axvline(best, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    ax.plot([best], [best_loss], marker="o", markersize=8, markerfacecolor=SURFACE_COLOR,
            markeredgecolor=VAL_COLOR, markeredgewidth=1.5, zorder=4)
    # Near the right edge, put the note on the left of the line so it stays inside the plot.
    near_end = best >= 0.8 * last_epoch
    ax.annotate(f"best val: epoch {best}", (best, best_loss), xytext=(-6 if near_end else 6, 22),
                textcoords="offset points", ha="right" if near_end else "left", fontsize=9, color=TEXT_PRIMARY)


def _plot_losses(ax: plt.Axes, history: list[EpochRecord] | tuple[EpochRecord, ...]) -> None:
    epochs = [h.epoch for h in history]
    train_losses = [h.train_loss for h in history]
    ax.plot(epochs, train_losses, color=TRAIN_COLOR, linewidth=LINE_WIDTH, label="train loss")
    _direct_label(ax, epochs[-1], train_losses[-1], "train", TRAIN_COLOR, y_offset=7)

    val_points = [(h.epoch, h.val_loss) for h in history if h.val_loss is not None]
    if val_points:
        val_epochs, val_losses = zip(*val_points)
        ax.plot(val_epochs, val_losses, color=VAL_COLOR, linewidth=LINE_WIDTH, label="val loss")
        _direct_label(ax, val_epochs[-1], val_losses[-1], "val", VAL_COLOR, y_offset=-7)
        best = best_val_epoch(history)
        _mark_best_val(ax, best, dict(val_points)[best], epochs[-1])

    all_losses = train_losses + [v for _, v in val_points]
    if all(v > 0 for v in all_losses):
        ax.set_yscale("log")
        # Plain-text tick labels: the default mathtext labels (10^{-3}) need a U+2212 glyph many fonts lack.
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0e}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("masked MSE", color=TEXT_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    _style_axis(ax)


def plot_training_history(history: list[EpochRecord] | tuple[EpochRecord, ...], path: Path | str) -> Path:
    """학습 곡선 PNG를 저장하고 경로를 반환한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # ASCII minus: some default fonts lack U+2212, which garbles log-scale ticks like 10^-3.
    with plt.rc_context({"axes.unicode_minus": False}):
        fig, (loss_ax, lr_ax) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE_COLOR
        )
        try:
            _plot_losses(loss_ax, history)
            loss_ax.set_title("Mamba training curve", loc="left", fontsize=12, color=TEXT_PRIMARY)
            _plot_learning_rate(lr_ax, history)
            fig.tight_layout()
            fig.savefig(path, dpi=150, facecolor=SURFACE_COLOR)
        finally:
            plt.close(fig)
    return path


def _plot_learning_rate(ax: plt.Axes, history: list[EpochRecord] | tuple[EpochRecord, ...]) -> None:
    rates = [h.learning_rate for h in history]
    ax.plot([h.epoch for h in history], rates, color=TEXT_SECONDARY, linewidth=LINE_WIDTH)
    ax.set_ylim(0, max(rates) * 1.2)  # zero-based so a constant rate reads as flat, not as noise
    ax.set_ylabel("learning rate", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlabel("epoch", color=TEXT_SECONDARY, fontsize=9)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    _style_axis(ax)
