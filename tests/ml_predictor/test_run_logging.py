import logging

import pytest

from auto_stock.ml_predictor.run_logging import configure_run_logging


@pytest.fixture
def isolated_root_logger():
    """configure_run_logging mutates the root logger — restore it so other tests are unaffected."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    auto_stock_logger = logging.getLogger("auto_stock")
    saved_auto_level = auto_stock_logger.level
    root.handlers = []
    yield
    for handler in root.handlers:
        handler.close()
    root.handlers, root.level = saved_handlers, saved_level
    auto_stock_logger.setLevel(saved_auto_level)


def test_writes_project_info_logs_to_timestamped_file(tmp_path, isolated_root_logger):
    path = configure_run_logging("collect", log_dir=tmp_path)

    logging.getLogger("auto_stock.ml_predictor.collection").info("[KRX] 10/200 로드=10")

    assert path.parent == tmp_path
    assert path.name.startswith("collect_") and path.suffix == ".log"
    assert "[KRX] 10/200 로드=10" in path.read_text(encoding="utf-8")


def test_third_party_info_logs_are_kept_out_of_the_file(tmp_path, isolated_root_logger):
    # pykrx logs via the root logger at INFO with a broken format string — it must not reach the file.
    path = configure_run_logging("collect", log_dir=tmp_path)

    logging.info("pykrx noise %s")
    logging.getLogger("pykrx").info("more noise")
    logging.getLogger("pykrx").warning("third-party warning kept")

    text = path.read_text(encoding="utf-8")
    assert "noise" not in text
    assert "third-party warning kept" in text
