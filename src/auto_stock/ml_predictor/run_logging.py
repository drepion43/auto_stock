"""학습/수집 스크립트 공용 로그 설정 — 화면 + 타임스탬프 로그 파일에 같은 내용을 남긴다.

root 로거는 WARNING으로 둔다: pykrx가 root 로거에 잘못된 포맷의 `logging.info(args, kwargs)`를
호출해서, root를 INFO로 올리면 "--- Logging error ---" 덤프가 로그를 덮는다(2026-10-14 실측).
프로젝트 로거(`auto_stock.*` — 수집 진행 줄, epoch 줄)만 INFO로 올린다. 서드파티의 WARNING 이상은
그대로 남는다."""

import logging
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_DIR = Path("logs/ml_training")
_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_run_logging(prefix: str, log_dir: Path | str = DEFAULT_LOG_DIR) -> Path:
    """콘솔 + `{log_dir}/{prefix}_{YYYYmmdd_HHMMSS}.log`(UTF-8) 핸들러를 root에 붙이고 파일 경로를 반환한다."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    logging.getLogger("auto_stock").setLevel(logging.INFO)
    return path
