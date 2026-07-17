"""Central logging configuration.

Calling :func:`setup_logging` (idempotent) attaches two handlers to the root
logger:

* a console handler (so the CLI still prints to the terminal), and
* a file handler that persists every record to ``logs/pipeline.log``.

All modules acquire a logger via :func:`get_logger` so that messages are
uniformly formatted and archived for later inspection / reporting.
"""

import logging
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_file = LOGS_DIR / "pipeline.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
