"""Run logging: a console+file logger, and a small CSV metrics history logger."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Return a logger that writes to console and, optionally, a log file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """Appends per-epoch metric dicts to a CSV file, writing the header on first use."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        is_new_file = not self.csv_path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)
