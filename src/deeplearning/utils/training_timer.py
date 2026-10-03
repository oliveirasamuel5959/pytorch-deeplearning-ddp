```python
"""Training time monitoring utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class TrainingTime:
    """Timing information for a training run."""

    total_seconds: float = 0.0
    epoch_seconds: list[float] | None = None

    def __post_init__(self) -> None:
        if self.epoch_seconds is None:
            self.epoch_seconds = []

    @property
    def average_epoch_seconds(self) -> float:
        """Return the average epoch duration."""
        if not self.epoch_seconds:
            return 0.0

        return sum(self.epoch_seconds) / len(self.epoch_seconds)

    @property
    def total_minutes(self) -> float:
        """Return total training time in minutes."""
        return self.total_seconds / 60.0

    @property
    def total_hours(self) -> float:
        """Return total training time in hours."""
        return self.total_seconds / 3600.0

    def throughput(self, samples: int) -> float:
        """Return training throughput in samples per second."""
        if self.total_seconds <= 0:
            return 0.0

        return samples / self.total_seconds


class TrainingTimer:
    """Measure training time for single-GPU or distributed training."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self._run_start: float | None = None
        self._epoch_start: float | None = None
        self._epoch_times: list[float] = []

    @staticmethod
    def _synchronize() -> None:
        """Synchronize CUDA and distributed workers before timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def start(self) -> None:
        """Start the total training timer."""
        self._synchronize()
        self._run_start = time.perf_counter()

    def start_epoch(self) -> None:
        """Start timing the current epoch."""
        self._synchronize()
        self._epoch_start = time.perf_counter()

    def end_epoch(self) -> float:
        """Stop the epoch timer and return elapsed seconds."""
        self._synchronize()

        if self._epoch_start is None:
            raise RuntimeError(
                "start_epoch() must be called before end_epoch()."
            )

        elapsed = time.perf_counter() - self._epoch_start

        self._epoch_times.append(elapsed)
        self._epoch_start = None

        return elapsed

    def stop(self) -> TrainingTime:
        """Stop the total timer."""
        self._synchronize()

        if self._run_start is None:
            raise RuntimeError(
                "start() must be called before stop()."
            )

        total = time.perf_counter() - self._run_start

        return TrainingTime(
            total_seconds=total,
            epoch_seconds=self._epoch_times.copy(),
        )
