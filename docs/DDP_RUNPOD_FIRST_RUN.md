# PyTorch DDP on RunPod

This document records the working Distributed Data Parallel (DDP) configuration used to train PyTorch models on a RunPod instance with **2 × NVIDIA L4 GPUs**.

## Environment

| Component | Configuration |
|---|---|
| Platform | RunPod |
| GPUs | 2 × NVIDIA L4 |
| NVIDIA Driver | 570.195.03 |
| Host CUDA support | 12.8 |
| PyTorch | 2.11.0 |
| PyTorch CUDA | cu128 |
| DDP backend | NCCL |
| DDP processes | 2 |
| Batch size / rank | 128 |
| Effective batch size | 256 |
| NCCL P2P | Disabled |

## 1. CUDA / PyTorch Compatibility

The RunPod host supports CUDA 12.8, so PyTorch must use a compatible CUDA 12.8 wheel.

`pyproject.toml`:

```toml
[tool.uv]
package = true
environments = ["sys_platform == 'linux'"]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128" },
]
torchvision = [
    { index = "pytorch-cu128" },
]
```

Dependencies:

```toml
"torch==2.11.0",
"torchvision==0.26.0",
```

Verify the environment:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected:

```text
2.11.0
12.8
True
2
```

## 2. DDP Launch

Current command:

```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --train-mode ddp \
  --nproc-per-node=2 \
  --device cuda \
  --run-name emnist_ddp
```

For `N` GPUs, use:

```bash
--nproc-per-node=N
```

For the current setup, two processes are created:

```text
rank 0 → GPU 0
rank 1 → GPU 1
```

## 3. Rank / GPU Mapping

For a single-node 2-GPU setup:

```text
Process 0
    rank       = 0
    local_rank = 0
    device     = cuda:0
    GPU        = NVIDIA L4 #0

Process 1
    rank       = 1
    local_rank = 1
    device     = cuda:1
    GPU        = NVIDIA L4 #1
```

`rank` identifies the process globally. `local_rank` identifies its GPU on the current node. `world_size` is the total number of processes.

A DDP worker should use:

```python
local_rank = int(os.environ.get("LOCAL_RANK", rank))

ddP_setup(
    rank=rank,
    world_size=world_size,
    local_rank=local_rank,
)

device = resolve_device(
    cfg.train.device,
    local_rank=local_rank,
)
```

Correct GPU initialization:

```python
torch.cuda.set_device(local_rank)

dist.init_process_group(
    backend="nccl",
    rank=rank,
    world_size=world_size,
    device_id=torch.device(f"cuda:{local_rank}"),
)
```

The explicit `device_id` prevents NCCL from having to guess which device a process is using.

## 4. NCCL Configuration on RunPod

The default NCCL P2P path timed out in this RunPod environment.

The working workaround is:

```bash
export NCCL_P2P_DISABLE=1
```

After this change, a minimal NCCL `all_reduce` test succeeded:

```text
RANK 0: after all_reduce = 1.0
RANK 1: after all_reduce = 1.0
```

NCCL then uses alternative communication paths such as SHM/socket communication.

This is an environment-specific workaround. It does not mean NCCL itself is broken.

## 5. GPU Topology and Diagnostics

Check topology:

```bash
nvidia-smi topo -m
```

The current environment reports:

```text
GPU 0 <-> GPU 1 = SYS
```

There is no NVLink between the GPUs.

Monitor both GPUs during training:

```bash
watch -n 1 nvidia-smi
```

Or:

```bash
watch -n 1 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv'
```

## 6. DistributedSampler

DDP partitions the training dataset between processes using `DistributedSampler`.

Conceptually:

```text
                    Training dataset
                         4908
                           │
                  DistributedSampler
                     /           \
                    /             \
                 Rank 0          Rank 1
                 GPU 0           GPU 1
                  │                │
               2454             2454
                batches          batches
```

The `2454` value is the number of DataLoader batches for each rank, not necessarily the number of images.

With:

```text
batch_size_per_rank = 128
world_size = 2
```

the effective batch size is:

```text
128 × 2 = 256
```

Call `set_epoch()` every epoch:

```python
if isinstance(train_loader.sampler, DistributedSampler):
    train_loader.sampler.set_epoch(epoch)
```

This allows the sampler to reshuffle consistently between epochs.

## 7. DDP Architecture

```text
                         RunPod
              ┌──────────────────────────┐
              │       NVIDIA L4 × 2      │
              │                          │
              │   GPU 0        GPU 1     │
              │   cuda:0       cuda:1    │
              └─────┬────────────┬───────┘
                    │            │
                 rank 0       rank 1
                    │            │
                    ▼            ▼
                 Model 0      Model 1
                    │            │
                 DataLoader   DataLoader
                    │            │
                  2454          2454
                 batches       batches
                    │            │
                    └─────┬──────┘
                          │
                    NCCL all-reduce
                          │
                    gradient sync
                          │
                     next batch
```

Each rank owns a model replica. During backpropagation, DDP synchronizes gradients between replicas.

## 8. Progress Bars for Both GPUs

For DDP, display one progress bar per rank:

```python
if distributed:
    progress = tqdm(
        loader,
        desc=f"Rank {rank} / GPU {local_rank}: Epoch {epoch} [train]",
        position=rank,
        leave=True,
        disable=not show_progress,
    )
else:
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} [train]",
        leave=False,
        disable=not show_progress,
    )
```

Expected output:

```text
Rank 0 / GPU 0: Epoch 1 [train]: 100%|...| 2454/2454 [00:37<00:00, 64.64it/s]
Rank 1 / GPU 1: Epoch 1 [train]: 100%|...| 2454/2454 [00:38<00:00, 64.10it/s]
```

Using `position=rank` prevents both workers from overwriting the same terminal line.

## 9. GPU Assignment Verification

After moving the model to the device:

```python
model = model.to(device)

print(
    f"[DDP] rank={context.rank} "
    f"local_rank={context.local_rank} "
    f"device={device} "
    f"gpu={torch.cuda.get_device_name(device)}",
    flush=True,
)
```

Expected:

```text
[DDP] rank=0 local_rank=0 device=cuda:0 gpu=NVIDIA L4
[DDP] rank=1 local_rank=1 device=cuda:1 gpu=NVIDIA L4
```

Use `nvidia-smi` for actual utilization and memory verification.

## 10. Training Time Measurement

CUDA operations are asynchronous, so timing should synchronize CUDA before taking measurements.

For DDP, workers should also synchronize with `dist.barrier()`.

A suitable timer implementation is:

```python
"""Training time monitoring utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class TrainingTime:
    total_seconds: float = 0.0
    epoch_seconds: list[float] | None = None

    def __post_init__(self) -> None:
        if self.epoch_seconds is None:
            self.epoch_seconds = []

    @property
    def average_epoch_seconds(self) -> float:
        if not self.epoch_seconds:
            return 0.0
        return sum(self.epoch_seconds) / len(self.epoch_seconds)

    @property
    def total_minutes(self) -> float:
        return self.total_seconds / 60.0

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600.0

    def throughput(self, samples: int) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return samples / self.total_seconds


class TrainingTimer:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self._run_start: float | None = None
        self._epoch_start: float | None = None
        self._epoch_times: list[float] = []

    @staticmethod
    def _synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def start(self) -> None:
        self._synchronize()
        self._run_start = time.perf_counter()

    def start_epoch(self) -> None:
        self._synchronize()
        self._epoch_start = time.perf_counter()

    def end_epoch(self) -> float:
        self._synchronize()

        if self._epoch_start is None:
            raise RuntimeError("start_epoch() must be called before end_epoch().")

        elapsed = time.perf_counter() - self._epoch_start
        self._epoch_times.append(elapsed)
        self._epoch_start = None
        return elapsed

    def stop(self) -> TrainingTime:
        self._synchronize()

        if self._run_start is None:
            raise RuntimeError("start() must be called before stop().")

        total = time.perf_counter() - self._run_start

        return TrainingTime(
            total_seconds=total,
            epoch_seconds=self._epoch_times.copy(),
        )
```

## 11. Timing Integration

```python
from utils.training_timer import TrainingTimer

timer = TrainingTimer(device)
timer.start()

for epoch in range(1, cfg.train.epochs + 1):
    timer.start_epoch()

    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)

    train_metrics = train_one_epoch(...)
    val_metrics = evaluate(...)

    epoch_time = timer.end_epoch()

    if context.is_main_process:
        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"epoch_time={epoch_time:.2f}s"
        )

training_time = timer.stop()
```

## 12. Metrics JSON

Store timing together with the existing metrics:

```text
outputs/
├── emnist_single_gpu/
│   └── metrics/
│       └── metrics.json
└── emnist_ddp/
    └── metrics/
        └── metrics.json
```

Example:

```json
{
  "training": {
    "epochs": 2,
    "batch_size_per_rank": 128,
    "effective_batch_size": 256,
    "world_size": 2
  },
  "performance": {
    "total_training_seconds": 78.4,
    "total_training_minutes": 1.31,
    "average_epoch_seconds": 39.2,
    "epoch_seconds": [38.5, 39.9],
    "training_samples": 4908,
    "throughput_samples_per_second": 1252.6
  }
}
```

The global number of samples should be used when calculating distributed throughput, rather than accidentally using only the local rank's sample count.

## 13. Performance Comparison

A simple comparison utility:

```python
import json
from pathlib import Path
from typing import Any


def load_metrics(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compare_training_runs(
    single_gpu_metrics: str | Path,
    ddp_metrics: str | Path,
) -> dict[str, float]:
    single = load_metrics(single_gpu_metrics)
    ddp = load_metrics(ddp_metrics)

    single_time = single["performance"]["total_training_seconds"]
    ddp_time = ddp["performance"]["total_training_seconds"]

    if ddp_time <= 0:
        raise ValueError("DDP training time must be greater than zero.")

    speedup = single_time / ddp_time
    time_reduction = (single_time - ddp_time) / single_time * 100.0

    return {
        "single_gpu_seconds": single_time,
        "ddp_seconds": ddp_time,
        "speedup": speedup,
        "time_reduction_percent": time_reduction,
    }
```

### Speedup

```text
speedup = single_gpu_training_time / ddp_training_time
```

### Time reduction

```text
time_reduction =
    (single_gpu_time - ddp_time)
    / single_gpu_time
    × 100
```

## 14. Experimental Consideration

The current configuration uses:

```text
Single GPU:
    batch_size = 128

2-GPU DDP:
    batch_size_per_rank = 128
    world_size = 2

Effective DDP batch size:
    128 × 2 = 256
```

Therefore, a direct timing comparison changes both GPU count and effective batch size.

For a controlled scaling experiment, keep the effective batch size constant when appropriate. For example:

```text
Single GPU:
    batch size = 128

2-GPU DDP:
    batch size/rank = 64

Effective batch size:
    64 × 2 = 128
```

Which comparison is appropriate depends on whether the goal is practical throughput or controlled scaling analysis.

## 15. Working Training Output

The working setup produced output similar to:

```text
Rank 0 / GPU 0: Epoch 1 [train]: 100%|...| 2454/2454 [00:37<00:00, 64.64it/s]
Rank 1 / GPU 1: Epoch 1 [train]: 100%|...| 2454/2454 [00:38<00:00, 64.10it/s]

Rank 0 / GPU 0: Epoch 1 [validation]: 100%|...| 273/273 [00:19<00:00, 14.03it/s]
Rank 1 / GPU 1: Epoch 1 [validation]: 100%|...| 273/273 [00:19<00:00, 13.91it/s]

Epoch 001 | train_loss=0.5890 train_acc=0.8039 | val_loss=0.4053 val_acc=0.8526
New best val_acc=0.8526, checkpoint saved
```

Both ranks are processing their own batches concurrently.

## 16. Troubleshooting Checklist

### CUDA unavailable

Check:

```bash
nvidia-smi
```

and:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Make sure the PyTorch CUDA wheel is compatible with the host driver.

### NCCL timeout

Try:

```bash
export NCCL_P2P_DISABLE=1
```

Then rerun the NCCL communication test.

### Only one GPU is active

Verify:

```text
--nproc-per-node=2
```

and check rank mapping:

```text
rank=0 local_rank=0 device=cuda:0
rank=1 local_rank=1 device=cuda:1
```

Also monitor:

```bash
nvidia-smi
```

### Only rank 0 progress bar appears

Do not condition progress display only on:

```python
context.is_main_process
```

Use `distributed` plus `position=rank` so each DDP worker has its own progress bar.

### NCCL device mapping warning

Make sure both are used before/while initializing NCCL:

```python
torch.cuda.set_device(local_rank)
```

and:

```python
dist.init_process_group(
    backend="nccl",
    rank=rank,
    world_size=world_size,
    device_id=torch.device(f"cuda:{local_rank}"),
)
```

## 17. Recommended Repository Structure

```text
project/
├── configs/
│   └── default.yaml
├── scripts/
│   └── train.py
├── utils/
│   ├── __init__.py
│   ├── device.py
│   ├── performance.py
│   └── training_timer.py
├── outputs/
│   ├── emnist_single_gpu/
│   │   └── metrics/
│   │       └── metrics.json
│   └── emnist_ddp/
│       └── metrics/
│           └── metrics.json
├── pyproject.toml
└── DDP_RUNPOD.md
```

## 18. Quick Reference

Check GPUs:

```bash
nvidia-smi
```

Check topology:

```bash
nvidia-smi topo -m
```

Monitor GPUs:

```bash
watch -n 1 nvidia-smi
```

Disable the problematic P2P path for this RunPod environment:

```bash
export NCCL_P2P_DISABLE=1
```

Start DDP:

```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --train-mode ddp \
  --nproc-per-node=2 \
  --device cuda \
  --run-name emnist_ddp
```

Effective batch size:

```text
effective_batch_size = batch_size_per_rank × world_size
```

Current:

```text
128 × 2 = 256
```

Speedup:

```text
speedup = single_gpu_time / ddp_time
```

## 19. Summary

The working RunPod DDP configuration depends on three key points:

1. **Match PyTorch CUDA to the host:** RunPod CUDA 12.8 → PyTorch cu128.
2. **Explicitly map each process to its GPU:** rank 0 → `cuda:0`, rank 1 → `cuda:1`.
3. **Disable the problematic NCCL P2P path:**

```bash
export NCCL_P2P_DISABLE=1
```

With these settings, the 2-GPU DDP job successfully runs both ranks concurrently, partitions data using `DistributedSampler`, synchronizes gradients through NCCL, displays both GPU workers, and records timing information for single-GPU versus DDP performance analysis.
