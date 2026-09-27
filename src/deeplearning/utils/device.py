"""Device selection and distributed process-group helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Process information shared by the training and logging layers."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def resolve_device(preference: str = "auto", local_rank: int | None = None) -> torch.device:
    """Resolve ``auto``, CPU, CUDA, or MPS to a concrete device.

    DDP workers pass ``local_rank`` so each process selects its assigned GPU.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            preference = "cuda"
        elif torch.backends.mps.is_available():
            preference = "mps"
        else:
            preference = "cpu"

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        return torch.device("cuda" if local_rank is None else f"cuda:{local_rank}")

    return torch.device(preference)


def get_distributed_context() -> DistributedContext:
    """Read the active process-group state, or return a single-process context."""
    if dist.is_available() and dist.is_initialized():
        return DistributedContext(
            rank=dist.get_rank(),
            local_rank=int(os.environ.get("LOCAL_RANK", dist.get_rank())),
            world_size=dist.get_world_size(),
            enabled=True,
        )
    return DistributedContext()


def ddp_setup(
    rank: int,
    world_size: int,
    local_rank: int | None = None,
    backend: str = "nccl",
) -> DistributedContext:
    """Initialize a process group using launcher-provided rendezvous settings."""
    if not dist.is_available():
        raise RuntimeError("This PyTorch installation does not provide torch.distributed")
    if dist.is_initialized():
        return get_distributed_context()
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with NCCL requires CUDA")
        if local_rank is None:
            local_rank = rank
        torch.cuda.set_device(local_rank)
    else:
        local_rank = rank if local_rank is None else local_rank

    # torchrun supplies these values. Direct spawning supplies them in the parent.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(rank, local_rank, world_size, enabled=True)


def ddp_cleanup() -> None:
    """Destroy the process group if this process initialized one."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
