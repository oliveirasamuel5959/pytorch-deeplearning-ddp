"""Device selection helper."""
import torch
import os
from torch.distributed import init_process_group

def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve 'auto' | 'cpu' | 'cuda' | 'mps' into a concrete torch.device,
    falling back gracefully if the requested backend is unavailable."""
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def ddp_setup(rank: int, world_size: int) -> None:
    """Initialize the distributed process group for DDP training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group("nccl", rank=rank, world_size=world_size)
