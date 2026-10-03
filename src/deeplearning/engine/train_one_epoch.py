"""Single training epoch with optional distributed metric reduction."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    show_progress: bool = True,
    rank=0,
    local_rank=0,
    distributed: bool = False,
) -> dict[str, float]:
    """Run one full pass and return globally reduced loss and accuracy."""
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

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
        
    for images, targets in progress:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.numel()
        loss_sum += loss.item() * batch_size
        correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
        sample_count += batch_size
        if show_progress:
            progress.set_postfix(loss=loss.item())

    stats = torch.tensor(
        [loss_sum, float(correct), float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    total_samples = max(stats[2].item(), 1.0)
    return {
        "loss": stats[0].item() / total_samples,
        "accuracy": stats[1].item() / total_samples,
    }
