"""Evaluation loop with globally reduced metrics and gathered predictions."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    collect_predictions: bool = False,
    desc: str = "eval",
    show_progress: bool = True,
    rank: int = 0,
    local_rank: int = 0,
    distributed: bool = False,
) -> dict:
    """Evaluate a loader and reduce metrics across all active DDP ranks."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    if distributed:
        progress = tqdm(
            loader,
            desc=f"Rank {rank} / GPU {local_rank}: {desc}",
            position=rank,
            leave=True,
            disable=not show_progress,
        )
    else:
        progress = tqdm(loader, desc=desc, disable=not show_progress)
        
    for images, targets in progress:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.numel()
        loss_sum += loss.item() * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        sample_count += batch_size

        if collect_predictions:
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    stats = torch.tensor(
        [loss_sum, float(correct), float(sample_count)],
        dtype=torch.float64,
        device=device,
    )
    if distributed and dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    if collect_predictions and distributed:
        gathered: list[dict[str, list[int]] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, {"y_true": y_true, "y_pred": y_pred})
        if dist.get_rank() == 0:
            y_true = [value for item in gathered if item for value in item["y_true"]]
            y_pred = [value for item in gathered if item for value in item["y_pred"]]
        else:
            y_true, y_pred = [], []

    total_samples = max(stats[2].item(), 1.0)
    result = {
        "loss": stats[0].item() / total_samples,
        "accuracy": stats[1].item() / total_samples,
    }
    if collect_predictions:
        result["y_true"] = y_true
        result["y_pred"] = y_pred
    return result
