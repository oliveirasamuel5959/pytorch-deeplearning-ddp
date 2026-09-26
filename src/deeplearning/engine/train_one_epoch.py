"""Single training epoch."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeplearning.utils.metrics import accuracy


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Run one full pass over `loader`, updating `model` weights.

    Returns a dict with mean 'loss' and 'accuracy' for the epoch.
    """
    model.train()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0

    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for images, targets in progress:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_acc = accuracy(logits.detach(), targets)
        running_loss += loss.item()
        running_acc += batch_acc
        n_batches += 1
        progress.set_postfix(loss=loss.item(), acc=batch_acc)

    return {
        "loss": running_loss / max(n_batches, 1),
        "accuracy": running_acc / max(n_batches, 1),
    }
