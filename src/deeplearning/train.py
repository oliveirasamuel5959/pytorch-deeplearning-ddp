"""Single training epoch."""
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

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    collect_predictions: bool = False,
    desc: str = "eval",
) -> dict:
    """Run inference over `loader` without gradient updates.

    Returns a dict with 'loss', 'accuracy', and, if `collect_predictions`,
    'y_true' and 'y_pred' lists for building a classification report / confusion matrix.
    """
    model.eval()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    progress = tqdm(loader, desc=desc, leave=False)
    for images, targets in progress:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item()
        running_acc += accuracy(logits, targets)
        n_batches += 1

        if collect_predictions:
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    result = {
        "loss": running_loss / max(n_batches, 1),
        "accuracy": running_acc / max(n_batches, 1),
    }
    if collect_predictions:
        result["y_true"] = y_true
        result["y_pred"] = y_pred
    return result