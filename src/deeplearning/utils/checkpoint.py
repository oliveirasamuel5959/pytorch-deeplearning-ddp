"""Model checkpoint save/load utilities."""
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a full training checkpoint (model + optimizer + metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics or {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into `model` (and `optimizer` if given). Returns the raw payload."""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def find_best_checkpoint(
    checkpoint_dir: str | Path,
    pattern: str = "*.pt",
    metric_key: str = "loss",
    mode: str = "min",
) -> Path:
    """Scan checkpoint_dir for files matching `pattern`, read each checkpoint's
    stored `metrics[metric_key]` payload, and return the path with the best value.

    Reads metrics from inside each checkpoint (not the filename), so it's exact
    and doesn't depend on your naming convention.
    """
    checkpoint_dir = Path(checkpoint_dir)
    candidates = list(checkpoint_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching '{pattern}' in {checkpoint_dir}")

    best_path, best_value = None, None
    for path in candidates:
        payload = torch.load(path, map_location="cpu")
        metrics = payload.get("metrics", {})
        if metric_key not in metrics:
            continue
        value = metrics[metric_key]
        is_better = best_value is None or (
            value < best_value if mode == "min" else value > best_value
        )
        if is_better:
            best_path, best_value = path, value

    if best_path is None:
        raise ValueError(f"No checkpoint in {checkpoint_dir} had metric '{metric_key}'")

    return best_path

def remove_previous(checkpoint_dir: Path, suffix: str) -> None:
    """Delete any existing checkpoint file ending in `suffix` (e.g. '_best.pt' or '_last.pt')
    before a new one is saved, so only the single most recent one of that kind is kept."""
    for old_file in checkpoint_dir.glob(f"*{suffix}"):
        old_file.unlink()