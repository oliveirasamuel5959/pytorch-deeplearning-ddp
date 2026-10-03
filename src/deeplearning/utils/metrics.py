"""Evaluation metrics: accuracy, classification report, confusion matrix plotting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix

from deeplearning.utils.training_timer import TrainingTime


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a batch of logits vs. integer targets."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_classification_report(
    y_true: list[int],
    y_pred: list[int],
    output_path: str | Path | None = None,
    class_names: Sequence[str] | None = None,
) -> dict:
    """Compute a sklearn classification report (precision/recall/F1 per class).
    Optionally writes it as JSON to `output_path`."""
    report_kwargs = {"output_dict": True, "zero_division": 0}
    if class_names is not None:
        report_kwargs.update(labels=list(range(len(class_names))), target_names=list(class_names))
    report = classification_report(y_true, y_pred, **report_kwargs)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: str | Path,
    class_names: list[str] | None = None,
    normalize: bool = True,
) -> None:
    """Compute and save a confusion matrix heatmap as a PNG."""
    labels = list(range(len(class_names))) if class_names is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig_size = max(6, len(cm) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_names is not None and len(class_names) <= 30:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_history(
    history: Mapping[str, Sequence[float]],
    output_path: str | Path,
) -> None:
    """Plot train/validation loss and accuracy and save the figure."""
    required = ("train_loss", "val_loss", "train_accuracy", "val_accuracy")
    missing = [key for key in required if key not in history]
    if missing:
        raise KeyError(f"Training history is missing: {', '.join(missing)}")

    epochs = history.get("epoch", range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Training history")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_performance_metrics(
    training_time: TrainingTime,
    train_samples: int,
    cfg,
    output_path: str | Path,
) -> None:
    """Save performance metrics as a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    performance_metrics = {
        "total_training_seconds": training_time.total_seconds,
        "total_training_minutes": training_time.total_minutes,
        "total_training_hours": training_time.total_hours,
        "average_epoch_seconds": training_time.average_epoch_seconds,
        "epoch_seconds": training_time.epoch_seconds,
        "training_samples": train_samples,
        "throughput_samples_per_second": training_time.throughput(
            train_samples * cfg.train.epochs
        ),
    }
    
    with open(output_path, "w") as f:
        json.dump(performance_metrics, f, indent=4)