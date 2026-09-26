"""Evaluation metrics: accuracy, classification report, confusion matrix plotting."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a batch of logits vs. integer targets."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_classification_report(
    y_true: list[int],
    y_pred: list[int],
    output_path: str | Path | None = None,
) -> dict:
    """Compute a sklearn classification report (precision/recall/F1 per class).
    Optionally writes it as JSON to `output_path`."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
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
    cm = confusion_matrix(y_true, y_pred)
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
