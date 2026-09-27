"""Reusable, non-interactive plots for datasets and inference results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch


def prepare_image_to_plot(
  img: torch.Tensor,
  mean: Sequence[float],
  std: Sequence[float],
) -> torch.Tensor:
  """Undo normalization and convert a ``(C, H, W)`` image for matplotlib."""
  mean_tensor = torch.as_tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
  std_tensor = torch.as_tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)

  img = (img.detach() * std_tensor + mean_tensor).clamp(0, 1)
  img = img.permute(1, 2, 0).cpu()
  if img.shape[-1] == 1:
    img = img.squeeze(-1)
  return img


def _label_name(label: int, classes: Sequence[str]) -> str:
  return classes[label] if 0 <= label < len(classes) else str(label)


def _save_grid(fig: plt.Figure, output_path: str | Path) -> None:
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.tight_layout()
  fig.savefig(output_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def plot_dataset_samples(
  images: torch.Tensor,
  labels: torch.Tensor,
  classes: Sequence[str],
  mean: Sequence[float],
  std: Sequence[float],
  output_path: str | Path,
  max_samples: int = 32,
) -> None:
  """Save a grid of normalized images with their dataset class labels."""
  count = min(max_samples, len(images), len(labels))
  if count == 0:
    raise ValueError("At least one image and label are required")

  columns = min(8, count)
  rows = (count + columns - 1) // columns
  fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.8, rows * 2.0), squeeze=False)

  for index, ax in enumerate(axes.flat):
    if index >= count:
      ax.axis("off")
      continue
    image = prepare_image_to_plot(images[index], mean, std)
    label_index = int(labels[index].item())
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_title(f"{_label_name(label_index, classes)} ({label_index})", fontsize=9)
    ax.axis("off")

  fig.suptitle("Dataset samples", y=1.02)
  _save_grid(fig, output_path)


def plot_predictions_grid(
  images: torch.Tensor,
  actual_labels: Sequence[int] | torch.Tensor,
  predicted_labels: Sequence[int] | torch.Tensor,
  classes: Sequence[str],
  mean: Sequence[float],
  std: Sequence[float],
  output_path: str | Path,
  max_samples: int = 32,
) -> None:
  """Save a green/red grid comparing actual and predicted class labels."""
  actual = torch.as_tensor(actual_labels).flatten()
  predicted = torch.as_tensor(predicted_labels).flatten()
  count = min(max_samples, len(images), len(actual), len(predicted))
  if count == 0:
    raise ValueError("At least one image and prediction are required")

  columns = min(8, count)
  rows = (count + columns - 1) // columns
  fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 2.2), squeeze=False)

  for index, ax in enumerate(axes.flat):
    if index >= count:
      ax.axis("off")
      continue
    image = prepare_image_to_plot(images[index], mean, std)
    actual_index = int(actual[index].item())
    predicted_index = int(predicted[index].item())
    correct = actual_index == predicted_index
    color = "green" if correct else "red"
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    ax.set_title(
      f"actual: {_label_name(actual_index, classes)}\n"
      f"pred: {_label_name(predicted_index, classes)}",
      color=color,
      fontsize=9,
    )
    ax.axis("off")

  fig.suptitle("Test predictions (green = correct, red = incorrect)", y=1.02)
  _save_grid(fig, output_path)


def emnist_visualization(
  images: torch.Tensor,
  labels: torch.Tensor,
  classes: Sequence[str],
  mean: Sequence[float],
  std: Sequence[float],
  output_path: str | Path = "outputs/dataset_samples.png",
) -> None:
  """Backward-compatible wrapper for the EMNIST sample visualization."""
  plot_dataset_samples(images, labels, classes, mean, std, output_path)
