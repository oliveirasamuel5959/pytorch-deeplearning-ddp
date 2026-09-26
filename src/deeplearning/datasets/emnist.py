"""EMNIST dataset loading: download, transforms, and dataloader construction.

Usage:
    from deeplearning.datasets import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg.data)
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Number of classes per official EMNIST split.
EMNIST_NUM_CLASSES = {
    "byclass": 62,
    "bymerge": 47,
    "balanced": 47,
    "letters": 27,
    "digits": 10,
    "mnist": 10,
}

# EMNIST images are stored transposed relative to how they should be viewed;
# this matches torchvision's own documented fix.
class _TransposeEMNIST:
  """Fixes EMNIST's transposed image orientation (picklable, unlike a lambda)."""
  def __call__(self, img):
    return img.transpose(2, 1)

_FIX_ORIENTATION = _TransposeEMNIST()

_MEAN, _STD = (0.1751,), (0.3332)  # approximate EMNIST-balanced statistics

def _build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
  """Return (train_transform, eval_transform)."""
  base = [
    transforms.ToTensor(),
    _FIX_ORIENTATION,
    transforms.Normalize(_MEAN, _STD),
  ]
  eval_transform = transforms.Compose(base)

  if augment:
    train_transform = transforms.Compose(
      [
        transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        *base,
      ]
    )
  else:
    train_transform = eval_transform

  return train_transform, eval_transform


def build_dataloaders(data_cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for the configured EMNIST split.

    `data_cfg` is a `deeplearning.config.DataConfig` (or any object with the
    same attributes: root, split, val_fraction, batch_size, num_workers, augment).
    """
    train_tf, eval_tf = _build_transforms(data_cfg.augment)

    full_train = datasets.EMNIST(
      root=data_cfg.root,
      split=data_cfg.split,
      train=True,
      download=True,
      transform=train_tf,
    )
    test_set = datasets.EMNIST(
      root=data_cfg.root,
      split=data_cfg.split,
      train=False,
      download=True,
      transform=eval_tf,
    )

    val_size = int(len(full_train) * data_cfg.val_fraction)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    # The val split shares the augmented dataset object; swap in eval transform
    # by wrapping the underlying dataset reference used for validation.
    val_set.dataset = datasets.EMNIST(
      root=data_cfg.root,
      split=data_cfg.split,
      train=True,
      download=False,
      transform=eval_tf,
    )

    loader_kwargs = dict(
      batch_size=data_cfg.batch_size,
      num_workers=data_cfg.num_workers,
      pin_memory=torch.cuda.is_available(),  # only pin when there's a GPU to benefit
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
