"""EMNIST dataset loading: download, transforms, and dataloader construction.

Usage:
    from deeplearning.datasets import build_dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(cfg.data)
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler, Subset, random_split
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

_MEAN, _STD = (0.1751,), (0.3332,)  # approximate EMNIST-balanced statistics


def get_class_names(dataset: Dataset) -> list[str]:
  """Return human-readable labels from a torchvision dataset or ``Subset``."""
  while isinstance(dataset, Subset):
    dataset = dataset.dataset

  classes = getattr(dataset, "classes", None)
  if classes is None:
    raise AttributeError("The dataset does not expose a 'classes' attribute")
  return [str(class_name) for class_name in classes]


def get_emnist_class_names(root: str, split: str) -> list[str]:
  """Load EMNIST metadata and return its class labels."""
  dataset = datasets.EMNIST(root=root, split=split, train=False, download=True)
  return get_class_names(dataset)


class DistributedEvalSampler(Sampler[int]):
  """Shard evaluation data without padding or duplicating examples."""

  def __init__(self, dataset: Dataset) -> None:
    if not dist.is_available() or not dist.is_initialized():
      raise RuntimeError("DistributedEvalSampler requires an initialized process group")
    self.dataset_size = len(dataset)
    self.rank = dist.get_rank()
    self.world_size = dist.get_world_size()

  def __iter__(self):
    return iter(range(self.rank, self.dataset_size, self.world_size))

  def __len__(self) -> int:
    if self.rank >= self.dataset_size:
      return 0
    return (self.dataset_size - self.rank + self.world_size - 1) // self.world_size

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
    distributed = data_cfg.train_mode == "ddp" and dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    if distributed and rank != 0:
      # Rank zero downloads first; other ranks wait before opening the files.
      dist.barrier()
    download = not distributed or rank == 0

    full_train = datasets.EMNIST(
      root=data_cfg.root,
      split=data_cfg.split,
      train=True,
      download=download,
      transform=train_tf,
    )
    test_set = datasets.EMNIST(
      root=data_cfg.root,
      split=data_cfg.split,
      train=False,
      download=download,
      transform=eval_tf,
    )
    if distributed and rank == 0:
      dist.barrier()

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

    if data_cfg.train_mode == "ddp":
      train_sampler = DistributedSampler(train_set, shuffle=True)
      val_sampler = DistributedEvalSampler(val_set)
      test_sampler = DistributedEvalSampler(test_set)
    else:
      train_sampler = val_sampler = test_sampler = None

    train_loader = DataLoader(
      train_set,
      shuffle=train_sampler is None,
      sampler=train_sampler,
      **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, sampler=test_sampler, **loader_kwargs)

    return train_loader, val_loader, test_loader
