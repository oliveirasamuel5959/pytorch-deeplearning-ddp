#!/usr/bin/env python
"""CLI entrypoint for training an EMNIST model.

Usage:
    uv run scripts/train.py --config configs/default.yaml
    uv run scripts/train.py --config configs/default.yaml --epochs 20 --lr 0.0005
"""

from __future__ import annotations

import argparse
import json
import sys

from deeplearning.config import TrainConfig
from deeplearning.datasets.emnist import build_dataloaders, EMNIST_NUM_CLASSES
from deeplearning.engine.train import run_training
from deeplearning.models.cnn_mlp import build_model
from deeplearning.utils.device import resolve_device
from deeplearning.utils.seed import set_seed
from deeplearning.utils.logger import get_logger

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train a CNN on EMNIST")
  parser.add_argument("--config", type=str, default="configs/default.yaml")
  parser.add_argument("--run-name", type=str, default=None)
  parser.add_argument("--epochs", type=int, default=None)
  parser.add_argument("--lr", type=float, default=None)
  parser.add_argument("--batch-size", type=int, default=None)
  parser.add_argument("--split", type=str, default=None, help="EMNIST split, e.g. balanced, digits, letters")
  parser.add_argument("--model", type=str, default=None, help="Model registry key, e.g. simple_cnn, deep_cnn")
  parser.add_argument("--device", type=str, default=None, help="auto | cpu | cuda | mps")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  cfg = TrainConfig.from_yaml(args.config)

  overrides = {
      "run_name": args.run_name,
      "train.epochs": args.epochs,
      "train.lr": args.lr,
      "train.device": args.device,
      "data.batch_size": args.batch_size,
      "data.split": args.split,
      "model.name": args.model,
  }
  cfg = cfg.apply_overrides(overrides)

  # Keep num_classes consistent with the chosen split unless the user set it explicitly.
  if args.split is not None:
    cfg.model.num_classes = EMNIST_NUM_CLASSES.get(args.split, cfg.model.num_classes)

  set_seed(cfg.train.seed)
  device = resolve_device(cfg.train.device)

  model = build_model(cfg.model.input_channels, cfg.model.num_classes)

  summary = run_training(model, cfg, device)

  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  sys.exit(main())
