"""Typed configuration objects loaded from YAML, with dotted-key CLI overrides."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

@dataclass
class DataConfig:
  root: str = "data/"
  split: str = "balanced"
  val_fraction: float = 0.1
  batch_size: int = 128
  num_workers: int = 4
  augment: bool = True


@dataclass
class ModelConfig:
  name: str = "cnn_classifier"
  input_channels: int = 1
  num_classes: int = 62
  dropout: float = 0.3


@dataclass
class TrainSettings:
  epochs: int = 10
  lr: float = 1e-3
  weight_decay: float = 1e-4
  optimizer: str = "adam"
  scheduler: str = "cosine"
  early_stopping_patience: int = 5
  seed: int = 42
  device: str = "auto"


@dataclass
class OutputConfig:
  root: str = "outputs/"
  save_every_epoch: bool = False


@dataclass
class TrainConfig:
  run_name: str = "emnist_cnn"
  data: DataConfig = field(default_factory=DataConfig)
  model: ModelConfig = field(default_factory=ModelConfig)
  train: TrainSettings = field(default_factory=TrainSettings)
  output: OutputConfig = field(default_factory=OutputConfig)

  @classmethod
  def from_yaml(cls, path: str | Path) -> "TrainConfig":
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return cls.from_dict(raw)

  @classmethod
  def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
    return cls(
      run_name=raw.get("run_name", "emnist_cnn"),
      data=DataConfig(**raw.get("data", {})),
      model=ModelConfig(**raw.get("model", {})),
      train=TrainSettings(**raw.get("train", {})),
      output=OutputConfig(**raw.get("output", {})),
    )

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  def apply_overrides(self, overrides: dict[str, Any]) -> "TrainConfig":
    """Apply dotted-key overrides, e.g. {'train.lr': 0.0005}. Returns a new config."""
    cfg = copy.deepcopy(self)
    for dotted_key, value in overrides.items():
      if value is None:
        continue
      section, _, key = dotted_key.partition(".")
      if not key:
        setattr(cfg, section, value)
        continue
      target = getattr(cfg, section)
      setattr(target, key, value)
    return cfg

  def run_dir(self) -> Path:
    return Path(self.output.root) / self.run_name
