#!/usr/bin/env python
"""Train an EMNIST model in single-process or DistributedDataParallel mode."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

import torch
import torch.multiprocessing as mp

from deeplearning.config import TrainConfig
from deeplearning.datasets.emnist import EMNIST_NUM_CLASSES
from deeplearning.engine.train import run_training
from deeplearning.models.cnn_mlp import build_model
from deeplearning.utils.device import ddp_cleanup, ddp_setup, get_distributed_context, resolve_device
from deeplearning.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on EMNIST")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--train-mode", choices=["single", "ddp"], default=None)
    parser.add_argument("--nproc-per-node", type=int, default=None, help="GPU processes when using direct Python DDP")
    parser.add_argument("--master-port", type=int, default=None, help="Rendezvous port for direct Python DDP")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--split", type=str, default=None, help="EMNIST split, e.g. balanced, digits, letters")
    parser.add_argument("--model", type=str, default=None, help="Model registry key used at training time")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    return parser.parse_args()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _validate_ddp_request(world_size: int, device_preference: str) -> None:
    gpu_count = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("The requested DDP process count must be at least one")
    if device_preference not in {"auto", "cuda"}:
        raise ValueError("DDP requires CUDA/NCCL; use --train-mode single for CPU or MPS training")
    if gpu_count == 0:
        raise RuntimeError("DDP was requested, but no CUDA devices are available")
    if gpu_count < world_size:
        raise RuntimeError(
            f"DDP requested {world_size} GPU processes, but only {gpu_count} CUDA devices are available"
        )
    if world_size < 2:
        raise ValueError("DDP requires at least two processes; use --train-mode single for one GPU")


def _run_worker(rank: int, world_size: int, cfg: TrainConfig) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ddp_setup(rank=rank, world_size=world_size, local_rank=local_rank)
    try:
        set_seed(cfg.train.seed)
        device = resolve_device(cfg.train.device, local_rank=local_rank)
        model = build_model(cfg.model.input_channels, cfg.model.num_classes)
        summary = run_training(model, cfg, device)
        context = get_distributed_context()
        if context.is_main_process:
            print(json.dumps(summary, indent=2))
    finally:
        ddp_cleanup()


def _run_single(cfg: TrainConfig) -> None:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    model = build_model(cfg.model.input_channels, cfg.model.num_classes)
    summary = run_training(model, cfg, device)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    cfg = cfg.apply_overrides(
        {
            "run_name": args.run_name,
            "data.train_mode": args.train_mode,
            "train.epochs": args.epochs,
            "train.lr": args.lr,
            "train.device": args.device,
            "data.batch_size": args.batch_size,
            "data.split": args.split,
            "model.name": args.model,
        }
    )

    if args.split is not None:
        cfg.model.num_classes = EMNIST_NUM_CLASSES.get(args.split, cfg.model.num_classes)

    launched_by_torchrun = "RANK" in os.environ or "LOCAL_RANK" in os.environ
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.data.train_mode == "single":
        if launched_by_torchrun:
            raise RuntimeError("A torchrun process group is active; use --train-mode ddp")
        if args.nproc_per_node is not None:
            raise ValueError("--nproc-per-node is only valid with --train-mode ddp")
        _run_single(cfg)
        return

    if launched_by_torchrun:
        if args.nproc_per_node is not None and args.nproc_per_node != launched_world_size:
            raise ValueError("--nproc-per-node does not match torchrun's WORLD_SIZE")
        _validate_ddp_request(launched_world_size, cfg.train.device)
        _run_worker(
            rank=int(os.environ["RANK"]),
            world_size=launched_world_size,
            cfg=cfg,
        )
        return

    world_size = args.nproc_per_node or torch.cuda.device_count()
    _validate_ddp_request(world_size, cfg.train.device)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port or _free_port())
    mp.spawn(_run_worker, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == "__main__":
    sys.exit(main())
