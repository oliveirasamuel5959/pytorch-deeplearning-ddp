# Distributed Data Parallel Implementation Plan

## Current Issues

The existing DDP path needs correction before multi-GPU training:

1. `scripts/train.py` initializes one process while using the number of GPUs as `world_size`.
2. `--train-mode` is parsed but is not applied to the configuration.
3. The local GPU rank is not used when selecting the CUDA device.
4. The process-group setup hardcodes the rendezvous address and port instead of respecting `torchrun` environment variables.
5. Distributed samplers do not receive `set_epoch(epoch)`.
6. Validation and test metrics are local to each rank instead of globally reduced.
7. Early stopping and checkpoint decisions are not synchronized.
8. DDP checkpoint files omit metrics and save keys that cannot be loaded directly into a DDP wrapper.
9. All ranks can write logs, plots, reports, and checkpoints concurrently.
10. The process group is not destroyed after training.

## Implementation

1. Support both `torchrun` and direct Python execution with `torch.multiprocessing.spawn`.
2. Use `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` for `torchrun` workers.
3. Add `--nproc-per-node` for direct multi-GPU spawning.
4. Validate CUDA availability and requested GPU count, failing clearly when unavailable.
5. Select `cuda:<local_rank>` and initialize NCCL from the launcher environment.
6. Add explicit rank, local-rank, world-size, and main-process state.
7. Call `DistributedSampler.set_epoch(epoch)` before every training epoch.
8. Reduce sample-weighted loss, correct predictions, and sample counts across ranks.
9. Synchronize early stopping and best-checkpoint decisions using global metrics.
10. Save portable unwrapped model weights and full metadata from rank zero only.
11. Make checkpoint loading work for both plain and DDP-wrapped models.
12. Gather test predictions for rank-zero reports and plots.
13. Restrict metrics, plots, reports, checkpoints, and final summaries to rank zero.
14. Log DDP mode, rank, world size, GPU count, effective batch size, epochs per rank, and output paths.
15. Destroy the process group in a `finally` block.

## Commands

Single GPU:

```bash
python scripts/train.py --config configs/emnist.yaml --train-mode single --device cuda
```

Using `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/emnist.yaml --train-mode ddp --device cuda
```

Direct Python spawning:

```bash
python scripts/train.py --config configs/emnist.yaml \
  --train-mode ddp --nproc-per-node=2 --device cuda
```

Each rank runs every configured epoch over a different data shard. The effective global batch size is `batch_size * world_size`. Checkpoints remain loadable by the existing single-GPU Predictor and are written under `outputs/<run_name>/` by rank zero.
