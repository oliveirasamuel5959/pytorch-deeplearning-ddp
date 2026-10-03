"""Full training orchestration for single-process and DDP execution."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from deeplearning.config import TrainConfig
from deeplearning.datasets.emnist import build_dataloaders, get_class_names
from deeplearning.engine.evaluate import evaluate
from deeplearning.engine.train_one_epoch import train_one_epoch
from deeplearning.utils.checkpoint import find_best_checkpoint, load_checkpoint, remove_previous, save_checkpoint
from deeplearning.utils.device import get_distributed_context, verify_gpu_assign
from deeplearning.utils.logger import MetricsLogger, get_logger
from deeplearning.utils.metrics import compute_classification_report, plot_confusion_matrix, plot_training_history, save_performance_metrics
from deeplearning.utils.plots import plot_dataset_samples, plot_predictions_grid
from deeplearning.utils.training_timer import TrainingTimer


def _build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.train.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if cfg.train.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay
        )
    raise ValueError(f"Unknown optimizer '{cfg.train.optimizer}'")


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    if cfg.train.scheduler == "none":
        return None
    if cfg.train.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.train.epochs // 3), gamma=0.1)
    if cfg.train.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    raise ValueError(f"Unknown scheduler '{cfg.train.scheduler}'")


def _barrier(context_enabled: bool) -> None:
    if context_enabled:
        dist.barrier()


def _broadcast_stop(should_stop: bool, device: torch.device, context_enabled: bool) -> bool:
    if not context_enabled:
        return should_stop
    value = torch.tensor(int(should_stop), device=device)
    dist.broadcast(value, src=0)
    return bool(value.item())


def run_training(
    model: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    
    timer = TrainingTimer(device=device)
    timer.start()
    
    """Train a model and write rank-zero artifacts under ``cfg.run_dir()``."""
    context = get_distributed_context()
    ddp_enabled = cfg.data.train_mode == "ddp"
    if ddp_enabled and not context.enabled:
        raise RuntimeError("DDP training requires an initialized process group")
    if not ddp_enabled and context.enabled and context.world_size > 1:
        raise RuntimeError("A multi-process process group is active, but train_mode is not 'ddp'")

    run_dir = cfg.run_dir()
    plots_dir = run_dir / "plots"
    if context.is_main_process:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
    _barrier(context.enabled)

    if context.is_main_process:
        logger = get_logger("deeplearning.train", log_file=run_dir / "logs" / "train.log")
        metrics_logger: MetricsLogger | None = MetricsLogger(run_dir / "metrics" / "history.csv")
    else:
        logger = logging.getLogger(f"deeplearning.train.rank{context.rank}")
        logger.disabled = True
        metrics_logger = None

    logger.info(
        "Training mode: %s | rank=%d/%d | device=%s | available_gpus=%d | epochs_per_rank=%d | "
        "batch_size_per_rank=%d | effective_batch_size=%d",
        "DDP" if ddp_enabled else "single",
        context.rank,
        context.world_size,
        device,
        torch.cuda.device_count(),
        cfg.train.epochs,
        cfg.data.batch_size,
        cfg.data.batch_size * context.world_size,
    )
    logger.info("Checkpoints, metrics, and plots are owned by rank 0 under %s", run_dir)
    logger.info("Building dataloaders for split '%s'", cfg.data.split)

    train_loader, val_loader, test_loader = build_dataloaders(cfg.data)
    class_names = class_names or get_class_names(test_loader.dataset)

    if context.is_main_process:
        sample_images, sample_labels = next(iter(test_loader))
        plot_dataset_samples(
            sample_images,
            sample_labels,
            class_names,
            mean=(0.1751,),
            std=(0.3332,),
            output_path=plots_dir / "dataset_samples.png",
        )

    model.to(device)
    # verify_gpu_assign(device, logger, context.rank, context.local_rank)
    
    if ddp_enabled:
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = float("-inf")
    epochs_without_improvement = 0
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, cfg.train.epochs + 1):
        timer.start_epoch()
        
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        
        train_metrics = train_one_epoch(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            epoch,
            show_progress=True,
            rank=context.rank,
            local_rank=context.local_rank,
            distributed=ddp_enabled,
        )
        
        val_metrics = evaluate(
            model, 
            val_loader, 
            criterion, 
            device,
            desc=f"Epoch {epoch} [validation]",
            show_progress=True,
            rank=context.rank,
            local_rank=context.local_rank,
            distributed=ddp_enabled,
        )

        if scheduler is not None:
            scheduler.step()

        if context.is_main_process:
            logger.info(
                "Epoch %03d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            )
            assert metrics_logger is not None
            metrics_logger.log(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            if context.is_main_process:
                remove_previous(run_dir / "checkpoints", "_best.pt")
                save_checkpoint(
                    run_dir / "checkpoints" / (
                        f"{cfg.run_name}_epoch_{epoch:03d}_valacc{val_metrics['accuracy']:.4f}_"
                        f"valloss{val_metrics['loss']:.4f}_best.pt"
                    ),
                    model,
                    optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    extra={"train_mode": cfg.data.train_mode, "world_size": context.world_size},
                )
                logger.info("New best val_acc=%.4f, checkpoint saved", best_val_acc)
        else:
            epochs_without_improvement += 1

        if context.is_main_process:
            remove_previous(run_dir / "checkpoints", "_last.pt")
            save_checkpoint(
                run_dir / "checkpoints" / f"{cfg.run_name}_epoch_{epoch:03d}_last.pt",
                model,
                optimizer,
                epoch=epoch,
                metrics=val_metrics,
                extra={"train_mode": cfg.data.train_mode, "world_size": context.world_size},
            )
            if cfg.output.save_every_epoch:
                save_checkpoint(
                    run_dir / "checkpoints" / f"{cfg.run_name}_epoch_{epoch:03d}.pt",
                    model,
                    optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    extra={"train_mode": cfg.data.train_mode, "world_size": context.world_size},
                )
        _barrier(context.enabled)

        should_stop = epochs_without_improvement >= cfg.train.early_stopping_patience
        should_stop = _broadcast_stop(should_stop, device, context.enabled)
        if should_stop:
            if context.is_main_process:
                logger.info("Early stopping at epoch %d", epoch)
            break
    
    
    training_time = timer.stop()
    
    if context.is_main_process:
        plot_training_history(history, plots_dir / "training_history.png")
        
        logger.info(
            f"Training completed | "
            f"total_time={training_time.total_seconds:.2f}s "
            f"({training_time.total_minutes:.2f} min) | "
            f"average_epoch={training_time.average_epoch_seconds:.2f}s"
        )
        
        train_samples = len(train_loader)
        
        save_performance_metrics(
            training_time=training_time,
            train_samples=train_samples,
            cfg=cfg,
            output_path=run_dir / "metrics" / "performance_metrics.json",
        )
        
    _barrier(context.enabled)

    best_checkpoint = find_best_checkpoint(
        run_dir / "checkpoints",
        pattern=f"{cfg.run_name}_epoch_*_best.pt",
        metric_key="loss",
        mode="min",
    ) if context.is_main_process else None
    checkpoint_object = [str(best_checkpoint) if best_checkpoint else None]
    if context.enabled:
        dist.broadcast_object_list(checkpoint_object, src=0)
    best_checkpoint = Path(checkpoint_object[0])
    load_checkpoint(best_checkpoint, model, map_location=device)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        collect_predictions=True,
        desc="test",
        show_progress=context.is_main_process,
    )

    if context.is_main_process:
        compute_classification_report(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            output_path=run_dir / "metrics" / "classification_report.json",
            class_names=class_names,
        )
        plot_confusion_matrix(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            output_path=run_dir / "metrics" / "confusion_matrix.png",
            class_names=class_names,
        )

        sample_images, sample_labels = next(iter(test_loader))
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.eval()
        with torch.no_grad():
            sample_predictions = raw_model(sample_images.to(device)).argmax(dim=1).cpu()
        plot_predictions_grid(
            sample_images,
            sample_labels,
            sample_predictions,
            class_names,
            mean=(0.1751,),
            std=(0.3332,),
            output_path=plots_dir / "test_predictions.png",
        )
        logger.info("Test accuracy: %.4f | Test loss: %.4f", test_metrics["accuracy"], test_metrics["loss"])
    _barrier(context.enabled)

    return {
        "best_val_accuracy": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "run_dir": str(run_dir),
    }
