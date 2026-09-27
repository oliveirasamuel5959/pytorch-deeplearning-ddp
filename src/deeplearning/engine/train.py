"""Full training orchestration: epoch loop, checkpointing, early stopping, logging."""

from __future__ import annotations


# Deep learning imports
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Standard library imports
import os
from pathlib import Path

# Local imports
from deeplearning.config import TrainConfig
from deeplearning.datasets.emnist import build_dataloaders, get_class_names
from deeplearning.engine.evaluate import evaluate
from deeplearning.engine.train_one_epoch import train_one_epoch
from deeplearning.inference import Predictor
from deeplearning.utils.checkpoint import find_best_checkpoint, remove_previous, save_checkpoint
from deeplearning.utils.checkpoint import load_checkpoint
from deeplearning.utils.logger import MetricsLogger, get_logger
from deeplearning.utils.metrics import (
    compute_classification_report,
    plot_confusion_matrix,
    plot_training_history,
)
from deeplearning.utils.plots import plot_dataset_samples


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


def run_training(
    model: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """Run the full training loop and write checkpoints/metrics/plots under
    `cfg.run_dir()`. Returns a summary dict with best val accuracy and final test metrics.
    """
    run_dir = cfg.run_dir()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("deeplearning.train", log_file=run_dir / "logs" / "train.log")
    metrics_logger = MetricsLogger(run_dir / "metrics" / "history.csv")
    
    logger.info(f"Building dataloaders for split '{cfg.data.split}' with batch_size={cfg.data.batch_size}...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg.data)
    class_names = class_names or get_class_names(test_loader.dataset)

    sample_images, sample_labels = next(iter(test_loader))
    plot_dataset_samples(
        sample_images,
        sample_labels,
        class_names,
        mean=(0.1751,),
        std=(0.3332,),
        output_path=plots_dir / "dataset_samples.png",
    )

    if cfg.data.train_mode == "ddp":
        model = model.to(device)
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    else:
        model.to(device)
        
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    logger.info(f"Training run '{cfg.run_name}' on device={device} for {cfg.train.epochs} epochs")

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device, desc=f"Epoch {epoch} [validation]")

        if scheduler is not None:
            scheduler.step()

        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )
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

        # Save checkpoint for this epoch if requested
        if device.index == 0 and cfg.output.save_every_epoch:
            save_checkpoint(
                run_dir / "checkpoints" / f"{cfg.run_name}_epoch_{epoch:03d}_{val_metrics['accuracy']:.4f}.pt",
                model, optimizer, cfg.data.train_mode, epoch, val_metrics,
            )

        is_best = val_metrics["accuracy"] > best_val_acc
        
        # Save best checkpoint if this is the best validation accuracy so far
        if device.index == 0 and is_best:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            
            remove_previous(run_dir / "checkpoints", "_best.pt")
            save_checkpoint(
                run_dir / "checkpoints" / f"{cfg.run_name}_epoch_{epoch:03d}_valacc{val_metrics['accuracy']:.4f}_valloss{val_metrics['loss']:.4f}_best.pt", 
                model, optimizer, cfg.data.train_mode, epoch, val_metrics)
            logger.info(f"[OK] New best val_acc={best_val_acc:.4f}, checkpoint saved")
        else:
            epochs_without_improvement += 1

        # remove_previous(run_dir / "checkpoints", "_last.pt")
        # save_checkpoint(run_dir / "checkpoints" / f"{cfg.run_name}_epoch_{epoch:03d}_valacc{val_metrics['accuracy']:.4f}_valloss{val_metrics['loss']:.4f}_last.pt", model, optimizer, cfg.data.train_mode, epoch, val_metrics)

        if epochs_without_improvement >= cfg.train.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {epochs_without_improvement} epochs)")
            break

    plot_training_history(history, plots_dir / "training_history.png")

    
    best_ckpt = find_best_checkpoint(
        run_dir / "checkpoints",
        pattern=f"{cfg.run_name}_epoch_*_best.pt",
        metric_key="loss",
        mode="min",
    )

    logger.info(f"Training complete. Evaluating on test set with best checkpoint {best_ckpt}...")
    # load_checkpoint(run_dir / "checkpoints" / f"{cfg.run_name}_epoch_{epoch:03d}_valacc{val_metrics['accuracy']:.4f}_valloss{val_metrics['loss']:.4f}_best.pt", model, map_location=device)
    load_checkpoint(best_ckpt, model, map_location=device)
    test_metrics = evaluate(model, test_loader, criterion, device, collect_predictions=True, desc="test")

    compute_classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
        output_path=run_dir / "metrics" / "classification_report.json",
        class_names=class_names,
    )
    plot_confusion_matrix(
        test_metrics["y_true"], test_metrics["y_pred"],
        output_path=run_dir / "metrics" / "confusion_matrix.png",
        class_names=class_names,
    )

    predictor = Predictor(
        checkpoint_path=best_ckpt,
        model_cfg=cfg.model,
        device=str(device),
        class_names=class_names,
    )
    predictor.predict_dataloader(
        test_loader,
        output_path=plots_dir / "test_predictions.png",
    )

    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f} | Test loss: {test_metrics['loss']:.4f}")

    return {
        "best_val_accuracy": best_val_acc,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "run_dir": str(run_dir),
    }
