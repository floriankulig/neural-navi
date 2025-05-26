#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Architecture Training Pipeline for Multimodal Braking Prediction
Trains all specified architecture combinations with DDP and comprehensive logging.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Wandb for logging
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not available, using local logging only")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from src.model.factory import create_model_variant
from datasets.data_loaders import create_multimodal_dataloader, calculate_class_weights


# Configure logging
def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


# Training configuration
ARCHITECTURE_COMBINATIONS = [
    ("simple", "concat", "lstm"),
    ("simple", "concat", "transformer"),
    ("simple", "cross_attention", "lstm"),
    ("simple", "cross_attention", "transformer"),
    ("simple", "query", "lstm"),
    ("simple", "query", "transformer"),
    ("attention", "concat", "lstm"),
    ("attention", "concat", "transformer"),
    ("attention", "cross_attention", "lstm"),
    ("attention", "cross_attention", "transformer"),
    ("attention", "query", "lstm"),
    ("attention", "query", "transformer"),
]

# Fixed hyperparameters for fair comparison
HYPERPARAMETERS = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "num_heads": 8,
    "dropout_prob": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 512,  # Global batch size across all GPUs
    "epochs": 50,
    "patience": 10,
    "grad_clip_norm": 1.0,
    "warmup_epochs": 5,
}

# Multi-task learning weights
TASK_WEIGHTS = {
    "brake_1s": 1.0,  # Primary task
    "brake_2s": 0.3,  # Secondary tasks
    "brake_4s": 0.3,
    # TODO: Add coasting prediction tasks
}

TARGET_HORIZONS = list(TASK_WEIGHTS.keys())

# Dataset configuration
TELEMETRY_INPUT_DIM = 5  # [SPEED, RPM, ACCELERATOR_POS_D, ENGINE_LOAD, GEAR]
DETECTION_INPUT_DIM_PER_BOX = (
    6  # [confidence, x1, y1, x2, y2, area] (no class_id by default)
)

DEFAULT_DATA_DIR = "data/datasets/multimodal"
DEFAULT_OUTPUT_DIR = "data/models/multimodal"


class MultiTaskLoss(nn.Module):
    """Multi-task loss with weighted BCE for different prediction horizons."""

    def __init__(
        self, task_weights: Dict[str, float], class_weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.task_weights = task_weights
        self.class_weights = class_weights
        self.criterions = {}

        for task_name in task_weights.keys():
            if task_name in class_weights:
                # Use class weights for imbalanced data
                pos_weight = class_weights[task_name][1]  # Positive class weight
                self.criterions[task_name] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterions[task_name] = nn.BCEWithLogitsLoss()

    def forward(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss.

        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with target labels

        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        total_loss = 0.0

        for task_name, weight in self.task_weights.items():
            if task_name in predictions and task_name in targets:
                # Calculate task-specific loss
                pred = predictions[task_name].squeeze(-1)  # Remove last dimension
                target = targets[task_name].float()

                task_loss = self.criterions[task_name](pred, target)
                losses[f"loss_{task_name}"] = task_loss

                # Add to total loss with weight
                total_loss += weight * task_loss

        losses["loss_total"] = total_loss
        return losses


class ModelTrainer:
    """Handles training of a single architecture variant."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.rank = rank
        self.world_size = world_size
        self.use_wandb = use_wandb
        self.is_main_process = rank == 0

        # Create output directory
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        if self.is_main_process:
            self.logger = setup_logging(self.output_dir / "training.log")
        else:
            self.logger = logging.getLogger(__name__)

        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_stats = []

        # Setup device and DDP
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        if self.is_main_process:
            self.logger.info(f"🎯 Training architecture: {config['arch_name']}")
            self.logger.info(f"   📁 Output directory: {self.output_dir}")
            self.logger.info(f"   🔧 Device: {self.device}")
            self.logger.info(f"   🌍 World size: {self.world_size}")

        # Initialize wandb
        if self.use_wandb and self.is_main_process and WANDB_AVAILABLE:
            wandb.init(
                project="neural-navi-multimodal",
                name=config["arch_name"],
                config=config,
                dir=str(self.output_dir),
            )

    def setup_model_and_optimizers(self, class_weights: Dict[str, torch.Tensor]):
        """Setup model, loss function, and optimizers."""
        # Create model
        model_config = {
            "encoder_type": self.config["encoder_type"],
            "fusion_type": self.config["fusion_type"],
            "decoder_type": self.config["decoder_type"],
            "telemetry_input_dim": TELEMETRY_INPUT_DIM,
            "detection_input_dim_per_box": DETECTION_INPUT_DIM_PER_BOX,
            "embedding_dim": HYPERPARAMETERS["embedding_dim"],
            "hidden_dim": HYPERPARAMETERS["hidden_dim"],
            "attention_num_heads": HYPERPARAMETERS["num_heads"],
            "dropout_prob": HYPERPARAMETERS["dropout_prob"],
            "prediction_horizons": [1, 2, 4],  # Only train on these horizons
            "max_detections": 12,
            "max_seq_length": 20,
        }

        self.model = create_model_variant(model_config)
        self.model = self.model.to(self.device)

        if self.is_main_process:
            total_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self.logger.info(f"🤖 Model parameters: {total_params:,}")

        # Wrap with DDP
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank])

        # Setup loss function
        self.loss_fn = MultiTaskLoss(TASK_WEIGHTS, class_weights)
        self.loss_fn = self.loss_fn.to(self.device)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=HYPERPARAMETERS["learning_rate"],
            weight_decay=HYPERPARAMETERS["weight_decay"],
        )

        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=self.is_main_process,
        )

        # Setup mixed precision scaler
        self.scaler = GradScaler()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_main_process:
            return

        # Get model state dict (handle DDP wrapper)
        model_state = (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "training_stats": self.training_stats,
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer and scheduler states
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            # Load training state
            self.current_epoch = checkpoint["epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self.training_stats = checkpoint.get("training_stats", [])

            if self.is_main_process:
                self.logger.info(
                    f"✅ Loaded checkpoint from epoch {self.current_epoch}"
                )

            return True

        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"❌ Failed to load checkpoint: {e}")
            return False

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {f"loss_{task}": 0.0 for task in TASK_WEIGHTS.keys()}
        epoch_losses["loss_total"] = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            telemetry_seq = batch["telemetry_seq"].to(self.device, non_blocking=True)
            detection_seq = batch["detection_seq"].to(self.device, non_blocking=True)
            detection_mask = batch["detection_mask"].to(self.device, non_blocking=True)

            targets = {}
            for task_name in TASK_WEIGHTS.keys():
                targets[task_name] = batch["targets"][task_name].to(
                    self.device, non_blocking=True
                )

            # Forward pass with mixed precision
            with autocast():
                predictions = self.model(telemetry_seq, detection_seq, detection_mask)
                losses = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses["loss_total"]).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), HYPERPARAMETERS["grad_clip_norm"]
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name] += loss_value.item()
            num_batches += 1

            # Log progress
            if self.is_main_process and batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {losses['loss_total'].item():.4f}"
                )

        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches

        return epoch_losses

    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        epoch_losses = {f"loss_{task}": 0.0 for task in TASK_WEIGHTS.keys()}
        epoch_losses["loss_total"] = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                telemetry_seq = batch["telemetry_seq"].to(
                    self.device, non_blocking=True
                )
                detection_seq = batch["detection_seq"].to(
                    self.device, non_blocking=True
                )
                detection_mask = batch["detection_mask"].to(
                    self.device, non_blocking=True
                )

                targets = {}
                for task_name in TASK_WEIGHTS.keys():
                    targets[task_name] = batch["targets"][task_name].to(
                        self.device, non_blocking=True
                    )

                # Forward pass
                with autocast():
                    predictions = self.model(
                        telemetry_seq, detection_seq, detection_mask
                    )
                    losses = self.loss_fn(predictions, targets)

                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    epoch_losses[loss_name] += loss_value.item()
                num_batches += 1

        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches

        return epoch_losses

    def train(self, train_loader, val_loader):
        """Main training loop."""
        if self.is_main_process:
            self.logger.info(f"🚀 Starting training for {self.config['arch_name']}")
            self.logger.info(f"   📊 Training batches: {len(train_loader)}")
            self.logger.info(f"   📊 Validation batches: {len(val_loader)}")

        start_time = time.time()

        for epoch in range(self.current_epoch, HYPERPARAMETERS["epochs"]):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Training phase
            train_losses = self.train_epoch(train_loader)

            # Validation phase
            val_losses = self.validate_epoch(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_losses["loss_total"])

            # Check for improvement
            current_val_loss = val_losses["loss_total"]
            is_best = current_val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Log progress
            if self.is_main_process:
                epoch_time = time.time() - start_time
                self.logger.info(
                    f"📊 Epoch {epoch + 1}/{HYPERPARAMETERS['epochs']} completed"
                )
                self.logger.info(f"   🚆 Train Loss: {train_losses['loss_total']:.4f}")
                self.logger.info(f"   🔍 Val Loss: {val_losses['loss_total']:.4f}")
                self.logger.info(f"   ⏱️ Time: {epoch_time:.1f}s")
                self.logger.info(f"   🏆 Best Val Loss: {self.best_val_loss:.4f}")
                self.logger.info(
                    f"   ⌛ Patience: {self.patience_counter}/{HYPERPARAMETERS['patience']}"
                )

                # Wandb logging
                if self.use_wandb and WANDB_AVAILABLE:
                    log_dict = {
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    log_dict.update({f"train_{k}": v for k, v in train_losses.items()})
                    log_dict.update({f"val_{k}": v for k, v in val_losses.items()})
                    wandb.log(log_dict)

            # Store training stats
            self.training_stats.append(
                {
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "is_best": is_best,
                }
            )

            # Early stopping
            if self.patience_counter >= HYPERPARAMETERS["patience"]:
                if self.is_main_process:
                    self.logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break

        # Save final training logs
        if self.is_main_process:
            logs_path = self.output_dir / "training_logs.json"
            with open(logs_path, "w") as f:
                json.dump(self.training_stats, f, indent=2)

            total_time = time.time() - start_time
            self.logger.info(f"✅ Training completed in {total_time:.1f}s")
            self.logger.info(
                f"💾 Best model saved with validation loss: {self.best_val_loss:.4f}"
            )

            if self.use_wandb and WANDB_AVAILABLE:
                wandb.finish()


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_single_architecture(
    rank: int,
    world_size: int,
    encoder_type: str,
    fusion_type: str,
    decoder_type: str,
    data_dir: str,
    output_dir: str,
    use_wandb: bool = False,
    resume_from: Optional[str] = None,
):
    """Train a single architecture variant."""
    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)

    try:
        # Create architecture configuration
        arch_name = f"{encoder_type}_{fusion_type}_{decoder_type}"
        config = {
            "arch_name": arch_name,
            "encoder_type": encoder_type,
            "fusion_type": fusion_type,
            "decoder_type": decoder_type,
            **HYPERPARAMETERS,
        }

        # Setup output directory
        arch_output_dir = Path(output_dir) / arch_name

        # Create trainer
        trainer = ModelTrainer(
            config=config,
            output_dir=arch_output_dir,
            rank=rank,
            world_size=world_size,
            use_wandb=use_wandb,
        )

        # Load data
        if trainer.is_main_process:
            trainer.logger.info("📚 Loading dataset...")

        # Calculate per-GPU batch size
        per_gpu_batch_size = HYPERPARAMETERS["batch_size"] // world_size

        # Create dataloaders
        train_loader = create_multimodal_dataloader(
            h5_file_path=f"{data_dir}/train.h5",
            batch_size=per_gpu_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            use_class_features=False,  # Use Boxy-style features by default
            target_horizons=TARGET_HORIZONS,
        )

        val_loader = create_multimodal_dataloader(
            h5_file_path=f"{data_dir}/val.h5",
            batch_size=per_gpu_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            use_class_features=False,
            target_horizons=TARGET_HORIZONS,
        )

        # Wrap with distributed sampler
        if world_size > 1:
            train_sampler = DistributedSampler(
                train_loader.dataset, rank=rank, num_replicas=world_size
            )
            val_sampler = DistributedSampler(
                val_loader.dataset, rank=rank, num_replicas=world_size, shuffle=False
            )

            # Recreate dataloaders with distributed samplers
            train_loader = create_multimodal_dataloader(
                h5_file_path=f"{data_dir}/train.h5",
                batch_size=per_gpu_batch_size,
                shuffle=False,  # Sampler handles shuffling
                num_workers=8,
                pin_memory=True,
                use_class_features=False,
                target_horizons=TARGET_HORIZONS,
                sampler=train_sampler,
            )

            val_loader = create_multimodal_dataloader(
                h5_file_path=f"{data_dir}/val.h5",
                batch_size=per_gpu_batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                use_class_features=False,
                target_horizons=TARGET_HORIZONS,
                sampler=val_sampler,
            )

        # Calculate class weights
        if trainer.is_main_process:
            trainer.logger.info("⚖️ Calculating class weights...")
        class_weights = calculate_class_weights(train_loader.dataset)

        # Move class weights to device
        for task_name in class_weights:
            class_weights[task_name] = class_weights[task_name].to(trainer.device)

        # Setup model and optimizers
        trainer.setup_model_and_optimizers(class_weights)

        # Resume from checkpoint if specified
        if resume_from and Path(resume_from).exists():
            trainer.load_checkpoint(Path(resume_from))

        # Start training
        trainer.train(train_loader, val_loader)

    finally:
        # Cleanup distributed training
        if world_size > 1:
            cleanup_distributed()


def main():
    """Main entry point for multi-architecture training."""
    parser = argparse.ArgumentParser(
        description="Train all multimodal architecture variants"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to prepared dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Train specific architecture (format: encoder_fusion_decoder)",
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--resume-from", type=str, help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use",
    )

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    required_files = ["train.h5", "val.h5", "dataset_config.json"]
    for file_name in required_files:
        if not (data_dir / file_name).exists():
            print(f"❌ Required file not found: {data_dir / file_name}")
            sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎯 Multi-Architecture Training Pipeline")
    print(f"   📁 Data directory: {data_dir}")
    print(f"   📁 Output directory: {output_dir}")
    print(f"   🌍 World size: {args.world_size}")
    print(f"   📊 Use wandb: {args.use_wandb}")

    # Determine architectures to train
    if args.architecture:
        # Train single architecture
        try:
            encoder, fusion, decoder = args.architecture.split("_")
            architectures = [(encoder, fusion, decoder)]
        except ValueError:
            print(f"❌ Invalid architecture format: {args.architecture}")
            print("   Expected format: encoder_fusion_decoder")
            sys.exit(1)
    else:
        # Train all architectures
        architectures = ARCHITECTURE_COMBINATIONS

    print(f"🎯 Training {len(architectures)} architecture(s):")
    for i, (encoder, fusion, decoder) in enumerate(architectures, 1):
        print(f"   {i}. {encoder}_{fusion}_{decoder}")

    # Train each architecture
    for i, (encoder_type, fusion_type, decoder_type) in enumerate(architectures, 1):
        arch_name = f"{encoder_type}_{fusion_type}_{decoder_type}"
        print(f"\n{'='*60}")
        print(f"🚀 Training architecture {i}/{len(architectures)}: {arch_name}")
        print(f"{'='*60}")

        if args.world_size > 1:
            # Multi-GPU training with DDP
            import torch.multiprocessing as mp

            mp.spawn(
                train_single_architecture,
                args=(
                    args.world_size,
                    encoder_type,
                    fusion_type,
                    decoder_type,
                    str(data_dir),
                    str(output_dir),
                    args.use_wandb,
                    args.resume_from,
                ),
                nprocs=args.world_size,
            )
        else:
            # Single GPU training
            train_single_architecture(
                rank=0,
                world_size=1,
                encoder_type=encoder_type,
                fusion_type=fusion_type,
                decoder_type=decoder_type,
                data_dir=str(data_dir),
                output_dir=str(output_dir),
                use_wandb=args.use_wandb,
                resume_from=args.resume_from,
            )

        print(f"✅ Completed training: {arch_name}")

    print(f"\n🎉 All architectures trained successfully!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
