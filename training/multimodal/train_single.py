#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Architecture Training Script
Clean implementation for training one architecture on one GPU.
"""

import sys
import time
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from model.factory import create_model_variant
from datasets.data_loaders import create_multimodal_dataloader, calculate_class_weights


def setup_logging(log_file: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """Multi-task loss with weighted BCE for different prediction horizons."""

    def __init__(self, task_weights: dict, class_weights: dict):
        super().__init__()
        self.task_weights = task_weights
        self.criterions = {}

        for task_name, weight in task_weights.items():
            if task_name in class_weights:
                pos_weight = class_weights[task_name][1]  # Positive class weight
                self.criterions[task_name] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterions[task_name] = nn.BCEWithLogitsLoss()

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Calculate multi-task loss."""
        losses = {}
        total_loss = 0.0

        for task_name, weight in self.task_weights.items():
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name].squeeze(-1)
                target = targets[task_name].float()

                task_loss = self.criterions[task_name](pred, target)
                losses[f"loss_{task_name}"] = task_loss
                total_loss += weight * task_loss

        losses["loss_total"] = total_loss
        return losses


class SingleArchitectureTrainer:
    """Trainer for a single architecture variant."""

    def __init__(
        self, data_dir: str, output_dir: str, config: dict, device: str = "auto"
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.output_dir / "training.log")

        self.logger.info(f"🎯 Training: {config['arch_name']}")
        self.logger.info(f"   📁 Output: {self.output_dir}")
        self.logger.info(f"   🔧 Device: {self.device}")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = []

    def setup_dataloaders(self):
        """Setup train and validation dataloaders."""
        self.logger.info("📚 Setting up dataloaders...")

        # Train DataLoader
        self.train_loader = create_multimodal_dataloader(
            h5_file_path=str(self.data_dir / "train.h5"),
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            auto_normalize=True,
            img_width=1920,
            img_height=575,  # ROI height
            use_class_features=False,
            target_horizons=self.config["prediction_tasks"],
        )

        # Validation DataLoader
        self.val_loader = create_multimodal_dataloader(
            h5_file_path=str(self.data_dir / "val.h5"),
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            auto_normalize=True,
            img_width=1920,
            img_height=575,
            use_class_features=False,
            target_horizons=self.config["prediction_tasks"],
        )

        self.logger.info(f"✅ DataLoaders ready:")
        self.logger.info(f"   🚆 Train: {len(self.train_loader)} batches")
        self.logger.info(f"   🔍 Val: {len(self.val_loader)} batches")

    def setup_model(self):
        """Setup model, loss function, and optimizer."""
        self.logger.info("🤖 Setting up model...")

        # Create model
        model_config = {
            "encoder_type": self.config["encoder_type"],
            "fusion_type": self.config["fusion_type"],
            "decoder_type": self.config["decoder_type"],
            "telemetry_input_dim": 5,  # [SPEED, RPM, ACCELERATOR_POS_D, ENGINE_LOAD, GEAR]
            "detection_input_dim_per_box": 6,  # [confidence, x1, y1, x2, y2, area]
            "embedding_dim": self.config["embedding_dim"],
            "hidden_dim": self.config["hidden_dim"],
            "attention_num_heads": self.config["num_heads"],
            "dropout_prob": self.config["dropout_prob"],
            "prediction_tasks": self.config["prediction_tasks"],
            "max_detections": 12,
            "max_seq_length": 20,
        }

        self.model = create_model_variant(model_config)
        self.model = self.model.to(self.device)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"✅ Model: {total_params:,} parameters")

        # Setup loss function
        class_weights = calculate_class_weights(self.train_loader.dataset)

        # Task weights for multi-task learning
        task_weights = {
            "brake_1s": 1.0,
            "brake_2s": 0.8,
            "coast_1s": 0.4,
            "coast_2s": 0.3,
        }

        # Filter task weights to only include configured tasks
        filtered_weights = {
            task: weight
            for task, weight in task_weights.items()
            if task in self.config["prediction_tasks"]
        }

        self.loss_fn = MultiTaskLoss(filtered_weights, class_weights)
        self.loss_fn = self.loss_fn.to(self.device)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Setup mixed precision scaler
        self.scaler = GradScaler("cuda" if self.device.type == "cuda" else "cpu")

        self.logger.info("✅ Model setup complete")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "training_history": self.training_history,
        }

        # Save latest checkpoint
        checkpoint_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Saved best model: {best_path}")

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {}
        for task in self.config["prediction_tasks"]:
            epoch_losses[f"loss_{task}"] = 0.0
        epoch_losses["loss_total"] = 0.0

        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            telemetry = batch["telemetry_seq"].to(self.device, non_blocking=True)
            detections = batch["detection_seq"].to(self.device, non_blocking=True)
            mask = batch["detection_mask"].to(self.device, non_blocking=True)

            targets = {}
            for task_name in self.config["prediction_tasks"]:
                targets[task_name] = batch["targets"][task_name].to(
                    self.device, non_blocking=True
                )

            # Forward pass
            with autocast(device_type=self.device.type):
                predictions = self.model(telemetry, detections, mask)
                losses = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses["loss_total"]).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.get("grad_clip_norm", 1.0)
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name] += loss_value.item()
            num_batches += 1

            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {losses['loss_total'].item():.4f}"
                )

        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches

        return epoch_losses

    def validate_epoch(self, epoch: int):
        """Validate for one epoch."""
        self.model.eval()

        epoch_losses = {}
        for task in self.config["prediction_tasks"]:
            epoch_losses[f"loss_{task}"] = 0.0
        epoch_losses["loss_total"] = 0.0

        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                telemetry = batch["telemetry_seq"].to(self.device, non_blocking=True)
                detections = batch["detection_seq"].to(self.device, non_blocking=True)
                mask = batch["detection_mask"].to(self.device, non_blocking=True)

                targets = {}
                for task_name in self.config["prediction_tasks"]:
                    targets[task_name] = batch["targets"][task_name].to(
                        self.device, non_blocking=True
                    )

                # Forward pass
                with autocast(device_type=self.device.type):
                    predictions = self.model(telemetry, detections, mask)
                    losses = self.loss_fn(predictions, targets)

                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    epoch_losses[loss_name] += loss_value.item()
                num_batches += 1

        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches

        return epoch_losses

    def train(self):
        """Main training loop."""
        self.logger.info(f"🚀 Starting training for {self.config['epochs']} epochs")

        start_time = time.time()

        for epoch in range(self.config["epochs"]):
            self.current_epoch = epoch

            # Training phase
            train_losses = self.train_epoch(epoch)

            # Validation phase
            val_losses = self.validate_epoch(epoch)

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

            # Log epoch results
            self.logger.info(f"📊 Epoch {epoch + 1}/{self.config['epochs']}:")
            self.logger.info(f"   🚆 Train Loss: {train_losses['loss_total']:.4f}")
            self.logger.info(f"   🔍 Val Loss: {val_losses['loss_total']:.4f}")
            self.logger.info(f"   🏆 Best Val Loss: {self.best_val_loss:.4f}")
            self.logger.info(
                f"   ⌛ Patience: {self.patience_counter}/{self.config['patience']}"
            )
            self.logger.info(f"   📈 LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Store training history
            epoch_data = {
                "epoch": epoch,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "is_best": is_best,
            }
            self.training_history.append(epoch_data)

            # Early stopping
            if self.patience_counter >= self.config["patience"]:
                self.logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break

        # Save final training logs
        logs_path = self.output_dir / "training_logs.json"
        with open(logs_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_history = []
            for entry in self.training_history:
                serializable_entry = {
                    "epoch": int(entry["epoch"]),
                    "train_losses": {
                        k: float(v) for k, v in entry["train_losses"].items()
                    },
                    "val_losses": {k: float(v) for k, v in entry["val_losses"].items()},
                    "learning_rate": float(entry["learning_rate"]),
                    "is_best": bool(entry["is_best"]),
                }
                serializable_history.append(serializable_entry)

            json.dump(serializable_history, f, indent=2)

        total_time = time.time() - start_time
        self.logger.info(f"✅ Training completed in {total_time/60:.1f} minutes")
        self.logger.info(f"💾 Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train single architecture")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets/multimodal",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/multimodal",
        help="Output directory for model",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="simple_concat_lstm",
        help="Architecture name (encoder_fusion_decoder)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )

    args = parser.parse_args()

    # Parse architecture
    try:
        encoder, fusion, decoder = args.architecture.split("_")
    except ValueError:
        print(f"❌ Invalid architecture format: {args.architecture}")
        print("   Expected format: encoder_fusion_decoder")
        return False

    # Configuration
    config = {
        "arch_name": args.architecture,
        "encoder_type": encoder,
        "fusion_type": fusion,
        "decoder_type": decoder,
        # Model parameters
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_heads": 8,
        "dropout_prob": 0.15,
        # Training parameters
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-5,
        "epochs": args.epochs,
        "patience": 10,
        "grad_clip_norm": 1.0,
        "num_workers": 8,
        # Tasks
        "prediction_tasks": ["coast_1s", "coast_2s"],
    }

    # Setup output directory
    output_dir = Path(args.output_dir) / args.architecture

    # Initialize trainer
    trainer = SingleArchitectureTrainer(
        data_dir=args.data_dir, output_dir=str(output_dir), config=config
    )

    # Setup and train
    trainer.setup_dataloaders()
    trainer.setup_model()
    trainer.train()

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
