#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Architecture Training Script
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
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from model.factory import create_model_variant
from datasets.data_loaders import create_multimodal_dataloader, calculate_class_weights


# ====================================
# TRAINING CONFIGURATION
# ====================================

# Model Architecture
EMBEDDING_DIM = 128
HIDDEN_DIM = EMBEDDING_DIM * 2
NUM_HEADS = 8
DECODER_NUM_LAYERS = 2
DROPOUT_PROB = 0.15

# Training Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
PATIENCE = EPOCHS // 5
GRAD_CLIP_NORM = 0.5

# Task Configuration - Testing coast events (more frequent than brake)
PREDICTION_TASKS = ["coast_1s", "coast_2s", "coast_3s"]
TASK_WEIGHTS = {
    "coast_1s": 1.0,
    "coast_2s": 0.8,
    "coast_3s": 0.6,
}
CLASS_WEIGHT_MULTIPLIERS = {
    "coast_1s": 1.5,
    "coast_2s": 1.5,
    "coast_3s": 1.5,
}

# Learning Rate Scheduling
SCHEDULER_FACTOR = 0.7
SCHEDULER_PATIENCE = 8
MIN_LR = 1e-6

# Data Configuration
USE_CLASS_FEATURES = False
IMG_WIDTH = 1920
IMG_HEIGHT = 575

# Training Infrastructure
NUM_WORKERS = 8
PIN_MEMORY = True
MIXED_PRECISION = True
LOG_INTERVAL = 20


def setup_logging(log_file: str):
    """Setup logging configuration."""
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class MultiTaskLoss(nn.Module):
    """Multi-task loss with weighted BCE."""

    def __init__(self, class_weights: dict):
        super().__init__()
        self.task_weights = TASK_WEIGHTS
        self.criterions = {}

        for task_name in PREDICTION_TASKS:
            if task_name in class_weights:
                base_pos_weight = class_weights[task_name][1]
                multiplier = CLASS_WEIGHT_MULTIPLIERS.get(task_name, 1.0)
                adjusted_pos_weight = base_pos_weight * multiplier
                adjusted_pos_weight = max(
                    adjusted_pos_weight, class_weights[task_name][0]
                )  # Ensure it's not less than the negative class weight

                self.criterions[task_name] = nn.BCEWithLogitsLoss(
                    pos_weight=adjusted_pos_weight
                )
            else:
                self.criterions[task_name] = nn.BCEWithLogitsLoss()

    def forward(self, predictions: dict, targets: dict) -> dict:
        """Calculate multi-task loss."""
        losses = {}
        total_loss = 0.0

        for task_name in PREDICTION_TASKS:
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name].squeeze(-1)
                target = targets[task_name].float()

                task_loss = self.criterions[task_name](pred, target)
                losses[f"loss_{task_name}"] = task_loss

                task_weight = self.task_weights.get(task_name, 1.0)
                total_loss += task_weight * task_loss

        losses["loss_total"] = total_loss
        return losses


class Trainer:
    """Single architecture trainer."""

    def __init__(
        self, arch_name: str, data_dir: str, output_dir: str, device: str = "auto"
    ):
        self.arch_name = arch_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.output_dir / "training.log")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = []

        self.logger.info(f"🎯 Training: {arch_name}")
        self.logger.info(f"   📁 Output: {self.output_dir}")
        self.logger.info(f"   🔧 Device: {self.device}")
        self.logger.info(f"   🎯 Tasks: {PREDICTION_TASKS}")

    def setup_dataloaders(self):
        """Setup train and validation dataloaders."""
        self.logger.info("📚 Setting up dataloaders...")

        self.train_loader = create_multimodal_dataloader(
            h5_file_path=str(self.data_dir / "train.h5"),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            load_into_memory=True,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            use_class_features=USE_CLASS_FEATURES,
            target_horizons=PREDICTION_TASKS,
        )

        self.val_loader = create_multimodal_dataloader(
            h5_file_path=str(self.data_dir / "val.h5"),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            load_into_memory=True,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            use_class_features=USE_CLASS_FEATURES,
            target_horizons=PREDICTION_TASKS,
        )

        self.logger.info(f"✅ DataLoaders ready:")
        self.logger.info(f"   🚆 Train: {len(self.train_loader)} batches")
        self.logger.info(f"   🔍 Val: {len(self.val_loader)} batches")

    def setup_model(self):
        """Setup model, loss function, and optimizer."""
        self.logger.info("🤖 Setting up model...")

        # Parse architecture
        encoder_type, fusion_type, decoder_type = self.arch_name.split("_")

        # Create model
        model_config = {
            "encoder_type": encoder_type,
            "fusion_type": fusion_type,
            "decoder_type": decoder_type,
            "telemetry_input_dim": 10,  # 4 ranges + 6 onehot-gears
            "detection_input_dim_per_box": 6,  # confidence + bbox*4 + area
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "attention_num_heads": NUM_HEADS,
            "decoder_num_layers": DECODER_NUM_LAYERS,
            "dropout_prob": DROPOUT_PROB,
            "prediction_tasks": PREDICTION_TASKS,
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
        self.loss_fn = MultiTaskLoss(class_weights)
        self.loss_fn = self.loss_fn.to(self.device)

        # Log class weights
        for task_name in PREDICTION_TASKS:
            if task_name in class_weights:
                base_weight = class_weights[task_name][1].item()
                multiplier = CLASS_WEIGHT_MULTIPLIERS.get(task_name, 1.0)
                final_weight = base_weight * multiplier
                self.logger.info(f"   ⚖️ {task_name}: pos_weight={final_weight:.2f}")

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # Setup scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=MIN_LR,
        )

        # Setup mixed precision scaler
        if MIXED_PRECISION:
            self.scaler = GradScaler("cuda" if self.device.type == "cuda" else "cpu")
        else:
            self.scaler = None

        self.logger.info("✅ Model setup complete")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "arch_name": self.arch_name,
            "training_history": self.training_history,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save latest checkpoint
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"💾 New best model saved (val_loss: {self.best_val_loss:.4f})"
            )

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {f"loss_{task}": 0.0 for task in PREDICTION_TASKS}
        epoch_losses["loss_total"] = 0.0
        num_batches = 0

        train_pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Train]",
            unit="batch",
            leave=False,
        )

        for batch_idx, batch in enumerate(train_pbar):
            # Move data to device
            telemetry = batch["telemetry_seq"].to(self.device, non_blocking=True)
            detections = batch["detection_seq"].to(self.device, non_blocking=True)
            mask = batch["detection_mask"].to(self.device, non_blocking=True)

            targets = {}
            for task_name in PREDICTION_TASKS:
                targets[task_name] = batch["targets"][task_name].to(
                    self.device, non_blocking=True
                )

            # Forward pass
            if MIXED_PRECISION and self.scaler:
                with autocast(device_type=self.device.type):
                    predictions = self.model(telemetry, detections, mask)
                    losses = self.loss_fn(predictions, targets)
            else:
                predictions = self.model(telemetry, detections, mask)
                losses = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()

            if MIXED_PRECISION and self.scaler:
                self.scaler.scale(losses["loss_total"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                self.optimizer.step()

            # Accumulate losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name] += loss_value.item()
            num_batches += 1

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress_dict = {
                "loss": f"{losses['loss_total'].item():.4f}",
                "lr": f"{current_lr:.2e}",
            }
            train_pbar.set_postfix(progress_dict)

            # Log detailed progress
            if batch_idx % LOG_INTERVAL == 0 and batch_idx > 0:
                avg_loss = epoch_losses["loss_total"] / num_batches
                self.logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {losses['loss_total'].item():.4f}, "
                    f"Avg: {avg_loss:.4f}, "
                    f"LR: {current_lr:.2e}"
                )

        train_pbar.close()

        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches

        return epoch_losses

    def validate_epoch(self, epoch: int):
        """Validate for one epoch."""
        self.model.eval()

        epoch_losses = {f"loss_{task}": 0.0 for task in PREDICTION_TASKS}
        epoch_losses["loss_total"] = 0.0
        num_batches = 0

        val_pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [Val]",
            unit="batch",
            leave=False,
        )

        with torch.no_grad():
            for batch in val_pbar:
                # Move data to device
                telemetry = batch["telemetry_seq"].to(self.device, non_blocking=True)
                detections = batch["detection_seq"].to(self.device, non_blocking=True)
                mask = batch["detection_mask"].to(self.device, non_blocking=True)

                targets = {}
                for task_name in PREDICTION_TASKS:
                    targets[task_name] = batch["targets"][task_name].to(
                        self.device, non_blocking=True
                    )

                # Forward pass
                if MIXED_PRECISION:
                    with autocast(device_type=self.device.type):
                        predictions = self.model(telemetry, detections, mask)
                        losses = self.loss_fn(predictions, targets)
                else:
                    predictions = self.model(telemetry, detections, mask)
                    losses = self.loss_fn(predictions, targets)

                # Accumulate losses
                for loss_name, loss_value in losses.items():
                    epoch_losses[loss_name] += loss_value.item()
                num_batches += 1

                # Update progress bar
                progress_dict = {"val_loss": f"{losses['loss_total'].item():.4f}"}
                val_pbar.set_postfix(progress_dict)

        val_pbar.close()

        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches

        return epoch_losses

    def train(self):
        """Main training loop."""
        self.logger.info(f"🚀 Starting training for {EPOCHS} epochs")
        self.logger.info(f"   📊 Architecture: {self.arch_name}")
        self.logger.info(f"   📦 Batch size: {BATCH_SIZE}")
        self.logger.info(f"   📈 Learning rate: {LEARNING_RATE}")
        self.logger.info(f"   🎯 Tasks: {PREDICTION_TASKS}")

        start_time = time.time()

        epoch_pbar = tqdm(range(EPOCHS), desc="Training Progress", unit="epoch")

        for epoch in epoch_pbar:
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

            # Update epoch progress bar
            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_info = {
                "train_loss": f"{train_losses['loss_total']:.4f}",
                "val_loss": f"{val_losses['loss_total']:.4f}",
                "best": f"{self.best_val_loss:.4f}",
                "lr": f"{current_lr:.2e}",
                "patience": f"{self.patience_counter}/{PATIENCE}",
            }
            epoch_pbar.set_postfix(epoch_info)

            # Detailed logging
            self.logger.info(f"📊 Epoch {epoch + 1} Summary:")
            self.logger.info(f"   🚆 Train - Total: {train_losses['loss_total']:.4f}")
            for task in PREDICTION_TASKS:
                task_loss = train_losses.get(f"loss_{task}", 0.0)
                self.logger.info(f"     {task}: {task_loss:.4f}")

            self.logger.info(f"   🔍 Val - Total: {val_losses['loss_total']:.4f}")
            for task in PREDICTION_TASKS:
                task_loss = val_losses.get(f"loss_{task}", 0.0)
                self.logger.info(f"     {task}: {task_loss:.4f}")

            self.logger.info(f"   🏆 Best: {self.best_val_loss:.4f}")
            self.logger.info(f"   📈 LR: {current_lr:.6f}")

            # Store training history
            epoch_data = {
                "epoch": epoch,
                "train_losses": {k: float(v) for k, v in train_losses.items()},
                "val_losses": {k: float(v) for k, v in val_losses.items()},
                "learning_rate": float(current_lr),
                "is_best": is_best,
            }
            self.training_history.append(epoch_data)

            # Early stopping
            if self.patience_counter >= PATIENCE:
                self.logger.info(f"🛑 Early stopping at epoch {epoch + 1}")
                break

        epoch_pbar.close()

        # Save final training logs
        logs_path = self.output_dir / "training_logs.json"
        with open(logs_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

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
        help="Dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/multimodal",
        help="Output directory",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="simple_concat_lstm",
        help="Architecture (encoder_fusion_decoder)",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir) / args.architecture

    print(f"🎯 Training Configuration:")
    print(f"   🏗️ Architecture: {args.architecture}")
    print(f"   📦 Batch size: {BATCH_SIZE}")
    print(f"   📈 Learning rate: {LEARNING_RATE}")
    print(f"   🔄 Epochs: {EPOCHS}")
    print(f"   🎯 Tasks: {PREDICTION_TASKS}")
    print(f"   📁 Output: {output_dir}")

    # Initialize and run trainer
    trainer = Trainer(
        arch_name=args.architecture, data_dir=args.data_dir, output_dir=str(output_dir)
    )

    trainer.setup_dataloaders()
    trainer.setup_model()
    trainer.train()

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
