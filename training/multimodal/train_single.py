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

from utils.feature_config import (
    DETECTION_INPUT_DIM_PER_BOX,
    MAX_DETECTIONS_PER_FRAME,
    SEQUENCE_LENGTH,
    TELEMETRY_INPUT_DIM,
    PREDICTION_TASKS,
)
from model.factory import create_model_variant
from utils.debug import NaNDebugger
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
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 40
PATIENCE = EPOCHS // 5
GRAD_CLIP_NORM = 1

# Warmup for more stable training
WARMUP_EPOCHS = EPOCHS // 10
WARMUP_LR = 1e-6  # Noch kleinere LR für Warmup

# Learning Rate Scheduling
SCHEDULER_FACTOR = 0.85
SCHEDULER_PATIENCE = 6
MIN_LR = 1e-7

# Task Configuration - Testing coast events (more frequent than brake)
TASK_WEIGHTS = {
    "coast_1s": 1.0,
    # "coast_2s": 0.8,
}
CLASS_WEIGHT_MULTIPLIERS = {
    "coast_1s": 1.5,
    # "coast_2s": 1.5,
}


# Data Configuration

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


def monitor_gradients(model, epoch, batch_idx):
    """Monitor gradients for NaN detection."""
    total_norm = 0
    nan_params = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_params.append(name)
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1.0 / 2)

    if nan_params:
        logging.error(f"🚨 NaN gradients in epoch {epoch}, batch {batch_idx}:")
        for param_name in nan_params:
            logging.error(f"  - {param_name}")

    if total_norm > 100:  # Threshold for gradient explosion
        logging.warning(f"⚠️ Large gradient norm: {total_norm:.3f}")

    return total_norm, len(nan_params) > 0


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
            target_horizons=PREDICTION_TASKS,
        )

        self.val_loader = create_multimodal_dataloader(
            h5_file_path=str(self.data_dir / "val.h5"),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            load_into_memory=True,
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
            "telemetry_input_dim": TELEMETRY_INPUT_DIM,  # 4 ranges (+ 1 Brake Signal) + 6 onehot-gears
            "detection_input_dim_per_box": DETECTION_INPUT_DIM_PER_BOX,  # confidence + bbox*4 + area
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "attention_num_heads": NUM_HEADS,
            "decoder_num_layers": DECODER_NUM_LAYERS,
            "dropout_prob": DROPOUT_PROB,
            "prediction_tasks": PREDICTION_TASKS,
            "max_detections": MAX_DETECTIONS_PER_FRAME,
            "max_seq_length": SEQUENCE_LENGTH,
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

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = WARMUP_LR

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
        if epoch < WARMUP_EPOCHS:
            progress = (epoch + 1) / WARMUP_EPOCHS
            current_lr = WARMUP_LR + (LEARNING_RATE - WARMUP_LR) * progress
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
            self.logger.info(f"🔥 Warmup LR: {current_lr:.6f}")

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

            # Im Training Loop nach backward():
            if MIXED_PRECISION and self.scaler:
                self.scaler.scale(losses["loss_total"]).backward()
                self.scaler.unscale_(self.optimizer)

                # Monitor gradients BEFORE clipping
                grad_norm, has_nan_grads = monitor_gradients(
                    self.model, epoch, batch_idx
                )

                if has_nan_grads:
                    logging.error("🚨 Skipping optimizer step due to NaN gradients")
                    self.scaler.update()  # Still update scaler
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()

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

    def debug_detailed_batch(self):
        """Detailed debugging of problematic batch elements."""
        from utils.nan_debug import debug_model_forward, check_tensor_health

        self.model.eval()  # Set to eval to avoid dropout randomness

        # Get the same batch
        batch = next(iter(self.train_loader))

        telemetry = batch["telemetry_seq"].to(self.device)
        detections = batch["detection_seq"].to(self.device)
        mask = batch["detection_mask"].to(self.device)

        print(f"📊 Batch Analysis:")
        print(f"   Batch size: {telemetry.shape[0]}")
        print(f"   Sequence length: {telemetry.shape[1]}")
        print(f"   Max detections: {detections.shape[2]}")

        # Analyze detection masks per batch element
        mask_stats = []
        for b in range(mask.shape[0]):
            valid_detections_per_frame = (~mask[b]).sum(
                dim=1
            )  # Valid detections per time step
            total_valid = valid_detections_per_frame.sum().item()
            frames_with_no_detections = (valid_detections_per_frame == 0).sum().item()

            mask_stats.append(
                {
                    "batch_idx": b,
                    "total_valid_detections": total_valid,
                    "frames_with_no_detections": frames_with_no_detections,
                    "avg_detections_per_frame": (
                        total_valid / mask.shape[1] if mask.shape[1] > 0 else 0
                    ),
                }
            )

        # Sort by problematic cases (many frames without detections)
        mask_stats.sort(key=lambda x: x["frames_with_no_detections"], reverse=True)

        print(f"\n📋 Top 10 potentially problematic batch elements:")
        for i, stats in enumerate(mask_stats[:10]):
            print(
                f"   {i+1}. Batch {stats['batch_idx']}: "
                f"{stats['frames_with_no_detections']}/{mask.shape[1]} empty frames, "
                f"avg {stats['avg_detections_per_frame']:.1f} detections/frame"
            )

        # Debug model forward
        print(f"\n🔍 Running detailed model forward debugging...")
        output, nan_batch_indices = debug_model_forward(
            self.model, telemetry, detections, mask
        )

        if nan_batch_indices:
            print(f"\n🚨 NaN found in batch indices: {nan_batch_indices}")

            # Analyze the problematic batch elements
            for idx in nan_batch_indices[:5]:  # Analyze first 5 problematic ones
                print(f"\n📊 Analyzing problematic batch element {idx}:")

                # Check detection mask for this element
                element_mask = mask[idx]  # (seq_len, max_detections)
                valid_per_frame = (~element_mask).sum(dim=1)

                print(f"   Valid detections per frame: {valid_per_frame.tolist()}")
                print(
                    f"   Frames with 0 detections: {(valid_per_frame == 0).sum().item()}"
                )
                print(f"   Total valid detections: {valid_per_frame.sum().item()}")

                # Check telemetry for this element
                element_tel = telemetry[idx]
                print(
                    f"   Telemetry range: [{element_tel.min().item():.3f}, {element_tel.max().item():.3f}]"
                )
                element_det = detections[idx]
                print(
                    f"   Detections range: [{element_det.min().item():.3f}, {element_det.max().item():.3f}]"
                )

                # Check if telemetry has any extreme values
                if torch.isnan(element_tel).any():
                    print(f"   🚨 NaN in input telemetry!")
                if torch.isinf(element_tel).any():
                    print(f"   🚨 Inf in input telemetry!")

        return len(nan_batch_indices) == 0

    def debug_single_batch(self):
        """Enhanced debug to find NaN source."""
        self.model.eval()

        # Get one batch
        batch = next(iter(self.train_loader))

        telemetry = batch["telemetry_seq"].to(self.device)
        detections = batch["detection_seq"].to(self.device)
        mask = batch["detection_mask"].to(self.device)

        print("🔍 Testing model components step by step...")

        # Step 1: Input encoder
        print("\n📝 Step 1: Input Encoder")
        try:
            encoded_inputs = self.model.input_encoder(telemetry, detections, mask)

            for key, value in encoded_inputs.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    print(f"🚨 NaN in input encoder output: {key}")
                    return False
            print("✅ Input encoder OK")
        except Exception as e:
            print(f"❌ Input encoder failed: {e}")
            return False

        # Step 2: Fusion module (COMPLETE debugging)
        print("\n🔗 Step 2: Fusion Module - COMPLETE DEBUG")
        try:
            # Manual step-by-step fusion debugging
            tel_features = encoded_inputs["telemetry_features"]
            det_features = encoded_inputs["detection_features"]
            det_mask = encoded_inputs["detection_mask"]
            print(
                f"   Input shapes: tel={tel_features.shape}, det={det_features.shape}"
            )

            # Check if it's CrossModalAttentionFusion
            if hasattr(self.model.fusion_module, "tel_to_det_attention"):
                print("   Debugging CrossModalAttentionFusion step by step...")

                # Import the debug function
                from utils.nan_debug import debug_cross_modal_attention_step

                batch_size, seq_len, embedding_dim = tel_features.shape
                fused_list = []

                # Test ALL timestöeps manually
                for t in range(seq_len):
                    # Get the current sequence timestamp for verification
                    if t < tel_features.shape[1] and hasattr(batch, "timestamps"):
                        current_timestamp = (
                            batch["timestamps"][0][t].item()
                            if batch["timestamps"] is not None
                            else f"step_{t}"
                        )
                    else:
                        current_timestamp = f"step_{t}"

                    self.logger.info(f"🕒 Current timestamp: {current_timestamp}")
                    tel_t = tel_features[:, t].unsqueeze(1)
                    det_t = det_features[:, t]
                    mask_t = det_mask[:, t]

                    # Debug this specific timestep
                    if not debug_cross_modal_attention_step(tel_t, det_t, mask_t, t):
                        print(f"❌ Attention failed at timestep {t}")
                        return False

                    # Manual attention computation (mirroring the actual fusion)
                    attn_mask = ~mask_t
                    # Logging der Eingabetensoren
                    self.logger.info(f"Timestep {t}:")
                    self.logger.info(
                        f"  Query (tel_t) - min: {tel_t.min().item()}, max: {tel_t.max().item()}, mean: {tel_t.mean().item()}"
                    )
                    self.logger.info(
                        f"  Key/Value (det_t) - min: {det_t.min().item()}, max: {det_t.max().item()}, mean: {det_t.mean().item()}"
                    )
                    self.logger.info(
                        f"  Query contains nan: {torch.isnan(tel_t).any().item()}"
                    )
                    self.logger.info(
                        f"  Key/Value contains nan: {torch.isnan(det_t).any().item()}"
                    )
                    self.logger.info(
                        f"  Query contains inf: {torch.isinf(tel_t).any().item()}"
                    )
                    self.logger.info(
                        f"  Key/Value contains inf: {torch.isinf(det_t).any().item()}"
                    )
                    try:
                        tel_t = tel_t / 16
                        det_t = det_t / 16
                        # Logging der Eingabetensoren
                        self.logger.info(f"Scaled by / 16:")
                        self.logger.info(
                            f"  Query (tel_t) - min: {tel_t.min().item()}, max: {tel_t.max().item()}, mean: {tel_t.mean().item()}"
                        )
                        self.logger.info(
                            f"  Key/Value (det_t) - min: {det_t.min().item()}, max: {det_t.max().item()}, mean: {det_t.mean().item()}"
                        )
                        self.logger.info(
                            f"  Query contains nan: {torch.isnan(tel_t).any().item()}"
                        )
                        self.logger.info(
                            f"  Key/Value contains nan: {torch.isnan(det_t).any().item()}"
                        )
                        self.logger.info(
                            f"  Query contains inf: {torch.isinf(tel_t).any().item()}"
                        )
                        self.logger.info(
                            f"  Key/Value contains inf: {torch.isinf(det_t).any().item()}"
                        )
                        self.logger.info(f"Has valid detections: {mask_t.any()}")
                        self.logger.info(
                            f"Query zero rows: {(tel_t == 0).all(dim=-1).any()}"
                        )
                        self.logger.info(
                            f"Key zero rows: {(det_t == 0).all(dim=-1).any()}"
                        )
                        relevant_dets, attn_weights = (
                            self.model.fusion_module.tel_to_det_attention(
                                query=tel_t,
                                key=det_t,
                                value=det_t,
                                key_padding_mask=attn_mask,
                                need_weights=True,
                            )
                        )
                        # Logging der Attention-Gewichte
                        self.logger.info(
                            f"  Attention Weights - min: {attn_weights.min().item()}, max: {attn_weights.max().item()}, mean: {attn_weights.mean().item()}"
                        )
                        self.logger.info(
                            f"  Attention Weights contains nan: {torch.isnan(attn_weights).any().item()}"
                        )
                        self.logger.info(
                            f"  Attention Weights contains inf: {torch.isinf(attn_weights).any().item()}"
                        )

                        # Logging der Ausgabe
                        self.logger.info(
                            f"  Output (relevant_dets) - min: {relevant_dets.min().item()}, max: {relevant_dets.max().item()}, mean: {relevant_dets.mean().item()}"
                        )
                        self.logger.info(
                            f"  Output contains nan: {torch.isnan(relevant_dets).any().item()}"
                        )
                        self.logger.info(
                            f"  Output contains inf: {torch.isinf(relevant_dets).any().item()}"
                        )
                        relevant_dets = relevant_dets.squeeze(1)

                        if torch.isnan(attn_weights).any():
                            print(f"🚨 NaN in attn_weights at timestep {t}")
                        if torch.isnan(relevant_dets).any():
                            print(f"🚨 NaN in attention output at timestep {t}")
                            return False

                        # Apply norm
                        relevant_dets = (
                            self.model.fusion_module.norm_relevant_detections(
                                relevant_dets
                            )
                        )

                        if torch.isnan(relevant_dets).any():
                            print(
                                f"🚨 NaN after norm_relevant_detections at timestep {t}"
                            )
                            return False

                        # Concatenate features
                        features_to_fuse = [tel_t.squeeze(1), relevant_dets]
                        fused_input_t = torch.cat(features_to_fuse, dim=-1)

                        if torch.isnan(fused_input_t).any():
                            print(f"🚨 NaN after concatenation at timestep {t}")
                            return False

                        fused_list.append(fused_input_t)
                        print(f"   ✅ Timestep {t} OK")

                    except Exception as e:
                        print(f"❌ Manual attention failed at timestep {t}: {e}")
                        return False

                # Stack along time dimension
                print("   🔗 Stacking fused inputs...")
                fused_inputs = torch.stack(fused_list, dim=1)

                if torch.isnan(fused_inputs).any():
                    print("🚨 NaN after stacking fused inputs!")
                    return False

                print(f"   Fused inputs shape: {fused_inputs.shape}")
                print(
                    f"   Fused inputs range: [{fused_inputs.min().item():.4f}, {fused_inputs.max().item():.4f}]"
                )

                # Apply fusion MLP
                print("   🧠 Applying fusion MLP...")
                fused_output = self.model.fusion_module.fusion_mlp(fused_inputs)

                if torch.isnan(fused_output).any():
                    print("🚨 NaN after fusion MLP!")
                    return False

                print(
                    f"   MLP output range: [{fused_output.min().item():.4f}, {fused_output.max().item():.4f}]"
                )

                # Apply residual connection
                print("   🔄 Applying residual connection...")
                residual = self.model.fusion_module.residual_projection(fused_inputs)

                if torch.isnan(residual).any():
                    print("🚨 NaN in residual projection!")
                    return False

                print(
                    f"   Residual range: [{residual.min().item():.4f}, {residual.max().item():.4f}]"
                )

                fused_features = fused_output + residual

                if torch.isnan(fused_features).any():
                    print("🚨 NaN after adding residual!")
                    return False

                # Apply final norm
                print("   📏 Applying final normalization...")
                fused_features = self.model.fusion_module.norm_fusion(fused_features)

                if torch.isnan(fused_features).any():
                    print("🚨 NaN after final normalization!")
                    return False

                print(
                    f"   Final output range: [{fused_features.min().item():.4f}, {fused_features.max().item():.4f}]"
                )
                print("✅ Manual fusion computation successful!")

            # Now run actual fusion to compare
            print("   🏃 Running actual fusion module...")
            actual_fused_features = self.model.fusion_module(encoded_inputs)

            if torch.isnan(actual_fused_features).any():
                print("🚨 NaN in actual fusion module output!")
                print("   This is strange since manual computation worked...")
                return False

            print("✅ Fusion module OK")

        except Exception as e:
            print(f"❌ Fusion module failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 3: Output decoder
        print("\n📤 Step 3: Output Decoder")
        try:
            predictions = self.model.output_decoder(actual_fused_features)

            for key, value in predictions.items():
                if torch.isnan(value).any():
                    print(f"🚨 NaN in output decoder: {key}")
                    return False
            print("✅ Output decoder OK")

        except Exception as e:
            print(f"❌ Output decoder failed: {e}")
            return False

        print("✅ All components passed detailed testing!")
        return True


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
    # trainer.debug_training_step(
    #     trainer.train_loader.dataset[0:1], batch_idx=0
    # )  # Debugging a single batch
    # print("✅ Training setup complete. Debugging a single batch...")
    # trainer.train()  # Uncomment to run full training

    # DEBUG: Test single batch first
    print("🧪 Testing single batch...")
    if not trainer.debug_single_batch():
        print("❌ Single batch test failed!")
        return False

    print("✅ Single batch test passed, starting full training...")
    trainer.train()
    return True

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
