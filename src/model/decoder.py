# ====================================
# OutputDecoder Implementations
# ====================================

from pathlib import Path
import sys

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent  # Go up two levels: model/ -> src/ -> root/
sys.path.insert(0, str(project_root))

from src.model.base import OutputDecoder
import torch
import torch.nn as nn
from typing import Dict, List


class LSTMOutputDecoder(OutputDecoder):
    """
    Output decoder with LSTM architecture for sequence processing.

    Processes the fused features through an LSTM network and
    derives predictions for various time horizons.

    Args:
        input_dim (int): Dimensionality of the input features
        hidden_dim (int): Dimensionality of the LSTM hidden states
        num_layers (int): Number of LSTM layers
        dropout_prob (float): Dropout probability
        prediction_tasks (List[str]): List of prediction tasks (e.g., ["brake_1s", "brake_2s", "brake_4s"])
        include_brake_force (bool): Whether to perform regression for brake force
        include_uncertainty (bool): Whether to perform uncertainty estimation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_prob: float = 0.3,
        prediction_tasks: List[str] = ["brake_1s", "brake_2s", "brake_4s"],
        include_brake_force: bool = False,
        include_uncertainty: bool = False,
    ):
        super().__init__()

        self.prediction_tasks = prediction_tasks
        self.include_brake_force = include_brake_force
        self.include_uncertainty = include_uncertainty

        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Task-specific prediction heads (instead of horizon-based)
        self.task_heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for task_name in prediction_tasks  # brake_1s, coast_1s, etc.
            }
        )

        # Optional: Regression head for brake force
        if include_brake_force:
            self.brake_force_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # Brake force in range [0, 1]
            )

        # Optional: Uncertainty estimation
        if include_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Ensures positive uncertainty values
            )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the LSTMOutputDecoder.

        Args:
            fused_features (torch.Tensor): Fused features
                Shape: (batch_size, seq_len, input_dim)

        Returns:
            dict: Dictionary with predictions for different tasks
                For each task in prediction_tasks:
                "binary_{horizon}s": (batch_size, 1)

                Optional:
                "brake_force": (batch_size, 1)
                "uncertainty": (batch_size, 1)
        """
        # LSTM processing
        lstm_out, (lstm_h_n, _) = self.lstm(fused_features)
        final_state = lstm_h_n[-1]  # Last hidden state

        # Task-specific predictions
        predictions = {}
        for task_name in self.prediction_tasks:
            predictions[task_name] = self.task_heads[task_name](final_state)

        # Optional: Brake force regression
        if self.include_brake_force:
            predictions["brake_force"] = self.brake_force_head(final_state)

        # Optional: Uncertainty estimation
        if self.include_uncertainty:
            predictions["uncertainty"] = self.uncertainty_head(final_state)

        return predictions


class TransformerOutputDecoder(OutputDecoder):
    """
    Output decoder with Transformer architecture for sequence processing.

    Processes the fused features through Transformer encoder layers and
    derives predictions for various time horizons.

    Args:
        input_dim (int): Dimensionality of the input features
        hidden_dim (int): Dimensionality of the internal representation
        num_heads (int): Number of attention heads
        num_layers (int): Number of transformer encoder layers
        dropout_prob (float): Dropout probability
        prediction_tasks (List[str]): List of prediction tasks (e.g., ["brake_1s", "brake_2s", "brake_4s"])
        include_brake_force (bool): Whether to perform regression for brake force
        include_uncertainty (bool): Whether to perform uncertainty estimation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout_prob: float = 0.3,
        prediction_tasks: List[str] = ["brake_1s", "brake_2s", "brake_4s"],
        include_brake_force: bool = False,
        include_uncertainty: bool = False,
    ):
        super().__init__()

        self.prediction_tasks = prediction_tasks
        self.include_brake_force = include_brake_force
        self.include_uncertainty = include_uncertainty

        # Linear mapping to hidden_dim if input_dim doesn't match
        self.input_projection = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_prob,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )

        # Task-specific prediction heads (instead of horizon-based)
        self.task_heads = nn.ModuleDict(
            {
                task_name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for task_name in prediction_tasks  # brake_1s, coast_1s, etc.
            }
        )

        # Optional: Regression head for brake force
        if include_brake_force:
            self.brake_force_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # Brake force in range [0, 1]
            )

        # Optional: Uncertainty estimation
        if include_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Ensures positive uncertainty values
            )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the TransformerOutputDecoder.

        Args:
            fused_features (torch.Tensor): Fused features
                Shape: (batch_size, seq_len, input_dim)

        Returns:
            dict: Dictionary with predictions for different tasks
                For each task in prediction_tasks:
                "binary_{horizon}s": (batch_size, 1)

                Optional:
                "brake_force": (batch_size, 1)
                "uncertainty": (batch_size, 1)
        """
        # Projection to hidden_dim if necessary
        features = self.input_projection(fused_features)

        # Transformer processing
        transformer_out = self.transformer_encoder(features)

        # Use the last token of the sequence for predictions
        final_state = transformer_out[:, -1]

        # Task-specific predictions
        predictions = {}
        for task_name in self.prediction_tasks:
            predictions[task_name] = self.task_heads[task_name](final_state)

        # Optional: Brake force regression
        if self.include_brake_force:
            predictions["brake_force"] = self.brake_force_head(final_state)

        # Optional: Uncertainty estimation
        if self.include_uncertainty:
            predictions["uncertainty"] = self.uncertainty_head(final_state)

        return predictions
