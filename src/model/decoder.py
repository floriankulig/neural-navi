# ====================================
# OutputDecoder Implementations
# ====================================


from model.model import OutputDecoder
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
        prediction_horizons (List[int]): List of time horizons for predictions
        include_brake_force (bool): Whether to perform regression for brake force
        include_uncertainty (bool): Whether to perform uncertainty estimation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_prob: float = 0.3,
        prediction_horizons: List[int] = [1, 3, 5],
        include_brake_force: bool = False,
        include_uncertainty: bool = False,
    ):
        super().__init__()

        self.prediction_horizons = prediction_horizons
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

        # Binary prediction heads for different time horizons
        self.binary_heads = nn.ModuleDict(
            {
                f"horizon_{horizon}s": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for horizon in prediction_horizons
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
                For each horizon in prediction_horizons:
                "binary_{horizon}s": (batch_size, 1)

                Optional:
                "brake_force": (batch_size, 1)
                "uncertainty": (batch_size, 1)
        """
        # LSTM processing
        lstm_out, (lstm_h_n, _) = self.lstm(fused_features)
        final_state = lstm_h_n[-1]  # Last hidden state

        # Predictions for various time horizons
        predictions = {}
        for horizon in self.prediction_horizons:
            predictions[f"binary_{horizon}s"] = self.binary_heads[
                f"horizon_{horizon}s"
            ](final_state)

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
        prediction_horizons (List[int]): List of time horizons for predictions
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
        prediction_horizons: List[int] = [1, 3, 5],
        include_brake_force: bool = False,
        include_uncertainty: bool = False,
    ):
        super().__init__()

        self.prediction_horizons = prediction_horizons
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

        # Binary prediction heads for different time horizons
        self.binary_heads = nn.ModuleDict(
            {
                f"horizon_{horizon}s": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for horizon in prediction_horizons
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
                For each horizon in prediction_horizons:
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

        # Predictions for various time horizons
        predictions = {}
        for horizon in self.prediction_horizons:
            predictions[f"binary_{horizon}s"] = self.binary_heads[
                f"horizon_{horizon}s"
            ](final_state)

        # Optional: Brake force regression
        if self.include_brake_force:
            predictions["brake_force"] = self.brake_force_head(final_state)

        # Optional: Uncertainty estimation
        if self.include_uncertainty:
            predictions["uncertainty"] = self.uncertainty_head(final_state)

        return predictions
