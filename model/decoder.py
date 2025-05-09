# ====================================
# OutputDecoder Implementierungen
# ====================================


from model.model import OutputDecoder
import torch
import torch.nn as nn
from typing import Dict, List


class LSTMOutputDecoder(OutputDecoder):
    """
    Output-Decoder mit LSTM-Architektur für Sequenzverarbeitung.

    Verarbeitet die fusionierten Features durch ein LSTM-Netzwerk und
    leitet daraus Vorhersagen für verschiedene Zeithorizonte ab.

    Args:
        input_dim (int): Dimensionalität der Eingabe-Features
        hidden_dim (int): Dimensionalität der LSTM-Hidden-States
        num_layers (int): Anzahl der LSTM-Schichten
        dropout_prob (float): Dropout-Wahrscheinlichkeit
        prediction_horizons (List[int]): Liste der Zeithorizonte für Prädiktionen
        include_brake_force (bool): Ob eine Regression für die Bremskraft durchgeführt werden soll
        include_uncertainty (bool): Ob eine Unsicherheitsschätzung durchgeführt werden soll
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

        # LSTM für Sequenzverarbeitung
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Binäre Prädiktionsköpfe für verschiedene Zeithorizonte
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

        # Optional: Regressionskopf für Bremskraft
        if include_brake_force:
            self.brake_force_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # Bremskraft im Bereich [0, 1]
            )

        # Optional: Unsicherheitsschätzung
        if include_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Stellt positive Unsicherheitswerte sicher
            )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward-Pass durch den LSTMOutputDecoder.

        Args:
            fused_features (torch.Tensor): Fusionierte Features
                Shape: (batch_size, seq_len, input_dim)

        Returns:
            dict: Dictionary mit Vorhersagen für die verschiedenen Aufgaben
                Für jeden Zeithorizont in prediction_horizons:
                "binary_{horizon}s": (batch_size, 1)

                Optional:
                "brake_force": (batch_size, 1)
                "uncertainty": (batch_size, 1)
        """
        # LSTM-Verarbeitung
        lstm_out, (lstm_h_n, _) = self.lstm(fused_features)
        final_state = lstm_h_n[-1]  # Letzter Hidden State

        # Prädiktionen für verschiedene Zeithorizonte
        predictions = {}
        for horizon in self.prediction_horizons:
            predictions[f"binary_{horizon}s"] = self.binary_heads[
                f"horizon_{horizon}s"
            ](final_state)

        # Optional: Bremskraft-Regression
        if self.include_brake_force:
            predictions["brake_force"] = self.brake_force_head(final_state)

        # Optional: Unsicherheitsschätzung
        if self.include_uncertainty:
            predictions["uncertainty"] = self.uncertainty_head(final_state)

        return predictions


class TransformerOutputDecoder(OutputDecoder):
    """
    Output-Decoder mit Transformer-Architektur für Sequenzverarbeitung.

    Verarbeitet die fusionierten Features durch Transformer-Encoder-Layer und
    leitet daraus Vorhersagen für verschiedene Zeithorizonte ab.

    Args:
        input_dim (int): Dimensionalität der Eingabe-Features
        hidden_dim (int): Dimensionalität der internen Repräsentation
        num_heads (int): Anzahl der Attention-Heads
        num_layers (int): Anzahl der Transformer-Encoder-Schichten
        dropout_prob (float): Dropout-Wahrscheinlichkeit
        prediction_horizons (List[int]): Liste der Zeithorizonte für Prädiktionen
        include_brake_force (bool): Ob eine Regression für die Bremskraft durchgeführt werden soll
        include_uncertainty (bool): Ob eine Unsicherheitsschätzung durchgeführt werden soll
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

        # Lineares Mapping auf hidden_dim, falls input_dim nicht übereinstimmt
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

        # Binäre Prädiktionsköpfe für verschiedene Zeithorizonte
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

        # Optional: Regressionskopf für Bremskraft
        if include_brake_force:
            self.brake_force_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),  # Bremskraft im Bereich [0, 1]
            )

        # Optional: Unsicherheitsschätzung
        if include_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # Stellt positive Unsicherheitswerte sicher
            )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward-Pass durch den TransformerOutputDecoder.

        Args:
            fused_features (torch.Tensor): Fusionierte Features
                Shape: (batch_size, seq_len, input_dim)

        Returns:
            dict: Dictionary mit Vorhersagen für die verschiedenen Aufgaben
                Für jeden Zeithorizont in prediction_horizons:
                "binary_{horizon}s": (batch_size, 1)

                Optional:
                "brake_force": (batch_size, 1)
                "uncertainty": (batch_size, 1)
        """
        # Projektion auf hidden_dim, falls notwendig
        features = self.input_projection(fused_features)

        # Transformer-Verarbeitung
        transformer_out = self.transformer_encoder(features)

        # Verwende den letzten Token der Sequenz für die Vorhersagen
        final_state = transformer_out[:, -1]

        # Prädiktionen für verschiedene Zeithorizonte
        predictions = {}
        for horizon in self.prediction_horizons:
            predictions[f"binary_{horizon}s"] = self.binary_heads[
                f"horizon_{horizon}s"
            ](final_state)

        # Optional: Bremskraft-Regression
        if self.include_brake_force:
            predictions["brake_force"] = self.brake_force_head(final_state)

        # Optional: Unsicherheitsschätzung
        if self.include_uncertainty:
            predictions["uncertainty"] = self.uncertainty_head(final_state)

        return predictions
