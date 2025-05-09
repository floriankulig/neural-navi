# ====================================
# FusionModule Implementierungen
# ====================================


from typing import Optional, Dict
from model.model import FusionModule
import torch
import torch.nn as nn


class SimpleConcatenationFusion(FusionModule):
    """
    Einfache Fusion durch Konkatenation und MLP.

    Konkateniert die Feature-Vektoren der verschiedenen Modalitäten und
    verarbeitet diese durch ein Multi-Layer Perceptron.

    Args:
        embedding_dim (int): Dimensionalität der eingehenden Feature-Vektoren
        output_dim (int, optional): Dimensionalität des Ausgabe-Vektors.
            Standardmäßig embedding_dim * 2.
        dropout_prob (float): Dropout-Wahrscheinlichkeit
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.output_dim = output_dim or embedding_dim * 2

        # MLP für Fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward-Pass durch die SimpleConcatenationFusion.

        Args:
            encoded_inputs (dict): Dictionary mit encodierten Features
                "telemetry_features": (batch_size, seq_len, embedding_dim)
                "detection_features": (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: Fusionierte Features
                Shape: (batch_size, seq_len, output_dim)
        """
        telemetry = encoded_inputs["telemetry_features"]
        detections = encoded_inputs["detection_features"]

        # Einfache Konkatenation entlang der Feature-Dimension
        fused_features = torch.cat([telemetry, detections], dim=-1)

        # Verarbeitung durch MLP
        fused_features = self.fusion_mlp(fused_features)
        fused_features = self.norm(fused_features)

        return fused_features


class CrossModalAttentionFusion(FusionModule):
    """
    Fusion mit Cross-Modal-Attention-Mechanismus.

    Verwendet Multi-Head Attention, damit Telemetriedaten die relevanten
    Informationen aus den Detektionsdaten extrahieren können und umgekehrt.

    Args:
        embedding_dim (int): Dimensionalität der eingehenden Feature-Vektoren
        num_heads (int): Anzahl der Attention-Heads
        output_dim (int, optional): Dimensionalität des Ausgabe-Vektors.
            Standardmäßig embedding_dim * 2.
        dropout_prob (float): Dropout-Wahrscheinlichkeit
        bidirectional (bool): Ob Attention in beide Richtungen angewendet werden soll
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        output_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) muss durch "
                f"num_heads ({num_heads}) teilbar sein"
            )

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim or embedding_dim * 2
        self.bidirectional = bidirectional

        # Telemetrie -> Detektionen Attention
        self.tel_to_det_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Optional: Detektionen -> Telemetrie Attention (bidirektional)
        if bidirectional:
            self.det_to_tel_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout_prob,
                batch_first=True,
            )

        # Layernorms für Attention-Ausgaben
        self.norm_tel_attended = nn.LayerNorm(embedding_dim)
        if bidirectional:
            self.norm_det_attended = nn.LayerNorm(embedding_dim)

        # MLP für Fusion der Attention-Ergebnisse
        fusion_input_dim = embedding_dim * 3 if bidirectional else embedding_dim * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.norm_fusion = nn.LayerNorm(self.output_dim)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward-Pass durch die CrossModalAttentionFusion.

        Args:
            encoded_inputs (dict): Dictionary mit encodierten Features
                "telemetry_features": (batch_size, seq_len, embedding_dim)
                "detection_features": (batch_size, seq_len, embedding_dim)

        Returns:
            torch.Tensor: Fusionierte Features
                Shape: (batch_size, seq_len, output_dim)
        """
        telemetry = encoded_inputs["telemetry_features"]
        detections = encoded_inputs["detection_features"]

        # Telemetrie (Query) betrachtet Detektionen (Key, Value)
        tel_attended, _ = self.tel_to_det_attention(
            query=telemetry, key=detections, value=detections
        )

        # Residual-Verbindung und Normalisierung
        tel_attended = telemetry + tel_attended
        tel_attended = self.norm_tel_attended(tel_attended)

        if self.bidirectional:
            # Detektionen (Query) betrachten Telemetrie (Key, Value)
            det_attended, _ = self.det_to_tel_attention(
                query=detections, key=telemetry, value=telemetry
            )

            # Residual-Verbindung und Normalisierung
            det_attended = detections + det_attended
            det_attended = self.norm_det_attended(det_attended)

            # Konkatenieren und durch MLP führen
            fused_features = torch.cat([tel_attended, det_attended, detections], dim=-1)
        else:
            # Konkatenieren und durch MLP führen
            fused_features = torch.cat([tel_attended, detections], dim=-1)

        fused_features = self.fusion_mlp(fused_features)
        fused_features = self.norm_fusion(fused_features)

        return fused_features
