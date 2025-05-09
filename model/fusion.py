# ====================================
# FusionModule Implementations
# ====================================


from typing import Optional, Dict
from model.model import FusionModule
import torch
import torch.nn as nn


class SimpleConcatenationFusion(FusionModule):
    """
    Simple fusion through concatenation and MLP.

    Concatenates the feature vectors from different modalities and
    processes them through a Multi-Layer Perceptron.

    Args:
      embedding_dim (int): Dimensionality of the input feature vectors
      output_dim (int, optional): Dimensionality of the output vector.
        Default is embedding_dim * 2.
      dropout_prob (float): Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.output_dim = output_dim or embedding_dim * 2

        # MLP for fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the SimpleConcatenationFusion.

        Args:
          encoded_inputs (dict): Dictionary with encoded features
            "telemetry_features": (batch_size, seq_len, embedding_dim)
            "detection_features": (batch_size, seq_len, embedding_dim)

        Returns:
          torch.Tensor: Fused features
            Shape: (batch_size, seq_len, output_dim)
        """
        telemetry = encoded_inputs["telemetry_features"]
        detections = encoded_inputs["detection_features"]

        # Simple concatenation along the feature dimension
        fused_features = torch.cat([telemetry, detections], dim=-1)

        # Processing through MLP
        fused_features = self.fusion_mlp(fused_features)
        fused_features = self.norm(fused_features)

        return fused_features


class CrossModalAttentionFusion(FusionModule):
    """
    Fusion with Cross-Modal Attention mechanism.

    Uses Multi-Head Attention to allow telemetry data to extract relevant
    information from detection data and vice versa.

    Args:
      embedding_dim (int): Dimensionality of the input feature vectors
      num_heads (int): Number of attention heads
      output_dim (int, optional): Dimensionality of the output vector.
        Default is embedding_dim * 2.
      dropout_prob (float): Dropout probability
      bidirectional (bool): Whether to apply attention in both directions
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
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim or embedding_dim * 2
        self.bidirectional = bidirectional

        # Telemetry -> Detections Attention
        self.tel_to_det_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Optional: Detections -> Telemetry Attention (bidirectional)
        if bidirectional:
            self.det_to_tel_attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout_prob,
                batch_first=True,
            )

        # Layernorms for attention outputs
        self.norm_tel_attended = nn.LayerNorm(embedding_dim)
        if bidirectional:
            self.norm_det_attended = nn.LayerNorm(embedding_dim)

        # MLP for fusion of attention results
        fusion_input_dim = embedding_dim * 3 if bidirectional else embedding_dim * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.norm_fusion = nn.LayerNorm(self.output_dim)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the CrossModalAttentionFusion.

        Args:
          encoded_inputs (dict): Dictionary with encoded features
            "telemetry_features": (batch_size, seq_len, embedding_dim)
            "detection_features": (batch_size, seq_len, embedding_dim)

        Returns:
          torch.Tensor: Fused features
            Shape: (batch_size, seq_len, output_dim)
        """
        telemetry = encoded_inputs["telemetry_features"]
        detections = encoded_inputs["detection_features"]

        # Telemetry (Query) looks at Detections (Key, Value)
        tel_attended, _ = self.tel_to_det_attention(
            query=telemetry, key=detections, value=detections
        )

        # Residual connection and normalization
        tel_attended = telemetry + tel_attended
        tel_attended = self.norm_tel_attended(tel_attended)

        if self.bidirectional:
            # Detections (Query) look at Telemetry (Key, Value)
            det_attended, _ = self.det_to_tel_attention(
                query=detections, key=telemetry, value=telemetry
            )

            # Residual connection and normalization
            det_attended = detections + det_attended
            det_attended = self.norm_det_attended(det_attended)

            # Concatenate and pass through MLP
            fused_features = torch.cat([tel_attended, det_attended, detections], dim=-1)
        else:
            # Concatenate and pass through MLP
            fused_features = torch.cat([tel_attended, detections], dim=-1)

        fused_features = self.fusion_mlp(fused_features)
        fused_features = self.norm_fusion(fused_features)

        return fused_features
