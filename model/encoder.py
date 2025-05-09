# ====================================
# InputEncoder Implementations
# ====================================


from model.model import InputEncoder
import torch
import torch.nn as nn
import math
from typing import Dict


class SimpleInputEncoder(InputEncoder):
    """
    Simple encoder without complex attention mechanisms.

    Encodes telemetry data and object detections independently and aggregates
    the detections by taking the mean over all valid objects.

    Args:
      telemetry_input_dim (int): Dimension of telemetry data
      detection_input_dim_per_box (int): Dimension of properties for each object detection
      embedding_dim (int): Dimension of the generated embeddings
      dropout_prob (float): Dropout probability
      use_positional_encoding (bool): Whether to use positional encoding
      max_seq_length (int): Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        telemetry_input_dim: int,
        detection_input_dim_per_box: int,
        embedding_dim: int = 64,
        dropout_prob: float = 0.1,
        use_positional_encoding: bool = True,
        max_seq_length: int = 20,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_positional_encoding = use_positional_encoding

        # Telemetry embedding
        self.telemetry_embed = nn.Sequential(
            nn.Linear(telemetry_input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Detection embedding (per object)
        self.detection_embed = nn.Sequential(
            nn.Linear(detection_input_dim_per_box, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        # LayerNorm for outputs
        self.norm_telemetry = nn.LayerNorm(embedding_dim)
        self.norm_detections = nn.LayerNorm(embedding_dim)

        # Optional positional encoding for sequence data
        if use_positional_encoding:
            self.register_buffer(
                "position_encoding",
                self._create_position_encoding(max_seq_length, embedding_dim),
            )

    def _create_position_encoding(self, max_len, d_model):
        """Creates positional encoding for sequences as in the Transformer paper."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        return pe

    def forward(
        self,
        telemetry_seq: torch.Tensor,
        detection_seq: torch.Tensor,
        detection_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the SimpleInputEncoder.

        Args:
          telemetry_seq (torch.Tensor): Telemetry sequence data
            Shape: (batch_size, seq_len, telemetry_input_dim)
          detection_seq (torch.Tensor): Detection sequence data (with padding)
            Shape: (batch_size, seq_len, max_detections, detection_input_dim_per_box)
          detection_mask (torch.Tensor): Boolean mask for valid detections (True=valid)
            Shape: (batch_size, seq_len, max_detections)

        Returns:
          dict: Dictionary with encoded features
            "telemetry_features": (batch_size, seq_len, embedding_dim)
            "detection_features": (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = telemetry_seq.shape
        _, _, max_dets, det_feat_dim = detection_seq.shape

        # --- Telemetry encoding ---
        telemetry_embedded = self.telemetry_embed(telemetry_seq)

        # Add positional encoding if activated
        if self.use_positional_encoding:
            pos_enc = self.position_encoding[:, :seq_len, :]
            telemetry_embedded = telemetry_embedded + pos_enc

        telemetry_embedded = self.norm_telemetry(telemetry_embedded)

        # --- Detection encoding ---
        # Reshape for efficient processing
        detection_seq_flat = detection_seq.view(-1, det_feat_dim)
        embedded_detections_flat = self.detection_embed(detection_seq_flat)
        embedded_detections = embedded_detections_flat.view(
            batch_size, seq_len, max_dets, self.embedding_dim
        )

        # Aggregation of detections per time step (weighted average)
        valid_detections = detection_mask.unsqueeze(-1).float()
        detection_sum = (embedded_detections * valid_detections).sum(dim=2)
        detection_count = valid_detections.sum(dim=2) + 1e-6  # Avoids division by zero
        detection_aggregated = detection_sum / detection_count

        detection_aggregated = self.norm_detections(detection_aggregated)

        return {
            "telemetry_features": telemetry_embedded,
            "detection_features": detection_aggregated,
        }


class AttentionInputEncoder(InputEncoder):
    """
    Advanced encoder with internal self-attention for both modalities.

    Uses self-attention to capture relationships between objects as well as
    temporal patterns in telemetry data.

    Args:
      telemetry_input_dim (int): Dimension of telemetry data
      detection_input_dim_per_box (int): Dimension of properties for each object detection
      embedding_dim (int): Dimension of the generated embeddings
      attention_num_heads (int): Number of attention heads
      dropout_prob (float): Dropout probability
      max_detections (int): Maximum number of detections per time step
      max_seq_length (int): Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        telemetry_input_dim: int,
        detection_input_dim_per_box: int,
        embedding_dim: int = 64,
        attention_num_heads: int = 4,
        dropout_prob: float = 0.1,
        max_detections: int = 10,
        max_seq_length: int = 20,
    ):
        super().__init__()

        if embedding_dim % attention_num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"attention_num_heads ({attention_num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.max_detections = max_detections

        # Telemetry embedding
        self.telemetry_embed = nn.Linear(telemetry_input_dim, embedding_dim)

        # Detection embedding (per object)
        self.detection_embed = nn.Sequential(
            nn.Linear(detection_input_dim_per_box, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        # Temporal self-attention for telemetry
        self.telemetry_self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=attention_num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Object-to-object self-attention for detections
        self.object_self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=attention_num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # LayerNorms for residual connections
        self.norm_telemetry = nn.LayerNorm(embedding_dim)
        self.norm_telemetry_attended = nn.LayerNorm(embedding_dim)
        self.norm_objects = nn.LayerNorm(embedding_dim)
        self.norm_objects_attended = nn.LayerNorm(embedding_dim)

        # Positional encoding
        self.register_buffer(
            "temporal_position_encoding",
            self._create_temporal_position_encoding(max_seq_length, embedding_dim),
        )

    def _create_temporal_position_encoding(self, max_len, d_model):
        """Creates positional encoding for sequences as in the Transformer paper."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        return pe

    def forward(
        self,
        telemetry_seq: torch.Tensor,
        detection_seq: torch.Tensor,
        detection_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the AttentionInputEncoder.

        Args:
          telemetry_seq (torch.Tensor): Telemetry sequence data
            Shape: (batch_size, seq_len, telemetry_input_dim)
          detection_seq (torch.Tensor): Detection sequence data (with padding)
            Shape: (batch_size, seq_len, max_detections, detection_input_dim_per_box)
          detection_mask (torch.Tensor): Boolean mask for valid detections (True=valid)
            Shape: (batch_size, seq_len, max_detections)

        Returns:
          dict: Dictionary with encoded features
            "telemetry_features": (batch_size, seq_len, embedding_dim)
            "detection_features": (batch_size, seq_len, embedding_dim)
            "attention_weights": Optional - Dictionary with attention weights
        """
        batch_size, seq_len, _ = telemetry_seq.shape
        _, _, max_dets, det_feat_dim = detection_seq.shape

        # Store attention weights if requested
        attention_weights = {}
        store_attentions = kwargs.get("return_attentions", False)

        # --- Telemetry encoding ---
        telemetry_embedded = self.telemetry_embed(telemetry_seq)

        # Add positional encoding
        pos_enc = self.temporal_position_encoding[:, :seq_len, :]
        telemetry_embedded = telemetry_embedded + pos_enc
        telemetry_embedded = self.norm_telemetry(telemetry_embedded)

        # --- Telemetry self-attention (temporal) ---
        telemetry_attended, temp_attn_weights = self.telemetry_self_attention(
            query=telemetry_embedded, key=telemetry_embedded, value=telemetry_embedded
        )

        if store_attentions:
            attention_weights["telemetry_self_attention"] = temp_attn_weights

        # Residual connection and normalization
        telemetry_attended = telemetry_embedded + telemetry_attended
        telemetry_attended = self.norm_telemetry_attended(telemetry_attended)

        # --- Detection encoding ---
        # Reshape for efficient processing
        detection_seq_flat = detection_seq.view(-1, det_feat_dim)
        embedded_detections_flat = self.detection_embed(detection_seq_flat)
        embedded_detections = embedded_detections_flat.view(
            batch_size, seq_len, max_dets, self.embedding_dim
        )

        embedded_detections = self.norm_objects(embedded_detections)

        # --- Object-to-object self-attention (per time step) ---
        objects_attended = []
        object_attn_weights = []

        for t in range(seq_len):
            # Extract objects for this time step
            objects_t = embedded_detections[
                :, t
            ]  # (batch_size, max_dets, embedding_dim)
            mask_t = detection_mask[:, t]  # (batch_size, max_dets)

            # Create attention mask (True = ignore)
            obj_attn_mask = ~mask_t

            # Transpose for attention: (max_dets, batch_size, embedding_dim)
            objects_t_trans = objects_t.transpose(0, 1)

            # Apply self-attention
            objects_attended_t, attn_weights_t = self.object_self_attention(
                query=objects_t_trans,
                key=objects_t_trans,
                value=objects_t_trans,
                key_padding_mask=obj_attn_mask,
            )

            # Transpose back: (batch_size, max_dets, embedding_dim)
            objects_attended_t = objects_attended_t.transpose(0, 1)
            objects_attended.append(objects_attended_t)

            if store_attentions:
                object_attn_weights.append(attn_weights_t)

        # Stack along the time dimension
        objects_attended = torch.stack(objects_attended, dim=1)

        if store_attentions:
            attention_weights["object_self_attention"] = object_attn_weights

        # Aggregation of detections (with attention) per time step
        valid_detections = detection_mask.unsqueeze(-1).float()
        detection_sum = (objects_attended * valid_detections).sum(dim=2)
        detection_count = valid_detections.sum(dim=2) + 1e-6  # Avoids division by zero
        detection_aggregated = detection_sum / detection_count

        detection_aggregated = self.norm_objects_attended(detection_aggregated)

        result = {
            "telemetry_features": telemetry_attended,
            "detection_features": detection_aggregated,
        }

        if store_attentions:
            result["attention_weights"] = attention_weights

        return result
