# ====================================
# FusionModule Implementations
# ====================================


from typing import Optional, Dict
from model.model import FusionModule
import torch
import torch.nn as nn


class SimpleConcatenationFusion(FusionModule):
    """
    Simple fusion through aggregation of object detections and concatenation with telemetry.

    Processes each object detection, aggregates them, and then concatenates
    with telemetry features.

    Args:
      embedding_dim (int): Dimensionality of the input feature vectors
      output_dim (int, optional): Dimensionality of the output vector.
        Default is embedding_dim * 2.
      dropout_prob (float): Dropout probability
      aggregation_method (str): Method to aggregate object detections ('mean', 'max', 'attention')
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        aggregation_method: str = "mean",
    ):
        super().__init__()

        self.output_dim = output_dim or embedding_dim * 2
        self.aggregation_method = aggregation_method

        # MLP for fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.norm = nn.LayerNorm(self.output_dim)

        if aggregation_method == "mean":
            self.norm_detection = nn.LayerNorm(embedding_dim)

    def _aggregate_detections(self, detections, mask):
        """
        Aggregate object detections using the specified method.

        Args:
          detections: Tensor of shape (batch_size, seq_len, max_detections, embedding_dim)
          mask: Boolean mask of shape (batch_size, seq_len, max_detections)

        Returns:
          Aggregated detections of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, max_detections, embedding_dim = detections.shape

        if self.aggregation_method == "mean":
            # Mean pooling considering only valid detections
            # Replace masked (invalid) values with zeros
            masked_detections = detections * mask.unsqueeze(-1)

            # Sum and divide by number of valid detections (avoid div by zero)
            detection_sum = masked_detections.sum(dim=2)  # Sum along max_detections
            valid_count = mask.sum(dim=2, keepdim=True).clamp(
                min=1
            )  # Count valid, avoid div by zero
            return detection_sum / valid_count

        elif self.aggregation_method == "max":
            # Max pooling considering only valid detections
            # Replace masked (invalid) values with large negative values
            masked_detections = detections.clone()
            masked_detections[~mask.unsqueeze(-1).expand_as(masked_detections)] = float(
                "-inf"
            )

            # Max along detection dimension
            return torch.max(masked_detections, dim=2)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the SimpleConcatenationFusion.

        Args:
          encoded_inputs (dict): Dictionary with encoded features
            "telemetry_features": (batch_size, seq_len, embedding_dim)
            "detection_features": (batch_size, seq_len, max_detections, embedding_dim)
            "detection_mask": (batch_size, seq_len, max_detections)

        Returns:
          torch.Tensor: Fused features
            Shape: (batch_size, seq_len, output_dim)
        """
        telemetry = encoded_inputs["telemetry_features"]
        detections = encoded_inputs["detection_features"]
        mask = encoded_inputs["detection_mask"]

        # Aggregate detections
        aggregated_detections = self._aggregate_detections(detections, mask)

        if self.aggregation_method == "attention":
            aggregated_detections = self.norm_detection(aggregated_detections)

        # Simple concatenation along the feature dimension
        fused_features = torch.cat([telemetry, aggregated_detections], dim=-1)

        # Processing through MLP
        fused_features = self.fusion_mlp(fused_features)
        fused_features = self.norm(fused_features)

        return fused_features


class CrossModalAttentionFusion(FusionModule):
    """
    Fusion with unidirectional Cross-Modal Attention and Residual Connection.

    Uses Multi-Head Attention where telemetry queries which object detections
    are relevant for the current driving situation.
    Includes a residual connection around the fusion MLP for improved gradient flow.

    Args:
        embedding_dim (int): Dimensionality of the input feature vectors
        num_heads (int): Number of attention heads
        output_dim (int, optional): Dimensionality of the output vector
        dropout_prob (float): Dropout probability
        use_attention_weights (bool): Whether to use attention weights for additional features
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        output_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
        max_detections: int = 12,
        use_attention_weights: bool = False,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim or embedding_dim * 2
        self.use_attention_weights = use_attention_weights

        # Telemetry -> Detections Attention
        self.tel_to_det_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Layernorm for attended features
        self.norm_relevant_detections = nn.LayerNorm(embedding_dim)

        # Calculate fusion input dimension
        if use_attention_weights:
            # Attention weights can tell us about object importance distribution
            self.attention_processor = nn.Sequential(
                nn.Linear(max_detections, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
            )
            fusion_input_dim = embedding_dim * 2 + embedding_dim // 2
        else:
            fusion_input_dim = embedding_dim * 2  # telemetry + relevant_detections

        self.fusion_input_dim = fusion_input_dim

        # MLP for final fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Residual connection projection (if dimensions don't match)
        if fusion_input_dim != self.output_dim:
            self.residual_projection = nn.Linear(fusion_input_dim, self.output_dim)
        else:
            self.residual_projection = nn.Identity()

        # Layer normalization after residual connection
        self.norm_fusion = nn.LayerNorm(self.output_dim)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the CrossModalAttentionFusion with residual connection.

        Args:
          encoded_inputs (dict): Dictionary with encoded features

        Returns:
          torch.Tensor: Fused features (batch_size, seq_len, output_dim)
        """
        telemetry = encoded_inputs[
            "telemetry_features"
        ]  # [batch_size, seq_len, embedding_dim]
        detections = encoded_inputs[
            "detection_features"
        ]  # [batch_size, seq_len, max_detections, embedding_dim]
        mask = encoded_inputs["detection_mask"]  # [batch_size, seq_len, max_detections]

        batch_size, seq_len, embedding_dim = telemetry.shape
        _, _, max_detections, _ = detections.shape

        # Process each timestep
        fused_list = []

        for t in range(seq_len):
            # Get features for this timestep
            tel_t = telemetry[:, t].unsqueeze(1)  # [batch_size, 1, embedding_dim]
            det_t = detections[:, t]  # [batch_size, max_detections, embedding_dim]
            mask_t = mask[:, t]  # [batch_size, max_detections]

            # Attention mask (True = ignore)
            attn_mask = ~mask_t

            # Telemetry queries relevant detections
            relevant_dets, attn_weights = self.tel_to_det_attention(
                query=tel_t, key=det_t, value=det_t, key_padding_mask=attn_mask
            )

            relevant_dets = relevant_dets.squeeze(1)  # [batch_size, embedding_dim]
            relevant_dets = self.norm_relevant_detections(relevant_dets)

            # Concatenate telemetry with relevant detections
            features_to_fuse = [tel_t.squeeze(1), relevant_dets]

            # Optionally process attention weights
            if self.use_attention_weights:
                # attn_weights shape: [batch_size, 1, max_detections]
                attn_features = self.attention_processor(attn_weights.squeeze(1))
                features_to_fuse.append(attn_features)

            # Concatenate all features
            fused_input_t = torch.cat(features_to_fuse, dim=-1)

            fused_list.append(fused_input_t)

        # Stack along the time axis to get full sequence
        fused_inputs = torch.stack(fused_list, dim=1)

        # Apply fusion MLP
        fused_output = self.fusion_mlp(fused_inputs)

        # Apply residual connection
        residual = self.residual_projection(fused_inputs)
        fused_features = fused_output + residual

        # Final normalization after residual connection
        fused_features = self.norm_fusion(fused_features)

        return fused_features


class ObjectQueryFusion(FusionModule):
    """
    Advanced fusion module using learnable object queries to extract relevant information.

    Instead of direct cross-attention, this module uses a set of learnable queries
    that attend to both telemetry and detections separately, and then combines
    the information. This allows the model to focus on specific aspects of the
    driving scene in a more flexible way.

    Args:
      embedding_dim (int): Dimensionality of the input feature vectors
      num_queries (int): Number of specialized object queries
      num_heads (int): Number of attention heads
      output_dim (int, optional): Dimensionality of the output vector
      dropout_prob (float): Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int,
        num_queries: int = 8,
        num_heads: int = 4,
        output_dim: Optional[int] = None,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_queries = num_queries
        self.output_dim = output_dim or embedding_dim * 2

        # Learnable queries
        self.object_queries = nn.Parameter(torch.randn(1, num_queries, embedding_dim))

        # Multi-head attention for queries attending to telemetry
        self.query_to_tel_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Multi-head attention for queries attending to detections
        self.query_to_det_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Final attention to combine query outputs
        self.final_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Context vector for final attention
        self.context_vector = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Layernorms
        self.norm_queries_tel = nn.LayerNorm(embedding_dim)
        self.norm_queries_det = nn.LayerNorm(embedding_dim)
        self.norm_final = nn.LayerNorm(embedding_dim)

        # MLP for final projection
        self.output_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.norm_output = nn.LayerNorm(self.output_dim)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the ObjectQueryFusion.

        Args:
          encoded_inputs (dict): Dictionary with encoded features
            "telemetry_features": (batch_size, seq_len, embedding_dim)
            "detection_features": (batch_size, seq_len, max_detections, embedding_dim)
            "detection_mask": (batch_size, seq_len, max_detections)

        Returns:
          torch.Tensor: Fused features
            Shape: (batch_size, seq_len, output_dim)
        """
        telemetry = encoded_inputs["telemetry_features"]
        detections = encoded_inputs["detection_features"]
        mask = encoded_inputs["detection_mask"]

        batch_size, seq_len, embedding_dim = telemetry.shape

        # Process each timestep separately
        fused_features_list = []

        for t in range(seq_len):
            # Get features for this timestep
            tel_t = telemetry[:, t]  # [batch_size, embedding_dim]
            det_t = detections[:, t]  # [batch_size, max_detections, embedding_dim]
            mask_t = mask[:, t]  # [batch_size, max_detections]

            # Expand queries for batch size
            queries = self.object_queries.expand(
                batch_size, -1, -1
            )  # [batch_size, num_queries, embedding_dim]

            # --- Queries attend to telemetry ---
            # Reshape telemetry for attention
            tel_t = tel_t.unsqueeze(1)  # [batch_size, 1, embedding_dim]

            # Queries attend to telemetry
            queries_tel, _ = self.query_to_tel_attention(
                query=queries, key=tel_t, value=tel_t
            )

            queries_tel = self.norm_queries_tel(queries_tel)

            # --- Queries attend to detections ---
            # Create attention mask (True = ignore)
            attn_mask = ~mask_t

            # Queries attend to detections
            queries_det, _ = self.query_to_det_attention(
                query=queries, key=det_t, value=det_t, key_padding_mask=attn_mask
            )

            queries_det = self.norm_queries_det(queries_det)

            # --- Combine query information ---
            # Concatenate telemetry and detection query features
            combined_queries = torch.cat(
                [queries_tel, queries_det], dim=1
            )  # [batch_size, 2*num_queries, embedding_dim]

            # Expand context vector for batch size
            context = self.context_vector.expand(
                batch_size, -1, -1
            )  # [batch_size, 1, embedding_dim]

            # Final attention to combine information
            final_features, _ = self.final_attention(
                query=context, key=combined_queries, value=combined_queries
            )

            # Remove singleton dimension
            final_features = final_features.squeeze(1)  # [batch_size, embedding_dim]

            # Concatenate with original telemetry
            final_features = torch.cat(
                [tel_t.squeeze(1), final_features], dim=-1
            )  # [batch_size, embedding_dim*2]

            fused_features_list.append(final_features)

        # Stack to get full sequence
        fused_features = torch.stack(
            fused_features_list, dim=1
        )  # [batch_size, seq_len, embedding_dim*2]

        # Final MLP and normalization
        fused_features = self.output_mlp(fused_features)
        fused_features = self.norm_output(fused_features)

        return fused_features
