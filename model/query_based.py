# In model/encoder.py oder einer neuen Datei model/query_encoder.py


from typing import Dict, Optional
import torch
import torch.nn as nn
import math

from model.model import FusionModule, InputEncoder


class ObjectQueryEncoder(InputEncoder):
    """
    Advanced encoder using object queries to extract relevant scene information.

    Instead of simple aggregation, this encoder uses a set of learnable query vectors
    that extract relevant information from the detected objects through cross-attention.
    Each query can specialize in different aspects of the driving scene.

    Args:
        telemetry_input_dim (int): Dimension of telemetry data
        detection_input_dim_per_box (int): Dimension of properties for each object detection
        embedding_dim (int): Dimension of the generated embeddings
        num_queries (int): Number of object queries to use
        num_heads (int): Number of attention heads
        dropout_prob (float): Dropout probability
        max_seq_length (int): Maximum sequence length for positional encoding
    """

    def __init__(
        self,
        telemetry_input_dim: int,
        detection_input_dim_per_box: int,
        embedding_dim: int = 64,
        num_queries: int = 8,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
        max_seq_length: int = 20,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_queries = num_queries

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

        # Learnable object queries
        # These will be specialized to look for different aspects of the scene
        self.object_queries = nn.Parameter(torch.randn(1, num_queries, embedding_dim))

        # Cross-attention from queries to objects
        self.query_to_object_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # Self-attention for telemetry
        self.telemetry_self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # LayerNorms
        self.norm_telemetry = nn.LayerNorm(embedding_dim)
        self.norm_queries = nn.LayerNorm(embedding_dim)
        self.norm_telemetry_attended = nn.LayerNorm(embedding_dim)

        # Positional encoding for sequence data
        self.register_buffer(
            "temporal_position_encoding",
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
        Forward pass through the ObjectQueryEncoder.

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
                "query_features": (batch_size, seq_len, num_queries, embedding_dim)
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

        # --- Object query processing ---
        # Expand the object queries for each batch and timestep
        queries = self.object_queries.expand(
            batch_size, seq_len, self.num_queries, self.embedding_dim
        )

        # Process each time step separately
        attended_queries_list = []
        query_attn_weights_list = []

        for t in range(seq_len):
            # Get detections and mask for this time step
            detections_t = embedded_detections[
                :, t
            ]  # [batch_size, max_dets, embedding_dim]
            mask_t = detection_mask[:, t]  # [batch_size, max_dets]
            queries_t = queries[:, t]  # [batch_size, num_queries, embedding_dim]

            # Create attention mask (True = ignore)
            attn_mask = ~mask_t

            # We need to reshape for the attention module
            # [batch_size, num_queries, embedding_dim] -> [num_queries, batch_size, embedding_dim]
            queries_t_trans = queries_t.transpose(0, 1)
            # [batch_size, max_dets, embedding_dim] -> [max_dets, batch_size, embedding_dim]
            detections_t_trans = detections_t.transpose(0, 1)

            # Apply cross-attention: queries attend to objects
            queries_attended_t, attn_weights_t = self.query_to_object_attention(
                query=queries_t_trans,
                key=detections_t_trans,
                value=detections_t_trans,
                key_padding_mask=attn_mask,
            )

            # Reshape back to original shape
            # [num_queries, batch_size, embedding_dim] -> [batch_size, num_queries, embedding_dim]
            queries_attended_t = queries_attended_t.transpose(0, 1)

            # Add to lists
            attended_queries_list.append(queries_attended_t)
            if store_attentions:
                query_attn_weights_list.append(attn_weights_t)

        # Stack to get the full sequence
        attended_queries = torch.stack(
            attended_queries_list, dim=1
        )  # [batch_size, seq_len, num_queries, embedding_dim]

        # Apply LayerNorm to each query
        attended_queries = torch.stack(
            [
                self.norm_queries(attended_queries[:, :, i])
                for i in range(self.num_queries)
            ],
            dim=2,
        )

        if store_attentions:
            attention_weights["query_to_object_attention"] = query_attn_weights_list

        result = {
            "telemetry_features": telemetry_attended,
            "query_features": attended_queries,
        }

        if store_attentions:
            result["attention_weights"] = attention_weights

        return result


# In model/fusion.py


class QueryBasedFusion(FusionModule):
    """
    Fusion module for combining telemetry with object query features.

    This module performs cross-attention between telemetry and object queries,
    allowing the model to focus on the most relevant scene elements.

    Args:
        embedding_dim (int): Dimensionality of the input feature vectors
        num_queries (int): Number of object queries to process
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
        self.output_dim = output_dim or embedding_dim * 2
        self.num_queries = num_queries

        # Cross-attention from telemetry to queries
        self.tel_to_query_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

        # LayerNorm for attention outputs
        self.norm_tel_attended = nn.LayerNorm(embedding_dim)

        # MLP for fusion of attention results
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.norm_fusion = nn.LayerNorm(self.output_dim)

    def forward(self, encoded_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the QueryBasedFusion.

        Args:
            encoded_inputs (dict): Dictionary with encoded features
                "telemetry_features": (batch_size, seq_len, embedding_dim)
                "query_features": (batch_size, seq_len, num_queries, embedding_dim)

        Returns:
            torch.Tensor: Fused features
                Shape: (batch_size, seq_len, output_dim)
        """
        telemetry = encoded_inputs[
            "telemetry_features"
        ]  # [batch_size, seq_len, embedding_dim]
        queries = encoded_inputs[
            "query_features"
        ]  # [batch_size, seq_len, num_queries, embedding_dim]

        batch_size, seq_len, embedding_dim = telemetry.shape

        # Process each time step separately
        attended_tel_list = []

        for t in range(seq_len):
            # Get features for this time step
            tel_t = telemetry[:, t]  # [batch_size, embedding_dim]
            queries_t = queries[:, t]  # [batch_size, num_queries, embedding_dim]

            # Reshape telemetry for attention
            # [batch_size, embedding_dim] -> [batch_size, 1, embedding_dim]
            tel_t = tel_t.unsqueeze(1)

            # Transpose for attention
            # [batch_size, 1, embedding_dim] -> [1, batch_size, embedding_dim]
            tel_t_trans = tel_t.transpose(0, 1)
            # [batch_size, num_queries, embedding_dim] -> [num_queries, batch_size, embedding_dim]
            queries_t_trans = queries_t.transpose(0, 1)

            # Telemetry attends to queries
            tel_attended_t, _ = self.tel_to_query_attention(
                query=tel_t_trans,
                key=queries_t_trans,
                value=queries_t_trans,
            )

            # Reshape back
            # [1, batch_size, embedding_dim] -> [batch_size, 1, embedding_dim]
            tel_attended_t = tel_attended_t.transpose(0, 1)
            # [batch_size, 1, embedding_dim] -> [batch_size, embedding_dim]
            tel_attended_t = tel_attended_t.squeeze(1)

            # Add to list
            attended_tel_list.append(tel_attended_t)

        # Stack for full sequence
        attended_tel = torch.stack(
            attended_tel_list, dim=1
        )  # [batch_size, seq_len, embedding_dim]

        # Residual connection and normalization
        attended_tel = telemetry + attended_tel
        attended_tel = self.norm_tel_attended(attended_tel)

        # Concatenate original telemetry with attended features
        fused_features = torch.cat([telemetry, attended_tel], dim=-1)

        # Final MLP and normalization
        fused_features = self.fusion_mlp(fused_features)
        fused_features = self.norm_fusion(fused_features)

        return fused_features
