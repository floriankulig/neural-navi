# ====================================
# Abstract Base Classes
# ====================================

import torch.nn as nn


class InputEncoder(nn.Module):
    """Abstract base class for all input encoders."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")


class FusionModule(nn.Module):
    """Abstract base class for all fusion modules."""

    def forward(self, encoded_inputs):
        raise NotImplementedError("Subclasses must implement forward method")


class OutputDecoder(nn.Module):
    """Abstract base class for all output decoders."""

    def forward(self, fused_features):
        raise NotImplementedError("Subclasses must implement forward method")


class BrakingPredictionBaseModel(nn.Module):
    """
    Modular base architecture for braking prediction models.

    This class combines the three main components of a braking prediction model:
    1. Input Encoder: Processes the input data (telemetry and object detections)
    2. Fusion Module: Combines the encoded features from different modalities
    3. Output Decoder: Transforms the fused features into predictions

    Args:
      input_encoder (InputEncoder): Module for encoding input data
      fusion_module (FusionModule): Module for fusing encoded features
      output_decoder (OutputDecoder): Module for generating predictions
    """

    def __init__(self, input_encoder, fusion_module, output_decoder):
        super().__init__()
        self.input_encoder = input_encoder
        self.fusion_module = fusion_module
        self.output_decoder = output_decoder

    def forward(self, telemetry_seq, detection_seq, detection_mask, **kwargs):
        """
        Forward pass through the modularized model.

        Args:
          telemetry_seq (torch.Tensor): Telemetry sequence data
            Shape: (batch_size, seq_len, telemetry_input_dim)
          detection_seq (torch.Tensor): Detection sequence data (with padding)
            Shape: (batch_size, seq_len, max_detections, detection_input_dim_per_box)
          detection_mask (torch.Tensor): Boolean mask for valid detections (True=valid)
            Shape: (batch_size, seq_len, max_detections)
          **kwargs: Additional arguments passed to the components

        Returns:
          dict: Dictionary containing predictions
        """
        encoded_inputs = self.input_encoder(
            telemetry_seq, detection_seq, detection_mask, **kwargs
        )
        fused_features = self.fusion_module(encoded_inputs)
        predictions = self.output_decoder(fused_features)
        return predictions
