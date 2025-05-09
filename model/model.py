# ====================================
# Abstrakte Basisklassen
# ====================================

import torch.nn as nn


class InputEncoder(nn.Module):
    """Abstrakte Basisklasse für alle Input-Encoder."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")


class FusionModule(nn.Module):
    """Abstrakte Basisklasse für alle Fusionsmodule."""

    def forward(self, encoded_inputs):
        raise NotImplementedError("Subclasses must implement forward method")


class OutputDecoder(nn.Module):
    """Abstrakte Basisklasse für alle Output-Decoder."""

    def forward(self, fused_features):
        raise NotImplementedError("Subclasses must implement forward method")


class BrakingPredictionBaseModel(nn.Module):
    """
    Modulare Basisarchitektur für Bremsvorhersagemodelle.

    Diese Klasse kombiniert die drei Hauptkomponenten eines Bremsvorhersagemodells:
    1. Input Encoder: Verarbeitet die Eingabedaten (Telemetrie und Objektdetektionen)
    2. Fusion Module: Kombiniert die kodierten Features der verschiedenen Modalitäten
    3. Output Decoder: Wandelt die fusionierten Features in Vorhersagen um

    Args:
        input_encoder (InputEncoder): Modul zur Kodierung der Eingabedaten
        fusion_module (FusionModule): Modul zur Fusion der kodierten Features
        output_decoder (OutputDecoder): Modul zur Generierung der Vorhersagen
    """

    def __init__(self, input_encoder, fusion_module, output_decoder):
        super().__init__()
        self.input_encoder = input_encoder
        self.fusion_module = fusion_module
        self.output_decoder = output_decoder

    def forward(self, telemetry_seq, detection_seq, detection_mask, **kwargs):
        """
        Forward-Pass durch das modularisierte Modell.

        Args:
            telemetry_seq (torch.Tensor): Telemetrie-Sequenzdaten
                Shape: (batch_size, seq_len, telemetry_input_dim)
            detection_seq (torch.Tensor): Detektions-Sequenzdaten (mit Padding)
                Shape: (batch_size, seq_len, max_detections, detection_input_dim_per_box)
            detection_mask (torch.Tensor): Boolesche Maske für gültige Detektionen (True=gültig)
                Shape: (batch_size, seq_len, max_detections)
            **kwargs: Zusätzliche Argumente, die an die Komponenten weitergegeben werden

        Returns:
            dict: Wörterbuch mit Vorhersagen
        """
        encoded_inputs = self.input_encoder(
            telemetry_seq, detection_seq, detection_mask, **kwargs
        )
        fused_features = self.fusion_module(encoded_inputs)
        predictions = self.output_decoder(fused_features)
        return predictions
