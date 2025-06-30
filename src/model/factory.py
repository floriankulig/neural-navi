# ====================================
# Model Factory Helper
# ====================================


from typing import Any, Dict
import torch
import torch.nn as nn
from pathlib import Path
import sys


# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent  # Go up two levels: model/ -> src/ -> root/
sys.path.insert(0, str(project_root))

from src.utils.feature_config import (
    MAX_DETECTIONS_PER_FRAME,
    PREDICTION_TASKS,
    SEQUENCE_LENGTH,
    get_detection_input_dim_per_box,
    get_telemetry_input_dim,
)
from src.model.encoder import AttentionInputEncoder, SimpleInputEncoder
from src.model.fusion import (
    CrossModalAttentionFusion,
    ObjectQueryFusion,
    SimpleConcatenationFusion,
)
from src.model.decoder import LSTMOutputDecoder, TransformerOutputDecoder
from src.model.base import BrakingPredictionBaseModel
import time


def init_attention_weights(module):
    """Initialize attention modules for better stability."""
    if isinstance(module, nn.MultiheadAttention):
        # Xavier initialization with smaller variance
        nn.init.xavier_uniform_(module.in_proj_weight, gain=0.5)
        if module.in_proj_bias is not None:
            nn.init.constant_(module.in_proj_bias, 0.0)

        # Output projection
        nn.init.xavier_uniform_(module.out_proj.weight, gain=0.5)
        if module.out_proj.bias is not None:
            nn.init.constant_(module.out_proj.bias, 0.0)

    elif isinstance(module, nn.Linear):
        # Smaller initialization for linear layers in attention modules
        nn.init.xavier_uniform_(module.weight, gain=0.5)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def create_model_variant(config: Dict[str, Any]) -> BrakingPredictionBaseModel:
    """
    Erzeugt eine spezifische Modellvariante basierend auf der Konfiguration.

    Args:
        config (dict): Modellkonfiguration mit folgenden Schlüsseln:
            - "encoder_type": Typ des Input-Encoders ("simple", "attention")
            - "fusion_type": Typ des Fusion-Moduls ("concat", "cross_attention")
            - "decoder_type": Typ des Output-Decoders ("lstm", "transformer")
            - Weitere parameter-spezifische Schlüssel

    Returns:
        BrakingPredictionBaseModel: Konfiguriertes Modell
    """
    # --- Input-Encoder auswählen ---
    encoder_type = config.get("encoder_type", "simple")

    if encoder_type == "simple":
        encoder = SimpleInputEncoder(
            telemetry_input_dim=config["telemetry_input_dim"],
            detection_input_dim_per_box=config["detection_input_dim_per_box"],
            embedding_dim=config.get("embedding_dim", 64),
            dropout_prob=config.get("dropout_prob", 0.1),
            use_positional_encoding=config.get("use_positional_encoding", True),
            max_seq_length=config.get("max_seq_length", 20),
        )
    elif encoder_type == "attention":
        encoder = AttentionInputEncoder(
            telemetry_input_dim=config["telemetry_input_dim"],
            detection_input_dim_per_box=config["detection_input_dim_per_box"],
            embedding_dim=config.get("embedding_dim", 64),
            attention_num_heads=config.get("attention_num_heads", 4),
            dropout_prob=config.get("dropout_prob", 0.1),
            max_detections=config.get("max_detections", 12),
            max_seq_length=config.get("max_seq_length", 20),
        )
    else:
        raise ValueError(f"Unbekannter encoder_type: {encoder_type}")

    # --- Fusion-Modul auswählen ---
    fusion_type = config.get("fusion_type", "concat")

    if fusion_type == "concat":
        fusion = SimpleConcatenationFusion(
            embedding_dim=config.get("embedding_dim", 64),
            output_dim=config.get("fusion_output_dim", None),
            dropout_prob=config.get("dropout_prob", 0.1),
        )
    elif "cross" in fusion_type:
        fusion = CrossModalAttentionFusion(
            embedding_dim=config.get("embedding_dim", 64),
            num_heads=config.get("attention_num_heads", 4),
            output_dim=config.get("fusion_output_dim", None),
            dropout_prob=config.get("dropout_prob", 0.1),
            max_detections=config.get("max_detections", 12),
            use_attention_weights=config.get("use_attention_weights", False),
        )
    elif fusion_type == "query":
        fusion = ObjectQueryFusion(
            embedding_dim=config.get("embedding_dim", 64),
            num_queries=config.get("num_queries", 8),
            num_heads=config.get("attention_num_heads", 4),
            output_dim=config.get("fusion_output_dim", None),
            dropout_prob=config.get("dropout_prob", 0.1),
        )
    else:
        raise ValueError(f"Unbekannter fusion_type: {fusion_type}")

    # --- Output-Decoder auswählen ---
    decoder_type = config.get("decoder_type", "lstm")

    fusion_output_dim = config.get(
        "fusion_output_dim", config.get("embedding_dim", 64) * 2
    )
    # Get prediction tasks from config
    prediction_tasks = config.get("prediction_tasks", ["brake_1s", "brake_2s"])

    if decoder_type == "lstm":
        decoder = LSTMOutputDecoder(
            input_dim=fusion_output_dim,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("decoder_num_layers", 2),
            dropout_prob=config.get("dropout_prob", 0.3),
            prediction_tasks=prediction_tasks,
            include_brake_force=config.get("include_brake_force", False),
            include_uncertainty=config.get("include_uncertainty", False),
        )
    elif decoder_type == "transformer":
        decoder = TransformerOutputDecoder(
            input_dim=fusion_output_dim,
            hidden_dim=config.get("hidden_dim", 128),
            num_heads=config.get("attention_num_heads", 4),
            num_layers=config.get("decoder_num_layers", 2),
            dropout_prob=config.get("dropout_prob", 0.3),
            prediction_tasks=prediction_tasks,
            include_brake_force=config.get("include_brake_force", False),
            include_uncertainty=config.get("include_uncertainty", False),
        )
    else:
        raise ValueError(f"Unbekannter decoder_type: {decoder_type}")

    # --- Gesamtmodell erstellen ---
    model = BrakingPredictionBaseModel(encoder, fusion, decoder)

    # Apply special initialization for attention models
    if any(x in [encoder_type, fusion_type] for x in ["attention", "cross", "query"]):
        model.apply(init_attention_weights)
        print("🎯 Applied attention-specific weight initialization")

    return model


if __name__ == "__main__":
    # Konfigurationsbeispiel
    example_config = {
        # Modalitätsdimensionen
        "telemetry_input_dim": get_telemetry_input_dim(),
        "detection_input_dim_per_box": get_detection_input_dim_per_box(),
        # Architekturtypen
        "encoder_type": "simple",  # "simple" oder "attention"
        # "fusion_type": "cross_attention",  # "concat", "query" oder "cross_attention"
        "fusion_type": "concat",  # "concat", "query" oder  "cross_attention"
        "decoder_type": "transformer",  # "lstm" oder "transformer"
        # Allgemeine Parameter
        "embedding_dim": 64,
        "hidden_dim": 128,
        "attention_num_heads": 4,
        "decoder_num_layers": 4,
        "dropout_prob": 0.15,
        # Spezifische Parameter
        "prediction_tasks": PREDICTION_TASKS,  # Zeithorizonte in Sekunden
        "include_brake_force": False,
        "include_uncertainty": False,
        "use_attention_weights": True,
        "max_detections": MAX_DETECTIONS_PER_FRAME,
        "max_seq_length": SEQUENCE_LENGTH,
    }

    # Modell erstellen
    model = create_model_variant(example_config)

    # Dumme Eingaben für einen Test
    batch_size = 4
    seq_len = 10
    tel_dim = example_config["telemetry_input_dim"]
    det_dim = example_config["detection_input_dim_per_box"]
    max_dets = example_config["max_detections"]

    dummy_telemetry = torch.randn(batch_size, seq_len, tel_dim)
    dummy_detections = torch.randn(batch_size, seq_len, max_dets, det_dim)

    # Maske für gültige Detektionen (einige zufällig)
    dummy_mask = torch.zeros(batch_size, seq_len, max_dets, dtype=torch.bool)
    for b in range(batch_size):
        for s in range(seq_len):
            num_valid = torch.randint(0, max_dets + 1, (1,)).item()
            if num_valid > 0:
                dummy_mask[b, s, :num_valid] = True

    # Forward-Pass mit Zeitmessung

    with torch.no_grad():
        start_time = time.time()
        predictions = model(dummy_telemetry, dummy_detections, dummy_mask)
        end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference Zeit: {inference_time:.4f} Sekunden")

    # Ergebnisse ausgeben
    print("\n--- Modulares Brems-Prädiktionsmodell Test ---")
    print(
        f"Modellkonfiguration: {example_config['encoder_type']}-{example_config['fusion_type']}-{example_config['decoder_type']}"
    )
    print()

    print("Eingabe-Dimensionen:")
    print(f"Telemetrie: {dummy_telemetry.shape}")
    print(f"Detektionen: {dummy_detections.shape}")
    print(f"Maske: {dummy_mask.shape}")
    print()

    print("Vorhersage-Dimensionen:")
    for key, value in predictions.items():
        print(f"{key}: {value.shape}")

    # Parameteranzahl berechnen
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainierbare Parameter: {num_params:,}")

    # Modularitätsübersicht ausgeben
    print("\nModulstruktur:")
    print(f"- Input Encoder: {type(model.input_encoder).__name__}")
    print(f"- Fusion Module: {type(model.fusion_module).__name__}")
    print(f"- Output Decoder: {type(model.output_decoder).__name__}")
