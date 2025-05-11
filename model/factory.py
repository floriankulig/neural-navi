# ====================================
# Model Factory Helper
# ====================================


from typing import Any, Dict
from model.decoder import LSTMOutputDecoder, TransformerOutputDecoder
from model.encoder import AttentionInputEncoder, SimpleInputEncoder
from model.fusion import CrossModalAttentionFusion, SimpleConcatenationFusion
from model.model import BrakingPredictionBaseModel


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
            max_detections=config.get("max_detections", 10),
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
    elif fusion_type == "cross_attention":
        fusion = CrossModalAttentionFusion(
            embedding_dim=config.get("embedding_dim", 64),
            num_heads=config.get("attention_num_heads", 4),
            output_dim=config.get("fusion_output_dim", None),
            dropout_prob=config.get("dropout_prob", 0.1),
            bidirectional=config.get("bidirectional_fusion", False),
        )
    else:
        raise ValueError(f"Unbekannter fusion_type: {fusion_type}")

    # --- Output-Decoder auswählen ---
    decoder_type = config.get("decoder_type", "lstm")

    fusion_output_dim = config.get(
        "fusion_output_dim", config.get("embedding_dim", 64) * 2
    )

    if decoder_type == "lstm":
        decoder = LSTMOutputDecoder(
            input_dim=fusion_output_dim,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("decoder_num_layers", 2),
            dropout_prob=config.get("dropout_prob", 0.3),
            prediction_horizons=config.get("prediction_horizons", [1, 3, 5]),
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
            prediction_horizons=config.get("prediction_horizons", [1, 3, 5]),
            include_brake_force=config.get("include_brake_force", False),
            include_uncertainty=config.get("include_uncertainty", False),
        )
    else:
        raise ValueError(f"Unbekannter decoder_type: {decoder_type}")

    # --- Gesamtmodell erstellen ---
    model = BrakingPredictionBaseModel(encoder, fusion, decoder)
    return model
