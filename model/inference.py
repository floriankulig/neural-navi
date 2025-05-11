from pathlib import Path
import sys
import torch

# Add parent directory to path for imports from main project
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from model.factory import create_model_variant

if __name__ == "__main__":
    # Konfigurationsbeispiel
    example_config = {
        # Modalitätsdimensionen
        "telemetry_input_dim": 6,  # z.B. Geschwindigkeit, RPM, Gaspedalposition, Motorlast, Gang, etc.
        "detection_input_dim_per_box": 7,  # z.B. Klasse, Vertrauen, x1, y1, x2, y2, Fläche
        # Architekturtypen
        "encoder_type": "simple",  # "simple" oder "attention"
        "fusion_type": "cross_attention",  # "concat" oder "cross_attention"
        "decoder_type": "lstm",  # "lstm" oder "transformer"
        # Allgemeine Parameter
        "embedding_dim": 64,
        "hidden_dim": 128,
        "attention_num_heads": 4,
        "decoder_num_layers": 2,
        "dropout_prob": 0.3,
        # Spezifische Parameter
        "prediction_horizons": [1, 2, 4],  # Zeithorizonte in Sekunden
        "include_brake_force": True,
        "include_uncertainty": True,
        "bidirectional_fusion": False,
        "max_detections": 12,
        "max_seq_length": 20,
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

    # Forward-Pass
    with torch.no_grad():
        predictions = model(dummy_telemetry, dummy_detections, dummy_mask)

    # Ergebnisse ausgeben
    print("--- Modulares Brems-Prädiktionsmodell Test ---")
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
