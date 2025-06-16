import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# Configuration
RECORDINGS_DIR = Path("data/recordings")
SELECTED_RECORDINGS = [
    "2024-11-08_16-00-48",
    "2024-11-10_15-27-44",
]


def calculate_speed_gradient(
    speed_values: pd.Series, time_interval: float = 0.5
) -> pd.Series:
    """Calculate speed gradient (acceleration/deceleration) from speed values."""
    speed_diff = speed_values.diff()
    # Convert km/h to m/s and divide by time interval
    acceleration = (speed_diff / 3.6) / time_interval  # m/s²
    return acceleration


def analyze_recording(recording_path: Path) -> Dict[str, float]:
    """Analyze single recording and return correlation metrics."""
    telemetry_file = recording_path / "telemetry.csv"

    if not telemetry_file.exists():
        return {"error": f"No telemetry.csv found in {recording_path.name}"}

    # Load data
    df = pd.read_csv(telemetry_file)

    # Clean data - remove invalid entries
    df = df.dropna(subset=["SPEED", "BRAKE_SIGNAL"])
    df = df[df["SPEED"] >= 0]  # Remove negative speeds

    if len(df) < 10:
        return {"error": f"Insufficient data points in {recording_path.name}"}

    # Calculate speed gradient
    df["SPEED_GRADIENT"] = calculate_speed_gradient(df["SPEED"])

    # Convert BRAKE_SIGNAL to numeric (True/False -> 1/0)
    df["BRAKE_SIGNAL_NUMERIC"] = df["BRAKE_SIGNAL"].astype(int)

    # Remove NaN values from gradient calculation
    df = df.dropna(subset=["SPEED_GRADIENT"])

    if len(df) < 5:
        return {"error": f"Insufficient valid gradients in {recording_path.name}"}

    # Calculate correlations
    correlation_gradient = np.corrcoef(
        df["BRAKE_SIGNAL_NUMERIC"], df["SPEED_GRADIENT"]
    )[0, 1]

    # Additional metrics
    brake_events = df["BRAKE_SIGNAL_NUMERIC"].sum()
    total_samples = len(df)
    mean_deceleration_during_braking = df[df["BRAKE_SIGNAL_NUMERIC"] == 1][
        "SPEED_GRADIENT"
    ].mean()

    return {
        "correlation_brake_gradient": correlation_gradient,
        "brake_events": int(brake_events),
        "total_samples": int(total_samples),
        "brake_ratio": brake_events / total_samples,
        "mean_decel_while_braking": mean_deceleration_during_braking,
    }


def main():
    """Main analysis function."""
    print("🔍 Korrelationsanalyse: Bremssignal vs. Geschwindigkeitsgradient")
    print("=" * 60)

    all_correlations = []
    all_brake_events = 0
    all_samples = 0

    for recording_name in SELECTED_RECORDINGS:
        recording_path = RECORDINGS_DIR / recording_name

        if not recording_path.exists():
            print(f"❌ {recording_name}: Ordner nicht gefunden")
            continue

        print(f"\n📁 {recording_name}:")

        results = analyze_recording(recording_path)

        if "error" in results:
            print(f"   ❌ {results['error']}")
            continue

        # Print individual results
        correlation = results["correlation_brake_gradient"]
        print(
            f"   📊 Korrelation Bremssignal ↔ Geschwindigkeitsgradient: {correlation:.4f}"
        )
        print(
            f"   🛑 Bremsereignisse: {results['brake_events']} / {results['total_samples']} ({results['brake_ratio']:.1%})"
        )

        if not np.isnan(results["mean_decel_while_braking"]):
            print(
                f"   📉 Ø Verzögerung beim Bremsen: {results['mean_decel_while_braking']:.2f} m/s²"
            )

        # Collect for overall statistics
        if not np.isnan(correlation):
            all_correlations.append(correlation)
            all_brake_events += results["brake_events"]
            all_samples += results["total_samples"]

    # Overall statistics
    if all_correlations:
        print(f"\n📈 Gesamtstatistik:")
        print(f"   📊 Mittlere Korrelation: {np.mean(all_correlations):.4f}")
        print(f"   📊 Std.-Abweichung: {np.std(all_correlations):.4f}")
        print(
            f"   📊 Bereich: [{np.min(all_correlations):.4f}, {np.max(all_correlations):.4f}]"
        )
        print(
            f"   🛑 Gesamt Bremsereignisse: {all_brake_events} / {all_samples} ({all_brake_events/all_samples:.1%})"
        )
        print(f"   📁 Analysierte Aufnahmen: {len(all_correlations)}")
    else:
        print(f"\n❌ Keine gültigen Korrelationen berechnet")


if __name__ == "__main__":
    main()
