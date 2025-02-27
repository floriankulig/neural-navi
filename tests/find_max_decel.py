import os
import pandas as pd
import numpy as np

# NUM_GRADIENTS = 4
# NUM_GRADIENTS2 = 10


# def calculate_gradient(y, x=None, deg=1):
#     """Calculate the gradient of a function using regressions."""
#     # assert len(y) > 1, "At least two values are required to calculate the gradient."
#     if len(y) < NUM_GRADIENTS - 1:
#         return 0
#     speeds = np.array(y)
#     if not x:
#         times = np.linspace(0, len(speeds), len(speeds))
#     else:
#         times = np.array(x)

#     t0 = times[0]
#     normalized_times = [t - t0 for t in times]
#     fit_degree = deg
#     weights = np.exp(np.arange(len(speeds)))
#     weights = np.linspace(1, 2, len(speeds))

#     coeffs = np.polyfit(normalized_times, speeds, deg=fit_degree, w=weights)
#     base_slope = coeffs[fit_degree - 1]
#     change_rate = coeffs[0]
#     latest_speed_change_time = normalized_times[-1]
#     # Linear regression: f(x) = mx + b
#     if fit_degree == 1:
#         # Derivative: f'(x) = m
#         gradient = base_slope
#     else:  # fit_degree == 2
#         # Polynom: f(x) = ax^2 + bx + c
#         # Derivative: f'(x) = 2ax + b
#         gradient = 2 * change_rate * latest_speed_change_time + base_slope
#     return gradient


# Funktion zum Auflisten der CSV-Dateien im logs-Ordner
def list_recordings(directory):
    return [f for f in os.listdir(directory) if not f.endswith(".DS_Store")]


# Verzeichnis, in dem sich die Logs befinden
RECORDING_DIRECTORY = "recordings"

# Auflisten der verfügbaren Recording-Ordner
recordings = list_recordings(RECORDING_DIRECTORY)

if not recordings:
    print("Keine Recording-Ordner im 'recordings' Verzeichnis gefunden.")
    exit()

# TIME_FORMAT_LOG = "%Y-%m-%d %H-%M-%S-%f"

# for recording in recordings:
#     file_path = os.path.join(RECORDING_DIRECTORY, recording, "telemetry.csv")
#     if not os.path.exists(file_path):
#         print(f"Die Datei {file_path} existiert nicht.")
#         continue

#     # CSV-Datei einlesen
#     df = pd.read_csv(
#         file_path,
#         parse_dates=["Time"],
#         date_format=TIME_FORMAT_LOG,
#     )
#     # Calculate gear ratios
#     speed_gradients_brake = [
#         calculate_gradient(df["SPEED"].iloc[max(0, i - NUM_GRADIENTS) : i + 1])
#         for i in range(NUM_GRADIENTS, len(df))
#     ]
#     print(f"Min speed braking gradient for {recording}: {min(speed_gradients_brake)}")
#     print("-----------------------------------")


def clean_data():
    file_path = os.path.join(
        RECORDING_DIRECTORY, "2024-12-08_19-19-26", "telemetry.csv"
    )
    df = pd.read_csv(file_path)
    df = df[
        [
            "Time",
            "SPEED",
            "ACCELERATOR_POS_D",
            "BRAKE_SIGNAL",
            "PRE_BRAKING",
            "WHILE_BRAKING",
        ]
    ]
    df.to_csv("cleaned_data.csv", index=False)

    time_diffs = []
    for i in range(1, len(df)):
        # braking_speeds = df["WHILE_BRAKING"].iloc[i]
        # if len(braking_speeds) < 2:
        #     continue
        # [float(speed.strip()) for speed in braking_speeds[1:-1].split(",")]
        timestamps = df["WHILE_BRAKING"].iloc[i]
        if len(timestamps) < 3:
            continue
        timestamps = [
            float(t.strip().split(",")[0].strip("()"))
            for t in timestamps[1:-1].split("), (")
        ]
        if len(timestamps) > 1:
            time_diffs = [
                timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]
            avg_time_diff = sum(time_diffs) / len(time_diffs)
            time_diffs.append(avg_time_diff)

    print(np.mean(time_diffs))


clean_data()
