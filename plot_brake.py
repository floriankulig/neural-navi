import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import numpy as np
import seaborn as sns

NUM_GRADIENTS = 5


def calculate_gradient(y, x=None, deg=1, mode="exp"):
    """Calculate the gradient of a function using regressions."""
    # assert len(y) > 1, "At least two values are required to calculate the gradient."
    if len(y) < NUM_GRADIENTS - 1:
        return 0
    speeds = np.array(y)
    if not x:
        times = np.linspace(0, len(speeds), len(speeds))
    else:
        times = np.array(x)

    t0 = times[0]
    normalized_times = [t - t0 for t in times]
    fit_degree = 2 if len(speeds) > 3 else 1
    fit_degree = deg
    if mode == "exp":
        weights = np.exp(np.arange(len(speeds)))
    elif mode == "lin":
        weights = np.linspace(1, len(speeds), len(speeds))
    elif mode == "log":
        weights = np.linspace(1, 2, len(speeds))
    else:
        weights = None

    coeffs = np.polyfit(normalized_times, speeds, deg=fit_degree, w=weights)
    base_slope = coeffs[fit_degree - 1]
    change_rate = coeffs[0]
    latest_speed_change_time = normalized_times[-1]
    # Linear regression: f(x) = mx + b
    if fit_degree == 1:
        # Derivative: f'(x) = m
        gradient = base_slope
    else:  # fit_degree == 2
        # Polynom: f(x) = ax^2 + bx + c
        # Derivative: f'(x) = 2ax + b
        gradient = 2 * change_rate * latest_speed_change_time + base_slope
    return gradient


# from drive_recorder import TIME_FORMAT_LOG
TIME_FORMAT_LOG = "%Y-%m-%d %H-%M-%S-%f"


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

# Anzeigen der verfügbaren Dateien
print("Verfügbare Recording-Ordner:")
for i, dirname in enumerate(recordings, 1):
    print(f"{i}. {dirname}")

# Benutzereingabe für die Dateiauswahl
while True:
    try:
        choice = int(
            input("Bitte wählen Sie die Nummer der Datei, die Sie plotten möchten: ")
        )
        if 1 <= choice <= len(recordings):
            selected_recording = recordings[choice - 1]
            break
        else:
            print("Ungültige Auswahl. Bitte wählen Sie eine gültige Nummer.")
    except ValueError:
        print("Bitte geben Sie eine Zahl ein.")

# Vollständiger Pfad zur ausgewählten Datei
file_path = os.path.join(RECORDING_DIRECTORY, selected_recording, "telemetry.csv")

# CSV-Datei einlesen
df = pd.read_csv(
    file_path,
    parse_dates=["Time"],
    date_format=TIME_FORMAT_LOG,
)
df = df[3000:6000]  # Nur die ersten 250 Zeilen anzeigen
# df = df[100:350]  # Nur die ersten 250 Zeilen anzeigen

# Plot-Setup
fig, ax1 = plt.subplots(figsize=(len(df) / 12, 10))  # Ein breiterer Plot
sns.set_theme(style="whitegrid")

# Horizontale Linien zu den Werten hinzufügen
for value in range(-8, 8):
    ax1.axhline(y=value, color="lightgray", linestyle="--", linewidth=0.5)

ax1.axhline(y=0, color="black", linewidth=1)

# Calculate gear ratios
speed_gradients = [
    calculate_gradient(df["SPEED"].iloc[max(0, i - NUM_GRADIENTS) : i + 1], mode="lin")
    for i in range(NUM_GRADIENTS, len(df))
]
ax1.plot(
    df["Time"][NUM_GRADIENTS:], speed_gradients, label="Speed Gradient", color="green"
)
ax1.set_xlabel("Time")
ax1.set_ylabel("Speed Gradient")
ax1.legend(loc="upper left")
ax1.set_ylim(-8, 8)
ax1.axhline(y=min(speed_gradients), color="green", linewidth=1)
print(f"Minimale Geschwindigkeitsänderung: {min(speed_gradients)}")

# # Calculate gear ratios
# speed_gradients2 = [
#     calculate_gradient(df["SPEED"].iloc[max(0, i - NUM_GRADIENTS) : i + 1], mode="exp")
#     for i in range(NUM_GRADIENTS, len(df))
# ]
# ax1.plot(
#     df["Time"][NUM_GRADIENTS:], speed_gradients2, label="Speed Gradient", color="orange"
# )
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Speed Gradient")
# ax1.legend(loc="upper left")
# ax1.set_ylim(-8, 8)
# ax1.axhline(y=min(speed_gradients), color="orange", linewidth=1)
# print(f"Minimale Geschwindigkeitsänderung exp: {min(speed_gradients)}")

# Calculate gear ratios
speed_gradients3 = [
    calculate_gradient(df["SPEED"].iloc[max(0, i - NUM_GRADIENTS) : i + 1], mode="log")
    for i in range(NUM_GRADIENTS, len(df))
]
ax1.plot(
    df["Time"][NUM_GRADIENTS:], speed_gradients3, label="Speed Gradient", color="red"
)
ax1.set_xlabel("Time")
ax1.set_ylabel("Speed Gradient")
ax1.legend(loc="upper left")
ax1.set_ylim(-8, 8)
ax1.axhline(y=min(speed_gradients), color="red", linewidth=1)
print(f"Minimale Geschwindigkeitsänderung log: {min(speed_gradients)}")

# Calculate gear ratios
speed_gradients4 = [
    calculate_gradient(
        df["SPEED"].iloc[max(0, i - NUM_GRADIENTS) : i + 1], mode="testes"
    )
    for i in range(NUM_GRADIENTS, len(df))
]
ax1.plot(
    df["Time"][NUM_GRADIENTS:], speed_gradients4, label="Speed Gradient", color="pink"
)
ax1.set_xlabel("Time")
ax1.set_ylabel("Speed Gradient")
ax1.legend(loc="upper left")
ax1.set_ylim(-8, 8)
ax1.axhline(y=min(speed_gradients), color="pink", linewidth=1)
print(f"Minimale Geschwindigkeitsänderung log: {min(speed_gradients)}")

# # Calculate gear ratios
# speed_gradients2 = [
#     np.mean(np.gradient(df["SPEED"].iloc[max(0, i - NUM_GRADIENTS) : i + 1])[-2:-1])
#     for i in range(NUM_GRADIENTS, len(df))
# ]
# ax1.plot(
#     df["Time"][NUM_GRADIENTS:],
#     speed_gradients2,
#     label="Speed Gradient NP",
#     color="orange",
# )
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Speed Gradient")
# ax1.legend(loc="upper left")
# ax1.set_ylim(-8, 8)
# ax1.axhline(y=min(speed_gradients2), color="orange", linewidth=1)
# print(f"Minimale Geschwindigkeitsänderung mit np: {min(speed_gradients)}")


# Plot für die zweite Y-Achse (SPEED)
ax2 = ax1.twinx()  # zweite Y-Achse teilen
ax2.plot(df["Time"], df["SPEED"] / 2, label="SPEED", color="blue")
ax2.set_ylabel("SPEED")
ax2.legend(loc="upper right")
ax2.grid(False)
ax2.set_ylim(0, 100)


# Bremsphasen als transparente Balken markieren
ax1.fill_between(
    df["Time"],
    0,
    1,
    where=df["BRAKE_SIGNAL"],
    transform=ax1.get_xaxis_transform(),
    color="lightblue",
    alpha=0.2,
)


# X-Achse formatieren
plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
plt.gcf().autofmt_xdate()  # Rotiert und richtet die Tick-Labels aus

# Titel und Layout
plt.suptitle(f"Vehicle Data Analysis - Gear Ratios - {selected_recording}", fontsize=16)
plt.tight_layout()

# Speichern des Plots
output_filename = f"plots/brakeplot-{os.path.splitext(selected_recording)[0]}.png"
plt.savefig(output_filename)
# plt.show()
plt.close()

print(f"Der Plot wurde als '{output_filename}' gespeichert.")
