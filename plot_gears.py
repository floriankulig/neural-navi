import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import seaborn as sns
from features.gear import get_gear

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
df = df[0:250]  # Nur die ersten 250 Zeilen anzeigen
# df = df[100:350]  # Nur die ersten 250 Zeilen anzeigen

# Plot-Setup
fig, ax1 = plt.subplots(figsize=(len(df) / 12, 10))  # Ein breiterer Plot
sns.set_theme(style="whitegrid")

# # Horizontale Linien zu den Werten hinzufügen
for value in range(0, 100, 20):
    ax1.axhline(y=value, color="gray", linestyle="--", linewidth=0.5)
# for value in range(5, 105, 10):
#     ax1.axhline(y=value, color="lightgray", linestyle="--", linewidth=0.5)

pot_ratios = [9.5, 16.5, 27.5, 40.5, 53.5]
# pot_ratios = [52.25, 53.5, 54.75]
for ratio in pot_ratios:
    ax1.axhline(y=ratio, color="red", linestyle="--", linewidth=1)

# Calculate gear ratios
gear_ratios = (df["SPEED"] / df["RPM"]) * 1000
ax1.plot(df["Time"], gear_ratios, label="Gear Ratio", color="green")
ax1.set_xlabel("Time")
ax1.set_ylabel("Gear Ratios")
ax1.legend(loc="upper left")
ax1.set_ylim(0, 100)

# df["GEAR"] = df.apply(
#     lambda row: get_gear(
#         row["SPEED"], row["RPM"], row["ACCELERATOR_POS_D"], row["ENGINE_LOAD"]
#     ),
#     axis=1,
# )
# ax1.plot(df["Time"], df["GEAR"] * 10, label="Gear (calculated)", color="orange")

# Plot für die zweite Y-Achse (SPEED)
ax2 = ax1.twinx()  # zweite Y-Achse teilen
ax2.plot(df["Time"], df["SPEED"], label="SPEED", color="blue")
ax2.set_ylabel("SPEED")
ax2.legend(loc="upper right")
# ax2.grid(False)
ax2.set_ylim(0, 100)

# X-Achse formatieren
plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
plt.gcf().autofmt_xdate()  # Rotiert und richtet die Tick-Labels aus

# Titel und Layout
plt.suptitle(f"Vehicle Data Analysis - Gear Ratios - {selected_recording}", fontsize=16)
plt.tight_layout()

# Speichern des Plots
output_filename = f"plots/gearplot-{os.path.splitext(selected_recording)[0]}.png"
plt.savefig(output_filename)
# plt.show()
plt.close()

print(f"Der Plot wurde als '{output_filename}' gespeichert.")
