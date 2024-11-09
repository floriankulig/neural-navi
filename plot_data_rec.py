import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import seaborn as sns
from drive_recorder import TIME_FORMAT_LOG


# Funktion zum Auflisten der CSV-Dateien im logs-Ordner
def list_recordings(directory):
    return [f for f in os.listdir(directory)]


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
# df = df[:2000]  # Nur die ersten 250 Zeilen anzeigen

# Plot-Setup
fig, ax1 = plt.subplots(figsize=(len(df) / 10, 10))  # Ein breiterer Plot
sns.set_theme(style="whitegrid")

# Plot für die erste Y-Achse (Vehicle Speed, Accelerator Position, Engine Load)
ax1.plot(df["Time"], df["SPEED"], label="Vehicle Speed (km/h)", color="blue")
# Horizontale Linien zu den Werten hinzufügen
for value in [0, 20, 40, 60, 80, 100]:
    ax1.axhline(y=value, color="gray", linestyle="--", linewidth=0.5)
ax1.plot(
    df["Time"],
    df["ACCELERATOR_POS_D"],
    label="Accelerator Position (%)",
    color="green",
)
# Calculate gear ratios
# gear_ratios = (df["Vehicle Speed"] / df["RPM"]) * 1000
# ax1.plot(df["Time"], gear_ratios, label="Gear Ratio", color="brown")
# ax1.plot(df["Time"], df["Engine Load"], label="Engine Load (%)", color="orange")

ax1.set_xlabel("Time")
ax1.set_ylabel("Values (Speed, Accelerator, Engine Load)")
ax1.legend(loc="upper left")

# Plot für die zweite Y-Achse (RPM)
ax2 = ax1.twinx()  # zweite Y-Achse teilen
ax2.plot(df["Time"], df["RPM"], label="RPM", color="red")
ax2.set_ylabel("RPM")
ax2.legend(loc="upper right")

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
plt.suptitle(f"Vehicle Data Analysis - {selected_recording}", fontsize=16)
plt.tight_layout()

# Speichern des Plots
output_filename = f"plots/{os.path.splitext(selected_recording)[0]}.png"
plt.savefig(output_filename)
# plt.show()
plt.close()

print(f"Der Plot wurde als '{output_filename}' gespeichert.")
