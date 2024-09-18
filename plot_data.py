import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
import seaborn as sns


# Funktion zum Auflisten der CSV-Dateien im logs-Ordner
def list_csv_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".csv")]


# Verzeichnis, in dem sich die Logs befinden
LOG_DIRECTORY = "logs"

# Auflisten der verfügbaren CSV-Dateien
csv_files = list_csv_files(LOG_DIRECTORY)

if not csv_files:
    print("Keine CSV-Dateien im 'logs' Verzeichnis gefunden.")
    exit()

# Anzeigen der verfügbaren Dateien
print("Verfügbare CSV-Dateien:")
for i, file in enumerate(csv_files, 1):
    print(f"{i}. {file}")

# Benutzereingabe für die Dateiauswahl
while True:
    try:
        choice = int(
            input("Bitte wählen Sie die Nummer der Datei, die Sie plotten möchten: ")
        )
        if 1 <= choice <= len(csv_files):
            selected_file = csv_files[choice - 1]
            break
        else:
            print("Ungültige Auswahl. Bitte wählen Sie eine gültige Nummer.")
    except ValueError:
        print("Bitte geben Sie eine Zahl ein.")

# Vollständiger Pfad zur ausgewählten Datei
file_path = os.path.join(LOG_DIRECTORY, selected_file)

# CSV-Datei einlesen
df = pd.read_csv(file_path, parse_dates=["Time"])
# df = df[:250]  # Nur die ersten 250 Zeilen anzeigen

# Plot-Setup
fig, ax1 = plt.subplots(figsize=(len(df) / 10, 10))  # Ein breiterer Plot
sns.set_theme(style="whitegrid")

# Plot für die erste Y-Achse (Vehicle Speed, Accelerator Position, Engine Load)
ax1.plot(df["Time"], df["Vehicle Speed"], label="Vehicle Speed (km/h)", color="blue")
ax1.plot(
    df["Time"],
    df["Accelerator Position"],
    label="Accelerator Position (%)",
    color="green",
)
# Calculate gear ratios
gear_ratios = (df["Vehicle Speed"] / df["RPM"]) * 1000
ax1.plot(df["Time"], gear_ratios, label="Gear Ratio", color="brown")
ax1.plot(df["Time"], df["Engine Load"], label="Engine Load (%)", color="orange")

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
    where=df["Brake Signal"],
    transform=ax1.get_xaxis_transform(),
    color="lightblue",
    alpha=0.2,
)


# X-Achse formatieren
plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
plt.gcf().autofmt_xdate()  # Rotiert und richtet die Tick-Labels aus

# Titel und Layout
plt.suptitle(f"Vehicle Data Analysis - {selected_file}", fontsize=16)
plt.tight_layout()

# Speichern des Plots
output_filename = f"plots/{os.path.splitext(selected_file)[0]}.png"
plt.savefig(output_filename)
plt.show()
plt.close()

print(f"Der Plot wurde als '{output_filename}' gespeichert.")
