import obd
import time
import csv
import os
import serial.tools.list_ports
import numpy as np
from custom_commands import BRAKE_SIGNAL
from helpers import normalize, numeric_or_none

INTERVAL = 0.5

COMMANDS_TO_MONITOR = [
    obd.commands.SPEED,
    obd.commands.RPM,
    obd.commands.ACCELERATOR_POS_D,
    obd.commands.ENGINE_LOAD,
    BRAKE_SIGNAL,
]


def find_com_ports():
    all_ports = list(serial.tools.list_ports.comports())
    return [port for port in all_ports if "Serial" in port.description]


def watch_commands(connection):
    for command in COMMANDS_TO_MONITOR:
        connection.watch(command)


def main():
    print("Verfügbare COM-Ports:")
    ports = find_com_ports()

    port_choice = -1
    while not 0 <= port_choice < len(ports):
        for i, port in enumerate(ports):
            print(f"{i+1}: {port.device} - {port.description}")
        port_choice = (
            int(input("Wählen Sie die Nummer des zu verbindenden Ports: ")) - 1
        )
        if 0 <= port_choice < len(ports):
            break
        else:
            print("Ungültige Auswahl.")

    port = ports[port_choice].device
    print(f"Verbinde mit {port}...")
    connection = obd.Async(port)
    if not connection.is_connected():
        print("Verbindung fehlgeschlagen. Bitte überprüfen Sie Ihre ELM327-Verbindung.")
        return

    connection.supported_commands.add(BRAKE_SIGNAL)
    watch_commands(connection)

    print("Verbindung hergestellt. Starte das Logging...")
    try:
        connection.start()
        # Erstellen des logs-Ordners, falls er nicht existiert
        os.makedirs("logs", exist_ok=True)
        file_name = f"logs/log-{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        with open(file_name, "x", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Time",
                    "Vehicle Speed",
                    "RPM",
                    "Accelerator Position",
                    "Engine Load",
                    "Brake Signal",
                ]
            )
            while True:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                brake_signal = connection.query(BRAKE_SIGNAL)
                brake_signal_value = bool(brake_signal.value)
                rpm = connection.query(obd.commands.RPM)
                vehicle_speed = connection.query(obd.commands.SPEED)
                accelerator_pos = normalize(
                    connection.query(obd.commands.ACCELERATOR_POS_D),
                    [14.12, 82],
                    [0, 100],
                )
                engine_load = connection.query(obd.commands.ENGINE_LOAD)
                writer.writerow(
                    [
                        timestamp,
                        numeric_or_none(vehicle_speed),
                        numeric_or_none(rpm),
                        numeric_or_none(accelerator_pos),
                        numeric_or_none(engine_load),
                        brake_signal_value,
                    ]
                )
                print(
                    f"{timestamp} | {numeric_or_none(vehicle_speed)} KM/H | {numeric_or_none(rpm)} RPM | {numeric_or_none(accelerator_pos)} % | {numeric_or_none(engine_load)} % | {brake_signal_value}"
                )
                time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Beende das Logging...")
        connection.stop()
        print("Verbindung getrennt.")


if __name__ == "__main__":
    main()
