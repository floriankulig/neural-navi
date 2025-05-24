import obd
import time
import csv
import os
import serial.tools.list_ports
from features.custom_commands import BRAKE_SIGNAL
from helpers import normalize, numeric_or_none

INTERVAL = 0.5

CUSTOM_COMMANDS = [BRAKE_SIGNAL]

COMMANDS_TO_MONITOR = [
    obd.commands.SPEED,
    obd.commands.RPM,
    obd.commands.ACCELERATOR_POS_D,
    obd.commands.ENGINE_LOAD,
]
COMMANDS_TO_MONITOR.extend(CUSTOM_COMMANDS)


def find_com_ports():
    all_ports = list(serial.tools.list_ports.comports())
    return all_ports


def watch_commands(connection):
    for command in COMMANDS_TO_MONITOR:
        connection.watch(command)


def support_custom_commands(connection):
    for command in CUSTOM_COMMANDS:
        connection.supported_commands.add(command)


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

    support_custom_commands(connection)
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
                ["Time"] + [command.name for command in COMMANDS_TO_MONITOR]
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
                        accelerator_pos,
                        numeric_or_none(engine_load),
                        brake_signal_value,
                    ]
                )
                print(
                    f"{timestamp} | {numeric_or_none(vehicle_speed)} KM/H | {numeric_or_none(rpm)} RPM | {accelerator_pos} % | {numeric_or_none(engine_load)} % | {brake_signal_value}"
                )
                time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Beende das Logging...")
        connection.stop()
        print("Verbindung getrennt.")


if __name__ == "__main__":
    main()
