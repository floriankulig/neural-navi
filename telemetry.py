import obd
import serial.tools.list_ports
from custom_commands import BRAKE_SIGNAL
import time
from helpers import normalize, numeric_or_none
from datetime import datetime

CUSTOM_COMMANDS = [BRAKE_SIGNAL]

COMMANDS_TO_MONITOR = [
    obd.commands.SPEED,
    obd.commands.RPM,
    obd.commands.ACCELERATOR_POS_D,
    obd.commands.ENGINE_LOAD,
]
COMMANDS_TO_MONITOR.extend(CUSTOM_COMMANDS)


class TelemetryLogger:
    def __init__(self, timestamp_format):
        self.__selected_port_device = None
        self.connection = None
        self.timestamp_format = timestamp_format
        self.commands = COMMANDS_TO_MONITOR
        self.connect_to_ecu()

    def __watch_commands(self, connection):
        for command in COMMANDS_TO_MONITOR:
            connection.watch(command)

    def __support_custom_commands(self, connection):
        for command in CUSTOM_COMMANDS:
            connection.supported_commands.add(command)

    # Finds the available serial ports and returns a list of them
    def __find_serial_ports(self):
        all_ports = list(serial.tools.list_ports.comports())
        return all_ports
        return [port for port in all_ports if "Serial" in port.description]

    def __prompt_choose_port(self):
        print("Verfügbare Serielle Ports:")
        ports = self.__find_serial_ports()

        port_choice = -1
        while not 0 <= port_choice < len(ports):
            for i, port in enumerate(ports):
                print(f"{i+1}: {port.device} - {port.description}")
            port_choice = int(input("Nummer des zu verbindenden Ports wählen: ")) - 1
            if 0 <= port_choice < len(ports):
                break
            else:
                print("Ungültige Auswahl.")

        self.__selected_port_device = ports[port_choice].device

    def connect_to_ecu(self):
        # If no port is selected, prompt the user to choose one
        if not self.__selected_port_device:
            print("❌ Kein Port zum Verbinden ausgewählt.")
            self.__prompt_choose_port()

        self.connection = obd.Async(self.__selected_port_device)
        if not self.connection.is_connected():
            print("❌ Verbindung fehlgeschlagen. ELM327-Verbindung überprüfen.")
            return
        print(f"✅ Verbunden mit {self.__selected_port_device}.")
        self.__support_custom_commands(self.connection)
        self.__watch_commands(self.connection)

    def disconnect_from_ecu(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            print("✅⛓️‍💥 Verbindung zur ECU getrennt.")

    def start_logging(self):
        if not self.connection:
            print("❌ Keine Verbindung zur ECU vorhanden.")
            return
        self.connection.start()

    def stop_logging(self):
        if not self.connection:
            print("❌ Keine Verbindung zur ECU vorhanden.")
            return
        self.connection.stop()

    def read_data(self, with_timestamp=False, with_logs=False):
        if not self.connection:
            print("❌ Keine Verbindung zur ECU vorhanden.")
            return
        timestamp = datetime.now().strftime(self.timestamp_format)[:-5]
        vehicle_speed = self.connection.query(obd.commands.SPEED)
        rpm = self.connection.query(obd.commands.RPM)
        accelerator_pos = normalize(
            self.connection.query(obd.commands.ACCELERATOR_POS_D),
            [14.12, 82],
            [0, 100],
        )
        engine_load = self.connection.query(obd.commands.ENGINE_LOAD)
        brake_signal = self.connection.query(BRAKE_SIGNAL)
        brake_signal_value = bool(brake_signal.value)

        values = [
            timestamp if with_timestamp else None,
            numeric_or_none(vehicle_speed),
            numeric_or_none(rpm),
            accelerator_pos,
            numeric_or_none(engine_load),
            brake_signal_value,
        ]
        if not with_timestamp:
            values.pop(0)
        if with_logs:
            print(
                f"{timestamp} | {numeric_or_none(vehicle_speed)} KM/H | {numeric_or_none(rpm)} RPM | {accelerator_pos} % | {numeric_or_none(engine_load)} % | {brake_signal_value}"
            )
        return values


if __name__ == "__main__":
    logger = TelemetryLogger("%Y-%m-%d_%H-%M-%S-%f")
    logger.connect_to_ecu()
    logger.start_logging()
    try:
        while True:
            logger.read_data(with_timestamp=True, with_logs=True)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.stop_logging()
        logger.disconnect_from_ecu()
        print("🛑 Programm beendet.")
