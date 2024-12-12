import obd
import serial.tools.list_ports
from features.custom_commands import (
    BRAKE_SIGNAL,
    ACCERLERATOR_POS_MIN,
    ACCERLERATOR_POS_MAX,
)
from features.brake_force import BrakeForceCalculator
from features.gear import GearCalculator
import time
from helpers import normalize, numeric_or_none
from datetime import datetime

CUSTOM_COMMANDS = [BRAKE_SIGNAL]
DERIVED_VALUES = ["GEAR", "BRAKE_FORCE", "PRE_BRAKING", "WHILE_BRAKING"]

COMMANDS_TO_MONITOR = [
    obd.commands.SPEED,
    obd.commands.RPM,
    obd.commands.ACCELERATOR_POS_D,
    # obd.commands.ACCELERATOR_POS_E,
    obd.commands.ENGINE_LOAD,
    # obd.commands.MAF,
]
COMMANDS_TO_MONITOR.extend(CUSTOM_COMMANDS)


class TelemetryLogger:
    def __init__(self, timestamp_format):
        self.__selected_port_device = None
        self.connection = None
        self.timestamp_format = timestamp_format
        self.commands = COMMANDS_TO_MONITOR
        self.derived_values = DERIVED_VALUES
        self.connect_to_ecu()
        self.brake_force = BrakeForceCalculator(connection=self.connection)
        self.current_gear = GearCalculator()  # New instance

    def __watch_commands(self, connection):
        for command in COMMANDS_TO_MONITOR:
            connection.watch(command)

    def __support_custom_commands(self, connection):
        for command in CUSTOM_COMMANDS:
            connection.supported_commands.add(command)

    def __find_serial_ports(self):
        """Finds the available serial ports and returns a list of them"""
        all_ports = list(serial.tools.list_ports.comports())
        return all_ports

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

        self.connection = obd.Async(self.__selected_port_device, delay_cmds=0)
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

        # Batch query commands
        responses = [self.connection.query(cmd) for cmd in self.commands]

        # Process responses in one go
        values = [
            (
                normalize(resp, [ACCERLERATOR_POS_MIN, ACCERLERATOR_POS_MAX], [0, 100])
                if i in [2]  # Accelerator position(s)
                else (
                    bool(resp.value)
                    if i == 4  # Brake signal
                    else numeric_or_none(resp)  # Speed, RPM, Engine Load
                )
            )
            for i, resp in enumerate(responses)
        ]
        # Use gear calculator instead of direct function call
        calculated_gear = self.current_gear(
            values[0],  # vehicle_speed
            values[1],  # rpm
            values[2],  # accelerator_pos
            values[3],  # engine_load
        )
        (brake_force, pre_braking, while_braking) = self.brake_force()
        derived_values = [calculated_gear, brake_force, pre_braking, while_braking]
        values.extend(derived_values)

        if with_logs:
            print(
                f"{timestamp[:-2].replace('-', ':')}: {values[0]} KM/H | {values[1]} RPM | {values[2]} % | {values[3]} % | {values[4]} | {values[-4]} | {values[-3] * 100:.2f} %"
            )

        if with_timestamp:
            values.insert(0, timestamp)

        return values


if __name__ == "__main__":
    logger = TelemetryLogger("%Y-%m-%d_%H-%M-%S-%f")
    logger.start_logging()
    try:
        while True:
            logger.read_data(with_timestamp=True, with_logs=True)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.stop_logging()
        logger.disconnect_from_ecu()
        print("🛑 Programm beendet.")
