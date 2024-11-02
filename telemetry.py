import obd
import serial.tools.list_ports
from custom_commands import BRAKE_SIGNAL

CUSTOM_COMMANDS = [BRAKE_SIGNAL]

COMMANDS_TO_MONITOR = [
    obd.commands.SPEED,
    obd.commands.RPM,
    obd.commands.ACCELERATOR_POS_D,
    obd.commands.ENGINE_LOAD,
]
COMMANDS_TO_MONITOR.extend(CUSTOM_COMMANDS)


class TelemetryLogger:
    def __init__(self):
        self._selected_port_device = None
        self.connection = None

    def _watch_commands(connection):
        for command in COMMANDS_TO_MONITOR:
            connection.watch(command)

    def _support_custom_commands(connection):
        for command in CUSTOM_COMMANDS:
            connection.supported_commands.add(command)

    # Finds the available serial ports and returns a list of them
    def _find_serial_ports():
        all_ports = list(serial.tools.list_ports.comports())
        return [port for port in all_ports if "Serial" in port.description]

    def prompt_choose_port(self):
        print("Verfügbare COM-Ports:")
        ports = self._find_serial_ports()

        port_choice = -1
        while not 0 <= port_choice < len(ports):
            for i, port in enumerate(ports):
                print(f"{i+1}: {port.device} - {port.description}")
            port_choice = int(input("Nummer des zu verbindenden Ports wählen: ")) - 1
            if 0 <= port_choice < len(ports):
                break
            else:
                print("Ungültige Auswahl.")

        self._selected_port_device = ports[port_choice].device

    def connect_to_ecu(self):
        # If no port is selected, prompt the user to choose one
        if not self._selected_port_device:
            print("Kein Port zum Verbinden ausgewählt.")
            self.prompt_choose_port()
            return

        self.connection = obd.Async(self._selected_port_device)
        if not self.connection.is_connected():
            print("Verbindung fehlgeschlagen. ELM327-Verbindung überprüfen.")
            return
        print(f"Verbunden mit {self._selected_port_device}.")
        self._watch_commands(self.connection)
        self._support_custom_commands(self.connection)

    def disconnect_from_ecu(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Verbindung getrennt.")

    def start_logging(self):
        if not self.connection:
            print("Keine Verbindung zum ECU vorhanden.")
            return
        self.connection.start()


if __name__ == "__main__":
    logger = TelemetryLogger()
    logger.prompt_choose_port()
    logger.connect_to_ecu(logger._selected_port_device)
