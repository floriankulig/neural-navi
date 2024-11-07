import obd
import time
import serial.tools.list_ports
from custom_commands import BRAKE_SIGNAL


# Funktion zum Konvertieren von OBD-Daten in einen Rohdatenstring
def raw_string(messages):
    d = messages[0].data  # only operate on a single message
    d = d[2:]  # chop off mode and PID bytes
    v = int.from_bytes(d, "big")  # helper function for converting byte arrays to ints
    print(hex(v))
    return v  # return the value


def main():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        print(f"{port}: {port.device}")
    print("Verbinde mit ELM327...")
    # connection = obd.OBD("/dev/ttyACM0", fast=False) # Linux USB OBD2 connection
    connection = obd.OBD("/dev/rfcomm0", fast=False)  # Linux Bluetooth OBD2 connection

    if not connection.is_connected():
        print("Verbindung fehlgeschlagen. Bitte überprüfen Sie Ihre ELM327-Verbindung.")
        return

    while True:
        response = connection.query(BRAKE_SIGNAL, force=True)
        print(f"{bool(response.value)}")
        # time.sleep(0.05)

    connection.close()
    print("Verbindung geschlossen.")


if __name__ == "__main__":
    main()
