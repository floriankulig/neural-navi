import obd
import time
from custom_commands import BRAKE_SIGNAL


# Funktion zum Konvertieren von OBD-Daten in einen Rohdatenstring
def raw_string(messages):
    d = messages[0].data  # only operate on a single message
    d = d[2:]  # chop off mode and PID bytes
    v = int.from_bytes(d, "big")  # helper function for converting byte arrays to ints
    print(hex(v))
    return v  # return the value


def main():
    print("Verbinde mit ELM327...")
    connection = obd.OBD("COM6", fast=False)

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
