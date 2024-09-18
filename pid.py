import obd
import time
import serial.tools.list_ports


def find_elm327_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        # if "Standard Serial over Bluetooth link" in port.description:
        if "USB Serial" in port.description:
            print("Port gefunden: " + port.device)
            return port.device
    return None


def byte_encode_from_pid(pid):
    return ("22" + pid).encode("UTF-8")


# Funktion zum Konvertieren von OBD-Daten in einen Rohdatenstring
def raw_string(messages):
    d = messages[0].data  # only operate on a single message
    d = d[2:]  # chop off mode and PID bytes
    v = int.from_bytes(d, "big")  # helper function for converting byte arrays to ints
    return v  # return the value


# Verbindung zum OBD-Adapter herstellen
port = find_elm327_port()
if port is None:
    print("ELM327 Adapter nicht gefunden. Bitte überprüfen Sie die Verbindung.")
    exit()

# connection = obd.Async(port, fast=False)
print(f"Mit ELM327 auf Port {port} verbinden...")
connection = obd.OBD(port, fast=False)
print(f"Verbindung hergestellt.")
pid = str(input("Mode 22 PID eingeben: "))
cmd = obd.OBDCommand(
    "POT_BRAKE_PRESSURE", f"Mode 22 PID {pid}", byte_encode_from_pid(pid), 0, raw_string
)
connection.supported_commands.add(cmd)

while True:
    response = connection.query(cmd)
    print(f"PID 0x22{pid}: {response.value} / {hex(response.value)}")
    time.sleep(0.2)
