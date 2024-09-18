import obd 
import time
import serial.tools.list_ports

def find_elm327_port():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "Standard Serial over Bluetooth link" in port.description:
            return port.device
    return None

# Verbindung zum OBD-Adapter herstellen
port = find_elm327_port()
if port is None:
    print("ELM327 Adapter nicht gefunden. Bitte überprüfen Sie die Verbindung.")
    exit()


def decode_rpm(messages):
    """ decoder for RPM messages """
    d = messages[0].data # only operate on a single message
    print(d)
    d = d[2:] # chop off mode and PID bytes
    v = obd.utils.bytes_to_int(d) / 4.0  # helper function for converting byte arrays to ints
    return v

command = obd.OBDCommand("Brake Pressure", "Brake Pressure Sensor", b"221816", 4, decode_rpm)         

# connection = obd.Async(port, fast=False)
connection = obd.OBD(port, fast=False)

print(obd.commands.pid_getters())

# a callback that prints every new value to the console
def new_rpm(r):
    print(r.value.magnitude, " RPM")
def new_throttle(r):
    print(r.value.magnitude, " Throttle")

# connection.watch(obd.commands.RPM, callback=new_rpm)
# connection.watch(obd.commands.ACCELERATOR_POS_D, callback=new_throttle)
# connection.watch(obd.commands.DTC_RELATIVE_THROTTLE_POS)
# connection.watch(command, callback=new_rpm)
# connection.start()
# connection.stop()
# exit()


commands = [
    obd.commands.PIDS_A,
    obd.commands.PIDS_B,
    obd.commands.PIDS_C,
    obd.commands.MIDS_A,
    obd.commands.MIDS_B,
    obd.commands.MIDS_C,
    obd.commands.MIDS_D,
]
# the callback will now be fired upon receipt of new values

# time.sleep(60)
# connection.stop()
# # Daten auslesen und anzeigen
while True:
    for command in commands:
        response = connection.query(command)
        if response.is_null():
            print(f"{command.name}: Keine Daten verfügbar")
        else:
            print(f"{command.name}: {response.value.magnitude} {response.unit == 'percent' and '%' or ''}")
    time.sleep(0.2)

# Verbindung schließen
# connection.close()