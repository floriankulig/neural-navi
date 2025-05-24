import serial
import serial.tools.list_ports
import time

def find_com_ports():
    return list(serial.tools.list_ports.comports())

def test_connection(port, baudrate=38400, timeout=1):
    try:
        with serial.Serial(port, baudrate, timeout=timeout) as ser:
            print(f"Verbunden mit {port} bei {baudrate} baud")
            
            # Sende AT-Befehl
            ser.write(b'ATZ\r')
            time.sleep(1)
            
            # Lese Antwort
            response = ser.read(100)
            print(f"Antwort: {response}")
            
            if b'ELM' in response or b'OBD' in response:
                print("OBD2-Adapter erfolgreich erkannt!")
                return True
            else:
                print("Keine gültige OBD2-Antwort erhalten.")
                return False
    except serial.SerialException as e:
        print(f"Fehler beim Verbinden mit {port}: {e}")
        return False

def main():
    print("Verfügbare COM-Ports:")
    ports = find_com_ports()
    for i, port in enumerate(ports):
        print(f"{i+1}: {port.device} - {port.description} - {port.manufacturer} - {port.serial_number}")

    choice = int(input("Wählen Sie die Nummer des zu testenden Ports: ")) - 1
    if 0 <= choice < len(ports):
        test_connection(ports[choice].device)
    else:
        print("Ungültige Auswahl.")

if __name__ == "__main__":
    main()