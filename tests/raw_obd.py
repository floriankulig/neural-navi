import serial
import time

# Funktion, um serielle Kommunikation mit dem ELM327-Adapter zu starten
def init_serial_connection(port, baudrate=38400, timeout=1):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        if ser.is_open:
            print(f"Serielle Verbindung zu {port} geöffnet.")
        return ser
    except Exception as e:
        print(f"Fehler beim Öffnen der seriellen Verbindung: {e}")
        return None

# Funktion, um einen AT-Befehl zu senden
def send_at_command(ser, command):
    ser.write((command + '\r').encode())  # Befehl senden
    time.sleep(0.5)  # Wartezeit für die Antwort
    response = ser.read(ser.in_waiting).decode().replace('\r', '')  # Antwort lesen und '\r' entfernen
    return response.strip()

# Funktion, um einen OBD-Befehl zu senden und die Antwort zu loggen
def send_obd_command(ser, command):
    ser.write((command + '\r').encode())  # Befehl senden
    time.sleep(0.2)  # Wartezeit für die Antwort
    response = ser.read(ser.in_waiting).decode().replace('\r', '')  # Antwort lesen und '\r' entfernen
    return response.strip()

# ELM327 mit nötigen AT-Befehlen konfigurieren
def setup_elm327(ser):
    # AT-Befehle zur Initialisierung
    at_commands = [
        'AT Z',    # Reset
        'AT E0',   # Echo ausschalten
        'AT L0',   # Zeilenumbruch ausschalten
        'AT S0',   # Leerzeichen ausschalten
        'AT SP 0'  # Automatische Protokollerkennung
    ]

    for command in at_commands:
        response = send_at_command(ser, command)
        print(f"Gesendet: {command}, Antwort: {response}")
        time.sleep(1)

# Alle OBD-PIDs auslesen
def read_all_pids(ser, start_pid=0x0100, end_pid=0x01FF):
    pid_data = {}
    for pid in range(start_pid, end_pid + 1):
        command = f"01{pid:02X}"  # OBD-II Befehl in Hexadezimalformat
        response = send_obd_command(ser, command)
        
        if response:
            pid_data[pid] = response
            print(f"Befehl: {command}, Antwort: {response}")
        else:
            print(f"Befehl: {command}, keine Antwort erhalten.")

        time.sleep(0.2)  # Wartezeit zwischen den Befehlen, um Überlastung zu vermeiden
    
    return pid_data

# Vergleiche zwei Datensätze, um Unterschiede zu finden
def compare_pid_data(before, after):
    changes = {}
    for pid in before:
        if pid in after and before[pid] != after[pid]:
            changes[pid] = {
                'before': before[pid],
                'after': after[pid]
            }
    return changes

# Hauptlogik des Programms
def main():
    # COM4-Port öffnen (Anpassen, falls notwendig)
    ser = init_serial_connection('COM4')

    if ser is None:
        return

    try:
        # ELM327 konfigurieren
        setup_elm327(ser)

        # Initiale Daten auslesen (vor Pedalbetätigung)
        print("Lese initiale PID-Daten aus (vor Pedalbetätigung)...")
        data_before = read_all_pids(ser)

        # Warten, um das Bremspedal zu betätigen
        input("Drücke das Bremspedal und drücke Enter, um fortzufahren...")

        # Daten nach der Pedalbetätigung auslesen
        print("Lese PID-Daten nach der Pedalbetätigung aus...")
        data_after = read_all_pids(ser)

        # Vergleiche die beiden Datensätze und finde Unterschiede
        print("Vergleiche die Daten vor und nach der Pedalbetätigung...")
        changes = compare_pid_data(data_before, data_after)

        # Änderungen anzeigen
        if changes:
            print("Änderungen bei folgenden PIDs erkannt:")
            for pid, change in changes.items():
                print(f"PID: {pid:02X}, Vorher: {change['before']}, Nachher: {change['after']}")
        else:
            print("Keine Änderungen festgestellt.")

    except Exception as e:
        print(f"Fehler: {e}")
    finally:
        # Schließe die serielle Verbindung
        ser.close()
        print("Serielle Verbindung geschlossen.")

if __name__ == '__main__':
    main()
