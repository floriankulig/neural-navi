import obd
import time

def scan_mode_22_pids(connection):
    valid_pids = []
    
    for pid in range(0x0000, 0xFFFF):
        hex_pid = f"{pid:04X}"
        cmd = obd.OBDCommand("CUSTOM_PID", f"Mode 22 PID {hex_pid}", ("22" + pid).encode("UTF-8"), 0, raw_string)
        
        response = connection.query(cmd)
        
        if not response.is_null():
            print(f"Gültige PID gefunden: 22{hex_pid}")
            valid_pids.append(hex_pid)
        
        time.sleep(0.1)  # Kurze Pause, um das Steuergerät nicht zu überlasten
    
    return valid_pids

def raw_string(messages):
    return ' '.join([m.raw() for m in messages])

def main():
    print("Verbinde mit ELM327...")
    connection = obd.OBD("COM4")
    
    if not connection.is_connected():
        print("Verbindung fehlgeschlagen. Bitte überprüfen Sie Ihre ELM327-Verbindung.")
        return
    
    print("Verbindung hergestellt. Starte Scan der Mode 22 PIDs...")
    valid_pids = scan_mode_22_pids(connection)
    
    print("\nScan abgeschlossen. Gefundene gültige PIDs:")
    for pid in valid_pids:
        print(f"22{pid}")
    
    print("\nAbrufen der Daten für gültige PIDs:")
    for pid in valid_pids:
        cmd = obd.OBDCommand("CUSTOM_PID", f"Mode 22 PID {pid}", ("22" + pid).encode("UTF-8"), 0, raw_string)
        response = connection.query(cmd)
        print(f"PID 22{pid}: {response.value}")
    
    connection.close()
    print("Verbindung geschlossen.")

if __name__ == "__main__":
    main()