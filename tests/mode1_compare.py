import obd
import time


# Funktion zum Scannen von Mode 01 PIDs
def scan_mode_01_pids(connection):
    valid_pids = []

    # for pid in range(0x00, 0xFF):
    #     hex_pid = f"{pid:02X}"
    #     cmd = obd.OBDCommand("CUSTOM_PID", f"Mode 01 PID {hex_pid}", ("01" + hex_pid).encode("UTF-8"), 4, raw_string)

    #     response = connection.query(cmd, force=True)

    #     if not response.is_null():
    #         print(f"Gültige PID gefunden: 01{hex_pid}")
    #         valid_pids.append(hex_pid)

    #     time.sleep(0.05)  # Kurze Pause, um das Steuergerät nicht zu überlasten

    for pid in range(0x00, 0xFF):
        hex_pid = f"{pid:02X}"
        valid_pids.append(hex_pid)
    return valid_pids


# Funktion zum Konvertieren von OBD-Daten in einen Rohdatenstring
def raw_string(messages):
    d = messages[0].data  # only operate on a single message
    d = d[2:]  # chop off mode and PID bytes
    v = int.from_bytes(d, "big")  # helper function for converting byte arrays to ints
    return v  # return the value


# Lese alle PIDs aus, die im übergebenen Pid-Array enthalten sind
def read_all_pids(connection, pids):
    pid_data = {}
    for pid in pids:
        cmd = obd.OBDCommand(
            "CUSTOM_PID",
            f"Mode 01 PID {pid}",
            ("01" + pid).encode("UTF-8"),
            4,
            raw_string,
        )
        response = connection.query(cmd, force=True)
        if response:
            pid_data[pid] = response.value
        print(f"PID 01{pid}: {response.value}")
        time.sleep(0.05)
    return pid_data


# Vergleiche zwei Datensätze, um Unterschiede zu finden
def compare_pid_data(before, after):
    changes = {}
    for pid in before:
        if pid in after and before[pid] != after[pid]:
            changes[pid] = {"before": before[pid], "after": after[pid]}
    return changes


def main():
    print("Verbinde mit ELM327...")
    connection = obd.OBD("COM6", fast=False)

    if not connection.is_connected():
        print("Verbindung fehlgeschlagen. Bitte überprüfen Sie Ihre ELM327-Verbindung.")
        return

    print("010C".encode("UTF-8"))
    cmd = obd.OBDCommand(
        "CUSTOM_PID",
        f"Mode 010C PID",
        "010C".encode("UTF-8"),
        4,
        raw_string,
        obd.ECU.ALL,
    )
    response = connection.query(cmd, force=True)
    print(response.value / 4)
    # Warten, um Scan zu starten
    input("Stimmt die RPM? Drücke Enter, um fortzufahren...")

    print("Verbindung hergestellt. Starte Scan der Mode 01 PIDs...")
    valid_pids = scan_mode_01_pids(connection)

    print("\nScan abgeschlossen. Gefundene gültige PIDs:")
    for pid in valid_pids:
        print(f"0x01{pid}")

    # Initiale Daten auslesen (vor Pedalbetätigung)
    print("Lese initiale PID-Daten aus (vor Pedalbetätigung)...")
    data_before = read_all_pids(connection, valid_pids)

    # Warten, um das Bremspedal zu betätigen
    input("Drücke das Bremspedal und drücke Enter, um fortzufahren...")

    # Daten nach der Pedalbetätigung auslesen
    print("Lese PID-Daten nach der Pedalbetätigung aus...")
    data_after = read_all_pids(connection, valid_pids)

    # Vergleiche die beiden Datensätze und finde Unterschiede
    print("Vergleiche die Daten vor und nach der Pedalbetätigung...")
    changes = compare_pid_data(data_before, data_after)

    # Änderungen anzeigen
    if changes:
        print("Änderungen bei folgenden PIDs erkannt:")
        for pid, change in changes.items():
            difference = change["after"] - change["before"]
            difference_percent = int((abs(difference) / change["after"]) * 100)
            print(
                f"PID: {pid}, Vorher: {change['before']}, Nachher: {change['after']}, Differenz: {difference} ({difference_percent} %)"
            )
    else:
        print("Keine Änderungen festgestellt.")

    connection.close()
    print("Verbindung geschlossen.")


if __name__ == "__main__":
    main()
