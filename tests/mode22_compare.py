import obd
import time


# Funktion zum Scannen von Mode 22 PIDs
def scan_mode_22_pids(connection):
    valid_pids = []

    # for pid in range(0x0000, 0xFFFF):
    #     hex_pid = f"{pid:04X}"
    #     cmd = obd.OBDCommand("CUSTOM_PID", f"Mode 22 PID {hex_pid}", ("22" + hex_pid).encode(), 0, raw_string)

    #     response = connection.query(cmd, force=True)
    #     print(response)
    #     print(response.value)

    #     if not response.is_null():
    #         print(f"Gültige PID gefunden: 22{hex_pid}")
    #         valid_pids.append(hex_pid)

    # time.sleep(0.025)  # Kurze Pause, um das Steuergerät nicht zu überlasten
    # for pid in range(0x0000, 0x0800):
    for pid in range(0x3000, 0x4000):
        hex_pid = f"{pid:04X}"
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
    has_had_nonzeros = False
    for pid in pids:
        cmd = obd.OBDCommand(
            "CUSTOM_PID", f"Mode 22 PID {pid}", ("22" + pid).encode(), 0, raw_string
        )
        response = connection.query(cmd, force=True)
        if response:
            pid_data[pid] = response.value
            if response.value != 0:
                has_had_nonzeros = True
        print(
            f"PID 22{pid}: {response.value} / {hex(response.value) if type(response.value) == int else ''}"
        )
        # time.sleep(0.025)
    if not has_had_nonzeros:
        print("Nur 0-Daten erhalten.")
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

    print("Verbindung hergestellt. Starte Scan der Mode 22 PIDs...")
    valid_pids = scan_mode_22_pids(connection)

    # print("\nScan abgeschlossen. Gefundene gültige PIDs:")
    # for pid in valid_pids:
    #     print(f"0x22{pid}")

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

    try:
        # Änderungen anzeigen
        if changes:
            print("Änderungen bei folgenden PIDs erkannt:")
            for pid, change in changes.items():
                difference = 0
                try:
                    difference = int(change["after"]) - int(change["before"])
                except Exception as e:
                    difference = 0
                # difference_percent = int((abs(difference) / change["after"]) * 100)
                print(
                    f"PID: {pid}, Vorher: {change['before']}, Nachher: {change['after']}, Differenz: {difference}"
                )
        else:
            print("Keine Änderungen festgestellt.")
    except Exception as e:
        print(e)
    connection.close()
    print("Verbindung geschlossen.")


if __name__ == "__main__":
    main()
