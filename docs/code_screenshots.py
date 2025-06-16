import obd


def raw_string_decoder(messages):
    d = messages[0].data  # only operate on a single message
    d = d[2:]  # chop off mode and PID bytes
    v = int.from_bytes(d, "big")  # helper function for converting byte arrays to ints
    return v  # return the value


def scan_mode_22_pids(obd_connection):
    valid_pids = []
    for pid in range(0x0000, 0xFFFF):
        hex_pid = f"{pid:04X}"
        cmd = obd.OBDCommand(
            "CUSTOM_PID",
            f"Mode 22 PID {hex_pid}",
            ("22" + hex_pid).encode(),
            0,
            raw_string_decoder,
        )
        response = obd_connection.query(cmd, force=True)
        if not response.is_null():
            valid_pids.append(hex_pid)
    return valid_pids


def compare_pid_data(before, after):
    changes = {}
    for pid in before:
        if pid in after and before[pid] != after[pid]:
            changes[pid] = {"before": before[pid], "after": after[pid]}
    return changes


def decode_brake_signal(messages):
    d: bytes = messages[0].data
    d = d[3:]  # Remove 1 mode and 2 PID bytes from response
    v = d[-1] & 0x01  # Extract least significant bit of last byte
    return bool(v)
