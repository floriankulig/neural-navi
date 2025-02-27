from obd import OBDCommand


def decode_brake_signal(messages):
    d: bytes = messages[0].data  # only operate on a single message
    d = d[2:]  # chop off mode and PID bytes
    v = d[-1] & 0x01  # evaluate the first bit of the last byte
    # v = int.from_bytes(v, "big")  # helper function for converting byte arrays to ints
    return v  # return the value


BRAKE_SIGNAL = OBDCommand(
    "BRAKE_SIGNAL",
    "Whether the braking signal is on or not",
    b"223F9F",
    0,
    decode_brake_signal,
)

# Capped values as ECU doesn't deliver this value from 0 to 100
ACCERLERATOR_POS_MIN = 14.12
ACCERLERATOR_POS_MAX = 81.56
