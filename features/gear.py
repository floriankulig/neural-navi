# Gear Ratio Calculation
IDLE_RPM = 920
GEAR_RATIOS_PLOTTED = [9.5, 16.5, 27.5, 40.5, 53.5]
GEAR_RATIOS = [ratio / 1000 for ratio in GEAR_RATIOS_PLOTTED]
GEAR_RATIO_ERROR_VARIANCE_BASE = 1 / 1000
GEAR_RATIO_ERROR_VARIANCE_PROGRESSION = 0.025


def get_gear(vehicle_speed, rpm, accelerator_pos, engine_load):
    if (
        vehicle_speed is None
        or int(rpm) == 0  # catch None values and division by 0
        or accelerator_pos is None
        or engine_load is None
    ):
        return None
    gear_ratio = vehicle_speed / rpm

    # Standing still, assumed to be in neutral
    if vehicle_speed < 6:
        return 0

    # In true neutral
    if (
        vehicle_speed - 5 < gear_ratio * IDLE_RPM < vehicle_speed + 5
        and accelerator_pos < 1
        and engine_load > 1
    ):
        return 0

    # In gear (base case)
    GEAR_RATIO_ERROR_VARIANCE = (
        GEAR_RATIO_ERROR_VARIANCE_BASE
        + gear_ratio * GEAR_RATIO_ERROR_VARIANCE_PROGRESSION
    )
    # GEAR_RATIO_ERROR_VARIANCE = 1.25 / 1000
    for i, ratio in enumerate(GEAR_RATIOS):
        if (
            ratio - GEAR_RATIO_ERROR_VARIANCE
            < gear_ratio
            < ratio + GEAR_RATIO_ERROR_VARIANCE
        ):
            return i + 1

    return 0
