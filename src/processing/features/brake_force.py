from utils.config import ACCERLERATOR_POS_MAX, ACCERLERATOR_POS_MIN
from obd import Async as AsyncOBDConnection, commands, OBDResponse
from collections import deque
import numpy as np
import time
from utils.helpers import numeric_or_none, normalize
from processing.features.custom_commands import (
    BRAKE_SIGNAL,
)


class BrakeForceCalculator:
    BRAKE_SIGNAL_CMD = BRAKE_SIGNAL
    VEHICLE_SPEED_CMD = commands.SPEED
    ACCELERATOR_CMD = commands.ACCELERATOR_POS_D
    is_braking = False
    is_accelerating = False

    last_pre_braking_speeds = deque()  # Deque für die letzten 3 Sekunden
    QUEUE_TIME_WINDOW_PRE = 3  # Zeitfenster in Sekunden

    last_braking_speeds = deque()  # Deque für die letzte Sekunde
    QUEUE_TIME_WINDOW_WHILE = 1.15  # Zeitfenster in Sekunden

    # Cache frequently used functions
    timestamp = time.time

    # Assumptions
    MIN_BRAKING_DECEL = 0.5  # m/s^2
    AVG_UPDATING_INTERVAL = 0.25  # s
    # Typical maximum deceleration during emergency braking is around 9.81 m/s² (1g)
    # We'll use this as reference for 100% brake force
    MAX_DECELERATION = 9.81  # m/s²

    def __init__(self, connection: AsyncOBDConnection):
        self.connection = connection
        self.__watch_values()

    def __call__(self):
        if not self.is_braking:
            return (
                0.0,
                list(self.last_pre_braking_speeds),
                list(self.last_braking_speeds),
            )

        return (
            self.__calculate_brake_force(),
            list(self.last_pre_braking_speeds),
            list(self.last_braking_speeds),
        )

    def __watch_values(self):
        self.connection.watch(
            self.BRAKE_SIGNAL_CMD, callback=self.__keep_track_of_brake_signal
        )
        self.connection.watch(
            self.VEHICLE_SPEED_CMD, callback=self.__keep_track_of_vehicle_speed
        )
        self.connection.watch(
            self.ACCELERATOR_CMD, callback=self.__keep_track_of_accelerator_position
        )

    def __calculate_brake_force(self):
        pre_braking_speeds = [speed for _, speed in self.last_pre_braking_speeds]
        pre_braking_times = [timestamp for timestamp, _ in self.last_pre_braking_speeds]

        try:
            # If there are not enough speeds to calculate the base deceleration, assume 0
            if len(pre_braking_speeds) < 2:
                baseline_deceleration = 0
            else:
                # Gradient at the last point in time before braking
                baseline_deceleration = self.__calculate_gradient(
                    pre_braking_speeds, pre_braking_times
                )

            braking_speeds = [speed for _, speed in self.last_braking_speeds]
            braking_times = [timestamp for timestamp, _ in self.last_braking_speeds]

            # If there are not enough speeds to calculate the deceleration, take values from the pre-braking phase
            if len(braking_speeds) < 2 and len(pre_braking_speeds) >= 1:
                braking_speeds.insert(0, pre_braking_speeds[-1])
                braking_times.insert(0, pre_braking_times[-1])

            # If still not enough speeds to calculate the deceleration, assume 0
            # This can happen if the driver is immediately braking after accelerating
            if len(braking_speeds) < 2:
                braking_speeds.append(braking_speeds[-1] - self.MIN_BRAKING_DECEL)
                braking_times.append(braking_times[-1] + self.AVG_UPDATING_INTERVAL)

            # Gradient at the latest point in time while braking
            current_deceleration = self.__calculate_gradient(
                braking_speeds, braking_times, mode="double"
            )

            # Calculate the difference in deceleration
            deceleration_delta = current_deceleration - baseline_deceleration

            # Convert to brake force percentage
            # brake_force_percentage = min(
            #     1, max(0, (-deceleration_delta / self.MAX_DECELERATION))
            # )
            # Leave for normalization later
            brake_force_percentage = -deceleration_delta / self.MAX_DECELERATION

            return brake_force_percentage

        except:
            return 0.0

    def __keep_track_of_accelerator_position(self, new_accelerator_pos: OBDResponse):
        new_accelerator_pos_value = normalize(
            new_accelerator_pos,
            [ACCERLERATOR_POS_MIN, ACCERLERATOR_POS_MAX],
            [0, 100],
        )
        if new_accelerator_pos_value is None:
            return
        new_is_accelerating = new_accelerator_pos_value > 0

        # Reset the last braking speeds if the driver is not more accelerating
        # Don't reset if the driver is braking, as driver might be heel-and-toe shifting
        # and braking is not yet finished
        if self.is_accelerating and not new_is_accelerating and not self.is_braking:
            self.last_pre_braking_speeds.clear()

        # Reset the last braking speeds if the driver is going back to accelerating
        if not self.is_accelerating and new_is_accelerating and not self.is_braking:
            self.last_pre_braking_speeds.clear()

        self.is_accelerating = new_is_accelerating

    def __keep_track_of_brake_signal(self, new_brake_signal: OBDResponse):
        new_brake_signal_value = bool(new_brake_signal.value)

        # Reset the last braking speeds if the brake signal has just gone off
        # (i.e. the driver has stopped braking)
        if self.is_braking and not new_brake_signal_value:
            self.last_pre_braking_speeds.clear()
            self.last_braking_speeds.clear()

        self.is_braking = new_brake_signal_value

    def __keep_track_of_vehicle_speed(self, new_speed: OBDResponse):
        new_speed_value = numeric_or_none(new_speed)
        if new_speed_value is None:
            return

        timestamp = self.timestamp()

        # If the driver is braking, keep track of the speed
        # No matter of the accelerator position -> account for heel-and-toe shifts
        if self.is_braking:
            # Add the new speed with a timestamp to the queue
            self.last_braking_speeds.append((timestamp, new_speed_value))

            # Remove old speeds outside the time window
            self.__trim_queue(
                queue=self.last_braking_speeds,
                timeframe=self.QUEUE_TIME_WINDOW_WHILE,
                current_time=timestamp,
            )
        # If the driver is not braking AND NOT accelerating, keep track of the speed
        elif not self.is_accelerating:
            # Add the new speed with a timestamp to the queue
            self.last_pre_braking_speeds.append((timestamp, new_speed_value))

            # Remove old speeds outside the time window
            self.__trim_queue(
                queue=self.last_pre_braking_speeds,
                timeframe=self.QUEUE_TIME_WINDOW_PRE,
                current_time=timestamp,
            )

    def __trim_queue(self, queue, timeframe, current_time):
        """Remove entries from the queue that are older than the time window."""
        while len(queue) > 0 and (current_time - queue[0][0] > timeframe):
            queue.popleft()  # Remove the oldest entry

    def __calculate_gradient(self, y, x=None, deg=1, mode="lin"):
        """Calculate the gradient of a function using regressions."""
        assert len(y) > 1, "At least two values are required to calculate the gradient."
        speeds = np.array(y)
        if not x:
            times = np.linspace(0, len(speeds), len(speeds))
        else:
            times = np.array(x)

        t0 = times[0]
        normalized_times = [t - t0 for t in times]
        fit_degree = deg
        if mode == "exp":
            weights = np.exp(np.arange(len(speeds)))
        elif mode == "lin":
            weights = np.linspace(1, len(speeds), len(speeds))
        elif mode == "double":
            weights = np.linspace(1, 2, len(speeds))
        else:
            weights = None

        coeffs = np.polyfit(normalized_times, speeds, deg=fit_degree, w=weights)
        base_slope = coeffs[fit_degree - 1]
        change_rate = coeffs[0]
        latest_speed_change_time = normalized_times[-1]
        # Linear regression: f(x) = mx + b
        if fit_degree == 1:
            # Derivative: f'(x) = m
            gradient = base_slope
        else:  # fit_degree == 2
            # Polynom: f(x) = ax^2 + bx + c
            # Derivative: f'(x) = 2ax + b
            gradient = 2 * change_rate * latest_speed_change_time + base_slope
        return gradient
