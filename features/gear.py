class GearCalculator:
    # Constants moved from gear.py
    IDLE_RPM = 920
    GEAR_RATIOS_PLOTTED = [9.5, 16.5, 27.5, 40.5, 53.5]
    GEAR_RATIOS = [ratio / 1000 for ratio in GEAR_RATIOS_PLOTTED]
    GEAR_RATIO_ERROR_VARIANCE_BASE = 1 / 1000
    GEAR_RATIO_ERROR_VARIANCE_PROGRESSION = 0.025

    def __call__(
        self,
        vehicle_speed: float,
        rpm: float,
        accelerator_pos: float,
        engine_load: float,
    ) -> int:
        """
        Calculate the current gear based on vehicle parameters.

        Args:
            vehicle_speed: Current speed in km/h
            rpm: Current engine RPM
            accelerator_pos: Current accelerator position in %
            engine_load: Current engine load in %

        Returns:
            int: Current gear (0 = neutral, 1-5 = gears)
        """
        self.__get_gear(vehicle_speed, rpm, accelerator_pos, engine_load)

    def __calculate_gear_ratio(self, vehicle_speed: float, rpm: float) -> float:
        """Calculate the current gear ratio based on vehicle speed and RPM."""
        return vehicle_speed / rpm

    def __calculate_error_variance(self, gear_ratio: float) -> float:
        """Calculate the acceptable error variance for gear detection."""
        return (
            self.GEAR_RATIO_ERROR_VARIANCE_BASE
            + gear_ratio * self.GEAR_RATIO_ERROR_VARIANCE_PROGRESSION
        )

    def __is_standing_still(self, vehicle_speed: float) -> bool:
        """Check if the vehicle is standing still."""
        return vehicle_speed < 6

    def __is_in_true_neutral(
        self,
        vehicle_speed: float,
        gear_ratio: float,
        accelerator_pos: float,
        engine_load: float,
    ) -> bool:
        """Check if the vehicle is in true neutral."""
        return (
            vehicle_speed - 5 < gear_ratio * self.IDLE_RPM < vehicle_speed + 5
            and accelerator_pos < 1
            and engine_load > 1
        )

    def __get_gear(
        self,
        vehicle_speed: float,
        rpm: float,
        accelerator_pos: float,
        engine_load: float,
    ) -> int:
        # Input validation
        if (
            vehicle_speed is None
            or int(rpm) == 0  # catch None values and division by 0
            or accelerator_pos is None
            or engine_load is None
        ):
            return None

        # Calculate current gear ratio
        gear_ratio = self.__calculate_gear_ratio(vehicle_speed, rpm)

        # Check special cases
        if self.__is_standing_still(vehicle_speed):
            return 0

        if self.__is_in_true_neutral(
            vehicle_speed, gear_ratio, accelerator_pos, engine_load
        ):
            return 0

        # Calculate acceptable variance for gear detection
        error_variance = self.__calculate_error_variance(gear_ratio)

        # Check each gear ratio
        for i, ratio in enumerate(self.GEAR_RATIOS):
            if ratio - error_variance < gear_ratio < ratio + error_variance:
                return i + 1

        return 0
