from camera import Camera
from telemetry import TelemetryLogger
import time
import cv2
import os
import csv
from datetime import datetime

OUTPUT_PATH = "recordings"
TIME_FORMAT_FILES = "%Y-%m-%d_%H-%M-%S"
TIME_FORMAT_LOG = "%Y-%m-%d %H-%M-%S-%f"
CAPTURE_INTERVAL = 0.5  # 2 Hz


class DriveRecorder:
    def __init__(self, show_live_capture=False):
        self.show_live_capture = show_live_capture

        print("⌚🚗 Drive Recorder wird initialisiert...")
        self.camera_system = Camera(
            resolution=(1920, 1080), show_live_capture=show_live_capture
        )
        self.telemetry_logger = TelemetryLogger(timestamp_format=TIME_FORMAT_LOG)
        self.__create_output_folder()

    def __create_output_folder(self):
        """Creates the output folder for the recorded drive data."""
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    def start_recording(self, capture_interval=0.5):
        from concurrent.futures import ThreadPoolExecutor

        """Starts recording images and telemetry at the specified interval (in seconds)."""
        try:
            self.telemetry_logger.start_logging()
            timestamp_start = time.strftime(TIME_FORMAT_FILES)
            print(f"✅ Starte die Aufzeichnung um {timestamp_start}...")

            # Create a subfolder for the current recording session
            session_folder = os.path.join(OUTPUT_PATH, timestamp_start)
            os.makedirs(session_folder, exist_ok=True)
            telemetry_file = os.path.join(session_folder, "telemetry.csv")

            with ThreadPoolExecutor(max_workers=2) as executor:

                with open(telemetry_file, "x", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    # Write the header row with the command names
                    writer.writerow(
                        ["Time"]
                        + [command.name for command in self.telemetry_logger.commands]
                    )

                    # Cache frequently used functions
                    time_time = time.time
                    datetime_now = datetime.now
                    max_func = max
                    sleep_func = time.sleep

                    while True:
                        start_time = time_time()

                        # Prepare for parallel data capture
                        frame = telemetry_data = None

                        # Capture frame and telemetry data in parallel

                        frame_future = executor.submit(self.camera_system.capture_image)
                        telemetry_future = executor.submit(
                            self.telemetry_logger.read_data,
                            with_timestamp=False,
                            with_logs=True,
                        )

                        frame = frame_future.result()
                        telemetry_data = telemetry_future.result()

                        # Check for data integrity
                        if frame is None or not self.__check_obd_completeness(
                            telemetry_data
                        ):
                            print("⚠️ Fehler bei der Datenerfassung. Überspringe...")
                            cause = "No Frame" if frame is None else "Incomplete Data"
                            writer.writerow(
                                [cause]
                                + (["No Data"] * len(self.telemetry_logger.commands))
                            )
                            time_elapsed = time_time() - start_time
                            sleep_func(max_func(0, capture_interval - time_elapsed))
                            continue

                        # Write data to CSV file / image
                        timestamp_log = datetime_now().strftime(TIME_FORMAT_LOG)[:-5]
                        writer.writerow([timestamp_log] + telemetry_data)
                        image_filename = os.path.join(
                            session_folder, f"{timestamp_log}.jpg"
                        )
                        self.camera_system.save_image(frame, image_filename)

                        # Frequency control
                        time_elapsed = time_time() - start_time
                        sleep_func(max_func(0, capture_interval - (time_elapsed)))
        except KeyboardInterrupt:
            print("Recording interrupted by user.")
        except Exception as e:
            print(f"An error occurred during recording: {e}")

    def stop_recording(self):
        """Stops the recording."""
        self.camera_system.release_camera()
        self.telemetry_logger.stop_logging()
        self.telemetry_logger.disconnect_from_ecu()

    def __check_obd_completeness(self, obd_values: list) -> bool:
        """Checks if all OBD queries delivered a response."""
        return all(value is not None for value in obd_values)


def main():
    drive_recorder = DriveRecorder()
    input("Enter drücken, um die Aufzeichnung zu starten...")
    drive_recorder.start_recording(capture_interval=CAPTURE_INTERVAL)  # 2 Hz
    drive_recorder.stop_recording()


if __name__ == "__main__":
    main()
