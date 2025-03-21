from camera import Camera
from config import (
    CAPTURE_INTERVAL,
    RECORDING_OUTPUT_PATH,
    TIME_FORMAT_FILES,
    TIME_FORMAT_LOG,
)
from telemetry import TelemetryLogger
import time
import os
import csv
from datetime import datetime


class DriveRecorder:
    def __init__(self, show_live_capture=False, with_logs=False, compress_images=True):
        self.show_live_capture = show_live_capture
        self.with_logs = with_logs
        self.compress_images = compress_images

        print("⌚🚗 Drive Recorder wird initialisiert...")
        self.camera_system = Camera(
            resolution=(1920, 1080), show_live_capture=show_live_capture
        )
        self.telemetry_logger = TelemetryLogger(timestamp_format=TIME_FORMAT_LOG)
        self.session_folder = self.__create_output_folder()

    def __create_output_folder(self):
        """Creates the output folder for the recorded drive data."""
        timestamp_start = time.strftime(TIME_FORMAT_FILES)
        session_folder = os.path.join(RECORDING_OUTPUT_PATH, timestamp_start)
        os.makedirs(session_folder, exist_ok=True)
        return session_folder

    def start_recording(self, capture_interval=0.5):
        from concurrent.futures import ThreadPoolExecutor

        """Starts recording images and telemetry at the specified interval (in seconds)."""
        try:
            self.telemetry_logger.start_logging()
            timestamp_start = time.strftime(TIME_FORMAT_FILES)
            print(f"✅ Starte die Aufzeichnung um {timestamp_start}...")

            telemetry_file = os.path.join(self.session_folder, "telemetry.csv")

            with ThreadPoolExecutor(max_workers=2) as executor:

                with open(telemetry_file, "x", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    # Write the header row with the command names
                    writer.writerow(
                        ["Time"]
                        + [command.name for command in self.telemetry_logger.commands]
                        + self.telemetry_logger.derived_values
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
                        frame_future = executor.submit(
                            self.camera_system.capture_image,
                            compress=self.compress_images,
                        )
                        telemetry_future = executor.submit(
                            self.telemetry_logger.read_data,
                            with_timestamp=False,
                            with_logs=self.with_logs,
                        )

                        frame = frame_future.result()
                        telemetry_data = telemetry_future.result()
                        # As soon as the data is available, mark the timestamp
                        timestamp_log = datetime_now().strftime(TIME_FORMAT_LOG)[:-5]

                        # Check for data integrity
                        if frame is None or not self.__check_obd_completeness(
                            telemetry_data
                        ):
                            print("⚠️ Fehler bei der Datenerfassung. Überspringe...")
                            cause = "No Frame" if frame is None else "Incomplete Data"
                            writer.writerow(
                                [cause]
                                + (
                                    ["No Data"]
                                    * (
                                        len(self.telemetry_logger.commands)
                                        + len(self.telemetry_logger.derived_values)
                                    )
                                )
                            )
                            time_elapsed = time_time() - start_time
                            sleep_func(max_func(0, capture_interval - time_elapsed))
                            continue

                        # Write data to CSV file
                        writer.writerow([timestamp_log] + telemetry_data)

                        # Save image (with compression if enabled)
                        image_filename = os.path.join(
                            self.session_folder, f"{timestamp_log}.jpg"
                        )
                        self.__save_image_data(frame, image_filename)

                        # Frequency control
                        time_elapsed = time_time() - start_time
                        sleep_func(max_func(0, capture_interval - (time_elapsed)))
        except KeyboardInterrupt:
            print("Recording interrupted by user.")

    def stop_recording(self):
        """Stops the recording."""
        self.camera_system.release_camera()
        self.telemetry_logger.stop_logging()
        self.telemetry_logger.disconnect_from_ecu()

    def __check_obd_completeness(self, obd_values: list) -> bool:
        """Checks if all OBD queries delivered a response."""
        return all(
            value is not None
            for value in obd_values[: len(self.telemetry_logger.commands)]
        )

    def __save_image_data(self, frame, image_filename):
        try:
            # forward to save_image method in camera.py
            self.camera_system.save_image(
                frame, image_filename, with_logs=self.with_logs
            )
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Bildes: {str(e)}")
            print(f"  Dateiname: {image_filename}")
            # Try saving a backup image without compression
            alt_filename = os.path.join(
                self.session_folder, f"backup_{int(time.time())}.jpg"
            )
            try:
                self.camera_system.save_image(
                    frame, alt_filename, with_logs=self.with_logs
                )
                print(
                    f"✅ Bild stattdessen gespeichert als: {os.path.basename(alt_filename)}"
                )
            except Exception:
                print(f"❌ Konnte Bild auch nicht als Backup speichern")


def main():
    drive_recorder = DriveRecorder(with_logs=True, compress_images=True)
    input("Enter drücken, um die Aufzeichnung zu starten...")
    drive_recorder.start_recording(capture_interval=CAPTURE_INTERVAL)  # 2 Hz
    drive_recorder.stop_recording()


if __name__ == "__main__":
    main()
