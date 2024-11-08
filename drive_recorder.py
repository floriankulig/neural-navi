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
        """Startet die Aufzeichnung von Bildern und Telemetrie im angegebenen Intervall (in Sekunden)."""
        try:
            self.telemetry_logger.start_logging()
            timestamp_start = time.strftime(TIME_FORMAT_FILES)
            print(f"✅ Starte die Aufzeichnung um {timestamp_start}...")
            # Create a subfolder for the current recording session
            session_folder = os.path.join(OUTPUT_PATH, timestamp_start)
            os.makedirs(session_folder, exist_ok=True)
            telemetry_file = os.path.join(session_folder, "telemetry.csv")
            with open(telemetry_file, "x", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # Write the header row with the command names
                writer.writerow(
                    ["Time"]
                    + [command.name for command in self.telemetry_logger.commands]
                )
                while True:
                    start_time = time.time()

                    # CAMERA: Get data
                    frame = self.camera_system.capture_image()
                    timestamp_log = datetime.now().strftime(TIME_FORMAT_LOG)[:-5]
                    if frame is None:
                        # Frequency control
                        time_for_photo = time.time() - start_time
                        time.sleep(max(0, capture_interval - (time_for_photo)))
                        continue  # Skip this iteration if no frame was captured

                    # TELEMETRY: Get data
                    telemetry_data = self.telemetry_logger.read_data(
                        with_timestamp=False, with_logs=True
                    )
                    if not self.__check_obd_completeness(telemetry_data):
                        # Frequency control
                        time_for_photo_and_obd = time.time() - start_time
                        time.sleep(max(0, capture_interval - (time_for_photo_and_obd)))
                        continue  # Skip this iteration if no obd data is incomplete

                    # Write data to CSV file / image
                    writer.writerow([timestamp_log] + telemetry_data)

                    image_filename = os.path.join(
                        session_folder, f"{timestamp_log}.jpg"
                    )
                    self.camera_system.save_image(frame, image_filename)

                    # Frequency control
                    time_for_photo = time.time() - start_time
                    time.sleep(max(0, capture_interval - (time_for_photo)))
        except KeyboardInterrupt:
            print("Recording interrupted by user.")

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
