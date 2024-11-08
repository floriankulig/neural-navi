import platform
import time
import cv2
import numpy as np
import os


class Camera:
    def __init__(
        self,
        resolution=(1920, 1080),
        show_live_capture=False,
    ):
        # Detect if the system is a Raspberry Pi
        self.is_raspberry_pi = (
            platform.system() == "Linux" and "aarch64" in platform.machine()
        )

        # Initialize the camera based on the system
        self.camera = None
        self.show_live_capture = show_live_capture
        self.configure_camera(resolution=resolution)

    def configure_camera(self, resolution=(1920, 1080)):
        """Configures the camera based on the system (Raspberry Pi or others)."""
        if self.is_raspberry_pi:
            # Raspberry Pi spezifische Konfiguration mit PiCamera2
            from picamera2 import Picamera2

            self.camera = Picamera2()

            # Konfiguriere die PiCamera2 mit benutzerdefinierten Einstellungen
            config = self.camera.create_still_configuration(main={"size": resolution})
            self.camera.configure(config)
            print("📷 PiCamera2 konfiguriert.")

            if self.show_live_capture:
                from picamera2 import Preview

                self.camera.start_preview(Preview.QTGL)
            self.camera.start()

        else:
            # Standard-Webcam-Configuration for MacOS and Windows
            # Doesn't need configuration as it's only for testing purposes
            self.camera = cv2.VideoCapture(0)
            print("📷 Webcam konfiguriert.")
            if not self.camera.isOpened():
                raise RuntimeError("Error: Could not open webcam.")
        print("✅📷 Kamera verbunden.")

    def capture_image(self):
        """Captures an image from the camera and returns it as a NumPy array (unified image format for both solutions)."""
        if self.is_raspberry_pi:
            # Capture image from PiCamera2 and return as NumPy array
            frame = self.camera.capture_array()
            # Convert the image to RGB format (OpenCV uses BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Capture image from a webcam (using OpenCV)
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Bild konnte nicht erfasst werden.")
                return None
        return frame

    def save_image(self, frame, filename: str):
        """Saves the image in the specified folder with a timestamp in the filename."""
        cv2.imwrite(filename, frame)
        print(f"✅📷💾 Gespeichert: {filename}")

    def preview_image(self, frame):
        """Displays an image in an OpenCV window."""
        cv2.imshow("Live Capture", frame)

    def release_camera(self):
        """Releases the camera resources."""
        if self.is_raspberry_pi:
            self.camera.stop()
        else:
            self.camera.release()
        cv2.destroyAllWindows()


# Beispiel zur Verwendung der Klasse
if __name__ == "__main__":
    camera_system = Camera(show_live_capture=True)

    try:
        while True:
            image = camera_system.capture_image()
            time.sleep(0.5)
    except KeyboardInterrupt:
        camera_system.release_camera()
