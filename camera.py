import platform
import time
import cv2
import numpy as np
import os


class Camera:
    def __init__(
        self,
        timestamp_format,
        resolution=(1920, 1080),
    ):
        self.timestamp_format = timestamp_format
        # Detect if the system is a Raspberry Pi
        self.is_raspberry_pi = (
            platform.system() == "Linux" and "arm" in platform.machine()
        )

        # Initialize the camera based on the system
        self.camera = None
        self.configure_camera(resolution=resolution)

    def configure_camera(self, resolution=(1920, 1080)):
        """Configures the camera based on the system (Raspberry Pi or others)."""
        if self.is_raspberry_pi:
            # Raspberry Pi spezifische Konfiguration mit PiCamera2
            from picamera2 import Picamera2

            self.camera = Picamera2()

            # Konfiguriere die PiCamera2 mit benutzerdefinierten Einstellungen
            config = self.camera.create_still_configuration()
            config["resolution"] = resolution  # Defaults to (1920, 1080)
            config["controls"] = {
                "ExposureTime": 20000,  # Belichtungszeit in Mikrosekunden
                "AnalogueGain": 1.5,  # ISO-Verstärkung
                "AeEnable": False,  # Automatische Belichtung deaktivieren
                "AwbEnable": False,  # Automatischer Weißabgleich deaktivieren
                "Brightness": 0.5,  # Helligkeit
                "Contrast": 1.2,  # Kontrast
                "Saturation": 1.0,  # Sättigung
            }
            self.camera.configure(config)
            print("📷 PiCamera2 konfiguriert.")
            self.camera.start()

        else:
            # Standard-Webcam-Configuration for MacOS and Windows
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

    def preview_image(frame):
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
    camera_system = Camera()
    image = camera_system.capture_image()
    if image is not None:
        camera_system.save_image(image, "captured_image.jpg")
    camera_system.release_camera()
