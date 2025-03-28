import platform
import time
import cv2
import numpy as np

from config import DEFAULT_RESOLUTION, IMAGE_COMPRESSION_QUALITY
from imageprocessor import ImageProcessor


class Camera:
    def __init__(
        self,
        resolution=DEFAULT_RESOLUTION,
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

    def configure_camera(self, resolution=DEFAULT_RESOLUTION):
        """Configures the camera based on the system (Raspberry Pi or others)."""
        if self.is_raspberry_pi:
            # Raspberry Pi spezifische Konfiguration mit PiCamera2
            from picamera2 import Picamera2
            from libcamera import controls

            self.camera = Picamera2()

            # Konfiguriere die PiCamera2 mit benutzerdefinierten Einstellungen
            config = self.camera.create_video_configuration(main={"size": resolution})
            # config["controls"] = {
            #     "AeEnable": False,
            #     "AfMode": controls.AfModeEnum.Manual,
            #     "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,
            #     "Saturation": 1.1,
            #     "Sharpness": 1.2,
            #     "Contrast": 1.2,
            #     "ExposureTime": 1000,  # 1ms
            #     "FrameDurationLimits": (33333, 33333),
            # }
            config["controls"] = {
                "Saturation": 1.1,
                "Sharpness": 1.2,
                "Contrast": 1.2,
                # Aktiviere automatische Belichtung für verschiedene Lichtverhältnisse
                "AeEnable": True,
                # Autofokus auf Dauerbetrieb stellen für dynamische Szenen
                "AfMode": controls.AfModeEnum.Continuous,
                # Schnelle Rauschreduzierung für Echtzeitaufnahmen
                "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Fast,
                # Minimale und maximale Belichtungszeit (in Mikrosekunden)
                # Min: 1000 μs (1 ms) für helle Szenen
                # Max: 20000 μs (20 ms) für dunklere Szenen
                # "ExposureTime": [1000, 20000],
                "ExposureTime": 10000,  # 10ms (Kompromiss zwischen 1ms und 20ms)
                # Setze Mindestbildrate auf 20 FPS (50000 μs)
                "FrameDurationLimits": (20000, 50000),
                # Aktiviere HDR für bessere Dynamik zwischen hellen und dunklen Bereichen
                # (Falls von deiner Kamera unterstützt)
                "AwbEnable": True,  # Auto-Weißabgleich
                # Optional: ISO-Werte für bessere Low-Light-Performance
                # "AnalogueGain": [1.0, 4.0],  # Min und Max Gain
                "AnalogueGain": 2.0,
            }
            self.camera.configure(config)
            print("📷 PiCamera2 konfiguriert.")

            self.camera.start()

        else:
            # Standard-Webcam-Configuration for MacOS and Windows
            # Doesn't need configuration as it's only for testing purposes
            self.camera = cv2.VideoCapture(0)
            print("📷 Webcam konfiguriert.")
            if not self.camera.isOpened():
                raise RuntimeError("Error: Could not open webcam.")
        print("✅📷 Kamera verbunden.")

    def capture_image(self, compress=True):
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

        if not compress:
            return frame
        compressed_frame = ImageProcessor.compress_image(
            frame, quality=IMAGE_COMPRESSION_QUALITY
        )
        return compressed_frame

    def save_image(self, frame, filename: str, with_logs=False):
        """Saves the image in the specified folder with a timestamp in the filename."""
        cv2.imwrite(filename, frame)
        if with_logs:
            print(f"✅📷💾 Gespeichert: {filename.split('/')[-1]}")

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
    start_time = time.time()
    try:
        while True:
            image = camera_system.capture_image()
            if image is not None:
                camera_system.preview_image(image)
                camera_system.save_image(image, "test.jpg")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.5)
            current_time = time.time()
            elapsed_time = (current_time - start_time) * 1000  # Convert to milliseconds
            print(f"Time since last image: {elapsed_time:.2f} ms")
            start_time = current_time
    except KeyboardInterrupt:
        camera_system.release_camera()
