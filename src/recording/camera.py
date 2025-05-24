import platform
import time
import cv2
import numpy as np

from utils.config import (
    DEFAULT_RESOLUTION,
    IMAGE_COMPRESSION_QUALITY,
    DEFAULT_IMAGE_FOCUS,
    DEFAULT_IMAGE_ROI,
)
from processing.image_processor import ImageProcessor


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
            mode = self.camera.sensor_modes[1]
            fps = mode["fps"]
            assert fps >= 30, "Error: We want to record at least 30 fps."
            config = self.camera.create_video_configuration(
                main={"size": resolution},
                sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]},
            )
            config["controls"] = {
                "Saturation": 1.05,
                "Sharpness": 1.15,
                # "Contrast": 1.1,
                "AeEnable": True,
                # "FrameRate": 30,
                "FrameRate": mode["fps"],
                # "Brightness": 0.1,
                "HdrMode": controls.HdrModeEnum.SingleExposure,
                "AwbEnable": True,  # Auto-Weißabgleich
                # Autofocus
                "AfMode": controls.AfModeEnum.Continuous,
                "AfMetering": controls.AfMeteringEnum.Windows,
                "AfWindows": [DEFAULT_IMAGE_FOCUS],
            }
            self.camera.configure(config)
            print(f"📷 PiCamera2 bei {resolution} und {fps} FPS konfiguriert.")

            self.camera.start()
            print(self.camera.capture_metadata())

        else:
            # Standard-Webcam-Configuration for MacOS and Windows
            # Doesn't need configuration as it's only for testing purposes
            self.camera = cv2.VideoCapture(0)
            print("📷 Webcam konfiguriert.")
            if not self.camera.isOpened():
                raise RuntimeError("Error: Could not open webcam.")
        print("✅📷 Kamera verbunden.")

    def capture_image(self, compress=False):
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


def draw_calibration_guides(
    image, roi=DEFAULT_IMAGE_ROI, is_rgb=False, scale_factor=1.0
):
    """
    Draws calibration guides and ROI visualization on the image.

    Args:
        image: Input image as NumPy array
        roi: Tuple of (x, y, width, height) for the ROI
        is_rgb: Whether the image is in RGB format (True) or BGR format (False)
        scale_factor: Factor to scale the final image (e.g., 0.5 for half size)

    Returns:
        Image with calibration guides drawn
    """
    if image is None:
        return None

    # Create a copy of the image to avoid modifying the original
    overlay = image.copy()

    # Get image dimensions
    height, width = image.shape[:2]

    # ROI parameters
    roi_x, roi_y, roi_width, roi_height = roi

    # Define colors (adapted to color format)
    if is_rgb:
        GREEN = (0, 255, 0)  # RGB: Green for guide lines
        BLUE = (0, 255, 255)  # RGB: Blue for ROI
    else:
        GREEN = (0, 255, 0)  # BGR: Green for guide lines
        BLUE = (255, 255, 0)  # BGR: Blue for ROI

    # Line thickness
    line_thickness = 4

    # Draw ROI as semi-transparent rectangle
    cv2.rectangle(
        overlay, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), BLUE, -1
    )
    alpha = 0.2  # Transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw ROI border
    cv2.rectangle(
        image,
        (roi_x + 2, roi_y),
        (roi_x + roi_width - 2, roi_y + roi_height),
        BLUE,
        line_thickness,
    )

    # Horizontal guide line in the middle
    y_mid = height // 2
    cv2.line(image, (0, y_mid), (width, y_mid), GREEN, line_thickness)

    # Vertical guide lines at x=562 and x=1280
    line1_x = int(width * 0.2925)
    line2_x = int(width * 0.66)
    cv2.line(image, (line1_x, 0), (line1_x, height), GREEN, line_thickness)
    cv2.line(image, (line2_x, 0), (line2_x, height), GREEN, line_thickness)

    # Scale the image if needed
    if scale_factor != 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image


# Beispiel zur Verwendung der Klasse
if __name__ == "__main__":
    camera_system = Camera(show_live_capture=True)
    start_time = time.time()

    # Skalierungsoption
    scale_display = 0.33  # Auf 50% skalieren

    # Status für Skalierung anzeigen
    scale_enabled = True
    print(
        f"Vorschauskalierung: {scale_display if scale_enabled else 1.0}x (Drücke 'r' zum Umschalten)"
    )

    try:
        while True:
            image = camera_system.capture_image()
            if image is not None:
                # Kalibrierungslinien zeichnen
                # Bei Raspberry Pi ist das Bild im RGB-Format, sonst im BGR-Format
                is_rgb_format = camera_system.is_raspberry_pi

                # Skalierungsfaktor anwenden oder nicht, je nach Benutzereinstellung
                current_scale = scale_display if scale_enabled else 1.0

                image_with_guides = draw_calibration_guides(
                    image.copy(),
                    DEFAULT_IMAGE_ROI,
                    is_rgb=is_rgb_format,
                    scale_factor=current_scale,
                )

                # Bild mit Kalibrierungslinien anzeigen
                cv2.imshow("Camera Calibration", image_with_guides)

                # Tasteneingaben verarbeiten
                key = cv2.waitKey(1) & 0xFF

                # 'q' zum Beenden
                if key == ord("q"):
                    break

                # 's' zum Speichern des Originalbildes
                elif key == ord("s"):
                    camera_system.save_image(image, "calibration_test.jpg")
                    print("Bild gespeichert als calibration_test.jpg")

                # 'r' zum Umschalten der Skalierung
                elif key == ord("r"):
                    scale_enabled = not scale_enabled
                    print(
                        f"Vorschauskalierung: {scale_display if scale_enabled else 1.0}x"
                    )

            time.sleep(0.1)  # Kürzerer Delay für flüssigere Anzeige
            current_time = time.time()
            elapsed_time = (current_time - start_time) * 1000  # Convert to milliseconds
            print(f"Time since last image: {elapsed_time:.2f} ms")
            start_time = current_time
    except KeyboardInterrupt:
        camera_system.release_camera()
