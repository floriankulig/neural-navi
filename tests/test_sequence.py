import os
from datetime import datetime
from time import sleep
from picamera2 import Picamera2
import time
import cv2


def setup_camera():
    """Initialize and configure the camera"""
    camera = Picamera2()
    camera.resolution = (1920, 1080)  # Full HD
    camera.rotation = 0  # Adjust if needed
    return camera


def create_image_directory():
    """Create a directory for the current session using timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    directory = f"images/{timestamp}"
    os.makedirs(directory, exist_ok=True)
    return directory


def capture_images(camera, directory, interval=0.5):
    camera.start()
    time.sleep(2)
    """Continuously capture images with timestamps"""
    try:
        print(f"Starting image capture. Images will be saved to {directory}")
        print("Press CTRL+C to stop capturing")

        while True:
            # Generate timestamp for the current image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{directory}/{timestamp}.jpg"

            # Nimm das Bild auf
            image = camera.capture_array()
            cv2.imwrite("raw_image.jpg", image)
            # Konvertiere von BGR zu RGB für korrektes Speichern
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Speichere das Bild
            cv2.imwrite(filename, image_rgb)

            # Wait for the specified interval
            sleep(interval)

    except KeyboardInterrupt:
        print("\nCapture stopped by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        camera.stop()


def main():
    # Create the base 'images' directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    # Setup
    camera = setup_camera()
    directory = create_image_directory()

    # Give camera time to warm up
    print("Initializing camera...")
    sleep(2)

    # Start capturing
    capture_images(camera, directory)


if __name__ == "__main__":
    main()
