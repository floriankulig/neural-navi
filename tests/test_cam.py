from picamera2 import Picamera2
import cv2
import time
import numpy as np


def capture_image(save_path="captured_image.jpg"):
    """
    Nimmt ein Bild mit der PiCamera auf und speichert es.

    Args:
        save_path (str): Pfad, unter dem das Bild gespeichert werden soll

    Returns:
        numpy.ndarray: Das aufgenommene Bild als OpenCV-Array
    """
    # Initialisiere die Kamera
    picam2 = Picamera2()

    # Konfiguriere die Kamera (Full HD)
    config = picam2.create_still_configuration(
        main={
            "size": (1920, 1080),
            "framerate": 20,
        }
    )
    picam2.configure(config)

    # Starte die Kamera
    picam2.start()

    # Kurze Wartezeit für Belichtungsanpassung
    time.sleep(2)

    # Nimm das Bild auf
    image = picam2.capture_array()
    cv2.imwrite("raw_image.jpg", image)
    # Konvertiere von BGR zu RGB für korrektes Speichern
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Speichere das Bild
    cv2.imwrite(save_path, image_rgb)

    # Stoppe die Kamera
    picam2.stop()

    return image_rgb


if __name__ == "__main__":
    try:
        print("Starte Bildaufnahme...")
        image = capture_image()
        print("Bild wurde erfolgreich aufgenommen und gespeichert!")

        # Optional: Zeige das Bild an (wenn Sie einen Display haben)
        cv2.imshow("Aufgenommenes Bild", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {str(e)}")
