import cv2
import numpy as np

DEFAULT_IMAGE_ROI = (0, 320, 1920, 600)


class ImageProcessor:
    """Handles image processing operations for both recording and inference."""

    @staticmethod
    def crop_to_roi(image, roi=None):
        """
        Crops the image to the specified region of interest.

        Args:
            image: The input image as NumPy array
            roi: Tuple of (x, y, width, height) defining the region of interest.
                 Default is (0, 320, 1920, 600) for highway driving.

        Returns:
            The cropped image as NumPy array
        """
        if image is None:
            return None

        # Default ROI optimized for highway driving
        if roi is None:
            roi = DEFAULT_IMAGE_ROI

        x, y, width, height = roi

        # Safety checks for image boundaries
        img_height, img_width = image.shape[:2]
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = min(width, img_width - x)
        height = min(height, img_height - y)

        # Crop the image
        return image[y : y + height, x : x + width]

    @staticmethod
    def preprocess_for_yolo(image, target_size=(640, 640), roi=None):
        """
        Prepares an image for YOLO inference by cropping to ROI and resizing.

        Args:
            image: The input image as NumPy array
            target_size: Size to resize the image to (width, height)
            roi: Region of interest for cropping

        Returns:
            Preprocessed image ready for YOLO inference
        """
        if image is None:
            return None

        # First crop to ROI if specified
        if roi is not None:
            image = ImageProcessor.crop_to_roi(image, roi)

        # Resize to target size if needed (YOLOv11n might have specific requirements)
        if target_size is not None:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        return image
