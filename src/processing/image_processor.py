import cv2
import numpy as np

from utils.config import DEFAULT_IMAGE_ROI


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

    @staticmethod  # NOT USED YET
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

    @staticmethod
    def compress_image(
        image,
        quality=85,
        resize_factor=None,
        target_format="jpg",
    ):
        """
        Compresses an image by reducing quality and optionally resizing.

        Args:
            image: The input image as NumPy array
            quality: JPEG compression quality (0-100, higher is better quality)
            resize_factor: Optional factor to resize the image (e.g., 0.5 for half size)
            target_format: Output format ('jpg', 'png', etc.)

        Returns:
            Compressed image as MatLike object (NumPy array) or None if failed
        """
        if image is None:
            return None

        # Make a copy to avoid modifying the original
        processed_img = image.copy()

        # Resize if requested (actual resolution change)
        if resize_factor is not None and (0.25 < resize_factor < 1.0):
            height, width = processed_img.shape[:2]
            new_height = int(height * resize_factor)
            new_width = int(width * resize_factor)
            processed_img = cv2.resize(
                processed_img, (new_width, new_height), interpolation=cv2.INTER_AREA
            )

        # Set encoding parameters based on format
        if target_format.lower() == "jpg" or target_format.lower() == "jpeg":
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            ext = ".jpg"
        elif target_format.lower() == "png":
            # PNG compression level 0-9 (9 is highest compression)
            png_compression = min(9, int(9 * (1 - quality / 100)))
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
            ext = ".png"
        else:
            # Default to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            ext = ".jpg"

        # Encode image to bytes
        success, encoded_img = cv2.imencode(ext, processed_img, encode_params)

        if not success:
            return None

        # Convert bytes back to image
        compressed_img = cv2.imdecode(
            np.frombuffer(encoded_img, np.uint8), cv2.IMREAD_COLOR
        )
        return compressed_img

    @staticmethod
    def save_compressed_image(image, filename, quality=85, resize_factor=None):
        """
        Compresses and saves an image directly to a file.

        Args:
            image: The input image as NumPy array
            filename: Path where the image will be saved
            quality: JPEG compression quality (0-100)
            resize_factor: Optional factor to resize the image

        Returns:
            Tuple of (success, original_size, compressed_size)
        """
        if image is None:
            return False

        # Determine format from filename
        ext = filename.split(".")[-1].lower()
        if ext not in ["jpg", "jpeg", "png", "webp"]:
            ext = "jpg"  # Default to JPG

        # Now compress
        compressed_img = ImageProcessor.compress_image(
            image, quality=quality, resize_factor=resize_factor, target_format=ext
        )

        if compressed_img is None:
            return False

        # Save to file
        with open(filename, "wb") as f:
            f.write(compressed_img.tobytes())

        return True
