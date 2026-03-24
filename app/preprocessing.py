import cv2
import numpy as np
from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.feature import canny

class ImagePreprocessor:
    @staticmethod
    def grayscale(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def noise_reduction(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def thresholding(image):
        """
        Aggressive adaptive thresholding for high contrast.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )

    @staticmethod
    def sharpen(image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def resize_if_needed(image, max_size=2500):
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def deskew(image):
        """
        Detects the skew angle and rotates the image to straighten it.
        """
        # Best for documents: Use Hough Transform on the text lines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider horizontal-ish angles
                if abs(angle) < 45:
                    angles.append(angle)
        
        if not angles:
            return image
            
        median_angle = np.median(angles)
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        # 1. Resize if too large to prevent OOM
        resized = self.resize_if_needed(img)
        
        # 2. Deskew (Straighten tilted document)
        straightened = self.deskew(resized)
        
        # 3. Grayscale & Sharpen
        gray = self.grayscale(straightened)
        sharpened = self.sharpen(gray)
        
        # 4. Adaptive Thresholding (Binarization)
        # This significantly helps OCR in low-contrast areas
        binarized = self.thresholding(sharpened)
        
        return binarized
