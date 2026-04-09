import cv2
import numpy as np
from skimage.transform import rotate
from skimage.color import rgb2gray
from skimage.feature import canny

class ImagePreprocessor:
    @staticmethod
    def is_hopeless(image) -> tuple:
        """
        Fast-fail heuristic to detect blank or near-unreadable images.
        """
        if image is None: return True, "null_image"
        # Check for extreme brightness/darkness (near-blank)
        mean_val = np.mean(image)
        if mean_val < 5: return True, "pure_black_image"
        if mean_val > 250: return True, "pure_white_image"
        # Laplacian Variance for blur detection
        lat = cv2.Laplacian(image, cv2.CV_64F).var()
        if lat < 2.0: return True, "void_image_no_edges"
        return False, ""

    @staticmethod
    def is_clean_document(image) -> bool:
        """
        Detects if the image is high-contrast and low-noise.
        """
        if image is None: return False
        # If it's already grayscale, no need to convert
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        std = np.std(gray)
        return 40 < std < 100

    @staticmethod
    def grayscale(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def noise_reduction(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def adjust_contrast(image):
        """
        Using CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        for robust contrast enhancement in varying light.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def sharpen(image):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def resize_if_needed(image, max_size=2000):
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
        Robuster deskewing: detect orientation and straighten.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Invert (PaddleOCR likes black on white, but for lines we might need white on black)
        gray = cv2.bitwise_not(gray)
        
        # Threshold to keep only text
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Grab (x, y) coordinates of all non-zero pixels (the text)
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Handling different opencv versions of minAreaRect angle return
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

    @staticmethod
    def normalize(image):
        """Standardizes pixel intensities."""
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    @staticmethod
    def adaptive_threshold(image):
        """Fallback for extremely faded text."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def process_numpy(self, img, apply_clahe=True, apply_norm=True):
        if img is None:
            return None
            
        # 1. Resize if too large
        img = self.resize_if_needed(img)
        
        # 2. Grayscale
        gray = self.grayscale(img)
        
        # 3. Robust Contrast (CLAHE)
        if apply_clahe:
            gray = self.adjust_contrast(gray)
            
        # 4. Normalization
        if apply_norm:
            gray = self.normalize(gray)
        
        return gray

    def process(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        return self.process_numpy(img)
