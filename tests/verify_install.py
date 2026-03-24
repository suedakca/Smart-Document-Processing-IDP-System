from paddleocr import PaddleOCR
import logging

try:
    # Set logging to ERROR to avoid clutter
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    ocr = PaddleOCR(use_angle_cls=True, lang='tr')
    print("PaddleOCR successfully initialized.")
except Exception as e:
    print(f"Error initializing PaddleOCR: {str(e)}")
