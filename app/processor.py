from paddleocr import PaddleOCR
import os
import cv2
import logging
from .preprocessing import ImagePreprocessor

class DocumentProcessor:
    def __init__(self, lang='tr'):
        # Removing explicit side limits to avoid version-specific argument errors
        self.ocr = PaddleOCR(use_textline_orientation=False, lang=lang)
        self.preprocessor = ImagePreprocessor()
        self.logger = logging.getLogger(__name__)

    def process(self, img_path):
        """
        Processes an image and returns a list of results.
        Result format: [[bounding_box, [text, confidence]], ...]
        """
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at {img_path}")
            
            # 1. Preprocess image (resizes to 2500px if larger)
            processed_img = self.preprocessor.process(img_path)
            
            # 2. Save processed image to a temp file for OCR
            temp_processed_path = f"{img_path}_processed.png"
            cv2.imwrite(temp_processed_path, processed_img)
            
            # 3. Perform OCR on the resized/denoised image
            result = self.ocr.ocr(temp_processed_path)
            # self.logger.info(f"Raw OCR result: {result}")
            
            # 4. Cleanup temp processed file
            if os.path.exists(temp_processed_path):
                os.remove(temp_processed_path)
            
            if not result:
                return []
                
            extracted_data = []
            
            # Paddlex 3.4.0 often returns a list of dictionaries or a single dictionary
            pages = result if isinstance(result, list) else [result]
            
            for page in pages:
                if not isinstance(page, dict):
                    self.logger.warning(f"Unexpected page format: {type(page)}")
                    continue
                
                # Extract texts, scores, and boxes from dictionary keys
                texts = page.get("rec_texts", [])
                scores = page.get("rec_scores", [])
                boxes = page.get("dt_polys", []) # or rec_polys
                
                for i in range(len(texts)):
                    try:
                        extracted_data.append({
                            "text": texts[i],
                            "confidence": float(scores[i]) if i < len(scores) else 0.0,
                            "bbox": boxes[i] if i < len(boxes) else []
                        })
                    except Exception:
                        continue
                
            return extracted_data
        except Exception as e:
            self.logger.error(f"Error during OCR processing: {str(e)}")
            raise e
