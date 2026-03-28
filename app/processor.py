from paddleocr import PaddleOCR
import os
import cv2
import logging
from .preprocessing import ImagePreprocessor

class DocumentProcessor:
    def __init__(self, lang='tr'):
        # Removing explicit side limits to avoid version-specific argument errors
        # Enabling angle classification for better orientation handling
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
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
            
            # Handle List[List] format (Standard PaddleOCR)
            # Format: [[box, (text, score)], ...]
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list) and not isinstance(result[0][0], dict):
                for page in result:
                    for line in page:
                        if len(line) == 2:
                            box, (text, score) = line
                            extracted_data.append({
                                "text": text,
                                "confidence": float(score),
                                "bbox": box
                            })
                return extracted_data

            # Handle Dict format (PaddleX or newer versions)
            pages = result if isinstance(result, list) else [result]
            
            for page in pages:
                if isinstance(page, dict):
                    texts = page.get("rec_texts", [])
                    scores = page.get("rec_scores", [])
                    boxes = page.get("dt_polys", [])
                    
                    for i in range(len(texts)):
                        extracted_data.append({
                            "text": texts[i],
                            "confidence": float(scores[i]) if i < len(scores) else 0.0,
                            "bbox": boxes[i] if i < len(boxes) else []
                        })
                elif isinstance(page, list):
                    # Fallback for nested list without clear structure
                    for item in page:
                        if isinstance(item, list) and len(item) == 2:
                            box, (text, score) = item
                            extracted_data.append({"text": text, "confidence": float(score), "bbox": box})
                
            return extracted_data
        except Exception as e:
            self.logger.error(f"Error during OCR processing: {str(e)}")
            raise e
