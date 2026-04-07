from paddleocr import PaddleOCR
import os
import cv2
from loguru import logger
from .preprocessing import ImagePreprocessor

class DocumentProcessor:
    def __init__(self, lang='tr'):
        # Initialize PaddleOCR with angle classification
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        self.preprocessor = ImagePreprocessor()

    def process(self, img_paths: list) -> list:
        """
        Processes a list of image paths and returns flattened OCR results.
        Result format: [{"text": "", "confidence": 0.0, "bbox": [], "page": 1}, ...]
        """
        if isinstance(img_paths, str):
            img_paths = [img_paths]
            
        all_extracted_data = []
        
        for i, img_path in enumerate(img_paths):
            try:
                if not os.path.exists(img_path):
                    logger.error(f"Image not found: {img_path}")
                    continue
                
                # 1. Preprocess image
                processed_img = self.preprocessor.process(img_path)
                
                # 2. Save temp processed image
                temp_processed_path = f"{img_path}_step2.png"
                cv2.imwrite(temp_processed_path, processed_img)
                
                # 3. Perform OCR
                logger.info(f"Running OCR on page {i+1}...")
                result = self.ocr.ocr(temp_processed_path)
                
                # Cleanup temp file
                if os.path.exists(temp_processed_path):
                    os.remove(temp_processed_path)
                
                if not result:
                    continue
                
                # Handle standard List[List] or List[Dict] formats
                for page in result:
                    if not page: continue
                    for line in page:
                        # Case: [box, (text, score)]
                        if isinstance(line, list) and len(line) == 2:
                            box, (text, score) = line
                            all_extracted_data.append({
                                "text": text,
                                "confidence": float(score),
                                "bbox": box,
                                "page": i + 1
                            })
                        # Case: dict
                        elif isinstance(line, dict):
                            all_extracted_data.append({
                                "text": line.get("text", ""),
                                "confidence": float(line.get("confidence", 0.0)),
                                "bbox": line.get("bbox", []),
                                "page": i + 1
                            })
                            
            except Exception as e:
                logger.exception(f"OCR Error on page {i+1}: {str(e)}")
                
        return all_extracted_data
