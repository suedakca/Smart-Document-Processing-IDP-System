import os
import cv2
import numpy as np
from loguru import logger
from paddleocr import PaddleOCR

# Sophisticated import for PPStructure to handle version 2.9+ (V3) and older versions
try:
    # Try the newest V3 class first (PaddleOCR 2.9+)
    from paddleocr import PPStructureV3 as PPStructure
except ImportError:
    try:
        from paddleocr import PPStructure
    except ImportError:
        try:
            # Fallback to sub-module for some specific distributions
            from paddleocr.paddleocr import PPStructure
        except ImportError:
            PPStructure = None

from .preprocessing import ImagePreprocessor

class DocumentProcessor:
    def __init__(self, lang='tr'):
        # OCR Engine for standard text with high-sensitivity parameters
        self.ocr = PaddleOCR(
            use_angle_cls=True, 
            lang='tr',
            det_db_thresh=0.1,        # Ultra-sensitive threshold
            det_db_box_thresh=0.3,    # Lower box threshold
            det_db_unclip_ratio=2.0,  # Expand boxes to catch artifacts
            text_det_limit_side_len=1500 # Updated non-deprecated parameter
        )
        self.preprocessor = ImagePreprocessor()
        
        # Initialize Structure Engine safely
        self.structure_engine = None
        if PPStructure is not None:
            try:
                # Basic initialization without problematic parameters
                self.structure_engine = PPStructure(
                    det_db_thresh=0.1
                )
                logger.info("Table analysis engine (PPStructureV3) initialized.")
            except Exception as e:
                logger.warning(f"Failed to load PPStructure: {str(e)}. Table analysis will be skipped.")
                self.structure_engine = None
        else:
            logger.warning("PPStructure module not available. Layout analysis will be limited.")

    def _upscale_if_needed(self, img):
        """
        Dynamically upscales low-resolution images to help OCR detection.
        Capped at 1500px for performance on CPU.
        """
        h, w = img.shape[:2]
        max_dim = 1500
        if max(w, h) < max_dim:
            scale = max_dim / max(w, h)
            # Limit scale to avoid memory issues
            scale = min(scale, 2.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.info(f"Upscaling image for OCR: {w}x{h} -> {new_w}x{new_h} (Scale: {scale:.2f}x)")
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    def _format_table_markdown(self, table_info: dict) -> str:
        """
        Converts PaddleStructure table info to Markdown format.
        """
        html = table_info.get("res", {}).get("html", "")
        if not html: return ""
        return f"HTML_TABLE:\n{html}\n"

    def process(self, img_paths: list) -> tuple:
        """
        Processes a list of images and returns:
        (ocr_results, table_markdown_list)
        """
        if isinstance(img_paths, str):
            img_paths = [img_paths]
            
        all_ocr_results = []
        all_table_markdown = []
        
        for i, img_path in enumerate(img_paths):
            try:
                if not os.path.exists(img_path): continue
                
                # 1. Load and Upscale
                raw_img = cv2.imread(img_path)
                if raw_img is None: continue
                
                robust_img = self._upscale_if_needed(raw_img)
                processed_img = self.preprocessor.process_numpy(robust_img)
                
                temp_ocr_path = f"{img_path}_robust_ocr.png"
                cv2.imwrite(temp_ocr_path, processed_img)
                
                logger.info(f"Page {i+1}: Robust OCR processing (Pass 1)...")
                ocr_res = self.ocr.ocr(temp_ocr_path)
                
                # FALLBACK: If no text found, try Inverting Colors
                if not ocr_res or not ocr_res[0]:
                    logger.warning(f"Page {i+1}: Pass 1 failed. Attempting Pass 2 (Color Inversion)...")
                    inverted_img = cv2.bitwise_not(processed_img)
                    cv2.imwrite(temp_ocr_path, inverted_img)
                    ocr_res = self.ocr.ocr(temp_ocr_path)
                
                if ocr_res:
                    for page in ocr_res:
                        if not page: continue
                        for line in page:
                            if isinstance(line, list) and len(line) == 2:
                                box, (text, score) = line
                                all_ocr_results.append({"text": text, "confidence": float(score), "bbox": box, "page": i+1})
                
                # 2. Structure/Table Analysis
                if self.structure_engine:
                    logger.info(f"Page {i+1}: Layout Analysis (V3)...")
                    struct_res = self.structure_engine(temp_ocr_path)
                    for region in struct_res:
                        if region["type"] == "table":
                            md = self._format_table_markdown(region)
                            if md: all_table_markdown.append(f"Page {i+1} Table:\n{md}")
                else:
                    logger.warning(f"Page {i+1}: Skipping Layout Analysis (Engine Offline)")
                
                if os.path.exists(temp_ocr_path): os.remove(temp_ocr_path)
                            
            except Exception as e:
                logger.exception(f"Robust Processor Error on page {i+1}: {str(e)}")
                
        return all_ocr_results, "\n\n".join(all_table_markdown)
