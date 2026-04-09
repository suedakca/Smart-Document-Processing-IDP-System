try:
    from paddleocr import PaddleOCR, PPStructure
except ImportError:
    from paddleocr import PaddleOCR
    try:
        from paddleocr.paddleocr import PPStructure
    except ImportError:
        PPStructure = None
import os
import cv2
from loguru import logger
from .preprocessing import ImagePreprocessor

class DocumentProcessor:
    def __init__(self, lang='tr'):
        # OCR Engine for standard text
        self.ocr = PaddleOCR(use_angle_cls=True, lang='tr')
        self.preprocessor = ImagePreprocessor()
        
        # Initialize Structure Engine safely
        self.structure_engine = None
        if PPStructure is not None:
            try:
                # Basic initialization to avoid unsupported param errors
                self.structure_engine = PPStructure(show_log=False)
                logger.info("Table analysis engine (PPStructure) initialized.")
            except Exception as e:
                logger.warning(f"Failed to load PPStructure: {str(e)}. Table analysis will be skipped.")
                self.structure_engine = None

    def _format_table_markdown(self, table_info: dict) -> str:
        """
        Converts PaddleStructure table info to Markdown format.
        """
        html = table_info.get("res", {}).get("html", "")
        if not html: return ""
        
        # Note: In a real production system, you'd use a dedicated HTML-to-Markdown parser.
        # For simplicity and speed, we inject the raw HTML or a simplified text structure.
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
                
                # 1. OCR (Standard)
                processed_img = self.preprocessor.process(img_path)
                temp_ocr_path = f"{img_path}_ocr.png"
                cv2.imwrite(temp_ocr_path, processed_img)
                
                logger.info(f"Page {i+1}: Standard OCR...")
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
                    logger.info(f"Page {i+1}: Layout Analysis...")
                    struct_res = self.structure_engine(temp_ocr_path)
                    for region in struct_res:
                        if region["type"] == "table":
                            md = self._format_table_markdown(region)
                            if md: all_table_markdown.append(f"Page {i+1} Table:\n{md}")
                else:
                    logger.warning(f"Page {i+1}: Skipping Layout Analysis (Engine Offline)")
                
                if os.path.exists(temp_ocr_path): os.remove(temp_ocr_path)
                            
            except Exception as e:
                logger.exception(f"Processor Error on page {i+1}: {str(e)}")
                
        return all_ocr_results, "\n\n".join(all_table_markdown)
