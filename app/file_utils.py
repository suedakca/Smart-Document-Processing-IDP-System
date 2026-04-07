import fitz  # PyMuPDF
import cv2
import numpy as np
import os
import uuid
from loguru import logger

class FileHandler:
    @staticmethod
    def is_pdf(file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')

    @staticmethod
    def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> list:
        """
        Converts each page of a PDF into high-resolution images.
        Returns a list of image paths.
        """
        image_paths = []
        try:
            doc = fitz.open(pdf_path)
            prefix = str(uuid.uuid4())
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                
                img_name = f"{prefix}_page_{page_num + 1}.png"
                img_path = os.path.join(output_dir, img_name)
                
                # Use numpy to convert pixmap to image then save with CV2
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR) if pix.n == 3 else img_data
                
                cv2.imwrite(img_path, img_bgr)
                image_paths.append(img_path)
                logger.info(f"Page {page_num + 1} converted to {img_path}")
            
            doc.close()
        except Exception as e:
            logger.error(f"Error during PDF to Image conversion: {str(e)}")
            raise e
            
        return image_paths

    @staticmethod
    def cleanup(file_paths: list):
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Removed temp file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove file {path}: {str(e)}")
