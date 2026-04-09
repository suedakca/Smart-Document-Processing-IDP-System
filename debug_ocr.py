import os
import sys
import json
import cv2
from dotenv import load_dotenv
from loguru import logger

# Add root folder to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.processor import DocumentProcessor
from app.preprocessing import ImagePreprocessor

def test_ocr():
    load_dotenv()
    
    # 1. Initialize
    logger.info("Initializing DocumentProcessor...")
    processor = DocumentProcessor()
    preprocessor = ImagePreprocessor()
    
    # 2. Pick the latest file from uploads
    upload_dir = "uploads"
    files = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        logger.error("No files found in uploads/ directory.")
        return
        
    # Sort by modification time to get the latest
    files.sort(key=os.path.getmtime, reverse=True)
    target_file = files[0]
    
    logger.info(f"Testing OCR on: {target_file}")
    
    # 3. Test Preprocessing
    try:
        logger.info("Step 1: Preprocessing...")
        processed_img = preprocessor.process(target_file)
        debug_path = "debug_preprocessed.png"
        cv2.imwrite(debug_path, processed_img)
        logger.info(f"Preprocessed image saved to {debug_path}")
        
        # 4. Test OCR
        logger.info("Step 2: OCR Extraction...")
        ocr_results, table_md = processor.process([target_file])
        
        if ocr_results:
            logger.info(f"SUCCESS: Extracted {len(ocr_results)} lines of text.")
            for i, res in enumerate(ocr_results[:5]):
                logger.info(f"Line {i+1}: {res['text']} (Conf: {res['confidence']:.2f})")
        else:
            logger.warning("FAILURE: No text extracted. Check 'debug_preprocessed.png' for quality.")
            
        if table_md:
            logger.info("Table analysis detected tables.")
            
    except Exception as e:
        logger.exception(f"An error occurred during direct testing: {str(e)}")

if __name__ == "__main__":
    test_ocr()
