from paddleocr import PaddleOCR
import os
import sys
from loguru import logger

def test_ocr(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"ERROR: File not found: {image_path}")
            return
            
        print(f"Initializing PaddleOCR for image: {image_path}")
        ocr = PaddleOCR(use_angle_cls=True, lang='tr')
        
        print("Starting OCR extraction...")
        result = ocr.ocr(image_path)
        
        if not result or not result[0]:
            print("WARNING: OCR finished but NO TEXT was detected.")
        else:
            print(f"SUCCESS: Detected {len(result[0])} lines of text.")
            for i, line in enumerate(result[0]):
                print(f"Line {i+1}: {line[1][0]} (Conf: {line[1][1]:.2f})")
                
    except Exception as e:
        print(f"FATAL ERROR during OCR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use one of the uploaded files if available, or just check the engine
    test_image = "debug_binary.png" # Using a file we saw in the directory earlier
    if not os.path.exists(test_image):
        # Look for any jpg/png in uploads
        uploads = [f for f in os.listdir("uploads") if f.endswith(('.png', '.jpg', '.jpeg'))]
        if uploads:
            test_image = os.path.join("uploads", uploads[0])
            
    test_ocr(test_image)
