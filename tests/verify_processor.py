from app.processor import DocumentProcessor
import os
from loguru import logger

def test_processor_init():
    try:
        logger.info("Initializing DocumentProcessor...")
        dp = DocumentProcessor()
        
        # Check if structure_engine exists
        if hasattr(dp, "structure_engine"):
            logger.info(f"SUCCESS: structure_engine attribute exists. Value: {dp.structure_engine}")
        else:
            logger.error("FAILURE: structure_engine attribute is MISSING!")
            return False
            
        # Try a dummy process call if an image exists
        # dp.process(["some_image.jpg"])
        
        return True
    except Exception as e:
        logger.exception(f"FATAL ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    if test_processor_init():
        print("\n--- PASSED: DocumentProcessor is correctly initialized ---")
    else:
        print("\n--- FAILED: DocumentProcessor issues detected ---")
