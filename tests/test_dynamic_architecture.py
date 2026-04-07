import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.classifier import DocumentClassifier
from app.postprocessing import DataExtractor

async def main():
    load_dotenv()
    classifier = DocumentClassifier()
    extractor = DataExtractor()
    
    print("--- 1. Testing BANKING Category & Fee Rule ---")
    ocr_banking = [
        {"text": "HAVALE DEKONTU", "bbox": [[10, 10], [200, 10], [200, 30], [10, 30]]},
        {"text": "Masraf: 2,14", "bbox": [[10, 50], [100, 50], [100, 70], [10, 70]]},
        {"text": "BSMV: 0,11", "bbox": [[10, 80], [100, 80], [100, 100], [10, 100]]},
        {"text": "Toplam Masraf: 2,25", "bbox": [[10, 110], [250, 110], [250, 130], [10, 130]]}
    ]
    
    cat, conf = classifier.classify(ocr_banking)
    print(f"Detected Category: {cat} (Conf: {conf})")
    
    # Mock LLM data with correct mapping
    llm_mock = {"status": "SUCCESS", "confidence": 0.98, "data": {"fee_amount": 2.14, "tax_amount": 0.11, "fee_total": 2.25}}
    
    result = await extractor.extract(ocr_banking, category=cat, llm_data=llm_mock)
    print(f"Math Check: {result['engine_status']['math_check']}")
    print(f"Trust Score: {result['engine_status']['trust_score']}")

    print("\n--- 2. Testing COORDINATE FALLBACK (Missing Total) ---")
    # In this case, LLM "forgets" fee_total, but it's in OCR next to "Toplam Masraf"
    ocr_fallback = [
        {"text": "Toplam Masraf:", "bbox": [[10, 100], [150, 100], [150, 120], [10, 120]]},
        {"text": "2,25", "bbox": [[160, 100], [220, 100], [220, 120], [160, 120]]} # Directly to the right
    ]
    
    llm_incomplete = {"status": "SUCCESS", "confidence": 0.90, "data": {"fee_amount": 2.14, "tax_amount": 0.11}} # Missing fee_total
    
    result_fallback = await extractor.extract(ocr_fallback, category="BANKING", llm_data=llm_incomplete)
    print(f"Found via Fallback: {result_fallback['extracted_data']['financials']['primary_amount']}")
    print(f"Math Check: {result_fallback['engine_status']['math_check']}")
    print(f"Trust Score: {result_fallback['engine_status']['trust_score']}")

if __name__ == "__main__":
    asyncio.run(main())
