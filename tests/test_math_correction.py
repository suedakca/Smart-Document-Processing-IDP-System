import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.llm_utils import LLMHybridLayer
from app.postprocessing import DataExtractor

async def main():
    load_dotenv()
    llm_layer = LLMHybridLayer()
    extractor = DataExtractor()
    
    # Simulate OCR results with a math error
    # Case: Fee (140.00) + Tax (10.80) = 150.80
    # But OCR misread Total as 1S0.B0 (150.80) or similar
    ocr_results = [
        {"text": "İşlem Ücreti: 140,00", "bbox": [[10, 10], [100, 10], [100, 20], [10, 20]]},
        {"text": "BSMV: 10,80", "bbox": [[10, 30], [100, 30], [100, 40], [10, 40]]},
        {"text": "Toplam: 1S0.B0", "bbox": [[10, 50], [100, 50], [100, 60], [10, 60]]}
    ]
    
    print("--- Testing Math Correction ---")
    
    # 1. Initial extraction (will have mismatch)
    # Mocking fuzzy_data manually for internal extraction call logic
    fuzzy_data = {"fee_amount": "140,00", "tax_amount": "10,80", "total_amount": "1S0.B0"}
    
    # 2. Call extractor (now async)
    result = await extractor.extract(ocr_results, doc_type="receipt", llm_layer=llm_layer)
    
    print("\n--- Final Structured Result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n--- Analysis ---")
    print(f"Math Status: {result['validation']['math']['status']}")
    if result['validation']['math'].get('explanation'):
        print(f"Correction Explanation: {result['validation']['math']['explanation']}")
    print(f"Trust Score: {result['security']['trust_score']}")
    print(f"Security Status: {result['security']['status']}")

if __name__ == "__main__":
    asyncio.run(main())
