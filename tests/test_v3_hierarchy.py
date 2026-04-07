import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.postprocessing import DataExtractor

async def main():
    load_dotenv()
    extractor = DataExtractor()
    
    print("--- Testing Hierarchical Validation V3 (Decision Tree) ---")
    
    # 1. OCR Results (Mixed Numbers)
    ocr_results = [
        {"text": "YALNIZ İkiyüzelli Türk Lirası", "bbox": [[10, 10], [300, 10], [300, 30], [10, 30]]},
        {"text": "Işlem Tutarı: 250,00 TL", "bbox": [[10, 50], [250, 50], [250, 70], [10, 70]]},
        {"text": "Masraf: 2,14", "bbox": [[10, 90], [100, 90], [100, 110], [10, 110]]},
        {"text": "BSMV: 0,11", "bbox": [[10, 120], [100, 120], [100, 140], [10, 140]]},
        {"text": "Toplam Masraf: 2,25", "bbox": [[10, 150], [200, 150], [200, 170], [10, 170]]}
    ]
    
    # Simulate LLM output (assuming Pass 1 extraction)
    llm_mock = {
        "status": "SUCCESS",
        "confidence": 0.98,
        "data": {
            "transfer_amount": 250.00,
            "written_amount": "İkiyüzelli Türk Lirası",
            "fee_amount": 2.14,
            "tax_amount": 0.11,
            "fee_total": 2.25
        }
    }
    
    result = await extractor.extract(ocr_results, category="BANKING", llm_data=llm_mock)
    
    print(f"Document Status: {result['document_analysis']['status']}")
    print(f"Root Transaction Amount: {result['financial_hierarchy']['root_transaction']['amount']}")
    print(f"Text Confirmation: {result['financial_hierarchy']['root_transaction']['text_confirmation']}")
    print(f"Is Text Valid: {result['financial_hierarchy']['root_transaction']['is_valid']}")
    
    for adj in result['financial_hierarchy']['adjustments_and_fees']:
        print(f"Adjustment Group: {adj['group_name']}")
        print(f"Adjustment Total: {adj['total_impact']}")
        print(f"Math Status: {adj['math_status']}")
        
    print(f"Final Trust Score: {result['engine_report']['trust_score']}")
    
    # Check if 250.0 is selected correctly (The "Big Picture" test)
    assert result['financial_hierarchy']['root_transaction']['amount'] == 250.00
    assert result['financial_hierarchy']['adjustments_and_fees'][0]['total_impact'] == 2.25

if __name__ == "__main__":
    asyncio.run(main())
