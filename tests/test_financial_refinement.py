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
    
    print("--- 1. Testing Transfer Receipt Cluster ---")
    ocr_transfer = [
        {"text": "Hesaptan Çıkan: - 250,00 TL", "bbox": [[10, 10], [200, 10], [200, 20], [10, 20]]},
        {"text": "YALNIZ İki Yüz Elli Türk Lirası", "bbox": [[10, 30], [200, 30], [200, 40], [10, 40]]},
        {"text": "Masraf: 2,14", "bbox": [[10, 50], [100, 50], [100, 60], [10, 60]]},
        {"text": "BSMV: 0,11", "bbox": [[10, 70], [100, 70], [100, 80], [10, 80]]},
        {"text": "Masraf Toplamı: 2,25", "bbox": [[10, 90], [200, 90], [200, 100], [10, 100]]}
    ]
    
    llm_transfer_mock = {
        "status": "SUCCESS",
        "confidence": 0.94,
        "data": {
            "document_type": "transfer_receipt",
            "amount": -250.00,
            "written_amount": "İki Yüz Elli Türk Lirası",
            "fee_amount": 2.14,
            "tax_amount": 0.11,
            "fee_total": 2.25
        }
    }
    
    result_transfer = await extractor.extract(ocr_transfer, llm_data=llm_transfer_mock, llm_layer=llm_layer)
    print(json.dumps(result_transfer["validation"]["mathematical_integrity"], indent=2, ensure_ascii=False))
    print(f"Transfer Cluster Trust Score: {result_transfer['security']['trust_score']}")

    print("\n--- 2. Testing Invoice Cluster ---")
    ocr_invoice = [
        {"text": "Ara Toplam: 220,50", "bbox": [[10, 10], [200, 10], [200, 20], [10, 20]]},
        {"text": "KDV: 39,69", "bbox": [[10, 30], [200, 30], [200, 40], [10, 40]]},
        {"text": "Genel Toplam: 260,19 TL", "bbox": [[10, 50], [200, 50], [200, 60], [10, 60]]}
    ]
    
    llm_invoice_mock = {
        "status": "SUCCESS",
        "confidence": 0.92,
        "data": {
            "document_type": "invoice",
            "subtotal": 220.50,
            "tax_amount": 39.69,
            "amount": 260.19
        }
    }
    
    result_invoice = await extractor.extract(ocr_invoice, llm_data=llm_invoice_mock, llm_layer=llm_layer)
    print(json.dumps(result_invoice["validation"]["mathematical_integrity"], indent=2, ensure_ascii=False))
    print(f"Invoice Cluster Trust Score: {result_invoice['security']['trust_score']}")

if __name__ == "__main__":
    asyncio.run(main())
