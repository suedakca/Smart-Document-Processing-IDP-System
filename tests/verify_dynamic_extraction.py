import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.processor import DocumentProcessor
from app.llm_utils import LLMHybridLayer
from app.postprocessing import DataExtractor

async def main():
    load_dotenv()
    
    # 1. Initialize
    processor = DocumentProcessor()
    llm_layer = LLMHybridLayer()
    extractor = DataExtractor()
    
    # 2. Pick a sample image from uploads
    sample_img = "uploads/0e398957-7fca-4300-9d69-6b3ba4ab178e.jpg"
    if not os.path.exists(sample_img):
        print(f"Sample image {sample_img} not found. Please ensure it exists.")
        return

    print(f"--- Processing {sample_img} ---")
    
    # 3. Perform OCR
    ocr_results = processor.process(sample_img)
    raw_text = [d["text"] for d in ocr_results]
    print(f"Extracted {len(raw_text)} lines of text.")
    
    # 4. LLM Dynamic Extraction
    print("Sending to Gemini for dynamic extraction...")
    llm_output = await llm_layer.extract_dynamic_json(raw_text)
    
    # 5. Final Extraction
    final_result = extractor.extract(ocr_results, llm_data=llm_output)
    
    # 6. Print Results
    print("\n--- Final Structured Result ---")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
    
    print("\n--- Confidence Report ---")
    print(f"Trust Score: {final_result['security']['trust_score']}")
    print(f"Status: {final_result['security']['status']}")
    if final_result['security']['anomalies']:
        print(f"Anomalies: {final_result['security']['anomalies']}")

if __name__ == "__main__":
    asyncio.run(main())
