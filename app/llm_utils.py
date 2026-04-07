import os
import json
import re
from loguru import logger
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from .schemas import ExtractionResult

class LLMHybridLayer:
    """
    Advanced hybrid layer using Gemini 1.5 Flash for Hierarchical Validation V3.
    Integrated with Pydantic for strict schema enforcement.
    """
    def __init__(self, api_key=None, model_name="gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name=model_name)
            except Exception as e:
                logger.error(f"Error initializing Gemini: {str(e)}")
                self.model = None
        else:
            self.model = None
            logger.warning("No LLM_API_KEY found. LLM features will be disabled.")

    def _clean_llm_json(self, text: str) -> str:
        """Extracts JSON from markdown code blocks or raw text."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Fallback: find first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1].strip()
            
        return text.strip()

    async def extract_dynamic_json(self, raw_text_list: List[str]) -> Dict[str, Any]:
        """
        Main extraction pipeline for V3 Decision Tree.
        Supports multi-page text aggregation and Pydantic validation.
        """
        if not self.model:
            return {"status": "SKIPPED", "msg": "No API Key provided"}

        full_text = "\n".join(raw_text_list)
        
        # Enhanced prompt for multi-page support and strict JSON output
        extract_prompt = f"""
        Analyze the following OCR text from a document (which may have multiple pages) and extract financial/legal data.
        Return ONLY a JSON object matching the requested schema. Do not include any explanation.

        REQUIRED FIELDS:
        - 'document_analysis': {{'type': 'BANKING|RETAIL|ID|CONTRACT', 'status': 'VERIFIED|REVIEW_REQUIRED'}}
        - 'financial_hierarchy': 
            - 'root_transaction': {{'amount': float, 'label': str, 'text_confirmation': str, 'is_valid': bool}}
            - 'adjustments_and_fees': list of {{'group_name': str, 'total_impact': float, 'breakdown': dict, 'math_status': 'MATCH|MISMATCH'}}
        - 'engine_report': {{'trust_score': float, 'logic_applied': 'Hierarchical_Validation_V3'}}

        TEXT:
        {full_text}
        """

        try:
            # Note: Using synchronous generate_content as SDK async support varies by version.
            # In a production environment, this should be wrapped in run_in_executor.
            response = self.model.generate_content(extract_prompt)
            data_str = self._clean_llm_json(response.text)
            
            try:
                raw_data = json.loads(data_str)
                # Validate with Pydantic
                validated_data = ExtractionResult(**raw_data)
                
                return {
                    "status": "SUCCESS",
                    "data": validated_data.dict(),
                    "confidence": validated_data.engine_report.trust_score
                }
            except Exception as parse_err:
                logger.error(f"JSON Parsing/Validation Error: {str(parse_err)}")
                logger.debug(f"Failed Data String: {data_str}")
                return {"status": "ERROR", "msg": f"Validation failed: {str(parse_err)}"}

        except Exception as e:
            logger.error(f"Gemini Extraction Error: {str(e)}")
            return {"status": "ERROR", "msg": str(e)}
