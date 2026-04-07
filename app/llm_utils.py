import os
import json
import re
from loguru import logger
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
from .schemas import ExtractionResult
from .masking import PIIMasker

class LLMHybridLayer:
    """
    Advanced hybrid layer using Gemini 1.5 Flash for Hierarchical Validation V3.
    Features: Jinja2 Templating, Few-Shot Learning, PII Masking (KVKK).
    """
    def __init__(self, api_key=None, model_name="gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.masker = PIIMasker(use_presidio=False) # Presidio requires heavy spacy models
        
        # Setup Jinja2
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
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

    def _get_few_shot_examples(self, category: str) -> list:
        """Returns few-shot examples based on document category."""
        examples = {
            "BANKING": [
                {
                    "input": "Para Transferi. Tutar: 1.250,00 TL. Alıcı: Ahmet Yılmaz. IBAN: TR12 0001 0002...",
                    "output": {
                        "document_analysis": {"type": "BANKING_DEKONT", "status": "VERIFIED"},
                        "financial_hierarchy": {
                            "root_transaction": {"amount": 1250.0, "label": "TUTAR", "text_confirmation": "Bin İki Yüz Elli", "is_valid": True},
                            "adjustments_and_fees": []
                        },
                        "engine_report": {"trust_score": 0.99, "logic_applied": "FewShot_Match"}
                    }
                }
            ]
        }
        return examples.get(category, [])

    def _clean_llm_json(self, text: str) -> str:
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match: return json_match.group(1).strip()
        start = text.find('{'); end = text.rfind('}')
        if start != -1 and end != -1: return text[start:end+1].strip()
        return text.strip()

    async def extract_dynamic_json(self, raw_text_list: List[str], table_markdown: str = "", category: str = "UNKNOWN", mask_pii: bool = False) -> Dict[str, Any]:
        """
        Main extraction pipeline with Masking and Templating.
        """
        if not self.model:
            return {"status": "SKIPPED", "msg": "No API Key provided"}

        full_text = "\n".join(raw_text_list)
        
        # 1. Mask PII if requested
        pii_mapping = {}
        if mask_pii:
            full_text, pii_mapping = self.masker.mask(full_text)
            if table_markdown:
                table_markdown, _ = self.masker.mask(table_markdown)
        
        # 2. Render Template
        try:
            template = self.jinja_env.get_template('extraction_prompt.j2')
            prompt = template.render(
                role="auditor",
                category=category,
                full_text=full_text,
                table_markdown=table_markdown,
                has_tables=bool(table_markdown),
                few_shot=self._get_few_shot_examples(category)
            )
        except Exception as e:
            logger.error(f"Template rendering failed: {str(e)}")
            return {"status": "ERROR", "msg": "Prompt template error"}

        try:
            response = self.model.generate_content(prompt)
            data_str = self._clean_llm_json(response.text)
            
            # 3. Unmask result (Re-identification)
            if mask_pii and pii_mapping:
                data_str = self.masker.unmask(data_str, pii_mapping)
            
            raw_data = json.loads(data_str)
            validated = ExtractionResult(**raw_data)
            
            return {
                "status": "SUCCESS",
                "data": validated.dict(),
                "confidence": validated.engine_report.trust_score
            }
        except Exception as e:
            logger.error(f"Extraction Pipeline Error: {str(e)}")
            return {"status": "ERROR", "msg": str(e)}
