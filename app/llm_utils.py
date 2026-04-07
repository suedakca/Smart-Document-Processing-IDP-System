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
    def __init__(self, api_key=None, model_name=None):
        self.provider = os.getenv("LLM_PROVIDER", "GEMINI")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-1.5-flash")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.db = DatabaseClient()
        self.masker = PIIMasker(use_presidio=False)
        
        # Setup Jinja2
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
        if self.provider == "GEMINI" and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name=self.model_name)
        elif self.provider == "LOCAL":
            # Support for Ollama / vLLM via OpenAI Compatible Libs
            from openai import OpenAI
            self.model = OpenAI(
                base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
                api_key="ollama" 
            )
            logger.info(f"Local LLM initialized: {self.model_name}")
        else:
            self.model = None
            logger.warning("No LLM provider configured.")

    async def _get_dynamic_few_shot(self, category: str) -> list:
        """Fetches human-verified examples from DB for Self-Learning."""
        try:
            db_examples = self.db.get_verified_examples(category, limit=3)
            formatted = []
            for ex in db_examples:
                formatted.append({
                    "input": "DB_VERIFIED_EXAMPLE",
                    "output": json.loads(ex["corrected_json"])
                })
            return formatted
        except Exception as e:
            logger.error(f"Error fetching dynamic few-shot: {str(e)}")
            return []

    def _get_static_few_shot(self, category: str) -> list:
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
                        "engine_report": {"trust_score": 0.99, "logic_applied": "StaticFewShot"}
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
        if not self.model: return {"status": "SKIPPED", "msg": "No model configured"}

        full_text = "\n".join(raw_text_list)
        
        # 1. Mask PII
        pii_mapping = {}
        if mask_pii:
            full_text, pii_mapping = self.masker.mask(full_text)
            if table_markdown:
                table_markdown, _ = self.masker.mask(table_markdown)
        
        # 2. Build Hybrid Few-Shot (Static + DB Verified)
        static_examples = self._get_static_few_shot(category)
        dynamic_examples = await self._get_dynamic_few_shot(category)
        all_examples = static_examples + dynamic_examples

        # 3. Render Template
        try:
            template = self.jinja_env.get_template('extraction_prompt.j2')
            prompt = template.render(
                role="auditor",
                category=category,
                full_text=full_text,
                table_markdown=table_markdown,
                has_tables=bool(table_markdown),
                few_shot=all_examples
            )
        except Exception as e:
            logger.error(f"Template rendering failed: {str(e)}")
            return {"status": "ERROR", "msg": "Prompt template error"}

        # 4. Generate Content
        try:
            if self.provider == "GEMINI":
                response = self.model.generate_content(prompt)
                data_str = response.text
            else:
                # OpenAI/Local Provider
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                data_str = response.choices[0].message.content

            data_str = self._clean_llm_json(data_str)
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
