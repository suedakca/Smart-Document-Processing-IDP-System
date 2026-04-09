import os
import json
import re
import httpx
from loguru import logger
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader
from .schemas import ExtractionResult
from .masking import PIIMasker
from .db_client import DatabaseClient

class LLMHybridLayer:
    """
    Advanced hybrid layer using direct Gemini 1.5 Flash REST v1 API.
    Bypasses SDK versioning issues for production stability.
    """
    def __init__(self, api_key=None, model_name=None):
        self.provider = os.getenv("LLM_PROVIDER", "GEMINI")
        # Support for the newest, high-performance 2.5 flash model
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.db = DatabaseClient()
        self.masker = PIIMasker(use_presidio=False)
        
        # Setup Jinja2
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
        # Base REST URL for Gemini v1 Stable
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent?key={self.api_key}"
        
        if self.provider == "GEMINI":
            logger.info(f"LLM initialized: Gemini v1 REST ({self.model_name})")
        elif self.provider == "LOCAL":
            from openai import OpenAI
            self.model = OpenAI(
                base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1"),
                api_key="ollama" 
            )
            logger.info(f"Local LLM initialized: {self.model_name}")
        else:
            logger.warning("No LLM provider configured.")

    def probe_model(self) -> dict:
        """Performs a real 'Active Probe' via REST to verify credentials with retries."""
        if self.provider != "GEMINI":
             return {"status": "ONLINE", "msg": "Provider check skipped (Non-Gemini)"}
        
        import time
        max_retries = 2
        for attempt in range(max_retries):
            try:
                payload = {"contents": [{"parts": [{"text": "hi"}]}], "generationConfig": {"maxOutputTokens": 1}}
                with httpx.Client(timeout=10.0) as client:
                    res = client.post(self.gemini_url, json=payload)
                    if res.status_code == 200:
                        return {"status": "ONLINE", "msg": "v1 REST Probe successful"}
                    elif res.status_code in [429, 503] and attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return {"status": "DEGRADED", "msg": f"HTTP {res.status_code}: {res.text}"}
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                logger.warning(f"LLM Probe failed: {str(e)}")
                return {"status": "DEGRADED", "msg": str(e)}
        return {"status": "DEGRADED", "msg": "Exhausted retries"}

    async def _get_dynamic_few_shot(self, category: str) -> list:
        """Fetches human-verified examples from DB for Self-Learning."""
        try:
            db_examples = self.db.get_verified_examples(category, limit=3)
            formatted = []
            for ex in db_examples:
                formatted.append({
                    "input": ex["raw_text"] if ex["raw_text"] else "SAMPLE_DOC_TEXT",
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
                    "input": "Para Transferi. Tarih: 22.08.2018. Ref No: 987654321. Tutar: 1.250,00 TL. Alıcı: Ahmet Yılmaz. IBAN: TR12 0001 0002...",
                    "output": {
                        "document_analysis": {
                            "type": "BANKING_DEKONT", 
                            "status": "VERIFIED",
                            "transaction_id": "987654321",
                            "transaction_date": "2018-08-22",
                            "receiver_iban": "TR1200010002...",
                            "currency": "TRY"
                        },
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
        full_text = "\n".join(raw_text_list)
        
        # 1. Mask PII
        pii_mapping = {}
        if mask_pii:
            full_text, pii_mapping = self.masker.mask(full_text)
            if table_markdown:
                table_markdown, _ = self.masker.mask(table_markdown)
        
        # 2. Build Hybrid Few-Shot
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

        # 4. Generate Content via REST with Exponential Backoff
        max_retries = 5
        retry_delay = 2.0  # seconds
        import random
        import asyncio
        
        for attempt in range(max_retries):
            try:
                if self.provider == "GEMINI":
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.1, "topP": 0.95}
                    }
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        res = await client.post(self.gemini_url, json=payload)
                        
                        # Handle transient errors (Rate limit or high demand)
                        if res.status_code in [429, 503] and attempt < max_retries - 1:
                            jitter = random.uniform(0.5, 2.0)
                            wait_time = retry_delay + jitter
                            logger.warning(f"Gemini {res.status_code} (Transient). Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            retry_delay *= 2  # Exponential backoff
                            continue
                            
                        if res.status_code != 200:
                            raise Exception(f"Gemini API Error {res.status_code}: {res.text}")
                        
                        data = res.json()
                        data_str = data['candidates'][0]['content']['parts'][0]['text']
                else:
                    return {"status": "ERROR", "msg": "Provider mismatch in REST logic"}

                data_str = self._clean_llm_json(data_str)
                if mask_pii and pii_mapping:
                    data_str = self.masker.unmask(data_str, pii_mapping)
                
                raw_data = json.loads(data_str)
                validated = ExtractionResult(**raw_data)
                
                return {
                    "status": "SUCCESS",
                    "data": validated.dict(),
                    "confidence": validated.engine_report.get("trust_score", 0.0)
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    jitter = random.uniform(0.5, 2.0)
                    wait_time = retry_delay + jitter
                    logger.warning(f"Extraction attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    retry_delay *= 2
                    continue
                
                logger.error(f"Extraction Pipeline Error after {max_retries} attempts: {str(e)}")
                return {
                    "status": "ERROR", 
                    "msg": f"Failed after {max_retries} attempts: {str(e)}"
                }
