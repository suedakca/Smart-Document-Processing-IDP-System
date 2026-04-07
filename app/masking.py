import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from loguru import logger

class PIIMasker:
    """
    Handles detection and masking of sensitive PII (Personally Identifiable Information).
    Supports TC Kimlik, IBAN and Names.
    """
    def __init__(self, use_presidio=False):
        self.use_presidio = use_presidio
        if self.use_presidio:
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
            except Exception as e:
                logger.error(f"Failed to initialize Presidio: {str(e)}")
                self.use_presidio = False
        
        # Regex patterns for Turkish-specific PII
        self.patterns = {
            "TC_KIMLIK": r"\b[1-9]{1}[0-9]{9}[02468]{1}\b",
            "IBAN": r"\bTR\d{2}\s?(?:\d{4}\s?){5}\d{2}\b",
            "PHONE": r"\b(?:\+90|0)?\s?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}\b",
        }

    def mask(self, text: str) -> tuple:
        """
        Masks PII in the given text. Returns (masked_text, mapping).
        """
        if not text: return "", {}
        
        mapping = {}
        masked_text = text
        
        # 1. Regex based masking for structured patterns
        for entity, pattern in self.patterns.items():
            matches = re.findall(pattern, masked_text)
            for i, match in enumerate(set(matches)):
                placeholder = f"[{entity}_{i}]"
                mapping[placeholder] = match
                masked_text = masked_text.replace(match, placeholder)
        
        # 2. Presidio based masking for Names (if enabled)
        if self.use_presidio:
            try:
                results = self.analyzer.analyze(text=masked_text, entities=["PERSON"], language='en') # Spacy model dependent
                anonymized_result = self.anonymizer.anonymize(
                    text=masked_text,
                    analyzer_results=results,
                    operators={"PERSON": OperatorConfig("replace", {"new_value": "[NAME]"})}
                )
                masked_text = anonymized_result.text
            except Exception as e:
                logger.warning(f"Presidio masking failed: {str(e)}")
                
        return masked_text, mapping

    def unmask(self, text: str, mapping: dict) -> str:
        """
        Restores original values from a mapping.
        """
        unmasked = text
        for placeholder, original in mapping.items():
            unmasked = unmasked.replace(placeholder, str(original))
        return unmasked
