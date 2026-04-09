import re
import math
from thefuzz import fuzz, process as fuzz_process

class ValidationEngine:
    @staticmethod
    def clean_financial_text(text):
        """Standardizes numeric strings for float conversion."""
        text = str(text) if text else ""
        text = re.sub(r'^[eE]\s*', '', text) 
        text = text.replace('TL', '').replace('$', '').replace('€', '6').replace('~', '')
        corrections = {
            'E': '6', 'O': '0', 'o': '0', 'B': '8', 'I': '1', 'i': '1', 'l': '1', 
            '|': '1', 'L': '1', 'G': '6', 'g': '9', 'Z': '2', 'A': '4'
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        text = re.sub(r'(\d)\s+([,.]\d)', r'\1\2', text)
        text = re.sub(r'([,.])\s+(\d)', r'\1\2', text)
        # Preserve only numbers, dots, and commas
        return re.sub(r"[^\d.,]", "", text)

    @staticmethod
    def to_float(val):
        if not val: return 0.0
        if isinstance(val, (int, float)): return abs(float(val))
        
        clean = ValidationEngine.clean_financial_text(str(val))
        if not clean: return 0.0
        
        # Guard against strings with too many dots/commas (like IBANs or doc numbers)
        if clean.count('.') + clean.count(',') > 2:
            return 0.0
            
        try:
            # Turkish Standard: 1.250,50
            if clean.count(',') == 1 and '.' not in clean:
                return abs(float(clean.replace(',', '.')))
            if "," in clean and "." in clean:
                dot_idx, comma_idx = clean.rfind('.'), clean.rfind(',')
                return abs(float(clean.replace(",", ""))) if dot_idx > comma_idx else abs(float(clean.replace(".", "").replace(",", ".")))
            
            # Simple float conversion
            return abs(float(clean))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def search_by_proximity(label_keywords, ocr_results, threshold=60):
        for label_kw in label_keywords:
            label_box = None
            for res in ocr_results:
                if fuzz.partial_ratio(label_kw.upper(), res["text"].upper()) > 85:
                    label_box = res["bbox"]
                    break
            
            if label_box is not None:
                lx_center = (label_box[0][0] + label_box[2][0]) / 2
                ly_center = (label_box[0][1] + label_box[2][1]) / 2
                lx_end = label_box[2][0]
                ly_end = label_box[2][1]
                
                candidates = []
                for res in ocr_results:
                    txt = res["text"]
                    if not any(c.isdigit() for c in txt): continue
                    
                    rb = res["bbox"]
                    rx_start = rb[0][0]
                    ry_start = rb[0][1]
                    rx_center = (rb[0][0] + rb[2][0]) / 2
                    ry_center = (rb[0][1] + rb[2][1]) / 2
                    
                    dist_x = rx_start - lx_end
                    if 0 < dist_x < 400 and abs(ry_center - ly_center) < threshold:
                        candidates.append((dist_x, res))
                        
                    dist_y = ry_start - ly_end
                    if 0 < dist_y < 150 and abs(rx_center - lx_center) < threshold:
                        candidates.append((dist_y, res))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    return candidates[0][1]["text"]
        return None

class DataExtractor:
    def __init__(self):
        self.validator = ValidationEngine()

    def _as_validated(self, val, conf=1.0, source="llm", evidence=None) -> dict:
        """Wraps a value into the ValidatedField structure."""
        return {
            "value": val,
            "confidence": conf,
            "is_valid": True,
            "review_required": False,
            "source": source,
            "evidence_text": evidence
        }

    def _find_evidence(self, target_val, ocr_results) -> str:
        """Finds the best matching OCR line for a given extracted value."""
        if not target_val or not ocr_results:
            return None
        
        target_str = str(target_val).upper()
        best_match = None
        best_score = 0
        
        for res in ocr_results:
            text = res["text"].upper()
            score = fuzz.partial_ratio(target_str, text)
            if score > best_score and score > 80:
                best_score = score
                best_match = res["text"]
        
        return best_match

    def validate_math(self, extracted_dict: dict) -> dict:
        """
        Validates the mathematical consistency of extracted values using the new schema.
        Note: This is now a lightweight check; the main ValidationEngine handles the report.
        """
        hierarchy = extracted_dict.get("financial_hierarchy", {})
        root = hierarchy.get("root_transaction", {})
        root_amt = root.get("amount", {}).get("value", 0.0)
        adjustments = hierarchy.get("adjustments_and_fees", [])
        
        if not adjustments or root_amt == 0:
            return extracted_dict

        first_adj = adjustments[0]
        adj_total = first_adj.get("total_impact", {}).get("value", 0.0)
        breakdown = first_adj.get("breakdown", {})
        sub_sum = sum(breakdown.values())
        
        is_match = abs(sub_sum - adj_total) < 0.05
        first_adj["math_status"] = "MATCH" if is_match else "MISMATCH"
        
        return extracted_dict

    async def extract(self, ocr_results, category="UNKNOWN", llm_data=None, llm_layer=None):
        raw_dynamic = llm_data.get("data", {}) if llm_data else {}
        
        if not raw_dynamic:
            return None

        # Transform raw values into ValidatedField structures
        try:
            # 1. Document Analysis
            analysis = raw_dynamic.get("document_analysis", {})
            structured_analysis = {
                "type": self._as_validated(analysis.get("type", category)),
                "status": self._as_validated(analysis.get("status", "VERIFIED")),
                "sender": self._as_validated(analysis.get("sender"), evidence=self._find_evidence(analysis.get("sender"), ocr_results)),
                "receiver": self._as_validated(analysis.get("receiver"), evidence=self._find_evidence(analysis.get("receiver"), ocr_results)),
                "description": self._as_validated(analysis.get("description"), evidence=self._find_evidence(analysis.get("description"), ocr_results)),
                "currency": self._as_validated(analysis.get("currency", "TRY")),
                "transaction_id": self._as_validated(analysis.get("transaction_id"), evidence=self._find_evidence(analysis.get("transaction_id"), ocr_results)),
                "transaction_date": self._as_validated(analysis.get("transaction_date"), evidence=self._find_evidence(analysis.get("transaction_date"), ocr_results)),
                "sender_iban": self._as_validated(analysis.get("sender_iban"), evidence=self._find_evidence(analysis.get("sender_iban"), ocr_results)),
                "receiver_iban": self._as_validated(analysis.get("receiver_iban"), evidence=self._find_evidence(analysis.get("receiver_iban"), ocr_results)),
            }

            # 2. Financial Hierarchy
            fin_h = raw_dynamic.get("financial_hierarchy", {})
            root_tx = fin_h.get("root_transaction", {})
            
            structured_root = {
                "amount": self._as_validated(self.validator.to_float(root_tx.get("amount", 0.0)), evidence=self._find_evidence(root_tx.get("amount"), ocr_results)),
                "label": self._as_validated(root_tx.get("label", "TUTAR")),
                "text_confirmation": self._as_validated(root_tx.get("text_confirmation", "N/A")),
                "is_valid": True
            }

            adjustments = []
            for adj in fin_h.get("adjustments_and_fees", []):
                adjustments.append({
                    "group_name": self._as_validated(adj.get("group_name", "FEE")),
                    "total_impact": self._as_validated(self.validator.to_float(adj.get("total_impact", 0.0)), evidence=self._find_evidence(adj.get("total_impact"), ocr_results)),
                    "breakdown": {k: self.validator.to_float(v) for k, v in adj.get("breakdown", {}).items()},
                    "math_status": "MATCH"
                })

            final_data = {
                "document_analysis": structured_analysis,
                "financial_hierarchy": {
                    "root_transaction": structured_root,
                    "adjustments_and_fees": adjustments
                },
                "engine_report": raw_dynamic.get("engine_report", {"trust_score": 0.0})
            }

            # Internal lightweight math validation
            final_data = self.validate_math(final_data)
            return final_data

        except Exception as e:
            import traceback
            print(f"Extraction Transform Error: {str(e)}")
            traceback.print_exc()
            return None
