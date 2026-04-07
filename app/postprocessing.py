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
        self.label_map = {
            "root": ["TUTAR", "TOPLAM", "TRANSFER", "ODENECEK", "GRAND TOTAL", "ISLEM TUTARI"],
            "written": ["YALNIZ", "YALNIZCA"],
            "fee": ["MASRAF", "KOMISYON", "ISLEM UCRETI"],
            "tax": ["BSMV", "KDV", "TAX", "VAT"]
        }

    def validate_math(self, extracted_dict: dict) -> dict:
        """
        Validates the mathematical consistency of extracted values.
        """
        hierarchy = extracted_dict.get("financial_hierarchy", {})
        root = hierarchy.get("root_transaction", {})
        root_amt = root.get("amount", 0.0)
        adjustments = hierarchy.get("adjustments_and_fees", [])
        
        if not adjustments or root_amt == 0:
            return extracted_dict

        # Check if sub-items sum to total_impact
        first_adj = adjustments[0]
        adj_total = first_adj.get("total_impact", 0.0)
        breakdown = first_adj.get("breakdown", {})
        sub_sum = sum(breakdown.values())
        
        is_internal_match = abs(sub_sum - adj_total) < 0.05
        is_root_match = abs(adj_total - root_amt) < 0.05
        
        if not is_root_match or not is_internal_match:
            extracted_dict["document_analysis"]["status"] = "REVIEW_REQUIRED"
            extracted_dict["engine_report"]["trust_score"] = min(extracted_dict["engine_report"]["trust_score"], 0.80)
            if "math_status" not in first_adj:
                # Add status only if missing or mismatch
                first_adj["math_status"] = "MISMATCH"
            extracted_dict["engine_report"]["logic_applied"] += " | Math_Mismatch_Detected"
        else:
            first_adj["math_status"] = "MATCH"
            
        return extracted_dict

    async def extract(self, ocr_results, category="UNKNOWN", llm_data=None, llm_layer=None):
        dynamic_content = llm_data.get("data", {}) if llm_data else {}
        
        if not dynamic_content:
            return {
                "document_analysis": {"type": category, "status": "ERROR"},
                "financial_hierarchy": {"root_transaction": {"amount": 0.0}, "adjustments_and_fees": []},
                "engine_report": {"trust_score": 0.0, "logic_applied": "None"}
            }

        # Apply Math Validation to LLM's dynamic content
        validated_extracted = self.validate_math(dynamic_content)
        
        return validated_extracted
