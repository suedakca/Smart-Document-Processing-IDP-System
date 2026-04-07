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
        # A valid float should have at most one decimal separator after cleaning
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
            # Fail gracefully, return 0.0 instead of crashing the process
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

    async def extract(self, ocr_results, category="UNKNOWN", llm_data=None, llm_layer=None):
        raw_texts = [d["text"] for d in ocr_results]
        dynamic_content = llm_data.get("data", {}) if llm_data else {}
        llm_conf = llm_data.get("confidence", 0.0) if llm_data else 0.0
        
        numeric_pool = []
        for res in ocr_results:
            val = self.validator.to_float(res["text"])
            if val > 0.0:
                numeric_pool.append(val)
        
        root_candidate = self.validator.to_float(dynamic_content.get("transfer_amount", dynamic_content.get("amount", 0)))
        if root_candidate == 0 and numeric_pool:
            label_find = self.validator.search_by_proximity(self.label_map["root"], ocr_results)
            if label_find:
                root_candidate = self.validator.to_float(label_find)
            else:
                root_candidate = max(numeric_pool)
        
        written_amount = dynamic_content.get("written_amount", "")
        text_is_valid = llm_conf > 0.9 and written_amount != ""
        
        residuals = [n for n in numeric_pool if abs(n - root_candidate) > 0.1]
        
        fee_group = {"group_name": "Transaction_Fees", "total_impact": 0.0, "breakdown": {}, "math_status": "NONE"}
        
        if category == "BANKING":
            ca = self.validator.to_float(dynamic_content.get("fee_amount", 0))
            cb = self.validator.to_float(dynamic_content.get("tax_amount", 0))
            tot = self.validator.to_float(dynamic_content.get("fee_total", 0))
            if ca > 0 or cb > 0:
                is_valid = abs(ca + cb - tot) < 0.05
                fee_group.update({
                    "total_impact": tot,
                    "breakdown": {"base_fee": ca, "tax_bsmv": cb},
                    "math_status": "MATCH" if is_valid else "MISMATCH"
                })
        elif category == "RETAIL":
            sub = self.validator.to_float(dynamic_content.get("subtotal", 0))
            tax = self.validator.to_float(dynamic_content.get("tax_amount", 0))
            actual = root_candidate
            if sub > 0:
                is_valid = abs(sub + tax - actual) < 0.05
                fee_group.update({
                    "group_name": "Retail_Tax_Breakdown",
                    "total_impact": actual,
                    "breakdown": {"subtotal": sub, "tax_kdv": tax},
                    "math_status": "MATCH" if is_valid else "MISMATCH"
                })

        trust_score = max(llm_conf, 0.85)
        if text_is_valid or fee_group["math_status"] == "MATCH":
            trust_score = 0.99 if text_is_valid and fee_group["math_status"] != "MISMATCH" else 0.97

        return {
            "document_analysis": {
                "type": f"{category}_DEKONT" if category == "BANKING" else category,
                "status": "VERIFIED" if trust_score >= 0.98 else "REVIEW_REQUIRED"
            },
            "financial_hierarchy": {
                "root_transaction": {
                    "amount": root_candidate,
                    "label": "TUTAR / TOTAL",
                    "text_confirmation": written_amount if written_amount else "NOT_FOUND",
                    "is_valid": text_is_valid
                },
                "adjustments_and_fees": [fee_group] if fee_group["math_status"] != "NONE" else []
            },
            "engine_report": {
                "trust_score": round(trust_score, 2),
                "logic_applied": "Hierarchical_Validation_V3"
            }
        }
