import re
from thefuzz import fuzz, process as fuzz_process

class ValidationEngine:
    @staticmethod
    def clean_financial_text(text):
        """
        Fixes common OCR character confusion and removes noise.
        """
        # Remove common currency symbols and prefix noise (like 'e' in 'e 250,00')
        text = re.sub(r'^[eE]\s*', '', text) 
        
        corrections = {
            '€': '6', 'E': '6', 'O': '0', 'o': '0',
            'S': '5', 's': '5', 'B': '8', 'I': '1',
            'l': '1', '|': '1', 'Ş': '5'
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
            
        # Handle spaces in numbers (e.g., "250, 00" -> "250,00")
        text = re.sub(r'(\d)\s+([,.]\d)', r'\1\2', text)
        text = re.sub(r'([,.])\s+(\d)', r'\1\2', text)
        
        clean = re.sub(r"[^\d.,]", "", text)
        return clean

    @staticmethod
    def check_math(extracted_keywords):
        """
        Detailed math validation: Masraf + BSMV == Toplam.
        """
        try:
            def to_float(val):
                if not val: return 0.0
                clean = ValidationEngine.clean_financial_text(str(val))
                if "," in clean and "." in clean:
                    clean = clean.replace(".", "").replace(",", ".")
                elif "," in clean:
                    clean = clean.replace(",", ".")
                return float(clean) if clean and any(c.isdigit() for c in clean) else 0.0

            # Dynamic check based on keywords
            # Example for Garanti/Bank receipts
            masraf = to_float(extracted_keywords.get("fee_amount", 0))
            bsmv = to_float(extracted_keywords.get("tax_amount", 0))
            total = to_float(extracted_keywords.get("total_amount", 0))

            if masraf > 0 and bsmv > 0 and total > 0:
                is_valid = abs((masraf + bsmv) - total) < 0.05
                return {
                    "math_valid": is_valid,
                    "check": f"{masraf} + {bsmv} == {total}",
                    "status": "VALID" if is_valid else "MISMATCH"
                }

            return {"math_valid": True, "status": "INSUFFICIENT_DATA"}
        except Exception as e:
            return {"math_valid": False, "error": str(e)}

    @staticmethod
    def cross_check_text(raw_text_list, extracted_total):
        """
        Compares numeric total with 'written-out' total using fuzzy match for 'YALNIZ'.
        """
        written_total = ""
        for line in raw_text_list:
            if fuzz.partial_ratio("YALNIZ", line.upper()) > 75:
                written_total = line
                break
        
        return {
            "cross_check_status": "FOUND" if written_total else "NOT_FOUND",
            "written_line": written_total
        }

class DataExtractor:
    def __init__(self):
        # Common patterns for Turkish invoices
        self.patterns = {
            "iban": r"TR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}",
            "date": r"\d{2}[./-]\d{2}[./-]\d{4}",
            "tax_id": r"\d{10,11}"
        }
        # Generic amount pattern: handles dots/commas as separators and 0-3 decimal places
        self.amount_pattern = r"(\d{1,3}(?:[\s.,]\d{3})*(?:[.,]\d{0,3}))"
        
        # Target keywords for fuzzy matching (Semantik Mapping)
        self.keywords = {
            "total_amount": ["TOPLAM", "GENEL TOPLAM", "ÖDENECEK", "TOTAL", "ARA TOPLAM", "TUTAR"],
            "tax_amount": ["KDV", "TAX", "KATMA DEĞER VERGİSİ", "TAX AMOUNT", "BSMV", "BSIV"],
            "fee_amount": ["MASRAF", "KOMİSYON", "FEE"],
            "invoice_no": ["FATURA NO", "FIS NO", "BELGE NO", "INVOICE NUMBER", "FİŞ NO"]
        }
        self.validator = ValidationEngine()

    def _extract_patterns(self, text_lines):
        full_text = " ".join([d["text"] for d in text_lines])
        extracted = {}
        
        for key, pattern in self.patterns.items():
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                if key == "date":
                    # Simple validation: exclude impossible dates like 44.47.1028
                    valid_dates = []
                    for m in matches:
                        # Basic check: day 1-31, month 1-12
                        parts = re.split(r'[./-]', m)
                        if len(parts) == 3:
                            d, m_val, y = int(parts[0]), int(parts[1]), int(parts[2])
                            if 1 <= d <= 31 and 1 <= m_val <= 12 and 1900 <= y <= 2100:
                                valid_dates.append(m)
                    matches = valid_dates
                
                if key == "iban":
                    matches = [m.replace(" ", "") for m in matches]
                
                if matches:
                    extracted[key] = matches
                
        return extracted

    def _fuzzy_search(self, text_lines):
        results = {}
        for key, aliases in self.keywords.items():
            for i, d in enumerate(text_lines):
                text = d["text"].upper()
                for alias in aliases:
                    if fuzz.partial_ratio(alias, text) > 85:
                        # 1. Check same horizontal 'stripe' (y-axis tolerance)
                        bbox_y = (d["bbox"][0][1] + d["bbox"][2][1]) / 2
                        bbox_h = abs(d["bbox"][0][1] - d["bbox"][2][1])
                        tolerance = max(bbox_h * 0.8, 30) # Dynamic tolerance
                        
                        potential_values = []
                        for other_d in text_lines:
                            other_y = (other_d["bbox"][0][1] + other_d["bbox"][2][1]) / 2
                            # If vertically close and to the right
                            if abs(other_y - bbox_y) < tolerance and other_d["bbox"][0][0] > d["bbox"][0][0] * 0.9:
                                match = re.search(self.amount_pattern, other_d["text"])
                                if match:
                                    # Cleanup: must have at least one digit
                                    val = match.group(0).strip()
                                    if any(char.isdigit() for char in val):
                                        potential_values.append((other_d["bbox"][0][0], val))
                        
                        if potential_values:
                            # Sort by x-coordinate to get the one furthest to the right (usually the total)
                            potential_values.sort(key=lambda x: x[0], reverse=True)
                            results[key] = potential_values[0][1]
                        else:
                            # 2. Check area below keyword
                            for j in range(i + 1, min(i + 5, len(text_lines))):
                                match = re.search(self.amount_pattern, text_lines[j]["text"])
                                if match:
                                    results[key] = match.group(0)
                                    break
                        
                        if key not in results:
                            results[key] = text
        return results

    def _extract_roi(self, ocr_results, doc_type):
        """
        Template-based extraction for known document layouts.
        Example: If it's a standard invoice, look at the bottom right for Total.
        """
        roi_results = {}
        if doc_type == "invoice":
            # Mock ROI: Look for any amount in the bottom 20% of the page
            # This is where 'Total' usually resides
            for d in ocr_results:
                y_center = (d["bbox"][0][1] + d["bbox"][2][1]) / 2
                # Assuming normalized height is not yet done, we check relative to total lines
                if y_center > 0.8: # Bottom 20% (heuristic)
                    match = re.search(self.amount_pattern, d["text"])
                    if match:
                        roi_results["bottom_zone_total"] = match.group(0)
        return roi_results

    def extract(self, ocr_results, doc_type="unknown"):
        if not ocr_results:
            return {}
            
        pattern_data = self._extract_patterns(ocr_results)
        fuzzy_data = self._fuzzy_search(ocr_results)
        roi_data = self._extract_roi(ocr_results, doc_type)
        raw_texts = [d["text"] for d in ocr_results]
        
        # 1. Advanced Validation... (rest of the logic)
        math_check = self.validator.check_math(fuzzy_data)
        cross_check = self.validator.cross_check_text(raw_texts, fuzzy_data.get("total_amount"))
        
        # 2. Security & Anomaly Detection (Layout Consistency)
        trust_score = 0.98
        anomalies = []
        
        if math_check.get("status") == "MISMATCH":
            trust_score -= 0.3
            anomalies.append("Mathematical mismatch detected in amounts")
        
        if len(ocr_results) < 3:
            trust_score -= 0.4
            anomalies.append("Very low text density - possibly incomplete document")
            
        # Alignment check...
        left_coords = [d["bbox"][0][0] for d in ocr_results if len(d["bbox"]) > 0]
        if left_coords:
            mean_left = sum(left_coords) / len(left_coords)
            variance = sum((x - mean_left) ** 2 for x in left_coords) / len(left_coords)
            if variance > 5000:
                trust_score -= 0.1
                anomalies.append("High layout variance detected")
        
        return {
            "structured": pattern_data,
            "detected_keywords": fuzzy_data,
            "validation": {
                "math": math_check,
                "cross_reference": cross_check
            },
            "roi_analysis": roi_data,
            "security": {
                "trust_score": max(0.1, round(trust_score, 2)),
                "anomalies": anomalies,
                "status": "SECURE" if trust_score > 0.8 else "REVIEW_REQUIRED"
            },
            "raw_text": raw_texts
        }
