import re

class DocumentClassifier:
    def __init__(self):
        # Professional-grade categorized keyword groups
        self.categories = {
            "BANKING": [
                "DEKONT", "HAVALE", "EFT", "TRANSFER", "IBAN", "HESAP NO", "SUBE KODU", "BANKASI", "MASRAF", "BSMV"
            ],
            "RETAIL": [
                "FATURA", "FIS", "BELGE NO", "KDV", "TAX", "TOTAL", "TOPLAM", "TUTAR", "ARA TOPLAM", "MATRAH", "PERAKENDE"
            ],
            "INSURANCE": [
                "POLICE", "ZEYILNAME", "SIGORTA", "PRİM", "ACENTE", "TEMİNAT", "YÜRÜRLÜK", "TAHSİLAT"
            ],
            "HR": [
                "MAAS", "BORDRO", "ÜCRET", "BRÜT", "NET", "KESINTI", "YASAL KESINTI", "SOSYAL GÜVENLIK", "CALISAN", "GÜNLÜK"
            ],
            "ID_CARD": [
                "T.C. KİMLİK", "NÜFUS CÜZDANI", "DOĞUM TARİHİ", "ANNE ADI", "BABA ADI", "SURNAME", "NAME"
            ]
        }

    def classify(self, text_lines):
        full_text = " ".join([d["text"].upper() for d in text_lines])
        scores = {cls: 0 for cls in self.categories}
        
        for cls, keywords in self.categories.items():
            for kw in keywords:
                # Weighted score: More occurrences = higher confidence
                # Exact matches in IDP are usually clear
                if kw in full_text:
                    scores[cls] += 1
        
        # Determine best match
        if not scores or all(s == 0 for s in scores.values()):
            return "UNKNOWN", 0.0
            
        best_match = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_match] / total_score if total_score > 0 else 0.0
        
        return best_match, round(confidence, 2)
