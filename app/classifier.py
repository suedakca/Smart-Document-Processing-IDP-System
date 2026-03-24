import re

class DocumentClassifier:
    def __init__(self):
        self.classes = {
            "invoice": [
            "FATURA", "FIS", "BELGE NO", "IBAN", "KDV", "TAX", "TOTAL", "TOPLAM", "TUTAR", "SAYIN"
            ],
            "id_card": [
                "T.C. KİMLİK", "NÜFUS CÜZDANI", "DOĞUM TARİHİ", "ANNE ADI", "BABA ADI", "SURNAME", "NAME"
            ],
            "contract": [
                "SÖZLEŞME", "PROTOKOL", "TARAFLAR", "MADDE", "YÜKÜMLÜLÜK", "TAAHHÜT", "IMZA"
            ],
            "receipt": [
                "DEKONT", "ISLEM TARIHI", "FIS NO", "GÖNDEREN", "ALICI", "HAVALE", "EFT"
            ]
        }

    def classify(self, text_lines):
        full_text = " ".join([d["text"].upper() for d in text_lines])
        scores = {cls: 0 for cls in self.classes}
        
        for cls, keywords in self.classes.items():
            for kw in keywords:
                if kw in full_text:
                    scores[cls] += 1
        
        # Determine best match
        best_match = max(scores, key=scores.get)
        if scores[best_match] == 0:
            return "unknown", 0.0
            
        confidence = scores[best_match] / sum(scores.values()) if sum(scores.values()) > 0 else 0.0
        return best_match, confidence
