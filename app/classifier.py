import re
from typing import List, Dict, Tuple, Any

class DocumentClassifier:
    """
    Schema-aware hierarchical document classifier.
    Detects Domain -> Family -> Type -> Transaction.
    """
    def __init__(self):
        # Semantic mapping for Rule-based Scoring
        self.rules = {
            "BANKING": {
                "signals": ["DEKONT", "HAVALE", "EFT", "IBAN", "HESAP NUMARASI", "MÜŞTERİ NUMARASI", "MASRAF TUTARI", "BSMV", "TUTAR", "BANKASI"],
                "target_family": "RECEIPT"
            },
            "RETAIL": {
                "signals": ["FATURA", "FİŞ", "BELGE NO", "KDV", "TAX", "PERAKENDE", "SATIŞ"],
                "target_family": "INVOICE"
            }
        }

    def classify(self, text_lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Hierarchical classification based on semantic signals.
        Returns a structured result compatible with DetailedClassificationResult schema.
        """
        full_text = " ".join([str(d.get("text", "")).upper() for d in text_lines])
        
        # 1. Domain Detection
        domain_scores = {}
        for domain, meta in self.rules.items():
            found_signals = [sig for sig in meta["signals"] if sig in full_text]
            domain_scores[domain] = {
                "score": len(found_signals),
                "signals": found_signals,
                "family": meta["target_family"]
            }

        # 2. Determine Primary Domain
        best_domain = "UNKNOWN"
        max_score = 0
        signals = []
        family = "UNKNOWN"

        for domain, res in domain_scores.items():
            if res["score"] > max_score:
                max_score = res["score"]
                best_domain = domain
                signals = res["signals"]
                family = res["family"]

        # 3. Refine Document and Transaction Type (BANKING SPECIFIC)
        doc_type = "UNKNOWN"
        trans_type = "UNKNOWN"

        if best_domain == "BANKING":
            if "HAVALE" in full_text:
                trans_type = "HAVALE"
                doc_type = "BANK_TRANSFER_RECEIPT"
            elif "EFT" in full_text:
                trans_type = "EFT"
                doc_type = "BANK_TRANSFER_RECEIPT"
            elif "DEKONT" in full_text:
                doc_type = "BANK_RECEIPT"

        # 4. Calculate Confidence
        total_signals = len(signals)
        confidence = min(round(total_signals / 5.0, 2), 1.0) if total_signals > 0 else 0.0

        return {
            "domain": best_domain,
            "document_family": family,
            "document_type": doc_type,
            "transaction_type": trans_type,
            "confidence": confidence,
            "classification_signals": signals
        }
