import re
import datetime
from typing import Dict, Any, List
from ..schemas import ValidationReport, ValidationCheck, ExtractionResult, ValidatedField

class ValidationEngine:
    """
    Deterministic validation layer for financial document cross-checking.
    Supports PASS, FAIL, and WARNING states.
    """
    
    @staticmethod
    def validate(extraction: Dict[str, Any]) -> ValidationReport:
        checks = []
        overall_status = "SUCCESS"
        
        # 1. Mathematical Consistency (Fee Calculation)
        # Hierarchy: adjustments_and_fees -> total_impact
        # Check if Fee + BSMV matches total_impact
        adj_list = extraction.get("financial_hierarchy", {}).get("adjustments_and_fees", [])
        for adj in adj_list:
            group_name = adj.get("group_name", {}).get("value", "UNKNOWN")
            total = adj.get("total_impact", {}).get("value", 0.0)
            breakdown = adj.get("breakdown", {})
            
            if breakdown:
                sum_breakdown = sum(breakdown.values())
                if abs(sum_breakdown - total) < 0.01:
                    checks.append(ValidationCheck(
                        name=f"math_check_{group_name}",
                        status="PASS",
                        details=f"Math matches for {group_name}: {sum_breakdown} == {total}"
                    ))
                else:
                    checks.append(ValidationCheck(
                        name=f"math_check_{group_name}",
                        status="FAIL",
                        details=f"Math MISMATCH for {group_name}: {sum_breakdown} vs {total}"
                    ))
                    overall_status = "ERROR"

        # 2. IBAN Format Validation (Turkish Specific TR...)
        # Extract from sender/receiver fields
        analysis = extraction.get("document_analysis", {})
        sender_raw = str(analysis.get("sender", {}).get("value", ""))
        receiver_raw = str(analysis.get("receiver", {}).get("value", ""))
        
        for name, val in [("sender", sender_raw), ("receiver", receiver_raw)]:
            if "IBAN" in val or len(val) > 20:
                # Basic TR IBAN pattern
                iban_match = re.search(r"TR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}", val)
                if iban_match:
                    checks.append(ValidationCheck(
                        name=f"iban_format_{name}",
                        status="PASS",
                        details=f"Valid TR IBAN found for {name}"
                    ))
                else:
                    checks.append(ValidationCheck(
                        name=f"iban_format_{name}",
                        status="WARNING",
                        details=f"Possible IBAN detected for {name} but format is non-standard"
                    ))
                    if overall_status != "ERROR": overall_status = "WARNING"

        # 3. OCR Confidence Warning
        # If any major field has confidence < 0.6, flag as WARNING
        root_tx = extraction.get("financial_hierarchy", {}).get("root_transaction", {})
        amount_conf = root_tx.get("amount", {}).get("confidence", 1.0)
        
        if amount_conf < 0.6:
            checks.append(ValidationCheck(
                name="low_confidence_warning",
                status="WARNING",
                details=f"Primary amount has low OCR confidence ({amount_conf:.2f})"
            ))
            if overall_status != "ERROR": overall_status = "WARNING"

        return ValidationReport(status=overall_status, checks=checks)
