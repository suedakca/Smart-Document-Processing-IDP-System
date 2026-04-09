from typing import List, Dict, Any, Tuple
from ..schemas import DecisionEngineResult, PipelineResult

# Configuration Thresholds
AUTO_APPROVE_MIN_TRUST = 0.90
REVIEW_MIN_TRUST = 0.60
HIGH_VALUE_THRESHOLD = 100000.0 # Escalation trigger

class DecisionEngine:
    """
    Deterministic enterprise decision engine.
    Evaluates extraction results against business rules to recommend next actions.
    """
    
    @staticmethod
    def evaluate_decision(pipeline_result: Dict[str, Any]) -> DecisionEngineResult:
        reasons = []
        action = "REVIEW_REQUIRED" # Default conservative action
        risk_score = 0.5 # Default middle risk
        
        status = pipeline_result.get("status")
        trust_score = pipeline_result.get("overall_trust_score", 0.0)
        stages = pipeline_result.get("stages", {})
        data = pipeline_result.get("data", {})
        val_report = pipeline_result.get("validation_report", {})
        
        # 1. Critical Field Analysis
        analysis = data.get("document_analysis", {})
        fin_h = data.get("financial_hierarchy", {})
        root = fin_h.get("root_transaction", {})
        
        amount = root.get("amount", {}).get("value", 0.0)
        sender = analysis.get("sender", {}).get("value")
        receiver = analysis.get("receiver", {}).get("value")
        
        # --- RULE SET A: REJECT ---
        if status == "FAILED" or stages.get("ocr", {}).get("status") == "FAILED":
            reasons.append("OCR stage failed or document status is FAILED")
            return DecisionEngineResult(
                recommended_action="REJECT",
                risk_score=1.0,
                requires_human_review=True,
                auto_process_allowed=False,
                decision_reason=reasons
            )
            
        if not amount or amount <= 0:
            reasons.append("Root transaction amount is missing or invalid")
            return DecisionEngineResult(
                recommended_action="REJECT",
                risk_score=0.95,
                requires_human_review=True,
                auto_process_allowed=False,
                decision_reason=reasons
            )

        # --- RULE SET B: ESCALATE ---
        is_escalated = False
        if amount >= HIGH_VALUE_THRESHOLD:
            reasons.append(f"Transaction amount ({amount}) exceeds escalation threshold")
            is_escalated = True
            
        if val_report.get("status") == "ERROR":
            reasons.append("Severe mathematical or logical mismatch detected in validation")
            is_escalated = True
            
        if not sender and not receiver:
            reasons.append("Both sender and receiver are missing from extraction")
            is_escalated = True

        if is_escalated:
            return DecisionEngineResult(
                recommended_action="ESCALATE",
                risk_score=0.85,
                requires_human_review=True,
                auto_process_allowed=False,
                decision_reason=reasons
            )

        # --- RULE SET C: AUTO_APPROVE / AUTO_BOOK ---
        is_perfect = True
        
        if trust_score < AUTO_APPROVE_MIN_TRUST:
            reasons.append(f"Trust score ({trust_score:.2f}) is below auto-approval threshold")
            is_perfect = False
            
        if val_report.get("status") != "SUCCESS":
            reasons.append(f"Validation report status is {val_report.get('status')}")
            is_perfect = False
            
        # Check for field-level flags
        flagged_fields = 0
        # Check analysis fields
        for field_name, field_obj in analysis.items():
            if isinstance(field_obj, dict) and field_obj.get("review_required"):
                flagged_fields += 1
                
        if flagged_fields > 0:
            reasons.append(f"Extraction contains {flagged_fields} flagged fields requiring review")
            is_perfect = False

        if is_perfect:
            reasons.append("High trust score and clean validation check")
            # Determine if it can be AUTO_BOOKED (Ready for downstream systems)
            can_book = amount > 0 and bool(sender) and bool(receiver)
            action = "AUTO_BOOK" if can_book else "AUTO_APPROVE"
            risk_score = 1.0 - trust_score
            
            return DecisionEngineResult(
                recommended_action=action,
                risk_score=round(risk_score, 4),
                requires_human_review=False,
                auto_process_allowed=True,
                decision_reason=reasons
            )

        # --- RULE SET D: REVIEW_REQUIRED (Fallback) ---
        reasons.append("Document requires manual verification due to warnings or moderate confidence")
        return DecisionEngineResult(
            recommended_action="REVIEW_REQUIRED",
            risk_score=round(0.4 + (1.0 - trust_score) * 0.4, 4),
            requires_human_review=True,
            auto_process_allowed=False,
            decision_reason=reasons
        )
