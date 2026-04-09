from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, TypeVar, Generic

T = TypeVar('T')

class ValidatedField(BaseModel, Generic[T]):
    """Enterprise-grade field with full audit trail."""
    value: T
    confidence: float = 1.0
    is_valid: bool = True
    review_required: bool = False
    source: str = "ocr"  # e.g., "ocr", "llm", "rule_based"
    evidence_text: Optional[str] = None
    reason_if_flagged: Optional[str] = None

class ValidationCheck(BaseModel):
    name: str
    status: str  # PASS, FAIL, WARNING
    details: str

class ValidationReport(BaseModel):
    status: str  # SUCCESS, WARNING, ERROR
    checks: List[ValidationCheck] = []

class FinancialAdjustment(BaseModel):
    group_name: ValidatedField[str]
    total_impact: ValidatedField[float]
    breakdown: Dict[str, float]
    math_status: str

class RootTransaction(BaseModel):
    amount: ValidatedField[float]
    label: ValidatedField[str]
    text_confirmation: ValidatedField[str]
    is_valid: bool = False

class FinancialHierarchy(BaseModel):
    root_transaction: RootTransaction
    adjustments_and_fees: List[FinancialAdjustment] = []

class DocumentAnalysis(BaseModel):
    type: ValidatedField[str]
    status: ValidatedField[str]
    sender: Optional[ValidatedField[str]] = None
    receiver: Optional[ValidatedField[str]] = None
    description: Optional[ValidatedField[str]] = None
    currency: Optional[ValidatedField[str]] = None
    transaction_id: Optional[ValidatedField[str]] = None
    transaction_date: Optional[ValidatedField[str]] = None
    sender_iban: Optional[ValidatedField[str]] = None
    receiver_iban: Optional[ValidatedField[str]] = None

class EngineReport(BaseModel):
    trust_score: float
    logic_applied: str = "Hierarchical_Validation_V3"

class DecisionEngineResult(BaseModel):
    """Result from the deterministic rule-based decision layer."""
    recommended_action: str  # AUTO_APPROVE, REVIEW_REQUIRED, REJECT, ESCALATE, AUTO_BOOK
    risk_score: float
    requires_human_review: bool
    auto_process_allowed: bool
    decision_reason: List[str]

class ExtractionResult(BaseModel):
    document_analysis: Dict[str, Any]
    financial_hierarchy: Dict[str, Any]
    engine_report: Dict[str, Any]
    validation_report: Optional[ValidationReport] = None
    decision_engine: Optional[DecisionEngineResult] = None

class DetailedClassificationResult(BaseModel):
    domain: str = ""
    document_family: str = ""
    document_type: str = ""
    transaction_type: str = ""
    confidence: float = 0.0
    classification_signals: List[str] = []

class StageReport(BaseModel):
    status: str
    details: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

class PipelineResult(BaseModel):
    status: str  # SUCCESS, FAILED, PARTIAL_FAILURE, REVIEW_REQUIRED, APPROVED
    overall_trust_score: float
    stages: Dict[str, StageReport]
    data: Optional[ExtractionResult] = None
    full_raw_text: Optional[str] = None
    processing_time: float
    validation_report: Optional[ValidationReport] = None
    decision_engine: Optional[DecisionEngineResult] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    current_stage: str = "QUEUED"
    last_stage: Optional[str] = None
    message: Optional[str] = None
    result: Optional[PipelineResult] = None
    error: Optional[str] = None
