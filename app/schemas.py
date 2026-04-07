from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FinancialAdjustment(BaseModel):
    group_name: str
    total_impact: float
    breakdown: Dict[str, float]
    math_status: str

class RootTransaction(BaseModel):
    amount: float
    label: str = "TUTAR / TOTAL"
    text_confirmation: str = "NOT_FOUND"
    is_valid: bool = False

class FinancialHierarchy(BaseModel):
    root_transaction: RootTransaction
    adjustments_and_fees: List[FinancialAdjustment] = []

class DocumentAnalysis(BaseModel):
    type: str
    status: str

class EngineReport(BaseModel):
    trust_score: float
    logic_applied: str = "Hierarchical_Validation_V3"

class ExtractionResult(BaseModel):
    document_analysis: DocumentAnalysis
    financial_hierarchy: FinancialHierarchy
    engine_report: EngineReport

class JobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[ExtractionResult] = None
    error: Optional[str] = None
