import os
# Prevent Numpy/OpenCV from hanging on macOS thread pool initialization
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Query
import shutil
import os
import uuid
from fastapi.responses import HTMLResponse, FileResponse
from loguru import logger
from .db_client import DatabaseClient
from .worker import celery_app, process_document_v2
from .schemas import JobStatus, ExtractionResult
from .auth import get_api_key, check_rate_limit
from .exporters import DocumentExporter
from .intelligence import IntelligenceEngine
from celery.result import AsyncResult

app = FastAPI(title="Smart IDP System", description="Corporate Learning IDP Platform")
db = DatabaseClient()
intel = IntelligenceEngine()
UPLOAD_DIR = "uploads"
EXPORT_DIR = "exports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/setup-key")
async def setup_key(name: str = "Admin"):
    """Temporary helper to generate an API Key for testing."""
    key = db.create_api_key(name, "Management_Key")
    return {"msg": "SAVE THIS KEY. It won't be shown again.", "api_key": key}

@app.get("/history")
async def get_history(limit: int = 10):
    try:
        return db.get_history(limit=limit)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    logger.info("--- IDP Platform Enterprise Health Check ---")
    from app.processor import DocumentProcessor
    from app.llm_utils import LLMHybridLayer
    
    proc = DocumentProcessor()
    llm = LLMHybridLayer()
    
    ocr_status = "ONLINE" if proc.ocr else "OFFLINE"
    struct_status = "ONLINE (V3)" if proc.structure_engine else "OFFLINE (Layout Analysis limited)"
    
    # Active LLM Probe
    logger.info("[LLM] Performing active probe...")
    llm_report = llm.probe_model()
    llm_status = llm_report["status"]
    llm_msg = llm_report["msg"]
    
    logger.info(f"OCR Engine: {ocr_status}")
    logger.info(f"Layout Engine: {struct_status}")
    logger.info(f"LLM Engine: {llm_status} ({llm_msg})")
    
    if "OFFLINE" in [ocr_status, llm_status]:
        logger.critical("CRITICAL DEPENDENCY MISSING. SYSTEM OPERATING IN DEGRADED MODE.")
    
    logger.info("--------------------------------------------")

@app.get("/analytics")
async def get_analytics():
    """Returns system-wide stats for the Dashboard."""
    return db.get_stats()

@app.post("/process", status_code=202)
async def process_document(
    file: UploadFile = File(...), 
    mask_pii: bool = False
):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".png", ".jpg", ".jpeg", ".pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Pass None as key_id
        job = process_document_v2.delay(temp_path, file.filename, mask_pii=mask_pii, key_id=None)
        return {"job_id": job.id, "status": "PENDING"}
        
    except Exception as e:
        logger.exception(f"Processing failed: {str(e)}")
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail="Task submission failed.")

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """
    Check status of an extraction job.
    """
    try:
        res = AsyncResult(job_id, app=celery_app)
        state = res.state
        
        result_data = None
        error_msg = None
        
        if state == "SUCCESS":
            try:
                result_data = res.result
            except Exception as e:
                logger.error(f"Failed to decode SUCCESS result: {str(e)}")
                state = "ERROR"
                error_msg = f"Decoding error: {str(e)}"
        elif state == "FAILURE":
            # Attempt to get the error message safely
            try:
                # result on a failed task often contains the string representation of the exception
                error_msg = str(res.result)
            except:
                error_msg = "Processing failed. Check worker logs for details."
            
        return {
            "job_id": job_id, 
            "status": state, 
            "result": result_data,
            "error": error_msg
        }
        
    except Exception as e:
        logger.error(f"Status check failed for {job_id}: {str(e)}")
        return {"job_id": job_id, "status": "PENDING", "msg": "Job state unknown or pending"}

@app.get("/export/{job_id}")
async def export_document(job_id: str, format: str = Query("csv", enum=["csv", "ubl"])):
    """Exports processed data to CSV or UBL-TR XML."""
    job = AsyncResult(job_id, app=celery_app)
    if not job.ready() or not job.successful():
        raise HTTPException(status_code=400, detail="Job not ready or failed.")
    
    data = job.result
    out_file = os.path.join(EXPORT_DIR, f"export_{job_id}.{format if format != 'ubl' else 'xml'}")
    
    if format == "csv":
        DocumentExporter.to_csv(data, out_file)
    else:
        DocumentExporter.to_ubl_tr(data, out_file)
        
    return FileResponse(out_file, filename=os.path.basename(out_file))

@app.post("/correct/{job_id}")
async def submit_correction(job_id: str, corrected_data: dict):
    """
    Accepts human-corrected extraction data to build a learning loop.
    This data is used as the 'Ground Truth' for future dynamic few-shot learning.
    """
    try:
        db.save_correction(job_id, corrected_data)
        logger.info(f"Human correction received for job {job_id}. Self-learning data updated.")
        return {"status": "SUCCESS", "msg": "Correction saved. The system will learn from this."}
    except Exception as e:
        logger.error(f"Failed to save correction for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not save correction.")

@app.get("/platform/intelligence")
async def get_platform_intelligence():
    """Returns high-level business intelligence from the platform."""
    return intel.get_platform_insights()

@app.get("/platform/anomalies")
async def get_platform_anomalies(limit: int = 5):
    """Returns detected anomalies across all extractions."""
    return intel.detect_anomalies(limit=limit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
