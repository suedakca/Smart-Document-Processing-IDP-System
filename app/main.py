from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Query
import shutil
import os
import uuid
from fastapi.responses import HTMLResponse, FileResponse
from loguru import logger
from .db_client import DatabaseClient
from .worker import process_document_task
from .schemas import JobStatus, ExtractionResult
from .auth import get_api_key
from .exporters import DocumentExporter
from celery.result import AsyncResult
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart IDP System", description="Corporate Asynchronous IDP API")
db = DatabaseClient()
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
async def get_history(limit: int = 10, key_id: int = Depends(get_api_key)):
    try:
        return db.get_history(limit=limit)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics(key_id: int = Depends(get_api_key)):
    """Returns system-wide stats for the Dashboard."""
    return db.get_stats()

@app.post("/process", status_code=202)
async def process_document(
    file: UploadFile = File(...), 
    mask_pii: bool = False, 
    key_id: int = Depends(get_api_key)
):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".png", ".jpg", ".jpeg", ".pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Pass key_id to track usage
        job = process_document_task.delay(temp_path, file.filename, mask_pii=mask_pii, key_id=key_id)
        return {"job_id": job.id, "status": "PENDING"}
        
    except Exception as e:
        logger.exception(f"Processing failed: {str(e)}")
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail="Task submission failed.")

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str, key_id: int = Depends(get_api_key)):
    try:
        job = AsyncResult(job_id)
        if job.ready():
            return {"job_id": job_id, "status": "SUCCESS" if job.successful() else "FAILURE", "result": job.result}
        return {"job_id": job_id, "status": job.status, "result": None}
    except Exception as e:
        raise HTTPException(status_code=404, detail="Job not found.")

@app.get("/export/{job_id}")
async def export_document(job_id: str, format: str = Query("csv", enum=["csv", "ubl"]), key_id: int = Depends(get_api_key)):
    """Exports processed data to CSV or UBL-TR XML."""
    job = AsyncResult(job_id)
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
async def submit_correction(job_id: str, corrected_data: dict, key_id: int = Depends(get_api_key)):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
