from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import uuid
from fastapi.responses import HTMLResponse
from loguru import logger
from .db_client import DatabaseClient
from .worker import process_document_task
from .schemas import JobStatus, ExtractionResult
from celery.result import AsyncResult
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart IDP System", description="High-Performance Asynchronous IDP API")
db = DatabaseClient()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/history")
async def get_history(limit: int = 10):
    try:
        return db.get_history(limit=limit)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", status_code=202)
async def process_document(file: UploadFile = File(...), mask_pii: bool = False):
    """
    Triggers an asynchronous document processing task.
    Returns 202 Accepted with a job_id.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".png", ".jpg", ".jpeg", ".pdf"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF or Image.")

    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Trigger Background Task
        # Note: Celery will handle the processing and DB storage
        job = process_document_task.delay(temp_path, file.filename, mask_pii=mask_pii)
        
        logger.info(f"Task {job.id} submitted for {file.filename} (KVKK: {mask_pii})")
        return {"job_id": job.id, "status": "PENDING"}
        
    except Exception as e:
        logger.exception(f"Initial processing failed: {str(e)}")
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail="Task submission failed.")

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """
    Returns the current status of a processing job.
    """
    try:
        job = AsyncResult(job_id)
        
        if job.ready():
            if job.successful():
                return {
                    "job_id": job_id,
                    "status": "SUCCESS",
                    "result": job.result
                }
            else:
                return {
                    "job_id": job_id,
                    "status": "FAILURE",
                    "error": str(job.result) if job.result else "Unknown task error"
                }
        else:
            return {
                "job_id": job_id,
                "status": job.status,
                "result": None
            }
    except Exception as e:
        logger.error(f"Error checking job {job_id}: {str(e)}")
        raise HTTPException(status_code=404, detail="Job ID not found or server error.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
