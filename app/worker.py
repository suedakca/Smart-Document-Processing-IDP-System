import os
from celery import Celery
from celery.schedules import crontab
from loguru import logger
from .processor import DocumentProcessor
from .llm_utils import LLMHybridLayer
from .classifier import DocumentClassifier
from .postprocessing import DataExtractor
from .db_client import DatabaseClient
from .file_utils import FileHandler
import asyncio

# Celery Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "idp_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Istanbul",
    enable_utc=True,
)

# Initialize Processors
doc_processor = DocumentProcessor()
classifier = DocumentClassifier()
llm_layer = LLMHybridLayer()
data_extractor = DataExtractor()
db = DatabaseClient()

@celery_app.task(name="process_document_task", bind=True)
def process_document_task(self, file_path, original_filename):
    """
    Background task to process a document (Image or PDF).
    """
    logger.info(f"Starting task {self.request.id} for {original_filename}")
    temp_images = []
    
    try:
        # 1. Convert to images if PDF
        if FileHandler.is_pdf(file_path):
            temp_images = FileHandler.pdf_to_images(file_path, os.path.dirname(file_path))
        else:
            temp_images = [file_path]
            
        # 2. OCR Processing
        ocr_results = doc_processor.process(temp_images)
        if not ocr_results:
            raise ValueError("No text extracted from document")
            
        raw_text_list = [d["text"] for d in ocr_results]
        
        # 3. Classification
        category, _ = classifier.classify(ocr_results)
        
        # 4. LLM & Data Extraction (Running async code in sync worker)
        # Using a new event loop for the async calls in this thread
        loop = asyncio.get_event_loop()
        llm_results = loop.run_until_complete(llm_layer.extract_dynamic_json(raw_text_list))
        
        extracted_data = loop.run_until_complete(data_extractor.extract(
            ocr_results,
            category=category,
            llm_data=llm_results,
            llm_layer=llm_layer
        ))
        
        # 5. Save to DB
        meta = extracted_data.get("document_analysis", {})
        report = extracted_data.get("engine_report", {})
        
        db.save_result(
            filename=original_filename,
            doc_type=meta.get("type", "UNKNOWN"),
            trust_score=report.get("trust_score", 0.0),
            result_dict=extracted_data
        )
        
        # 6. Cleanup
        FileHandler.cleanup([file_path] + (temp_images if FileHandler.is_pdf(file_path) else []))
        
        logger.info(f"Task {self.request.id} completed successfully")
        return extracted_data

    except Exception as e:
        logger.exception(f"Task {self.request.id} failed: {str(e)}")
        FileHandler.cleanup([file_path] + temp_images)
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise e

@celery_app.task(name="cleanup_uploads_task")
def cleanup_uploads_task():
    """
    Periodic task to clean up old files in the uploads directory.
    """
    import time
    upload_dir = "uploads"
    now = time.time()
    count = 0
    
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            f_path = os.path.join(upload_dir, f)
            # Remove files older than 24 hours
            if os.stat(f_path).st_mtime < now - 86400:
                try:
                    os.remove(f_path)
                    count += 1
                except:
                    pass
    logger.info(f"Cleanup completed. Removed {count} files.")

# Periodic Task Schedule
celery_app.conf.beat_schedule = {
    "cleanup-every-night": {
        "task": "cleanup_uploads_task",
        "schedule": crontab(hour=3, minute=0), # 3:00 AM
    },
}
