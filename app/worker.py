from dotenv import load_dotenv
load_dotenv()
import os
from celery import Celery
from celery.schedules import crontab
from loguru import logger
from .processor import DocumentProcessor
from .llm_utils import LLMHybridLayer
from .classifier import DocumentClassifier
from .postprocessing import DataExtractor
from . import celery_config
from .db_client import DatabaseClient
from .file_utils import FileHandler
import asyncio

# Initialize Celery with standardized config
celery_app = Celery("worker")
celery_app.config_from_object(celery_config)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Europe/Istanbul",
    enable_utc=True,
)

# Initialize Processors safely
try:
    logger.info("Initializing worker components...")
    doc_processor = DocumentProcessor()
    classifier = DocumentClassifier()
    llm_layer = LLMHybridLayer()
    data_extractor = DataExtractor()
    logger.info("Worker components initialized successfully.")
except Exception as e:
    logger.critical(f"FATAL ERROR DURING WORKER INITIALIZATION: {str(e)}")
    import traceback
    logger.critical(traceback.format_exc())
    # Forcing exit to avoid confusing Celery errors
    import sys
    sys.exit(1)
db = DatabaseClient()

@celery_app.task(name="process_document_v2", bind=True)
def process_document_v2(self, file_path, original_filename, mask_pii=False, key_id=None):
    """
    Background task to process a document (Image or PDF).
    """
    import time
    start_time = time.time()
    logger.info(f"[STEP 0] Task {self.request.id} started for {original_filename}")
    temp_images = []
    
    try:
        # 1. Convert to images if PDF
        logger.info(f"[STEP 1] File check: {file_path}")
        if FileHandler.is_pdf(file_path):
            logger.info("Converting PDF to images...")
            temp_images = FileHandler.pdf_to_images(file_path, os.path.dirname(file_path))
        else:
            temp_images = [file_path]
            
        # 2. Processor (OCR + TABLE)
        logger.info(f"[STEP 2] Starting OCR Processor on {len(temp_images)} page(s)...")
        ocr_results, table_md = doc_processor.process(temp_images)
        
        if not ocr_results:
            logger.warning("[STEP 2] OCR failed: No text extracted.")
            raise ValueError("No text extracted from document")
            
        logger.info(f"[STEP 2] OCR completed. Extracted {len(ocr_results)} lines.")
        raw_text_list = [d["text"] for d in ocr_results]
        
        # 3. Classification
        logger.info("[STEP 3] Starting classification...")
        category, _ = classifier.classify(ocr_results)
        logger.info(f"[STEP 3] Document classified as: {category}")
        
        # 4. LLM & Data Extraction
        logger.info("[STEP 4] Starting LLM Data Extraction...")
        
        # Safer asyncio handling for Celery workers
        async def run_extraction():
            llm_res = await llm_layer.extract_dynamic_json(
                raw_text_list, 
                table_markdown=table_md, 
                category=category, 
                mask_pii=mask_pii
            )
            extracted = await data_extractor.extract(
                ocr_results,
                category=category,
                llm_data=llm_res,
                llm_layer=llm_layer
            )
            return extracted

        try:
            extracted_data = asyncio.run(run_extraction())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            extracted_data = loop.run_until_complete(run_extraction())

        logger.info("[STEP 4] LLM Extraction completed.")
        
        # 5. Save to DB with extra metrics
        duration = time.time() - start_time
        meta = extracted_data.get("document_analysis", {})
        report = extracted_data.get("engine_report", {})
        
        db.save_result(
            filename=original_filename,
            doc_type=meta.get("type", "UNKNOWN"),
            trust_score=report.get("trust_score", 0.0),
            result_dict=extracted_data,
            processing_time=duration,
            key_id=key_id,
            raw_text=full_text  # Save source for learning loop
        )
        
        # 6. Cleanup
        FileHandler.cleanup([file_path] + (temp_images if FileHandler.is_pdf(file_path) else []))
        logger.info(f"[SUCCESS] Task {self.request.id} completed in {duration:.2f}s")
        return extracted_data

    except Exception as e:
        logger.error(f"FATAL ERROR in task {self.request.id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Cleanup if we have temp images
        if temp_images:
            FileHandler.cleanup(temp_images if FileHandler.is_pdf(file_path) else [])
            
        # Re-raise to let Celery handle the failure state naturally with the original error
        raise

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
