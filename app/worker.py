import os
# Prevent Numpy/OpenCV from hanging on macOS thread pool initialization
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv
load_dotenv()
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
    Enterprise-Grade pipeline with honest status propagation and weighted trust.
    """
    import time
    start_time = time.time()
    logger.info(f"[PIPELINE] Task {self.request.id} started for {original_filename}")
    
    # Initiative Global Pipeline State
    stages = {
        "file_processing": {"status": "NOT_STARTED"},
        "ocr": {"status": "NOT_STARTED"},
        "classification": {"status": "NOT_STARTED"},
        "llm_extraction": {"status": "NOT_STARTED"}
    }
    pipeline_status = "SUCCESS"
    temp_images = []
    
    try:
        # 1. File Handling
        stages["file_processing"]["status"] = "IN_PROGRESS"
        if FileHandler.is_pdf(file_path):
            temp_images = FileHandler.pdf_to_images(file_path, os.path.dirname(file_path))
        else:
            temp_images = [file_path]
        stages["file_processing"]["status"] = "SUCCESS"
            
        # 2. Processor (OCR + TABLE)
        stages["ocr"]["status"] = "IN_PROGRESS"
        proc_report = doc_processor.process(temp_images)
        metrics = proc_report["metrics"]
        ocr_results = proc_report["all_ocr_results"]
        table_md = proc_report["all_table_markdown"]
        
        stages["ocr"]["metrics"] = metrics
        if metrics["total_blocks"] == 0:
            logger.error(f"[STAGE: OCR] FAILED - Zero blocks across {metrics['page_count']} pages.")
            stages["ocr"]["status"] = "FAILED"
            stages["ocr"]["details"] = "Document appears blank or unreadable after 3-tier processing."
            
            # FAIL FAST
            final_res = {
                "status": "FAILED",
                "overall_trust_score": 0.0,
                "stages": stages,
                "processing_time": time.time() - start_time
            }
            FileHandler.cleanup([file_path] + temp_images)
            return final_res
        
        stages["ocr"]["status"] = "SUCCESS"
        raw_text_list = [d["text"] for d in ocr_results]
        mean_ocr_conf = sum([d["confidence"] for d in ocr_results]) / len(ocr_results)
            
        # 3. Classification
        stages["classification"]["status"] = "IN_PROGRESS"
        category, _ = classifier.classify(ocr_results)
        stages["classification"]["status"] = "SUCCESS" if category != "UNKNOWN" else "PARTIAL_SUCCESS"
        stages["classification"]["details"] = f"Detected Type: {category}"
        
        # 4. LLM & Data Extraction
        stages["llm_extraction"]["status"] = "IN_PROGRESS"
        
        async def run_extraction():
            llm_res = await llm_layer.extract_dynamic_json(
                raw_text_list, 
                table_markdown="\n".join(table_md), 
                category=category, 
                mask_pii=mask_pii
            )
            if llm_res.get("status") == "ERROR":
                return llm_res, None
            
            extracted = await data_extractor.extract(
                ocr_results,
                category=category,
                llm_data=llm_res,
                llm_layer=llm_layer
            )
            return llm_res, extracted

        try:
            llm_res, extracted_data = asyncio.run(run_extraction())
        except Exception as e:
            logger.error(f"Async loop error: {str(e)}")
            llm_res = {"status": "ERROR", "msg": str(e)}
            extracted_data = None

        llm_success_bit = 1.0
        if llm_res.get("status") == "ERROR":
            stages["llm_extraction"]["status"] = "FAILED"
            stages["llm_extraction"]["details"] = llm_res.get("msg")
            pipeline_status = "PARTIAL_FAILURE"
            llm_success_bit = 0.0
        else:
            stages["llm_extraction"]["status"] = "SUCCESS"

        # 5. Weighted Trust & Final State logic
        overall_trust = (mean_ocr_conf * 0.7) + (llm_success_bit * 0.3)
        if overall_trust < 0.4:
            logger.warning(f"[PIPELINE] Trust score {overall_trust:.2f} too low. Degrading to PARTIAL_FAILURE.")
            pipeline_status = "PARTIAL_FAILURE"

        duration = time.time() - start_time
        
        # Construct Final Result Object
        final_pipeline_result = {
            "status": pipeline_status,
            "overall_trust_score": round(overall_trust, 4),
            "stages": stages,
            "data": extracted_data,
            "processing_time": round(duration, 2)
        }

        # 6. Database Persistence
        db.save_result(
            filename=original_filename,
            doc_type=category,
            trust_score=overall_trust,
            result_dict=final_pipeline_result,
            processing_time=duration,
            key_id=key_id,
            raw_text=" ".join(raw_text_list)
        )
        
        # 7. Final Summary & Cleanup
        FileHandler.cleanup([file_path] + temp_images)
        
        # Log Concise Structured Summary
        ocr_dur = final_pipeline_result["stages"]["ocr"].get("metrics", {}).get("total_ocr_duration", 0)
        total_blocks = final_pipeline_result["stages"]["ocr"].get("metrics", {}).get("total_blocks", 0)
        passes_stats = final_pipeline_result["stages"]["ocr"].get("metrics", {}).get("pass_results", [])
        avg_passes = sum([len(p["results"]) for p in passes_stats]) / len(passes_stats) if passes_stats else 0
        
        logger.info(f"--- [TASK SUMMARY] {original_filename} ---")
        logger.info(f"Status: {pipeline_status} | Trust: {overall_trust:.2f}")
        logger.info(f"OCR: {total_blocks} blocks | Duration: {ocr_dur:.2f}s | Avg Passes: {avg_passes:.1f}")
        logger.info(f"Total Processing: {duration:.2f}s")
        logger.info(f"------------------------------------------")
        
        return final_pipeline_result

    except Exception as e:
        logger.exception(f"CRITICAL ERROR in pipeline: {str(e)}")
        FileHandler.cleanup([file_path] + temp_images)
        return {
            "status": "FAILED",
            "overall_trust_score": 0.0,
            "stages": stages,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

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
