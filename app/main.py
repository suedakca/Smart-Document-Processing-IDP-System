from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import uuid
from fastapi.responses import HTMLResponse
import logging
from .preprocessing import ImagePreprocessor
from .processor import DocumentProcessor
from .postprocessing import DataExtractor
from .classifier import DocumentClassifier
from dotenv import load_dotenv
load_dotenv() 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart IDP System", description="Intelligent Document Processing API")

# Initialize processors
preprocessor = ImagePreprocessor()
doc_processor = DocumentProcessor()
data_extractor = DataExtractor()
classifier = DocumentClassifier()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    # 1. Save uploaded file
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Preprocess (Optional: PaddleOCR handles raw pretty well, but we can denoise)
        # processed_image_path = preprocessor.process(temp_path)
        
        # 3. Perform OCR
        ocr_results = doc_processor.process(temp_path)
        
        # 4. Classify Document
        doc_type, doc_conf = classifier.classify(ocr_results)
        
        # 5. Extract and Validate Structured Data
        extracted_data = data_extractor.extract(ocr_results, doc_type=doc_type)
        
        # 6. Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "metadata": {
                "document_type": doc_type,
                "classification_confidence": doc_conf,
                "file_id": file_id
            },
            "extraction": extracted_data
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
