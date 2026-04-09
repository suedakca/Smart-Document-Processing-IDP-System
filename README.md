# 🚀 Smart IDP Learning System & Data Intelligence Platform

This system is an end-to-end platform designed to go beyond standard document processing. It extracts high-fidelity data from various corporate documents (invoices, receipts, contracts, passports, etc.), generates **Enterprise Intelligence** from this data, and features a **Self-Learning Loop** that improves accuracy based on user corrections.

---

## 🧠 Platform Vision: From Pipelines to Intelligence

This platform is not just a data transmission channel (pipeline); it is designed as a **Data Asset** that feeds on every transaction and evolves through human-in-the-loop feedback.

- **Self-Evolution:** Dynamically learns the relationship between raw OCR text and human-validated ground truth.
- **Cross-Doc Intelligence:** Goes beyond individual documents to capture trends, spend patterns, and anomalies across the entire document pool.
- **Expert Orchestration:** Combines the structural analysis power of **PaddleOCR v4** with the semantic intelligence of **Google Gemini 1.5 Pro/Flash** in a hybrid architecture.
- **Robust OCR Engine:** Integrated automatic image upscaling and color inversion fallback to handle low-resolution or dark-mode documents (even 34KB highly-compressed files).

---

## 🛠 Technical Stack

- **Framework:** FastAPI & Celery (Asynchronous & Scalable)
- **Engine:** PaddleOCR v4 & PP-StructureV3 (Table & Layout Analysis)
- **Intelligence:** Google Gemini 1.5 Flash / Local Llama 3
- **Storage:** SQLite (Data Store) & Redis (Broker & Rate Limiter)
- **Aesthetics:** Modern Vanilla CSS/JS Dashboard with real-time analytics.

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Python 3.10+**
- **Redis Server** (Required for task orchestration and rate limiting)
- **Tesseract OCR** (Optional fallback)

### 2. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration (.env)
Create a `.env` file in the root directory (refer to `.env.example`):
```env
LLM_API_KEY=your_gemini_api_key
LLM_PROVIDER=GEMINI # or LOCAL for Ollama
LLM_MODEL=gemini-1.5-flash

# Redis Configuration (Critical)
REDIS_URL=redis://localhost:6379/1
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Platform Settings
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

### 4. Running the Platform
For full functionality, ensure the following services are running in separate terminals:

#### A. Redis (Prerequisite)
```bash
redis-server
```

#### B. Celery Worker (Intelligence Layer)
Start the worker with settings optimized for macOS and low-latency processing:
```bash
PYTHONPATH=. .venv/bin/celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

#### C. API & Dashboard (Gateway)
```bash
.venv/bin/python -m app.main
```

---

## 📖 User Roadmap

1.  **Ingestion:** Open the Dashboard (`http://localhost:8000`) and upload a document.
2.  **Robust Processing:** Even if the document is low-resolution, the system will upscale and analyze it using the hybrid OCR/Structure engine.
3.  **Correction (Learning):** If there is an error in the AI output, correct it in the JSON editor and click **"Teach System."** The platform will save this as a "Ground Truth" example for future Dynamic Few-Shot learning.
4.  **Intelligence Tab:** Visit the **Analytics & Learning** tab to monitor efficiency gains, automation rates, and detected spending anomalies.

---

## 🛡 Security & Compliance

- **GDPR/KVKK:** Built-in PII (Personally Identifiable Information) masking layer.
- **Rate Limiting:** Protects the API with a 20 requests/minute limit per API key.
- **Environment Stability:** Automatic thread-pool management ensures reliable execution on macOS Silicon.

---

## 🆘 Troubleshooting

### 1. "No text extracted from document"
This platform now includes **Robust OCR**. If a file is too small or low-res, it is automatically upscaled. If extraction still fails, it may be due to extreme blur. Check `logs/` forPass 1/Pass 2 results.

### 2. Module Import Hangs (macOS)
If the system freezes during `import numpy`, ensure you are using the latest entry points (`main.py` or `worker.py`) which include the environment fix: `os.environ["OPENBLAS_NUM_THREADS"] = "1"`.

---
*Developed by: Sueda Akça | Platform Version: 5.0.0 (Robust Edition)*