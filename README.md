# 🚀 Smart IDP Learning System & Data Intelligence Platform

Bu sistem, standart bir döküman işleme motorunun ötesinde, her türlü kurumsal dökümandan (fatura, dekont, sözleşme, pasaport vb.) veri ayıklayan, bu verilerden kurumsal zekâ üreten ve **kendi hatalarından ders çıkaran** uçtan uca bir platformdur.

---

## 🧠 Platform Vizyonu: Pipeline'dan Zekâya
Sistem, sadece veriyi bir noktadan diğerine taşıyan bir kanal (pipeline) değil; her işlemde beslenen, insan geri bildirimiyle gelişen bir **veri varlığı (data asset)** olarak tasarlanmıştır.

- **Self-Evolution:** Ham OCR metni ile insan doğrulaması arasındaki ilişkiyi dinamik olarak öğrenir.
- **Cross-Doc Intelligence:** Tekil belgelerin ötesinde, tüm doküman havuzundaki trendleri ve anomalileri yakalar.
- **Expert Orchestration:** PaddleOCR'ın yapısal analiz gücü ile Gemini 1.5 Flash'ın semantik zekâsını hibrit bir yapıda birleştirir.

---

## 🛠 Teknik Mimari
- **Hız:** FastAPI & Celery (Asenkron & Ölçeklenebilir)
- **Motor:** PaddleOCR v4 & PP-Structure (Tablo & Layout Analizi)
- **Zekâ:** Google Gemini 1.5 Flash / Yerel Llama 3
- **Hafıza:** SQLite (Sonuçlar) & Redis (Hız Sınırı & Kuyruk)

---

## 🚀 Kurulum ve Çalıştırma Rehberi

### 1. Ön Gereksinimler
- **Python 3.10+**
- **Redis Server** (Kuyruk ve hız sınırlaması için aktif olmalıdır)

### 2. Ortam Hazırlığı
```bash
# Sanal ortam oluşturun
python3 -m venv .venv
source .venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 3. Yapılandırma (.env)
Proje kök dizininde bir `.env` dosyası oluşturun ve şu değerleri tanımlayın:
```env
LLM_API_KEY=your_gemini_api_key
LLM_PROVIDER=GEMINI # veya LOCAL (Ollama için)
LLM_MODEL=gemini-1.5-flash

# Redis Yapılandırması (Kritik)
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Platform Ayarları
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

### 4. Sistemi Başlatma
Sistemin çalışması için üç ayrı terminalde şu servislerin aktif olması gerekir:

#### A. Redis (Prerequisite)
```bash
redis-server
```

#### B. Celery Worker (Zekâ Katmanı)
Worker'ı macOS için optimize edilmiş parametrelerle başlatın:
```bash
PYTHONPATH=. .venv/bin/celery -A app.worker.celery_app worker --loglevel=info --pool=solo --without-gossip --without-mingle --without-heartbeat
```

#### C. API & Dashboard (Gateway)
```bash
.venv/bin/python -m app.main
```

---

## 📖 Kullanıcı Yol Haritası

1.  **Giriş:** Dashboard açıldığında (`http://localhost:8000`) bir API Key tanımlayın veya `/setup-key` endpoint'ini kullanın.
2.  **Analiz:** Bir belge yükleyin. Sistem OCR, Tablo Analizi ve LLM katmanlarını kullanarak veriyi hiyerarşik JSON formatına getirir.
3.  **Doğrulama (Öğrenme):** AI çıktısında bir hata varsa, JSON editöründe düzeltin ve **"Sisteme Öğret"** butonuna basın. Sistem bu düzeltmeyi `raw_text` ile eşitleyerek hafızasına (Dynamic Few-Shot) kaydeder.
4.  **Intelligence:** **Analytics & Learning** sekmesine giderek sistemin verimlilik artışını, otomasyon oranını ve tespit edilen anomalileri izleyin.

---

## 🛡 Güvenlik ve Uyumluluk
- **KVKK:** PII (Kişisel Veri) maskeleme katmanı mevcuttur.
- **Rate Limit:** Kullanıcı başına 20 istek/dakika sınırı ile API güvenliği sağlanır.

## 🆘 Sorun Giderme (Troubleshooting)

### 1. AttributeError: 'DocumentProcessor' object has no attribute 'structure_engine'
Bu hata, PaddleStructure kütüphanesinin tam yüklenemediği durumlarda oluşur. Sistem bu durumu artık güvenli bir şekilde (`None` atayarak) yönetmektedir. Eğer tablo analizi istiyorsanız `paddleocr` ve `paddlepaddle` kütüphanelerinin güncel olduğundan emin olun.

### 2. ValueError: Exception information must include the exception type
Celery ve Redis arasındaki serileştirme (pickle) uyuşmazlığından kaynaklanır. Çözüm için:
- `.env` içinde `CELERY_RESULT_BACKEND` için ayrı bir DB (örneğin `/1`) kullanın.
- Görevlerin mutlaka bir sözlük (dict) döndürdüğünden ve `raise` kullanımının temiz olduğundan emin olun.

---
*Geliştiren: Sueda Akça | Platform Versiyonu: 4.5.1*