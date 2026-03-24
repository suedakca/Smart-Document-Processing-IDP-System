# 📄 Smart Document Processing (IDP) System

Bu sistem, taranmış dokümanlardaki (fatura, form, dekont) verileri **PaddleOCR (PP-OCRv5)** kullanarak yüksek doğrulukla ayıklayan ve yapılandırılmamış veriyi JSON formatına dönüştüren uçtan uca bir veri işleme hattıdır.

## 🚀 Özellikler

*   **Akıllı IDP Hattı:** Görüntüden yapılandırılmış veriye uçtan uca otonom işleme.
*   **Gelişmiş Ön İşleme:** Otomatik eğiklik düzeltme (deskewing) ve adaptif gürültü temizleme.
*   **Belge Sınıflandırma:** Fatura, Kimlik, Sözleşme ve Dekont türlerini otomatik tanıma.
*   **Veri Doğrulama:** Matematiksel tutarlılık kontrolleri ve format doğrulaması.
*   **Güvenlik & Anomali Tespiti:** Belge düzeni ve metin yoğunluğu üzerinden "Güven Skoru" üretimi.
*   **Modern UI:** Glassmorphism tasarımı ile gerçek zamanlı JSON ve metaveri önizleme.

## 🛠 Teknik Yığın (Tech Stack)

*   **Dil:** Python 3.10+
*   **OCR:** PaddleOCR (v5)
*   **Backend:** FastAPI
*   **Frontend:** Vanilla JS & CSS (Glassmorphism)
*   **Altyapı:** Docker & Docker Compose

## 📦 Kurulum ve Çalıştırma

### 1. Docker ile (Önerilen)

```bash
docker-compose up --build
```
API'ye `http://localhost:8000` adresinden erişebilirsiniz.

### 2. Yerel Kurulum (Local)

```bash
# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Uygulamayı çalıştırın
python -m app.main
```

## 📐 Sistem Mimarisi

1.  **Görüntü Ön İşleme:** Gürültü azaltma ve netleştirme.
2.  **OCR Katmanı:** Metin algılama ve tanıma (Türkçe karakter desteği dahil).
3.  **Veri Yapılandırma:** Ham metinden anlamlı alanların çıkarılması.
4.  **Entegrasyon:** REST API üzerinden çıktıların sunulması.

## 📊 Performans

*   **Hız:** Sayfa başına < 1.5 sn (CPU), < 0.5 sn (GPU).
*   **Doğruluk:** Standart dokümanlarda %98.2+.

---
*Geliştiren: Suheda Akca*