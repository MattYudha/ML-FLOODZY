# 🌊 Floodzy - Real-time Pendeteksi/Prediksi  dan Potensi Banjir & Monitoring Cuaca di Indonesia

<div align="center">
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/XGBoost-0066B2?style=for-the-badge&logo=xgboost" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker" alt="Docker"/>
</p>




Floodzy-ML: Sistem Prediksi Banjir Berbasis Machine Learning
Versi Proyek: 0.3.0

📝 Deskripsi Proyek
Floodzy-ML adalah sebuah sistem end-to-end yang dirancang untuk memprediksi potensi dan ketinggian banjir menggunakan pendekatan machine learning. Proyek ini dibangun dengan fokus pada kualitas produksi, mencakup validasi domain yang ketat, strategi splitting data yang time-aware dan spatio-temporal, serta alur kerja CI/CD untuk pengujian dan deployment otomatis.

Sistem ini diekspos melalui REST API yang dibangun menggunakan FastAPI dan siap untuk di-deploy sebagai kontainer Docker yang minimalis dan efisien.

✨ Fitur Utama
API Cepat dan Andal: Dibangun di atas FastAPI, menyediakan latensi rendah dan dokumentasi otomatis.

Tiga Model Prediktif:

Potensi Banjir (Regresi Logistik): Memberikan probabilitas dan label risiko (RENDAH, SEDANG, TINGGI).

Ketinggian Banjir (Random Forest): Memprediksi ketinggian air dalam sentimeter.

Potensi Banjir (XGBoost): Model alternatif dengan performa tinggi untuk klasifikasi potensi banjir.

Keamanan: Dukungan untuk autentikasi berbasis API key.

Validasi Input (Feature Contract): Menerapkan batasan pada fitur masukan untuk mencegah data anomali dan meningkatkan stabilitas model.

MLOps Terintegrasi:

CI/CD dengan GitHub Actions: Otomatisasi pengujian, analisis coverage, dan build Docker image.

Monitoring Otomatis: Deteksi data drift dan prediction stability secara berkala untuk memastikan model tetap relevan.

Deployment Siap Produksi: Dikemas dalam Docker dengan multi-stage build untuk ukuran image yang optimal.

🏗️ Struktur Proyek
Struktur direktori dirancang untuk memisahkan antara logika API, kode machine learning, dan komponen pendukung lainnya.
```
floodzy-ml/
├── api/
│   └── main.py             # Logika utama API (FastAPI)
├── ml/
│   ├── ml_config.py        # Konfigurasi model dan fitur
│   ├── schemas.py          # Skema data (Pydantic)
│   ├── inference.py        # Skrip untuk inferensi/prediksi
│   ├── train_*.py          # Skrip untuk melatih model (LR, RF, XGB)
│   └── ...
├── tests/                  # Unit dan integration tests
├── data/                   # Sampel data untuk training dan testing
├── models/                 # Artefak model yang telah dilatih
├── reports/                # Laporan hasil analisis (metrics, plots)
├── .github/workflows/      # Alur kerja CI/CD dan monitoring
├── requirements.txt        # Dependensi Python
├── Dockerfile              # Konfigurasi build Docker image
└── README.md               # Dokumentasi ini

```
Tentu, ini adalah versi README.md yang telah dirapikan sesuai permintaan Anda, dengan memastikan semua bagian yang relevan seperti struktur file, contoh kode, dan perintah terminal berada di dalam blok kode (```) untuk keterbacaan maksimal.

Floodzy-ML: Sistem Prediksi Banjir Berbasis Machine Learning
Versi Proyek: 0.3.0

📝 Deskripsi Proyek
Floodzy-ML adalah sebuah sistem end-to-end yang dirancang untuk memprediksi potensi dan ketinggian banjir menggunakan pendekatan machine learning. Proyek ini dibangun dengan fokus pada kualitas produksi, mencakup validasi domain yang ketat, strategi splitting data yang time-aware dan spatio-temporal, serta alur kerja CI/CD untuk pengujian dan deployment otomatis.

Sistem ini diekspos melalui REST API yang dibangun menggunakan FastAPI dan siap untuk di-deploy sebagai kontainer Docker yang minimalis dan efisien.

✨ Fitur Utama
API Cepat dan Andal: Dibangun di atas FastAPI, menyediakan latensi rendah dan dokumentasi otomatis.

Tiga Model Prediktif:

Potensi Banjir (Regresi Logistik): Memberikan probabilitas dan label risiko (RENDAH, SEDANG, TINGGI).

Ketinggian Banjir (Random Forest): Memprediksi ketinggian air dalam sentimeter.

Potensi Banjir (XGBoost): Model alternatif dengan performa tinggi untuk klasifikasi potensi banjir.

Keamanan: Dukungan untuk autentikasi berbasis API key.

Validasi Input (Feature Contract): Menerapkan batasan pada fitur masukan untuk mencegah data anomali dan meningkatkan stabilitas model.

MLOps Terintegrasi:

CI/CD dengan GitHub Actions: Otomatisasi pengujian, analisis coverage, dan build Docker image.

Monitoring Otomatis: Deteksi data drift dan prediction stability secara berkala untuk memastikan model tetap relevan.

Deployment Siap Produksi: Dikemas dalam Docker dengan multi-stage build untuk ukuran image yang optimal.

🏗️ Struktur Proyek
Struktur direktori dirancang untuk memisahkan antara logika API, kode machine learning, dan komponen pendukung lainnya.

floodzy-ml/
├── api/
│   └── main.py             # Logika utama API (FastAPI)
├── ml/
│   ├── ml_config.py        # Konfigurasi model dan fitur
│   ├── schemas.py          # Skema data (Pydantic)
│   ├── inference.py        # Skrip untuk inferensi/prediksi
│   ├── train_*.py          # Skrip untuk melatih model (LR, RF, XGB)
│   └── ...
├── tests/                  # Unit dan integration tests
├── data/                   # Sampel data untuk training dan testing
├── models/                 # Artefak model yang telah dilatih
├── reports/                # Laporan hasil analisis (metrics, plots)
├── .github/workflows/      # Alur kerja CI/CD dan monitoring
├── requirements.txt        # Dependensi Python
├── Dockerfile              # Konfigurasi build Docker image
└── README.md               # Dokumentasi ini
🚀 Model dan API
Kontrak Fitur (Feature Contract)
Untuk menjaga integritas data, setiap fitur masukan memiliki batasan (bounds) yang telah ditentukan berdasarkan justifikasi domain.

```
Fitur	Tipe	Unit	Rentang	Catatan
curah_hujan_24h	float	mm / 24h	0 .. 1000	Batas atas fisik untuk mencegah outlier
kecepatan_angin	float	km/h	0 .. 200	Di atas level badai kuat
suhu	float	°C	-20 .. 55	Rentang operasional sensor
kelembapan	float	%	0 .. 100	Sesuai definisi fisika
ketinggian_air_cm	float	cm	0 .. 1000	Batas aman untuk lingkungan urban
tren_air_6h	float	cm (Δ)	-200 .. 200	Perubahan wajar dalam 6 jam
mdpl	float	m	-430 .. 3000	Disesuaikan per geografi
jarak_sungai_m	float	m	0 .. 50000	>50 km dampak menurun
jumlah_banjir_5th	int	count	0 .. 100	Sebagai guardrail

```
Tentu, ini adalah versi README.md yang telah dirapikan sesuai permintaan Anda, dengan memastikan semua bagian yang relevan seperti struktur file, contoh kode, dan perintah terminal berada di dalam blok kode (```) untuk keterbacaan maksimal.Floodzy-ML: Sistem Prediksi Banjir Berbasis Machine LearningVersi Proyek: 0.3.0📝 Deskripsi ProyekFloodzy-ML adalah sebuah sistem end-to-end yang dirancang untuk memprediksi potensi dan ketinggian banjir menggunakan pendekatan machine learning. Proyek ini dibangun dengan fokus pada kualitas produksi, mencakup validasi domain yang ketat, strategi splitting data yang time-aware dan spatio-temporal, serta alur kerja CI/CD untuk pengujian dan deployment otomatis.Sistem ini diekspos melalui REST API yang dibangun menggunakan FastAPI dan siap untuk di-deploy sebagai kontainer Docker yang minimalis dan efisien.✨ Fitur UtamaAPI Cepat dan Andal: Dibangun di atas FastAPI, menyediakan latensi rendah dan dokumentasi otomatis.Tiga Model Prediktif:Potensi Banjir (Regresi Logistik): Memberikan probabilitas dan label risiko (RENDAH, SEDANG, TINGGI).Ketinggian Banjir (Random Forest): Memprediksi ketinggian air dalam sentimeter.Potensi Banjir (XGBoost): Model alternatif dengan performa tinggi untuk klasifikasi potensi banjir.Keamanan: Dukungan untuk autentikasi berbasis API key.Validasi Input (Feature Contract): Menerapkan batasan pada fitur masukan untuk mencegah data anomali dan meningkatkan stabilitas model.MLOps Terintegrasi:CI/CD dengan GitHub Actions: Otomatisasi pengujian, analisis coverage, dan build Docker image.Monitoring Otomatis: Deteksi data drift dan prediction stability secara berkala untuk memastikan model tetap relevan.Deployment Siap Produksi: Dikemas dalam Docker dengan multi-stage build untuk ukuran image yang optimal.🏗️ Struktur ProyekStruktur direktori dirancang untuk memisahkan antara logika API, kode machine learning, dan komponen pendukung lainnya.floodzy-ml/
├── api/
│   └── main.py             # Logika utama API (FastAPI)
├── ml/
│   ├── ml_config.py        # Konfigurasi model dan fitur
│   ├── schemas.py          # Skema data (Pydantic)
│   ├── inference.py        # Skrip untuk inferensi/prediksi
│   ├── train_*.py          # Skrip untuk melatih model (LR, RF, XGB)
│   └── ...
├── tests/                  # Unit dan integration tests
├── data/                   # Sampel data untuk training dan testing
├── models/                 # Artefak model yang telah dilatih
├── reports/                # Laporan hasil analisis (metrics, plots)
├── .github/workflows/      # Alur kerja CI/CD dan monitoring
├── requirements.txt        # Dependensi Python
├── Dockerfile              # Konfigurasi build Docker image
└── README.md               # Dokumentasi ini
🚀 Model dan APIKontrak Fitur (Feature Contract)Untuk menjaga integritas data, setiap fitur masukan memiliki batasan (bounds) yang telah ditentukan berdasarkan justifikasi domain.FiturTipeUnitRentangCatatancurah_hujan_24hfloatmm / 24h0 .. 1000Batas atas fisik untuk mencegah outlierkecepatan_anginfloatkm/h0 .. 200Di atas level badai kuatsuhufloat°C-20 .. 55Rentang operasional sensorkelembapanfloat%0 .. 100Sesuai definisi fisikaketinggian_air_cmfloatcm0 .. 1000Batas aman untuk lingkungan urbantren_air_6hfloatcm (Δ)-200 .. 200Perubahan wajar dalam 6 jammdplfloatm-430 .. 3000Disesuaikan per geografijarak_sungai_mfloatm0 .. 50000>50 km dampak menurunjumlah_banjir_5thintcount0 .. 100Sebagai guardrailEndpoint APIBerikut adalah daftar endpoint yang tersedia:GET /healthzDeskripsi: Memeriksa status kesehatan dan versi API.Respons Sukses (200):

```
{"status": "ok", "version": "0.3.0"}
```
POST /predict/flood-potential

Deskripsi: Memprediksi potensi banjir menggunakan model Regresi Logistik.

Request Body: FloodFeaturesIn (JSON)

Respons Sukses (200): FloodPotentialOut (JSON)

```
{
  "label": 1,
  "probability": 0.75,
  "risk_label": "HIGH",
  "model_version": "lr",
  "features_used": [...],
  "latency_ms": 15.2,
  "cache_status": "miss"
}


```
POST /predict/flood-height

Deskripsi: Memprediksi ketinggian air banjir menggunakan model Random Forest.

Request Body: FloodFeaturesIn (JSON)

Respons Sukses (200): FloodHeightOut (JSON)


```
{
  "predicted_height_cm": 150.5,
  "model_version": "rf",
  "features_used": [...],
  "latency_ms": 25.8
}


```

POST /predict/flood-potential-xgb

Deskripsi: Memprediksi potensi banjir menggunakan model XGBoost.

Request Body: FloodFeaturesIn (JSON)

Respons Sukses (200): FloodPotentialOut (JSON)

GET /analysis/feature-importance

Deskripsi: Mengembalikan daftar fitur dan tingkat kepentingannya.

Respons Sukses (200): List[FeatureImportanceItem] (JSON)

🛠️ Instalasi dan Penggunaan
Prasyarat
Python 3.11+

Docker (opsional, untuk deployment)

Instalasi Dependensi


```
# Buat dan aktifkan virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependensi
pip install -r requirements.txt

```

Menjalankan Sesi Latihan Model
Skrip latihan tersedia di dalam direktori ml/.

```
# Contoh melatih model XGBoost
python ml/train_xgb.py --data data/sample_data.csv

```
Menjalankan API Secara Lokal
Gunakan uvicorn untuk menjalankan server API.

```
uvicorn api.main:app --reload --port 8000

```

Server akan aktif di http://localhost:8000. Anda dapat mengakses dokumentasi interaktif API di http://localhost:8000/docs.

📦 Deployment dengan Docker
Proyek ini dilengkapi dengan Dockerfile multi-stage untuk menciptakan image yang ringan dan aman.


```
# 1. Build Docker image
docker build -t floodzy-ml:latest .

# 2. Jalankan kontainer
docker run -d -p 8000:8000 --name floodzy-ml-app floodzy-ml:latest

```

⚙️ Otomatisasi dan MLOps
CI (Continuous Integration)
Alur kerja CI diatur dalam .github/workflows/ci.yml dan melakukan tugas-tugas berikut pada setiap push atau pull request:

Setup Environment: Mengatur versi Python yang sesuai.

Caching: Menyimpan cache dependensi untuk mempercepat build.

Install Dependencies: Memasang semua paket yang dibutuhkan.

Run Tests: Menjalankan unit tests dengan pytest dan menghitung test coverage.

Upload Artifacts: Mengunggah laporan coverage sebagai artefak.

Build Docker Image: Memastikan image dapat di-build tanpa kesalahan.

Monitoring
Alur kerja monitoring diatur dalam .github/workflows/monitor.yml dan berjalan setiap hari untuk:

Deteksi Data Drift: Membandingkan distribusi data produksi dengan data latihan.

Cek Stabilitas Prediksi: Memantau perubahan signifikan pada output model.

Laporan: Menghasilkan laporan dan menyimpannya sebagai artefak.

Alerting: Membuat issue di GitHub jika terdeteksi anomali, untuk investigasi lebih lanjut.
