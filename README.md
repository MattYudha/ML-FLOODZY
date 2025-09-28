# 🌊 Floodzy-Machine Learning (Hardened v2)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![CI](https://img.shields.io/badge/CI-Automated-success?style=flat-square)

*Production-ready flood prediction ML pipeline with domain validation, time-aware splitting, and CI/CD automation*

</div>

---

## 🎯 **Overview**

**Floodzy-ML Hardened v2** adalah sistem prediksi banjir berbasis machine learning yang dirancang untuk lingkungan produksi. Sistem ini menggabungkan validas data yang ketat, pembagian dataset yang time-aware (termasuk opsi spatio-temporal), caching CI yang optimal, dan Docker image yang minimal untuk deployment yang efisien ke project Flodozy.

### ✨ **Key Features**

- 🛡️ **Domain Validation**: Validasi input ketat dengan bounds checking
- ⏰ **Time-Aware Splitting**: Pembagian data berdasarkan waktu dan lokasi
- 🚀 **Production Ready**: API dengan FastAPI dan containerization
- 🔄 **CI/CD Pipeline**: Automated testing dengan coverage reporting
- 📊 **Multiple Models**: Logistic Regression, Random Forest, XGBoost
- 🐳 **Docker Optimized**: Multi-stage builds dengan minimal image size

---

## 📁 **Project Structure ML FLOODZY**

```
floodzy-ml-hardened-v2/
├── 🚀 api/
│   └── main.py                    # FastAPI application
├── 🧠 ml/
│   ├── ml_config.py              # Configuration & constants
│   ├── schemas.py                # Data validation schemas
│   ├── split.py                  # Dataset splitting strategies
│   ├── inference.py              # Model inference pipeline
│   ├── feature_importance.py     # Feature analysis tools
│   ├── train_lr.py               # Logistic Regression trainer
│   ├── train_rf.py               # Random Forest trainer
│   ├── train_rf_grid.py          # RF with GridSearch
│   └── train_xgb.py              # XGBoost trainer
├── 🧪 tests/
│   ├── test_api.py               # API endpoint tests
│   ├── test_features.py          # Feature validation tests
│   ├── test_schema_property.py   # Schema property tests
│   └── test_monotonic.py         # Model monotonicity tests
├── 📊 data/
│   └── sample_data.csv           # Sample dataset
├── 🎯 models/                    # Trained model artifacts
├── 📈 reports/                   # Metrics & visualization outputs
├── 📦 requirements.txt           # Python dependencies
├── 🐳 Dockerfile                 # Container definition
├── 🚫 .dockerignore              # Docker ignore patterns
└── ⚙️ .github/workflows/ci.yml   # CI/CD pipeline
```

---

## 🛡️ **Feature Contract & Validation**

Sistem menggunakan validasi domain yang ketat untuk memastikan kualitas input data:

<div align="center">

| 🏷️ **Feature** | 📊 **Type** | 📏 **Unit** | 📐 **Range** | 💡 **Justifikasi** |
|---|---|---|---|---|
| `curah_hujan_24h` | `float` | mm/24h | `0 - 1000` | Cap fisik mencegah outlier; ekstrem dunia >500mm |
| `kecepatan_angin` | `float` | km/h | `0 - 200` | Di atas level badai kuat, cukup konservatif |
| `suhu` | `float` | °C | `-20 - 55` | Rentang operasional normal |
| `kelembapan` | `float` | % | `0 - 100` | Definisi persentase kelembapan |
| `ketinggian_air_cm` | `float` | cm | `0 - 1000` | Urban biasanya <300cm; 1000cm sebagai cap |
| `tren_air_6h` | `float` | cm (Δ) | `-200 - 200` | Perubahan wajar dalam 6 jam |
| `mdpl` | `float` | m | `-430 - 3000` | Disesuaikan per negara/region |
| `jarak_sungai_m` | `float` | m | `0 - 50000` | >50km dampak langsung menurun |
| `jumlah_banjir_5th` | `int` | count | `0 - 100` | Guardrail untuk data historis |

</div>

> 💡 **Note**: Angka dapat di-tune berdasarkan data lokal. Tujuan utama: **mencegah garbage-in** dan meningkatkan stabilitas model.

---

## ⚡ **Dataset Splitting Strategies**

Sistem menyediakan 3 strategi pembagian dataset yang berbeda:

### 🕐 **1. Time-Based Split**
```python
time_based_split(df)  # Default chronological split by ratio
```

### 🪟 **2. Time Window Split**
```python
time_window_split(df, 'timestamp', '2024-12-31', '2025-01-31')
# Train ≤ Dec 2024, Val = Jan 2025, Test ≥ Feb 2025
```

### 🗺️ **3. Spatio-Temporal Holdout**
```python
spatio_temporal_holdout(df, 'region_id', ['REG-02','REG-05'])
# Region tertentu full di test untuk uji generalisasi spasial
```

---

## 🚀 **Quick Start**

### 📦 **1. Environment Setup**

```bash
# Clone repository
git clone <repository-url>
cd floodzy-ml-hardened-v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 🧠 **2. Model Training**

Train different models dengan satu command:

```bash
# Logistic Regression
python ml/train_lr.py --data data/sample_data.csv

# Random Forest
python ml/train_rf.py --data data/sample_data.csv

# Random Forest with GridSearch
python ml/train_rf_grid.py --data data/sample_data.csv

# XGBoost
python ml/train_xgb.py --data data/sample_data.csv
```

### 🌐 **3. API Server**

```bash
# Start FastAPI server
uvicorn api.main:app --reload --port 8000

# Test health endpoint
curl http://localhost:8000/healthz

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"curah_hujan_24h": 120.5, "kecepatan_angin": 25.0, ...}'
```

---

## 🐳 **Docker Deployment**

### 🏗️ **Build & Run**

```bash
# Build Docker image
docker build -t floodzy-ml:v2 .

# Run container
docker run -p 8000:8000 floodzy-ml:v2

# Check container health
curl http://localhost:8000/healthz
```

### 🎯 **Docker Features**

- ✅ **Multi-stage builds** untuk ukuran image minimal
- ✅ **Optimized .dockerignore** (exclude tests/ui/.github)
- ✅ **Production-ready** dengan security best practices
- ✅ **Health checks** terintegrasi

---

## 🧪 **Testing & Quality Assurance**

### 🔬 **Run Tests**

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ml --cov=api --cov-report=html

# Run specific test categories
pytest tests/test_api.py -v          # API tests
pytest tests/test_features.py -v     # Feature validation tests
pytest tests/test_monotonic.py -v    # Model monotonicity tests
```

### 📊 **Test Coverage**

Sistem mencakup berbagai aspek testing:

- 🔌 **API Endpoint Testing**: Validasi semua endpoint
- 🛡️ **Schema Validation**: Testing bounds checking
- 📈 **Model Monotonicity**: Ensuring logical model behavior
- ⚡ **Feature Engineering**: Input validation testing

---

## ⚙️ **CI/CD Pipeline**

### 🔄 **Automated Workflow**

Pipeline CI/CD menggunakan GitHub Actions dengan fitur:

- ✅ **actions/cache** untuk pip dependencies (build lebih cepat)
- ✅ **pytest-cov** untuk coverage reporting
- ✅ **coverage.xml** di-upload sebagai artifacts
- ✅ **Multi-platform testing** (Ubuntu, Windows, macOS)
- ✅ **Docker image building** dan testing

### 📈 **Quality Gates**

- 🎯 Minimum test coverage: **80%**
- ✅ All tests must pass
- 🐳 Docker build must succeed
- 📊 Code quality checks

---

## 📊 **Model Performance & Monitoring**

### 🎯 **Available Models**

| 🤖 **Model** | ⚡ **Speed** | 🎯 **Accuracy** | 💾 **Memory** | 📝 **Use Case** |
|---|---|---|---|---|
| **Logistic Regression** | ⚡⚡⚡ | 🎯🎯 | 💾 | Baseline, interpretable |
| **Random Forest** | ⚡⚡ | 🎯🎯🎯 | 💾💾 | Balanced performance |
| **XGBoost** | ⚡ | 🎯🎯🎯🎯 | 💾💾💾 | Best accuracy |

### 📈 **Feature Importance Analysis**

```bash
# Generate feature importance report
python ml/feature_importance.py --model models/best_model.pkl --output reports/
```

---

## 🤝 **Contributing**

We welcome contributions! Please follow these guidelines:

### 🔧 **Development Setup**

```bash
# Fork the repository
# Clone your fork
git clone <your-fork-url>
cd floodzy-ml-hardened-v2

# Create feature branch
git checkout -b feature/amazing-feature

# Make your changes
# Add tests for your changes
pytest tests/ -v

# Commit your changes
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

### 📋 **Pull Request Checklist**

- [ ] ✅ Tests pass (`pytest tests/`)
- [ ] 📊 Code coverage maintained (>80%)
- [ ] 📝 Documentation updated
- [ ] 🧹 Code follows style guidelines
- [ ] 🔍 No breaking changes (unless intentional)

---

## 📚 **Documentation**

### 🔗 **Quick Links**

- 📖 [API Documentation](http://localhost:8000/docs) (when server is running)
- 🧠 [Model Architecture](#model-performance--monitoring)
- 🛡️ [Feature Validation](#feature-contract--validation)
- 🐳 [Docker Guide](#docker-deployment)
- 🧪 [Testing Guide](#testing--quality-assurance)

### ❓ **FAQ**

<details>
<summary><strong>🤔 Bagaimana cara menambah model baru?</strong></summary>

1. Buat file `train_newmodel.py` di folder `ml/`
2. Follow pattern yang ada di `train_*.py` files
3. Tambahkan tests di `tests/`
4. Update dokumentasi

</details>

<details>
<summary><strong>🔧 Bagaimana cara mengubah feature bounds?</strong></summary>

Edit konstanta di `ml/ml_config.py` dan update tests di `tests/test_schema_property.py`

</details>

<details>
<summary><strong>🐳 Mengapa Docker image lambat build?</strong></summary>

Pastikan menggunakan `.dockerignore` dan pertimbangkan menggunakan Docker layer caching.

</details>

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 **Support & Contact**

<div align="center">

**Need help?** 

[![Issues](https://img.shields.io/badge/GitHub-Issues-red?style=flat-square&logo=github)](https://github.com/your-repo/issues)
[![Discussions](https://img.shields.io/badge/GitHub-Discussions-blue?style=flat-square&logo=github)](https://github.com/your-repo/discussions)
[![Email](https://img.shields.io/badge/Email-Contact-green?style=flat-square&logo=gmail)](mailto:support@floodzy-ml.com)

</div>

---

<div align="center">

**⭐ Star this repository if it helped you!**

Made with ❤️ by the Floodzy-ML Team

</div>
