# Floodzy-ML (Hardened v2)

Baseline produksi dengan validasi domain ketat, split time-aware (termasuk opsi spatio-temporal), CI caching + coverage, dan Docker image minimal.

## Struktur
```
floodzy-ml-hardened-v2/
├── api/main.py
├── ml/
│   ├── ml_config.py
│   ├── schemas.py
│   ├── split.py
│   ├── inference.py
│   ├── feature_importance.py
│   ├── train_lr.py
│   ├── train_rf.py
│   ├── train_rf_grid.py
│   └── train_xgb.py
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_schema_property.py
│   └── test_monotonic.py
├── data/sample_data.csv
├── models/  (artefak)
├── reports/ (metrics/plots)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── .github/workflows/ci.yml
```

## Feature Contract (Bounds & Justification)
| Feature | Type | Unit | Range | Catatan |
|---|---|---|---|---|
| curah_hujan_24h | float | mm / 24h | 0 .. 1000 | cap fisik untuk cegah outlier absurd; ekstrem dunia bisa >500 mm |
| kecepatan_angin | float | km/h | 0 .. 200 | di atas level badai kuat; cukup konservatif |
| suhu | float | °C | -20 .. 55 | rentang operasional |
| kelembapan | float | % | 0 .. 100 | definisi |
| ketinggian_air_cm | float | cm | 0 .. 1000 | urban biasanya <<300 cm; 1000 cm sebagai cap |
| tren_air_6h | float | cm (Δ) | -200 .. 200 | perubahan wajar 6 jam |
| mdpl | float | m | -430 .. 3000 | sesuaikan per-negara |
| jarak_sungai_m | float | m | 0 .. 50000 | >50 km dampak langsung menurun |
| jumlah_banjir_5th | int | count | 0 .. 100 | guardrail |

> Angka dapat di-tune berdasarkan data lokal. Tujuan utama: **mencegah garbage-in** dan meningkatkan stabilitas model.

## Splitting Methods
- **time_based_split(df)**: default (chronological split by ratio).
- **time_window_split(df, 'timestamp', '2024-12-31', '2025-01-31')**: train ≤ Dec 2024, val = Jan 2025, test ≥ Feb 2025.
- **spatio_temporal_holdout(df, 'region_id', ['REG-02','REG-05'])**: region tertentu full di test (uji generalisasi spasial).

## Training
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python ml/train_lr.py --data data/sample_data.csv
python ml/train_rf.py --data data/sample_data.csv
python ml/train_rf_grid.py --data data/sample_data.csv
python ml/train_xgb.py --data data/sample_data.csv
```

## API
```bash
uvicorn api.main:app --reload --port 8000
curl http://localhost:8000/healthz
```

## CI
- **actions/cache** untuk pip — build lebih cepat.
- **pytest-cov** → coverage.xml di-upload sebagai artefak.

## Docker
- Multi-stage base + runtime, **.dockerignore** ketat (tidak membawa tests/ui/.github).
