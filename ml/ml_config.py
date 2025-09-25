from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# === Domain Bounds (Hydrology/Meteorology guardrails) ===
# Referensi umum & alasan (ringkas):
# - RAIN_MAX_MM_24H: 500–700 mm/hari tercatat pada event ekstrem tropis; 1000 mm dipilih sebagai cap konservatif.
# - WIND_MAX_KMH: 200 km/j mencakup badai tropis kuat; jauh di atas urban typical Jakarta.
# - TEMP_MIN/MAX_C: batas operasional sensor & klimatologi lokal.
# - WATERLEVEL_MAX_CM: 1000 cm (10 m) cap konservatif untuk sungai/kanal; genangan urban biasanya << 300 cm.
# - WATER_DELTA_6H_ABS: perubahan ketinggian 6 jam yang masih realistis untuk sistem sungai/kanal perkotaan.
# - RIVER_DIST_MAX_M: >50 km kontribusi hidrologi langsung menurun; tetap dibiarkan longgar demi generalisasi.
DOMAIN_BOUNDS: Dict[str, float] = {
    "RAIN_MAX_MM_24H": 1000.0,
    "WIND_MAX_KMH": 200.0,
    "TEMP_MIN_C": -20.0,
    "TEMP_MAX_C": 55.0,
    "WATERLEVEL_MAX_CM": 1000.0,
    "WATER_DELTA_6H_ABS": 200.0,
    "ELEVATION_MIN_M": -430.0,
    "ELEVATION_MAX_M": 3000.0,
    "RIVER_DIST_MAX_M": 50000.0,
    "FLOODCOUNT_MAX": 100.0,
}

FEATURES: List[str] = [
    "curah_hujan_24h",
    "kecepatan_angin",
    "suhu",
    "kelembapan",
    "ketinggian_air_cm",
    "tren_air_6h",
    "mdpl",
    "jarak_sungai_m",
    "jumlah_banjir_5th",
]

TARGET_LR: str = "terjadi_banjir"
TARGET_RF: str = "ketinggian_air_banjir_aktual_cm"

LR_PARAMS: Dict[str, Any] = {"class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}
RF_PARAMS: Dict[str, Any] = {"n_estimators": 200, "max_depth": None, "random_state": 42, "n_jobs": -1}
RF_GRID = {"rf__n_estimators": [100, 200, 400], "rf__max_depth": [None, 10, 20, 40]}
XGB_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic", "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1,
}

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

LR_PIPELINE_NAME = "lr_pipeline.joblib"
RF_PIPELINE_NAME = "rf_pipeline.joblib"
XGB_PIPELINE_NAME = "xgb_pipeline.joblib"
FEATURE_LIST_JSON = "feature_names.json"

def timestamp_tag() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
