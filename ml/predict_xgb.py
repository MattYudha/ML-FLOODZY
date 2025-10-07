from xgboost import XGBClassifier
import pandas as pd
from pathlib import Path

# Path model hasil training
MODEL_PATH = Path("artifacts/xgb_floodzy_model.json")

# Muat model
model = XGBClassifier()
model.load_model(MODEL_PATH.as_posix())

# Contoh input baru
sample = pd.DataFrame([{
    "region_id": 3171,
    "rain_mm": 85.0,
    "river_level_cm": 300,
    "tide_cm": 118,
    "temperature_avg": 27.2,
    "humidity_avg": 90,
    "wind_speed": 2.5,
    "elevation_m": 8,
    "api_7d": 230.5
}])

# Prediksi probabilitas banjir
proba = model.predict_proba(sample)[0, 1]
label = int(proba >= 0.5)

print(f"Prediksi flood_event: {label} (probabilitas {proba:.3f})")
