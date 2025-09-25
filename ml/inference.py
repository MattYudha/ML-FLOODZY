import os, json, joblib, pandas as pd
from typing import Dict, Any, List
from ml.ml_config import MODELS_DIR, FEATURE_LIST_JSON

def _read_versions():
    path = os.path.join(MODELS_DIR, "versions.json")
    if os.path.exists(path):
        try:
            return json.loads(open(path,'r',encoding='utf-8').read())
        except Exception:
            return {}
    return {}

def get_model_version(model_key: str) -> str:
    return _read_versions().get(model_key, "unknown")

def load_feature_order() -> List[str]:
    path = os.path.join(MODELS_DIR, FEATURE_LIST_JSON)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["curah_hujan_24h","kecepatan_angin","suhu","kelembapan","ketinggian_air_cm","tren_air_6h","mdpl","jarak_sungai_m","jumlah_banjir_5th"]

def _df_from_payload(payload: Dict[str, Any], order: List[str]) -> pd.DataFrame:
    row = {k: payload.get(k) for k in order}
    miss = [k for k,v in row.items() if v is None]
    if miss: raise ValueError(f"Missing required features: {miss}")
    return pd.DataFrame([row], columns=order)

def predict_lr(payload: Dict[str, Any], threshold: float = 0.5):
    order = load_feature_order()
    pipe = joblib.load(os.path.join(MODELS_DIR, "lr_pipeline.joblib"))
    X = _df_from_payload(payload, order)
    proba = float(pipe.predict_proba(X)[0,1])
    label = int(proba >= threshold)
    return {"label": label, "probability": proba, "threshold": float(threshold), "features_order": order, "model_version": get_model_version('lr')}

def predict_rf(payload: Dict[str, Any]):
    order = load_feature_order()
    pipe = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))
    X = _df_from_payload(payload, order)
    height = float(pipe.predict(X)[0])
    return {"predicted_height_cm": height, "features_order": order, "model_version": get_model_version('rf')}

def predict_xgb(payload: Dict[str, Any], threshold: float = 0.5):
    order = load_feature_order()
    bundle = joblib.load(os.path.join(MODELS_DIR, "xgb_pipeline.joblib"))
    model = bundle["model"]
    features = bundle.get("features", order)
    X = _df_from_payload(payload, features).values
    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= threshold)
    return {"label": label, "probability": proba, "threshold": float(threshold), "features_order": features, "model_version": get_model_version('xgb')}
