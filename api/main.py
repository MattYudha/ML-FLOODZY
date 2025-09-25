from __future__ import annotations
from typing import List
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
import logging, os, time, json as _json

from ml.inference import predict_lr, predict_rf, predict_xgb
from ml.feature_importance import feature_importance
from ml.schemas import FloodFeaturesIn, FloodPotentialOut, FloodHeightOut, FeatureImportanceItem

APP_VERSION = "0.3.0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="Floodzy ML API", version=APP_VERSION)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,https://floodzy.vercel.app")
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

API_KEY = os.getenv("API_KEY")  # optional

def _json_log(**kwargs):
    try:
        LOGGER.info(_json.dumps(kwargs, ensure_ascii=False))
    except Exception:
        LOGGER.info(str(kwargs))

def _risk_label(p: float) -> str:
    if p < 0.33: return "LOW"
    if p < 0.66: return "MED"
    return "HIGH"

@app.get("/healthz")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.post("/predict/flood-potential", response_model=FloodPotentialOut)
def predict_flood_potential(body: FloodFeaturesIn, threshold: float = 0.5, request: Request | None = None, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY: raise HTTPException(status_code=401, detail="Unauthorized")
    if not (0.0 <= float(threshold) <= 1.0): raise HTTPException(status_code=422, detail="threshold must be in [0,1]")
    start = time.time()
    try: result = predict_lr(body.model_dump(), threshold=threshold)
    except Exception as e: raise HTTPException(status_code=422, detail=str(e))
    prob = float(result["probability"]); label=int(result["label"]); risk=_risk_label(prob)
    latency_ms = (time.time()-start)*1000.0; _json_log(endpoint="/predict/flood-potential", model="lr", latency_ms=latency_ms, prob=prob, label=label)
    return FloodPotentialOut(label=label, probability=prob, threshold=float(threshold), risk_label=risk, model_version="lr", features_used=result.get("features_order", []), latency_ms=latency_ms, cache_status="miss")

@app.post("/predict/flood-height", response_model=FloodHeightOut)
def predict_flood_height(body: FloodFeaturesIn, request: Request | None = None, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY: raise HTTPException(status_code=401, detail="Unauthorized")
    start = time.time()
    try: result = predict_rf(body.model_dump())
    except Exception as e: raise HTTPException(status_code=422, detail=str(e))
    latency_ms = (time.time()-start)*1000.0; _json_log(endpoint="/predict/flood-height", model="rf", latency_ms=latency_ms)
    return FloodHeightOut(predicted_height_cm=float(result["predicted_height_cm"]), model_version="rf", features_used=result.get("features_order", []), latency_ms=latency_ms)

@app.post("/predict/flood-potential-xgb", response_model=FloodPotentialOut)
def predict_flood_potential_xgb(body: FloodFeaturesIn, threshold: float = 0.5, request: Request | None = None, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY: raise HTTPException(status_code=401, detail="Unauthorized")
    if not (0.0 <= float(threshold) <= 1.0): raise HTTPException(status_code=422, detail="threshold must be in [0,1]")
    start = time.time()
    try: result = predict_xgb(body.model_dump(), threshold=threshold)
    except Exception as e: raise HTTPException(status_code=422, detail=str(e))
    prob = float(result["probability"]); label=int(result["label"]); risk=_risk_label(prob)
    latency_ms = (time.time()-start)*1000.0; _json_log(endpoint="/predict/flood-potential-xgb", model="xgb", latency_ms=latency_ms, prob=prob, label=label)
    return FloodPotentialOut(label=label, probability=prob, threshold=float(threshold), risk_label=risk, model_version="xgb", features_used=result.get("features_order", []), latency_ms=latency_ms, cache_status="miss")

@app.get("/analysis/feature-importance", response_model=List[FeatureImportanceItem])
def get_feature_importance():
    try: data = feature_importance()
    except Exception as e: raise HTTPException(status_code=501, detail=str(e))
    return [FeatureImportanceItem(**row) for row in data]
