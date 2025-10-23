# ml/schemas.py

from pydantic import BaseModel, Field
from typing import List

# PERBAIKAN: Skema ini sekarang cocok dengan data yang dikirim dari frontend
class FloodFeaturesIn(BaseModel):
    latitude: float = Field(..., example=-6.2088)
    longitude: float = Field(..., example=106.8456)
    water_level: int = Field(..., example=1, description="Numeric representation of water level")

class FloodPotentialOut(BaseModel):
    label: int = Field(..., example=1)
    probability: float = Field(..., example=0.85)
    threshold: float = Field(..., example=0.5)
    risk_label: str = Field(..., example="HIGH")
    model_version: str = Field(..., example="xgb")
    features_used: List[str] = Field(..., example=["latitude", "longitude", "water_level"])
    latency_ms: float = Field(..., example=50.5)
    cache_status: str = Field(..., example="miss")

class FloodHeightOut(BaseModel):
    predicted_height_cm: float = Field(..., example=75.5)
    model_version: str = Field(..., example="rf")
    features_used: List[str] = Field(...)
    latency_ms: float = Field(...)

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float