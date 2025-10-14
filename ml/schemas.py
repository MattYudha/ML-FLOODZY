from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Optional, List, Literal
import math
from ml.ml_config import DOMAIN_BOUNDS as B  # single source of truth for bounds

class FloodFeaturesIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: Optional[str] = Field(None, description="ISO8601 timestamp")
    lat: Optional[float] = Field(None, description="Latitude (deg)")
    lon: Optional[float] = Field(None, description="Longitude (deg)")
    region_id: Optional[str] = Field(None, description="Region identifier (optional)")

    curah_hujan_24h: float = Field(..., ge=0, le=B["RAIN_MAX_MM_24H"], description=f"Rainfall (mm/24h) [0..{B['RAIN_MAX_MM_24H']}]")
    kecepatan_angin: float = Field(..., ge=0, le=B["WIND_MAX_KMH"], description=f"Wind speed (km/h) [0..{B['WIND_MAX_KMH']}]")
    suhu: float = Field(..., ge=B["TEMP_MIN_C"], le=B["TEMP_MAX_C"], description=f"Temperature (°C) [{B['TEMP_MIN_C']}..{B['TEMP_MAX_C']}]")
    kelembapan: float = Field(..., ge=0, le=100, description="Relative humidity (%) 0..100")
    ketinggian_air_cm: float = Field(..., ge=0, le=B["WATERLEVEL_MAX_CM"], description=f"Current water level (cm) [0..{B['WATERLEVEL_MAX_CM']}]")
    tren_air_6h: float = Field(..., ge=-B["WATER_DELTA_6H_ABS"], le=B["WATER_DELTA_6H_ABS"], description=f"Water level delta 6h (cm) [-{B['WATER_DELTA_6H_ABS']}..{B['WATER_DELTA_6H_ABS']}]")
    mdpl: float = Field(..., ge=B["ELEVATION_MIN_M"], le=B["ELEVATION_MAX_M"], description=f"Elevation (m) [{B['ELEVATION_MIN_M']}..{B['ELEVATION_MAX_M']}]")
    jarak_sungai_m: float = Field(..., ge=0, le=B["RIVER_DIST_MAX_M"], description=f"Distance to river (m) [0..{B['RIVER_DIST_MAX_M']}]")
    jumlah_banjir_5th: int = Field(..., ge=0, le=B["FLOODCOUNT_MAX"], description=f"Flood count in last 5 years [0..{B['FLOODCOUNT_MAX']}]")

    @field_validator("curah_hujan_24h","kecepatan_angin","suhu","kelembapan","ketinggian_air_cm","tren_air_6h","mdpl","jarak_sungai_m")
    @classmethod
    def finite(cls, v):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            raise ValueError("Value must be finite (no NaN/inf).")
        return v

    ## --- PENAMBAHAN: Validasi Lintas Fitur ---
    @model_validator(mode='after')
    def check_domain_logic(self) -> 'FloodFeaturesIn':
        """
        Menambahkan validasi untuk kombinasi fitur yang tidak logis secara domain.
        """
        if self.suhu is not None and self.kelembapan is not None:
            # Contoh: Suhu sangat tinggi tidak mungkin terjadi bersamaan dengan kelembapan sangat tinggi
            if self.suhu > 40 and self.kelembapan > 70:
                raise ValueError("Kombinasi tidak logis: suhu > 40°C dan kelembapan > 70%")
        
        # Anda bisa menambahkan validasi lain di sini
        # if self.curah_hujan_24h > 200 and self.ketinggian_air_cm < 10:
        #     raise ValueError("Kombinasi tidak logis: Curah hujan sangat tinggi namun ketinggian air rendah")
            
        return self


class FloodPotentialOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    label: int
    probability: float
    threshold: float
    risk_label: Literal["LOW","MED","HIGH"]
    model_version: str
    features_used: List[str]
    latency_ms: float
    cache_status: Literal["hit","miss"]

class FloodHeightOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predicted_height_cm: float
    model_version: str
    features_used: List[str]
    latency_ms: float

class FeatureImportanceItem(BaseModel):
    feature: str
    importance_score: float