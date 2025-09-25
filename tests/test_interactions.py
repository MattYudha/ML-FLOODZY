import os, pytest
from ml.inference import predict_xgb

pytestmark = pytest.mark.skipif(not os.path.exists("models/xgb_pipeline.joblib"), reason="no XGB model")

def base_payload():
    return {
        "curah_hujan_24h": 50,
        "kecepatan_angin": 10,
        "suhu": 29,
        "kelembapan": 85,
        "ketinggian_air_cm": 60,
        "tren_air_6h": 5,
        "mdpl": 25,
        "jarak_sungai_m": 200,
        "jumlah_banjir_5th": 3,
    }

def test_high_level_but_falling_trend_should_not_increase_risk():
    p = base_payload()
    up = predict_xgb(p)["probability"]
    p_falling = dict(p, tren_air_6h=-10)  # surut
    down = predict_xgb(p_falling)["probability"]
    assert down <= up + 1e-6

def test_heavy_rain_far_from_river_should_be_lower_than_close():
    p_far = base_payload()
    p_far["curah_hujan_24h"] = 80
    p_far["jarak_sungai_m"] = 40000
    prob_far = predict_xgb(p_far)["probability"]

    p_near = dict(p_far, jarak_sungai_m=50)
    prob_near = predict_xgb(p_near)["probability"]

    assert prob_far <= prob_near + 1e-6
