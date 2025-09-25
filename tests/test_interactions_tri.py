import os, pytest
from ml.inference import predict_xgb

pytestmark = pytest.mark.skipif(not os.path.exists("models/xgb_pipeline.joblib"), reason="no XGB model")

def base_payload():
    return {
        "curah_hujan_24h": 60,  # tinggi
        "kecepatan_angin": 12,
        "suhu": 29,
        "kelembapan": 85,
        "ketinggian_air_cm": 80,  # tinggi
        "tren_air_6h": 10,       # naik
        "mdpl": 20,
        "jarak_sungai_m": 150,
        "jumlah_banjir_5th": 3,
    }

def test_three_feature_interaction_rising_vs_falling():
    p_up = base_payload()
    prob_up = predict_xgb(p_up)["probability"]  # level tinggi + hujan tinggi + tren naik

    p_down = dict(p_up, tren_air_6h=-15)        # level tinggi + hujan tinggi + tren turun signifikan
    prob_down = predict_xgb(p_down)["probability"]

    # Secara domain: saat tren turun, risiko tidak boleh lebih tinggi daripada saat tren naik (toleransi float kecil)
    assert prob_down <= prob_up + 1e-6
