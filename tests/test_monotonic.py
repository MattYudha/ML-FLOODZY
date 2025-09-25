import os, pytest
from ml.inference import predict_xgb

@pytest.mark.skipif(not os.path.exists("models/xgb_pipeline.joblib"), reason="no model")
@pytest.mark.parametrize("feature,base,delta", [("curah_hujan_24h", 10, 20), ("ketinggian_air_cm", 10, 20), ("tren_air_6h", 0, 10)])
def test_monotonic_non_decreasing_prob(feature, base, delta):
    payload = {"curah_hujan_24h":10,"kecepatan_angin":5,"suhu":29,"kelembapan":80,"ketinggian_air_cm":20,"tren_air_6h":2,"mdpl":25,"jarak_sungai_m":100,"jumlah_banjir_5th":1}
    p1 = predict_xgb(payload)["probability"]
    payload[feature] = base + delta
    p2 = predict_xgb(payload)["probability"]
    assert p2 >= p1 - 1e-6
