from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

def test_bad_threshold():
    payload = {"curah_hujan_24h":10,"kecepatan_angin":5,"suhu":29,"kelembapan":80,"ketinggian_air_cm":20,"tren_air_6h":2,"mdpl":25,"jarak_sungai_m":100,"jumlah_banjir_5th":1}
    r = client.post("/predict/flood-potential?threshold=2", json=payload)
    assert r.status_code == 422
