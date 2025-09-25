import pytest
from ml.schemas import FloodFeaturesIn

def test_schema_valid_single():
    f = FloodFeaturesIn(curah_hujan_24h=10, kecepatan_angin=5, suhu=29, kelembapan=80, ketinggian_air_cm=20, tren_air_6h=2, mdpl=25, jarak_sungai_m=100, jumlah_banjir_5th=1)
    assert f.kelembapan == 80

@pytest.mark.parametrize("val", [-1, 1000.1])
def test_rain_bounds(val):
    with pytest.raises(Exception):
        FloodFeaturesIn(curah_hujan_24h=val, kecepatan_angin=0, suhu=25, kelembapan=50, ketinggian_air_cm=10, tren_air_6h=0, mdpl=0, jarak_sungai_m=10, jumlah_banjir_5th=0)
