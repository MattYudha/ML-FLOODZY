from hypothesis import given, strategies as st
from ml.schemas import FloodFeaturesIn

float_pos = st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
float_speed = st.floats(min_value=0, max_value=200, allow_nan=False, allow_infinity=False)
float_temp = st.floats(min_value=-20, max_value=55, allow_nan=False, allow_infinity=False)
float_hum = st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
float_level = st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
float_delta = st.floats(min_value=-200, max_value=200, allow_nan=False, allow_infinity=False)
float_mdpl = st.floats(min_value=-430, max_value=3000, allow_nan=False, allow_infinity=False)
float_dist = st.floats(min_value=0, max_value=50000, allow_nan=False, allow_infinity=False)
int_count = st.integers(min_value=0, max_value=100)

@given(curah=float_pos, angin=float_speed, suhu=float_temp, hum=float_hum, level=float_level, delta=float_delta, mdpl=float_mdpl, dist=float_dist, cnt=int_count)
def test_schema_fuzz(curah, angin, suhu, hum, level, delta, mdpl, dist, cnt):
    obj = FloodFeaturesIn(curah_hujan_24h=curah, kecepatan_angin=angin, suhu=suhu, kelembapan=hum, ketinggian_air_cm=level, tren_air_6h=delta, mdpl=mdpl, jarak_sungai_m=dist, jumlah_banjir_5th=cnt)
    assert 0 <= obj.kelembapan <= 100
