from hypothesis import given, strategies as st
from ml.schemas import FloodFeaturesIn

float_latitude = st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False)
float_longitude = st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
int_water_level = st.integers(min_value=0, max_value=100)

@given(lat=float_latitude, lon=float_longitude, wl=int_water_level)
def test_schema_fuzz(lat, lon, wl):
    obj = FloodFeaturesIn(latitude=lat, longitude=lon, water_level=wl)
    assert -90 <= obj.latitude <= 90
    assert -180 <= obj.longitude <= 180
    assert 0 <= obj.water_level <= 100
