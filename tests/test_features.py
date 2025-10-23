import pytest
from ml.schemas import FloodFeaturesIn

def test_schema_valid_single():
    f = FloodFeaturesIn(latitude=-6.2088, longitude=106.8456, water_level=1)
    assert f.latitude == -6.2088
    assert f.longitude == 106.8456
    assert f.water_level == 1
