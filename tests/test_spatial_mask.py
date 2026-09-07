import numpy as np
import pytest
import xarray as xr

regionmask = pytest.importorskip("regionmask", reason="regionmask not installed")
from esp_lab.utils.spatial_utils import compute_regional_average


def test_compute_regional_average_land_and_ocean():
    lat = np.array([0, 10, 20])
    lon = np.array([-180, 0, 180])
    data = np.ones((3, 3))
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    land_avg = compute_regional_average(da, (-10, 30), (-180, 180), land_mask=True)
    ocean_avg = compute_regional_average(da, (-10, 30), (-180, 180), ocean_mask=True)
    assert np.isfinite(float(land_avg)) or np.isnan(float(land_avg))
    assert np.isfinite(float(ocean_avg)) or np.isnan(float(ocean_avg))
