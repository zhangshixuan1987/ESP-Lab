import numpy as np
import xarray as xr
from esp_lab.data_access_e3sm import e3sm_regional_mean, e3sm_region_mask


def test_e3sm_regional_mean_uniform_field():
    lat = np.linspace(-90, 90, 100)
    lon = np.linspace(0, 359, 100)
    data = np.ones((100, 100))
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    res = e3sm_regional_mean(da, [-10, 10, -10, 10])
    assert np.isclose(float(res), 1.0)
