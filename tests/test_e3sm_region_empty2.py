import numpy as np
import xarray as xr
from esp_lab.data_access_e3sm import e3sm_regional_mean, e3sm_region_mask


def test_e3sm_region_mask_negative_lon_on_0_360_grid():
    lat = np.linspace(-90, 90, 100)
    lon = np.linspace(0, 359, 100)
    data = np.ones((100, 100))
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    mask_pos = e3sm_region_mask(da, [170, 240, -5, 5])
    assert mask_pos.sum().values > 0


def test_e3sm_region_mask_neg_lon_converted_on_0_360_grid():
    lat = np.linspace(-90, 90, 100)
    lon = np.linspace(0, 359, 100)
    data = np.ones((100, 100))
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    mask_neg = e3sm_region_mask(da, [-170, -120, -5, 5])
    assert mask_neg.sum().values >= 0
