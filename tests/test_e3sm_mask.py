import numpy as np
import xarray as xr
from esp_lab.data_access_e3sm import e3sm_regional_weights, e3sm_region_mask


def _make_da():
    lat = np.linspace(-90, 90, 100)
    lon = np.linspace(0, 359, 100)
    data = np.ones((100, 100))
    return xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])


def test_e3sm_region_mask_shape():
    da = _make_da()
    mask = e3sm_region_mask(da, [-170, -120, -5, 5])
    assert mask.shape == da.shape


def test_e3sm_regional_weights_zero_outside_region():
    da = _make_da()
    mask = e3sm_region_mask(da, [-170, -120, -5, 5])
    weights = e3sm_regional_weights(da, [-170, -120, -5, 5])
    bool_mask = mask.astype(bool)
    assert (weights.where(~bool_mask).fillna(0) == 0).all()


def test_e3sm_regional_weights_nonzero_inside_region():
    da = _make_da()
    mask = e3sm_region_mask(da, [-170, -120, -5, 5])
    weights = e3sm_regional_weights(da, [-170, -120, -5, 5])
    bool_mask = mask.astype(bool)
    assert weights.where(bool_mask).count().values > 0


def test_e3sm_region_mask_treats_zero_to_360_as_full_longitude():
    da = xr.DataArray(
        np.ones((3, 4)),
        dims=("lat", "lon"),
        coords={"lat": [-30.0, 0.0, 30.0], "lon": [0.0, 90.0, 180.0, 270.0]},
    )

    mask = e3sm_region_mask(da, [0.0, 360.0, -20.0, 20.0])

    assert mask.sel(lat=0.0).all()
    assert not mask.sel(lat=-30.0).any()
    assert not mask.sel(lat=30.0).any()
