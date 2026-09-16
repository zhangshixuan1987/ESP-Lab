import numpy as np
import xarray as xr
from esp_lab.data_access_e3sm import e3sm_regional_mean, e3sm_region_mask


def _uniform_global_field():
    lat = np.linspace(-90, 90, 100)
    lon = np.linspace(0, 359, 100)
    return xr.DataArray(
        np.ones((lat.size, lon.size)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )


def test_e3sm_regional_mean_uniform_field():
    res = e3sm_regional_mean(_uniform_global_field(), [-10, 10, -10, 10])
    assert np.isclose(float(res), 1.0)


def test_e3sm_regional_mean_uniform_field_across_dateline():
    result = e3sm_regional_mean(_uniform_global_field(), [170, 290, -5, 5])

    assert np.isclose(float(result), 1.0)


def test_nino34_mask_is_nonempty_and_respects_latitude_bounds():
    field = _uniform_global_field()

    mask = e3sm_region_mask(field, [170, 240, -5, 5])

    assert int(mask.sum()) > 0, "Niño3.4 must contain grid cells on a 0–360 grid"
    selected_lat = field.lat.where(mask.any("lon"), drop=True)
    assert float(selected_lat.min()) >= -5
    assert float(selected_lat.max()) <= 5


def test_equivalent_longitude_conventions_select_identical_region():
    field = _uniform_global_field()

    positive = e3sm_region_mask(field, [190, 240, -5, 5])
    negative = e3sm_region_mask(field, [-170, -120, -5, 5])

    xr.testing.assert_identical(negative, positive)
