import numpy as np
import pytest
import xarray as xr
from esp_lab.utils.spatial_utils import (
    _build_regionmask_land_source,
    _normalize_lon_180,
    compute_regional_average,
)


def test_longitude_normalization_maps_equivalent_coordinates_consistently():
    longitudes = np.array([0.0, 180.0, 190.0, 350.0, 360.0])

    np.testing.assert_allclose(
        _normalize_lon_180(longitudes),
        [0.0, -180.0, -170.0, -10.0, 0.0],
    )


def test_regional_average_basic():
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(0, 360, 20)
    data = np.ones((10, 20))
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    res = compute_regional_average(da, (-10, 10), (10, 20))
    assert np.isclose(res.values, 1.0)


def test_land_and_ocean_regional_averages_preserve_uniform_field():
    pytest.importorskip("regionmask", reason="regionmask not installed")
    lat = np.arange(-85.0, 86.0, 10.0)
    lon = np.arange(-175.0, 176.0, 10.0)
    field = xr.DataArray(
        np.ones((lat.size, lon.size)),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )

    land = compute_regional_average(
        field, (-90, 90), (-180, 180), land_mask=True
    )
    ocean = compute_regional_average(
        field, (-90, 90), (-180, 180), ocean_mask=True
    )

    assert np.isclose(float(land), 1.0)
    assert np.isclose(float(ocean), 1.0)


def test_regionmask_land_source_has_geographic_coordinates_and_binary_values():
    pytest.importorskip("regionmask", reason="regionmask not installed")

    source = _build_regionmask_land_source(resolution=10.0)

    assert set(source.sftlf.dims) == {"lat", "lon"}
    assert float(source.lat.min()) > -90.0
    assert float(source.lat.max()) < 90.0
    assert float(source.lon.min()) >= 0.0
    assert float(source.lon.max()) < 360.0
    assert np.all(np.diff(source.lat) > 0)
    assert np.all(np.diff(source.lon) > 0)
    assert set(np.unique(source.sftlf)) == {0.0, 1.0}
