import numpy as np
import pytest
import xarray as xr

global_land_mask = pytest.importorskip("global_land_mask", reason="global_land_mask not installed")
globe = global_land_mask.globe


def test_global_land_mask_shape():
    lat = np.array([0, 10, 20])
    lon = np.array([-180, 0, 180])
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask = globe.is_land(lat_grid, lon_grid)
    assert mask.shape == lat_grid.shape
    assert mask.dtype == bool
