import numpy as np
import pytest
import xarray as xr

regionmask = pytest.importorskip("regionmask", reason="regionmask not installed")


def test_regionmask_land_mask_coords():
    resolution = 0.25
    data_var = "sftlf"
    lat_vals = np.arange(-90 + resolution / 2, 90, resolution)
    lon_vals = np.arange(0 + resolution / 2, 360, resolution)

    lat = xr.DataArray(lat_vals, dims=("lat",), coords={"lat": lat_vals}, name="lat",
                       attrs={"units": "degrees_north"})
    lon = xr.DataArray(lon_vals, dims=("lon",), coords={"lon": lon_vals}, name="lon",
                       attrs={"units": "degrees_east"})

    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(lon, lat=lat)
    sftlf = xr.where(mask.isnull(), 0.0, 1.0).rename(data_var)
    ds = xr.Dataset({data_var: sftlf})
    assert "lat" in ds.coords
    assert "lon" in ds.coords
    assert set(ds[data_var].values.ravel()).issubset({0.0, 1.0})
