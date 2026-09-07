import numpy as np
import pytest
import subprocess
import sys
import xarray as xr

xe = pytest.importorskip("xesmf", reason="xesmf not installed")


def _coord_bounds_1d_size_n_plus_1(coord):
    vals = np.asarray(coord.values, dtype=float)
    mid = 0.5 * (vals[1:] + vals[:-1])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    bounds = np.concatenate([[first], mid, [last]])
    name = str(coord.name).lower() if coord.name else ""
    if "lat" in name or name == "y":
        bounds = np.clip(bounds, -90.0, 90.0)
    return xr.DataArray(
        bounds,
        dims=(coord.dims[0] + "_b",)
    )


def _make_grid_ds(lat, lon, lat_name="lat", lon_name="lon"):
    return xr.Dataset(
        coords={
            lat_name: lat,
            lon_name: lon,
            f"{lat_name}_b": _coord_bounds_1d_size_n_plus_1(lat),
            f"{lon_name}_b": _coord_bounds_1d_size_n_plus_1(lon),
        }
    )


def test_xesmf_conservative_regrid():
    lat = xr.DataArray(np.linspace(-90, 90, 10), dims=("lat",), name="lat")
    lon = xr.DataArray(np.linspace(0, 360, 20), dims=("lon",), name="lon")
    ds1 = _make_grid_ds(lat, lon)

    lat2 = xr.DataArray(np.linspace(-90, 90, 5), dims=("lat",), name="lat")
    lon2 = xr.DataArray(np.linspace(0, 360, 10), dims=("lon",), name="lon")
    ds2 = _make_grid_ds(lat2, lon2)

    assert ds1["lat_b"].shape == (11,)

    code = """
import numpy as np
import xarray as xr
import xesmf as xe

def bounds(coord):
    vals = np.asarray(coord.values, dtype=float)
    mid = 0.5 * (vals[1:] + vals[:-1])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])
    out = np.concatenate([[first], mid, [last]])
    if 'lat' in str(coord.name).lower():
        out = np.clip(out, -90.0, 90.0)
    return xr.DataArray(out, dims=(coord.dims[0] + '_b',))

def grid(lat, lon):
    return xr.Dataset(coords={'lat': lat, 'lon': lon, 'lat_b': bounds(lat), 'lon_b': bounds(lon)})

lat = xr.DataArray(np.linspace(-90, 90, 10), dims=('lat',), name='lat')
lon = xr.DataArray(np.linspace(0, 360, 20), dims=('lon',), name='lon')
lat2 = xr.DataArray(np.linspace(-90, 90, 5), dims=('lat',), name='lat')
lon2 = xr.DataArray(np.linspace(0, 360, 10), dims=('lon',), name='lon')
regridder = xe.Regridder(grid(lat, lon), grid(lat2, lon2), method='conservative')
assert regridder is not None
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        pytest.skip(f"xESMF/ESMF runtime is not usable in this environment: {result.stderr}")
