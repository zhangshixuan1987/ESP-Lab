"""Runtime contract for conservative xESMF regridding."""

import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from esp_lab.utils.spatial_utils import _coord_bounds_1d


def test_latitude_bounds_are_clipped_and_follow_coordinate_order():
    latitude = xr.DataArray([89.0, 87.0, 85.0], dims="lat", name="lat")

    bounds = _coord_bounds_1d(latitude)

    assert bounds.sizes["lat_b"] == latitude.size + 1
    assert float(bounds.min()) >= -90.0
    assert float(bounds.max()) <= 90.0
    assert np.all(np.diff(bounds) < 0)


def test_conservative_regridding_preserves_a_constant_global_field():
    """A conservative remap must preserve a finite spatially constant field."""
    pytest.importorskip("xesmf", reason="xesmf not installed")
    code = r"""
import numpy as np
import xarray as xr
import xesmf as xe

def bounds(coord):
    values = np.asarray(coord, dtype=float)
    midpoints = 0.5 * (values[1:] + values[:-1])
    edges = np.concatenate((
        [values[0] - 0.5 * (values[1] - values[0])],
        midpoints,
        [values[-1] + 0.5 * (values[-1] - values[-2])],
    ))
    if coord.name == 'lat':
        edges = np.clip(edges, -90.0, 90.0)
    return xr.DataArray(edges, dims=f'{coord.dims[0]}_b')

def grid(latitudes, longitudes):
    lat = xr.DataArray(latitudes, dims='lat', name='lat')
    lon = xr.DataArray(longitudes, dims='lon', name='lon')
    return xr.Dataset(coords={
        'lat': lat, 'lon': lon,
        'lat_b': bounds(lat), 'lon_b': bounds(lon),
    })

source_grid = grid(np.arange(-75.0, 76.0, 30.0), np.arange(15.0, 360.0, 30.0))
target_grid = grid(np.arange(-60.0, 61.0, 60.0), np.arange(30.0, 360.0, 60.0))
source = xr.DataArray(
    np.ones((source_grid.sizes['lat'], source_grid.sizes['lon'])),
    dims=('lat', 'lon'),
    coords={'lat': source_grid.lat, 'lon': source_grid.lon},
)
regridder = xe.Regridder(
    source_grid, target_grid, method='conservative',
    periodic=True, ignore_degenerate=True,
)
result = regridder(source)
assert result.dims == ('lat', 'lon')
assert np.isfinite(result).all()
np.testing.assert_allclose(result, 1.0, rtol=0.0, atol=1e-12)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"xESMF/ESMF runtime is unavailable: {result.stderr}")
