"""MPAS cell-integrated OHC (J, kelvin) converts to EN4-style J m-2 per lat/lon box."""

import numpy as np
import xarray as xr

from esp_lab.utils.unit_conversion import (
    KELVIN_OFFSET,
    SEAWATER_DENSITY,
    SEAWATER_HEAT_CAPACITY,
    convert_mpas_ohc_to_jm2,
)


def _mesh(path, lat, lon, area, bottom_depth):
    xr.Dataset({
        "latCell": ("nCells", np.radians(lat)),
        "lonCell": ("nCells", np.radians(lon)),
        "areaCell": ("nCells", np.asarray(area, dtype=float)),
        "bottomDepth": ("nCells", np.asarray(bottom_depth, dtype=float)),
    }).to_netcdf(path)
    return path


def test_converts_cell_integrated_kelvin_ohc_to_jm2(tmp_path):
    # Two 10-degree boxes: a deep one (two cells) and a shelf box (one 200 m cell).
    mesh = _mesh(tmp_path / "mesh.nc", lat=[5, 5, 5], lon=[4, 6, 15],
                 area=[1.0e9, 3.0e9, 2.0e9], bottom_depth=[4000, 4000, 200])
    rho_cp = SEAWATER_DENSITY * SEAWATER_HEAT_CAPACITY
    ohc_degc = np.array([3.0e10, 1.0e10])  # target J m-2 (T in degC)
    mean_area = np.array([2.0e9, 2.0e9])
    depth = np.array([700.0, 200.0])
    # What MPAS reports: kelvin-based column heat times cell area.
    mpas = (ohc_degc + rho_cp * KELVIN_OFFSET * depth) * mean_area
    da = xr.DataArray(mpas[None, :], dims=("lat", "lon"), coords={"lat": [5.0], "lon": [5.0, 15.0]},
                      attrs={"units": "J"})

    out = convert_mpas_ohc_to_jm2(da, mesh_path=mesh)

    np.testing.assert_allclose(out.values[0], ohc_degc, rtol=1e-6)
    assert out.attrs["units"] == "J m-2"


def test_empty_boxes_use_nearest_cell_and_stay_lazy(tmp_path):
    mesh = _mesh(tmp_path / "mesh.nc", lat=[0.5], lon=[0.5], area=[1.0e9], bottom_depth=[5000])
    da = xr.DataArray(np.ones((2, 1, 2)), dims=("time", "lat", "lon"),
                      coords={"lat": [0.5], "lon": [0.5, 1.5]}).chunk({"time": 1})

    out = convert_mpas_ohc_to_jm2(da, mesh_path=mesh)

    assert out.chunks is not None  # still lazy
    values = out.compute().values
    assert np.isfinite(values).all() and np.allclose(values[..., 0], values[..., 1])
