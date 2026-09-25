"""Lazy unit conversions shared by prediction-skill workflows."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr

# MPAS-Ocean seawater constants (also used by the EN4 OHC700 product).
SEAWATER_DENSITY = 1026.0  # kg m-3
SEAWATER_HEAT_CAPACITY = 3996.0  # J kg-1 K-1
KELVIN_OFFSET = 273.15


def convert_kelvin_to_celsius(data):
    """Convert kelvin to degrees Celsius while preserving lazy evaluation."""
    converted = data - 273.15
    converted.attrs["units"] = r"$^\circ$C"
    return converted


def convert_precip_mps_to_mmday(data):
    """Convert precipitation from metres per second to millimetres per day."""
    converted = data * (1000.0 * 86400.0)
    converted.attrs["units"] = "mm/day"
    return converted


def convert_pa_to_hpa(data):
    """Convert pressure from pascals to hectopascals."""
    converted = data * 1.0e-2
    converted.attrs["units"] = "hPa"
    return converted


def no_unit_conversion(data, units=None):
    """Return a lazy copy and optionally standardize its unit attribute."""
    converted = data * 1.0
    if units is not None:
        converted.attrs["units"] = units
    return converted


def _box_edges(centers):
    centers = np.asarray(centers, dtype=float)
    step = float(np.mean(np.diff(centers)))
    return np.concatenate([centers - step / 2.0, [centers[-1] + step / 2.0]])


@lru_cache(maxsize=8)
def _mpas_box_means(mesh_path, lat_key, lon_key, depth_m):
    """Mean native cell area and column depth (capped at depth_m) per lat/lon box."""
    from scipy.spatial import cKDTree

    lat = np.asarray(lat_key, dtype=float)
    lon = np.mod(np.asarray(lon_key, dtype=float), 360.0)
    with xr.open_dataset(mesh_path) as mesh:
        lat_cell = np.degrees(mesh["latCell"].values)
        lon_cell = np.mod(np.degrees(mesh["lonCell"].values), 360.0)
        area = mesh["areaCell"].values.astype(float)
        depth = np.minimum(mesh["bottomDepth"].values.astype(float), float(depth_m))
    order = np.argsort(lon)
    iy = np.clip(np.digitize(lat_cell, _box_edges(lat)) - 1, 0, lat.size - 1)
    ix_sorted = np.clip(np.digitize(lon_cell, _box_edges(lon[order])) - 1, 0, lon.size - 1)
    ix = order[ix_sorted]
    flat = iy * lon.size + ix
    count = np.bincount(flat, minlength=lat.size * lon.size).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_area = np.bincount(flat, area, lat.size * lon.size) / count
        mean_depth = np.bincount(flat, depth, lat.size * lon.size) / count
    empty = count == 0
    if empty.any():
        # Boxes narrower than a cell (near the poles) take their nearest cell.
        def xyz(la, lo):
            la, lo = np.radians(la), np.radians(lo)
            return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])

        box_lat, box_lon = np.meshgrid(lat, lon, indexing="ij")
        _, nearest = cKDTree(xyz(lat_cell, lon_cell)).query(xyz(box_lat.ravel()[empty], box_lon.ravel()[empty]))
        mean_area[empty] = area[nearest]
        mean_depth[empty] = depth[nearest]
    shape = (lat.size, lon.size)
    return mean_area.reshape(shape), mean_depth.reshape(shape)


def convert_mpas_ohc_to_jm2(data, *, depth_m=700.0, mesh_path=None):
    """Convert MPAS-Ocean ``oceanHeatContentSfcTo{depth}m`` to J m-2 like EN4.

    The MPAS diagnostic is cell-integrated (J, i.e. multiplied by the native
    cell area) and uses temperature in kelvin. On a regridded lat/lon field this
    divides by the mean native cell area of each box and removes the kelvin
    offset rho0*cp*273.15*min(depth_m, bottom depth), giving rho0*cp*sum(T dz)
    with T in degC, EN4's definition.

    Anomalies (ACC, anomaly RMSE) depend only on the area scaling, which is
    accurate to about 1%. The absolute level is sensitive to the offset, which
    is ~20x the heat-content signal, so absolute bias/RMSE carry roughly
    10-30% uncertainty (larger at high latitudes).
    """
    if mesh_path is None:
        from esp_lab.diagnostics.native_eli import DEFAULT_MPAS_MESH_FILE

        mesh_path = DEFAULT_MPAS_MESH_FILE
    area, depth = _mpas_box_means(
        str(mesh_path), tuple(np.asarray(data["lat"].values, dtype=float)),
        tuple(np.asarray(data["lon"].values, dtype=float)), float(depth_m),
    )
    coords = {"lat": data["lat"], "lon": data["lon"]}
    area = xr.DataArray(area, dims=("lat", "lon"), coords=coords)
    offset = xr.DataArray(
        SEAWATER_DENSITY * SEAWATER_HEAT_CAPACITY * KELVIN_OFFSET * depth,
        dims=("lat", "lon"), coords=coords,
    )
    converted = data / area - offset
    converted.attrs = dict(data.attrs)
    converted.attrs["units"] = "J m-2"
    return converted


__all__ = [
    "convert_kelvin_to_celsius",
    "convert_mpas_ohc_to_jm2",
    "convert_pa_to_hpa",
    "convert_precip_mps_to_mmday",
    "no_unit_conversion",
]
