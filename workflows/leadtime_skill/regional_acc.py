"""Regional summaries and ACC-versus-lead plots for lead-time skill maps.

The curves in this module are area-weighted summaries of the gridded ACC
fields produced by the 1a ACC notebooks.  They are deliberately *not* the
correlation of spatially averaged forecast and observation anomalies; that is
a different diagnostic with a different interpretation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr


def natural_earth_land_mask(da: xr.DataArray) -> xr.DataArray:
    """Return a boolean Natural Earth land mask on a rectilinear lat/lon grid."""
    if da["lat"].ndim != 1 or da["lon"].ndim != 1:
        raise ValueError("Regional ACC curves require one-dimensional lat/lon coordinates.")
    import regionmask

    regions = getattr(regionmask.defined_regions, "natural_earth_v5_0_0", None)
    if regions is None:  # Compatibility with older supported regionmask releases.
        regions = regionmask.defined_regions.natural_earth_v4_1_0
    land_number = regions.land_110.mask(da["lon"], da["lat"], wrap_lon=360)
    return land_number.notnull().rename("land_mask")


def _region_mask(da: xr.DataArray, bounds: Sequence[float]) -> xr.DataArray:
    west, east, south, north = map(float, bounds)
    lon2d, lat2d = xr.broadcast(da["lon"] % 360, da["lat"])
    lon2d = lon2d.transpose("lat", "lon")
    lat2d = lat2d.transpose("lat", "lon")
    west, east = west % 360, east % 360
    lon_ok = xr.ones_like(lon2d, dtype=bool) if abs(east - west) < 1e-12 and abs(bounds[1] - bounds[0]) >= 360 else (
        (lon2d >= west) & (lon2d <= east) if west <= east else (lon2d >= west) | (lon2d <= east)
    )
    return ((lat2d >= south) & (lat2d <= north) & lon_ok).rename("region_mask")


def regional_acc(
    skill: xr.Dataset | xr.DataArray,
    regions: Mapping[str, Mapping[str, object]],
    *,
    area: xr.DataArray | None = None,
    land_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Area-weighted regional mean ACC for every lead.

    ``skill`` is a 1a ACC skill dataset (with ``corr``) or its ``corr`` array.
    For Land/Ocean regions pass a grid-aligned boolean ``land_mask``; when it
    is omitted, a Natural Earth 110m mask is generated.
    """
    corr = skill["corr"] if isinstance(skill, xr.Dataset) else skill
    if not {"L", "lat", "lon"}.issubset(corr.dims):
        raise ValueError("ACC input must have L, lat, and lon dimensions.")
    if area is None:
        area, _ = xr.broadcast(np.cos(np.deg2rad(corr["lat"])), corr["lon"])
    area = area.transpose("lat", "lon")
    needs_land = any(spec.get("mask") in {"land", "ocean"} for spec in regions.values())
    if needs_land and land_mask is None:
        land_mask = natural_earth_land_mask(corr)
    if land_mask is not None:
        land_mask, _ = xr.align(land_mask.astype(bool), corr.isel(L=0, drop=True), join="exact")

    curves = []
    for name, spec in regions.items():
        keep = _region_mask(corr, spec["bounds"])
        if spec.get("mask") == "land":
            keep = keep & land_mask
        elif spec.get("mask") == "ocean":
            keep = keep & ~land_mask
        weights = area.where(keep, 0.0)
        curves.append(corr.weighted(weights).mean(("lat", "lon"), skipna=True).expand_dims(region=[name]))
    return xr.concat(curves, dim="region").rename("regional_acc")


def regional_skill_metrics(
    skill: xr.Dataset,
    regions: Mapping[str, Mapping[str, object]],
    *,
    metrics: Sequence[str] = ("corr", "rmse"),
    area: xr.DataArray | None = None,
    land_mask: xr.DataArray | None = None,
    min_obs_std_fraction: float = 0.01,
) -> xr.Dataset:
    """Area-weighted regional summaries of gridded ACC and nRMSE fields.

    In 1a ACC cache products, ``rmse`` is grid-cell normalized RMSE (nRMSE),
    i.e. RMSE divided by the observed standard deviation ``sig_obs``.  Where
    the observations barely vary (e.g. HadISST SST held at the freezing point
    under sea ice) that ratio explodes and swamps any area mean, so nRMSE
    cells with ``sig_obs`` below ``min_obs_std_fraction`` times the field's
    median ``sig_obs`` are left out.  Set it to 0 to keep every cell.

    The output remains a spatial summary of gridded metrics rather than a
    score recomputed from regionally averaged anomaly time series.
    """
    missing = set(metrics).difference(skill.data_vars)
    if missing:
        raise KeyError(f"Skill dataset is missing requested metrics: {sorted(missing)}")
    fields = {name: skill[name] for name in metrics}
    if "rmse" in fields and "sig_obs" in skill.data_vars and min_obs_std_fraction > 0:
        sig_obs = skill["sig_obs"]
        floor = float(min_obs_std_fraction) * float(sig_obs.median(skipna=True))
        fields["rmse"] = fields["rmse"].where(sig_obs > floor)
    return xr.Dataset({
        name: regional_acc(data, regions, area=area, land_mask=land_mask)
        for name, data in fields.items()
    })


