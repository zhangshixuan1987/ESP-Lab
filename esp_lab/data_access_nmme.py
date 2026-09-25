"""
Utilities for accessing and preprocessing NMME (North American Multi-Model Ensemble)
hindcast data.

Expected file naming convention
--------------------------------
<datadir>/<field>-<modelname>.nc

Example
-------
/path/to/nmme/sst-COLA-RSMAS-CCSM4.nc

Notes
-----
- NMME files use an "S" coordinate for initialization time.
- Some NMME models store calendar metadata as "360" rather than "360_day";
  this module corrects that automatically.
- Regional SST anomalies are computed by removing a monthly climatology
  (ensemble-mean). For COLA-RSMAS-CCSM4 and NCEP-CFSv2, a split
  climatology is applied: 1982–1998 and 1999–2016.
"""

import fcntl
import os
from functools import lru_cache
from pathlib import Path
from typing import Union
from uuid import uuid4

import numpy as np
import xarray as xr


_SPLIT_CLIMO_MODELS = {"COLA-RSMAS-CCSM4", "NCEP-CFSv2"}
_SPLIT_PERIOD_1 = ("1982-01-01", "1998-12-01")
_SPLIT_PERIOD_2 = ("1999-01-01", "2016-12-01")
_DEFAULT_INIT_MONTHS = [2, 5, 8, 11]
NMME_SST_MASK_VERSION = 3
NMME_LAND_MASK_VERSION = 1

_KELVIN_UNITS = {"k", "kelvin", "kelvin_scale", "degree_k", "degrees_k"}
_CELSIUS_UNITS = {
    "c",
    "degc",
    "degree_c",
    "degrees_c",
    "celsius",
    "degree_celsius",
    "degrees_celsius",
}


@lru_cache(maxsize=8)
def _natural_earth_ocean_mask_values(
    longitude: tuple[float, ...], latitude: tuple[float, ...]
) -> np.ndarray:
    """Return a cached Natural Earth ocean mask for a regular grid."""
    import regionmask

    lon_2d, lat_2d = np.meshgrid(
        np.asarray(longitude),
        np.asarray(latitude),
    )
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    return np.asarray(
        land.mask(lon_2d, lat=lat_2d, wrap_lon=360).isnull(),
        dtype=bool,
    )


def _natural_earth_ocean_mask(
    sst: xr.DataArray, lon_name: str, lat_name: str
) -> xr.DataArray:
    lon = tuple(np.asarray(sst[lon_name], dtype=float).tolist())
    lat = tuple(np.asarray(sst[lat_name], dtype=float).tolist())
    values = _natural_earth_ocean_mask_values(lon, lat)
    return xr.DataArray(
        values,
        dims=(lat_name, lon_name),
        coords={lat_name: sst[lat_name], lon_name: sst[lon_name]},
        name="ocean_mask",
    )


def nmme_land_mask_path(model: str, fixed_dir: str | Path) -> Path:
    """Return the fixed-field path for one model's NMME land mask."""
    safe_model = model.replace("/", "_")
    return Path(fixed_dir) / f"sftlf.NMME.{safe_model}.nc"


def _cached_land_mask_matches_grid(
    dataset: xr.Dataset,
    sst: xr.DataArray,
    lon_name: str,
    lat_name: str,
    model: str,
) -> bool:
    if "sftlf" not in dataset:
        return False
    if dataset.attrs.get("nmme_land_mask_version") != NMME_LAND_MASK_VERSION:
        return False
    if dataset.attrs.get("model") != model:
        return False
    for name in (lon_name, lat_name):
        if name not in dataset.coords or dataset[name].shape != sst[name].shape:
            return False
        if not np.allclose(dataset[name], sst[name], rtol=0.0, atol=1.0e-10):
            return False
    return True


def load_or_create_nmme_land_mask(
    sst: xr.DataArray,
    *,
    model: str,
    fixed_dir: str | Path,
    lon_name: str,
    lat_name: str,
) -> tuple[xr.DataArray, Path]:
    """Load or atomically create a grid-validated land mask for one model."""
    path = nmme_land_mask_path(model, fixed_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if path.exists():
            try:
                with xr.open_dataset(path) as cached:
                    if _cached_land_mask_matches_grid(
                        cached, sst, lon_name, lat_name, model
                    ):
                        return cached["sftlf"].load().astype(bool), path
            except Exception:
                pass

        ocean_mask = _natural_earth_ocean_mask(sst, lon_name, lat_name)
        land_mask = (~ocean_mask).astype("int8").rename("sftlf")
        land_mask.attrs.update(
            long_name="Natural Earth land mask on the native NMME model grid",
            units="1",
            flag_values=np.asarray([0, 1], dtype="int8"),
            flag_meanings="ocean land",
        )
        output = land_mask.to_dataset()
        output.attrs.update(
            model=model,
            source="Natural Earth v5.0.0 land_110",
            nmme_land_mask_version=NMME_LAND_MASK_VERSION,
        )
        temporary = path.with_name(f".{path.name}.tmp.{uuid4().hex}")
        try:
            output.to_netcdf(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return land_mask.astype(bool), path


def mask_invalid_sst(
    sst: xr.DataArray,
    *,
    apply_land_mask: bool = True,
    lon_name: str | None = None,
    lat_name: str | None = None,
    model: str | None = None,
    fixed_dir: str | Path | None = None,
) -> xr.DataArray:
    """Mask nonphysical finite fill values in NMME SST data.

    Some member archives encode land as 0 K instead of a missing value.  This
    normalization belongs at the data-access boundary so every downstream SST
    diagnostic sees the same valid field.
    """
    valid = np.isfinite(sst)
    units = str(sst.attrs.get("units", "")).strip().lower()
    scale_min = sst.attrs.get("scale_min")
    try:
        scale_min_value = float(scale_min)
    except (TypeError, ValueError):
        scale_min_value = np.nan

    if units in _KELVIN_UNITS:
        lower = scale_min_value if np.isfinite(scale_min_value) else 260.0
        if lower < 100.0:
            lower += 273.15
        upper = 330.0
        sanity_description = f"{lower:g} <= SST <= {upper:g} K"
    elif units in _CELSIUS_UNITS:
        lower = scale_min_value if np.isfinite(scale_min_value) else -13.15
        if lower > 100.0:
            lower -= 273.15
        upper = 56.85
        sanity_description = f"{lower:g} <= SST <= {upper:g} degC"
    elif np.isfinite(scale_min_value):
        lower = scale_min_value
        upper = None
        sanity_description = f"SST >= archive scale_min ({lower:g})"
    else:
        lower = None
        upper = None
        sanity_description = "finite SST"

    if lower is not None:
        valid = valid & (sst >= lower)
    if upper is not None:
        valid = valid & (sst <= upper)

    if apply_land_mask:
        lon_name = lon_name or next(
            (name for name in ("X", "lon", "longitude") if name in sst.coords),
            None,
        )
        lat_name = lat_name or next(
            (name for name in ("Y", "lat", "latitude") if name in sst.coords),
            None,
        )
        if lon_name is None or lat_name is None:
            raise ValueError(
                "Could not identify longitude/latitude coordinates for the "
                "NMME SST land mask."
            )
        land_mask_file = None
        if model is not None and fixed_dir is not None:
            land_mask, land_mask_file = load_or_create_nmme_land_mask(
                sst,
                model=model,
                fixed_dir=fixed_dir,
                lon_name=lon_name,
                lat_name=lat_name,
            )
            ocean_mask = ~land_mask
        else:
            ocean_mask = _natural_earth_ocean_mask(sst, lon_name, lat_name)
        valid = valid & ocean_mask
    else:
        land_mask_file = None

    masked = sst.where(valid)
    masked.attrs.update(sst.attrs)
    masked.attrs.update(
        valid_sst_mask=sanity_description,
        nmme_sst_land_mask=str(bool(apply_land_mask)),
        nmme_sst_mask_version=NMME_SST_MASK_VERSION,
    )
    if land_mask_file is not None:
        masked.attrs["nmme_sst_land_mask_file"] = str(land_mask_file)
    return masked


def decode_cf_time(ds: xr.Dataset, time_var: str = "S") -> xr.Dataset:
    """
    Decode CF-style time, fixing common NMME calendar metadata issues.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (with undecoded time).
    time_var : str
        Name of the time/initialization coordinate, default "S".

    Returns
    -------
    xr.Dataset
        Dataset with decoded time coordinate.

    Raises
    ------
    ValueError
        If `time_var` is not found in `ds`.
    """
    if time_var not in ds.coords and time_var not in ds.dims:
        raise ValueError(
            f"Time variable '{time_var}' not found in dataset. "
            f"Available coordinates: {list(ds.coords)}"
        )

    ds = ds.copy()
    cal = ds[time_var].attrs.get("calendar", None)
    if cal == "360":
        ds[time_var].attrs["calendar"] = "360_day"

    try:
        return xr.decode_cf(ds, decode_times=True)
    except Exception as e:
        raise ValueError(
            f"xr.decode_cf failed for time variable '{time_var}': {e}"
        ) from e


def subset_init_month(
    nmme_regsst: xr.DataArray,
    init_month: int,
    time_dim: str = "S",
) -> xr.DataArray:
    """
    Select one initialization month and convert to SMYLE-like format:
      S -> Y (integer years)
      L becomes 1..N

    Parameters
    ----------
    nmme_regsst : xr.DataArray
        Multi-model or single-model SST anomaly with an initialization time dim.
    init_month : int
        Initialization month to select (1–12).
    time_dim : str, optional
        Name of the initialization time coordinate, default 'S'.

    Returns
    -------
    xr.DataArray
        Subset DataArray with 'Y' (integer year) and 'L' lead-time dimensions.

    Raises
    ------
    ValueError
        If `init_month` is out of range, `time_dim` is missing, or the
        resulting selection is empty.
    """
    if not (1 <= init_month <= 12):
        raise ValueError(f"init_month must be in 1–12, got {init_month}.")

    if time_dim not in nmme_regsst.dims and time_dim not in nmme_regsst.coords:
        raise ValueError(
            f"time_dim='{time_dim}' not found in DataArray. "
            f"Available dims: {list(nmme_regsst.dims)}"
        )

    nmme_time = nmme_regsst[time_dim]
    timekeep = nmme_time.where(nmme_time.dt.month == init_month).dropna(time_dim)

    if timekeep.sizes[time_dim] == 0:
        raise ValueError(
            f"No initialization times found for month={init_month} "
            f"in the '{time_dim}' coordinate."
        )

    da = nmme_regsst.sel({time_dim: timekeep}).rename({time_dim: "Y"})
    da = da.assign_coords(Y=da.Y.dt.year)

    if "L" in da.dims:
        da = da.assign_coords(L=np.arange(da.sizes["L"]) + 1)

    return da


