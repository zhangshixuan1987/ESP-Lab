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
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import xarray as xr

from esp_lab.paths import NMME_FIXED_DIR

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


def nmme_land_mask_path(model: str, fixed_dir: str | Path = NMME_FIXED_DIR) -> Path:
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
    fixed_dir: str | Path = NMME_FIXED_DIR,
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


def open_nmme_model(
    datadir: str,
    modelname: str,
    field: str = "sst",
    s_slice: Optional[Tuple[Any, Any]] = ("264", "684"),
    time_var: str = "S",
    apply_sst_land_mask: bool = True,
    fixed_dir: str | Path = NMME_FIXED_DIR,
) -> xr.Dataset:
    """
    Open one NMME model file and decode its time coordinate.

    Parameters
    ----------
    datadir : str
        Directory containing NMME model files.
    modelname : str
        Model name, e.g. 'COLA-RSMAS-CCSM4'.
    field : str, optional
        Variable name used in the filename pattern, default 'sst'.
    s_slice : tuple of (str or None), optional
        (start, end) values for the initialization time coordinate.
        Set to None to skip subsetting.
    time_var : str, optional
        Name of the initialization time coordinate, default 'S'.
    apply_sst_land_mask : bool, optional
        Apply the Natural Earth land mask when ``field='sst'``, default True.
    fixed_dir : path-like, optional
        Directory containing reusable per-model NMME land masks.

    Returns
    -------
    xr.Dataset
        Opened and time-decoded dataset.

    Raises
    ------
    FileNotFoundError
        If the expected file does not exist.
    ValueError
        If time decoding fails or `time_var` is not found.
    """
    if not isinstance(datadir, str):
        raise TypeError(f"datadir must be a string, got {type(datadir).__name__}.")
    if not isinstance(modelname, str) or not modelname:
        raise TypeError("modelname must be a non-empty string.")
    if not isinstance(field, str) or not field:
        raise TypeError("field must be a non-empty string.")

    file_path = Path(datadir) / f"{field}-{modelname}.nc"
    if not file_path.exists():
        raise FileNotFoundError(f"NMME model file not found: {file_path}")

    ds = xr.open_dataset(str(file_path), decode_times=False)

    if time_var not in ds.coords and time_var not in ds.dims:
        raise ValueError(
            f"time_var='{time_var}' not found in {file_path.name}. "
            f"Available coordinates: {list(ds.coords)}"
        )

    if s_slice is not None:
        if len(s_slice) != 2:
            raise ValueError(f"s_slice must have exactly 2 elements, got {len(s_slice)}.")
        ds = ds.sel({time_var: slice(*s_slice)})
        if ds.sizes.get(time_var, 0) == 0:
            warnings.warn(
                f"s_slice={s_slice} produced an empty selection along '{time_var}' "
                f"for model {modelname}."
            )

    ds = decode_cf_time(ds, time_var=time_var)
    if field.lower() == "sst" and field in ds:
        ds[field] = mask_invalid_sst(
            ds[field],
            apply_land_mask=apply_sst_land_mask,
            model=modelname,
            fixed_dir=fixed_dir,
        )
    return ds


def compute_regional_mean_sst(
    ds: xr.Dataset,
    regionlonlat: Iterable,
    field: str = "sst",
    lon_name: str = "X",
    lat_name: str = "Y",
) -> xr.DataArray:
    """
    Compute area-weighted regional mean SST.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    regionlonlat : sequence of length 4
        [lon1, lon2, lat1, lat2]. Longitudes are expected in [-180, 180];
        they are converted to the [0, 360) convention used by NMME files.
    field : str, optional
        Variable name in `ds`, default 'sst'.
    lon_name : str, optional
        Longitude dimension name, default 'X'.
    lat_name : str, optional
        Latitude dimension name, default 'Y'.

    Returns
    -------
    xr.DataArray
        Regional mean SST.

    Raises
    ------
    ValueError
        If inputs are invalid or the region selection is empty.
    """
    regionlonlat = list(regionlonlat)
    if len(regionlonlat) != 4:
        raise ValueError(
            f"regionlonlat must have exactly 4 elements [lon1, lon2, lat1, lat2], "
            f"got {len(regionlonlat)}."
        )
    lon1, lon2, lat1, lat2 = regionlonlat
    if lat1 > lat2:
        raise ValueError(f"lat1 ({lat1}) must be <= lat2 ({lat2}).")

    if field not in ds:
        raise ValueError(
            f"Field '{field}' not found in dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )
    if lat_name not in ds.dims and lat_name not in ds.coords:
        raise ValueError(f"Latitude dimension '{lat_name}' not found in dataset.")
    if lon_name not in ds.dims and lon_name not in ds.coords:
        raise ValueError(f"Longitude dimension '{lon_name}' not found in dataset.")

    # Convert [-180, 180] bounds to the [0, 360) convention used by NMME files
    lon_vals = ds[lon_name].values
    uses_360 = np.nanmin(lon_vals) >= 0 and np.nanmax(lon_vals) > 180
    if uses_360:
        lon1_conv = lon1 % 360
        lon2_conv = lon2 % 360
    else:
        lon1_conv = lon1
        lon2_conv = lon2

    da = ds[field].sel(
        {lat_name: slice(lat1, lat2), lon_name: slice(lon1_conv, lon2_conv)}
    )

    if da.sizes.get(lat_name, 0) == 0 or da.sizes.get(lon_name, 0) == 0:
        raise ValueError(
            f"Regional selection is empty for regionlonlat={regionlonlat}. "
            "Check that bounds overlap the data domain."
        )

    wgts = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(wgts).mean((lon_name, lat_name))


def remove_monthly_climatology(
    regsst: xr.DataArray,
    modelname: Optional[str] = None,
    split_models: Optional[set] = None,
    time_dim: str = "S",
    member_dim: str = "M",
) -> xr.DataArray:
    """
    Remove monthly climatology from regional SST.

    For models in `split_models` (default: COLA-RSMAS-CCSM4 and NCEP-CFSv2),
    two separate climatology periods are used:
      - Period 1: 1982–1998
      - Period 2: 1999–2016

    Parameters
    ----------
    regsst : xr.DataArray
        Regional mean SST with time and member dimensions.
    modelname : str, optional
        Name of the model, used to select split-climatology treatment.
    split_models : set, optional
        Set of model names requiring split climatology.
        Defaults to {'COLA-RSMAS-CCSM4', 'NCEP-CFSv2'}.
    time_dim : str, optional
        Time/initialization coordinate name, default 'S'.
    member_dim : str, optional
        Member dimension name, default 'M'.

    Returns
    -------
    xr.DataArray
        SST anomaly with climatology removed.

    Raises
    ------
    ValueError
        If required dimensions are not found in `regsst`.
    """
    if time_dim not in regsst.dims and time_dim not in regsst.coords:
        raise ValueError(
            f"time_dim='{time_dim}' not found in DataArray. "
            f"Available dims: {list(regsst.dims)}"
        )
    if member_dim not in regsst.dims:
        raise ValueError(
            f"member_dim='{member_dim}' not found in DataArray. "
            f"Available dims: {list(regsst.dims)}"
        )

    if split_models is None:
        split_models = _SPLIT_CLIMO_MODELS

    if modelname in split_models:
        sst1 = regsst.sel({time_dim: slice(*_SPLIT_PERIOD_1)})
        if sst1.sizes.get(time_dim, 0) == 0:
            warnings.warn(
                f"Split climatology period 1 ({_SPLIT_PERIOD_1}) is empty "
                f"for model {modelname}."
            )
        ssta1 = (
            sst1.groupby(f"{time_dim}.month")
            - sst1.groupby(f"{time_dim}.month").mean((time_dim, member_dim))
        )

        sst2 = regsst.sel({time_dim: slice(*_SPLIT_PERIOD_2)})
        if sst2.sizes.get(time_dim, 0) == 0:
            warnings.warn(
                f"Split climatology period 2 ({_SPLIT_PERIOD_2}) is empty "
                f"for model {modelname}."
            )
        ssta2 = (
            sst2.groupby(f"{time_dim}.month")
            - sst2.groupby(f"{time_dim}.month").mean((time_dim, member_dim))
        )

        regsst = xr.concat([ssta1, ssta2], dim=time_dim)
    else:
        regsst = (
            regsst.groupby(f"{time_dim}.month")
            - regsst.groupby(f"{time_dim}.month").mean((time_dim, member_dim))
        )

    return regsst


def load_nmme_multimodel_regsst(
    modelnames: List[str],
    datadir: str,
    regionlonlat: Iterable,
    field: str = "sst",
    s_slice: Optional[Tuple[Any, Any]] = ("264", "684"),
) -> xr.DataArray:
    """
    Load all NMME models, compute regional SST anomalies, and concatenate
    across model.

    Parameters
    ----------
    modelnames : list of str
        Model names to load.
    datadir : str
        Directory containing NMME model files.
    regionlonlat : sequence of length 4
        [lon1, lon2, lat1, lat2] for regional averaging.
    field : str, optional
        Variable name, default 'sst'.
    s_slice : tuple, optional
        Initialization time slice.

    Returns
    -------
    xr.DataArray
        Concatenated SST anomalies with a 'model' dimension.

    Raises
    ------
    ValueError
        If `modelnames` is empty or no models loaded successfully.
    """
    if not modelnames:
        raise ValueError("modelnames must be a non-empty list.")

    out = []
    failed = []

    for modelname in modelnames:
        try:
            ds = open_nmme_model(
                datadir=datadir,
                modelname=modelname,
                field=field,
                s_slice=s_slice,
                time_var="S",
            )

            regsst = compute_regional_mean_sst(ds, regionlonlat=regionlonlat, field=field)
            regsst = remove_monthly_climatology(regsst, modelname=modelname)

            mem = ds.sizes.get("M", None)
            warnings.warn(f"[NMME] {modelname}: M={mem}", stacklevel=2)

            regsst = regsst.expand_dims(model=[modelname])
            out.append(regsst)
        except Exception as e:
            warnings.warn(
                f"[NMME] Failed to load model '{modelname}': {e}. Skipping.",
                stacklevel=2,
            )
            failed.append(modelname)

    if not out:
        raise ValueError(
            f"No models loaded successfully. Failed models: {failed}"
        )

    if failed:
        warnings.warn(
            f"[NMME] The following models were skipped due to errors: {failed}",
            stacklevel=2,
        )

    return xr.concat(out, dim="model")


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


def attach_time_and_seasonalize(
    da: xr.DataArray,
    smyle_time: xr.DataArray,
    cal: Any,
    field_name: str = "sst",
) -> Tuple[xr.Dataset, xr.DataArray]:
    """
    Convert monthly hindcast DataArray to Dataset, attach verification time,
    and compute seasonal means using cal.mon_to_seas_dask.

    Parameters
    ----------
    da : xr.DataArray
        Monthly hindcast DataArray with a 'Y' dimension.
    smyle_time : xr.DataArray
        Verification time DataArray indexed by 'Y'.
    cal : object
        Calendar utility object with a `mon_to_seas_dask(ds)` method.
    field_name : str, optional
        Variable name to use in the output Dataset, default 'sst'.

    Returns
    -------
    ds : xr.Dataset
        Monthly dataset with attached verification time.
    seas : xr.DataArray
        Seasonally averaged DataArray.

    Raises
    ------
    TypeError
        If `cal` does not have a `mon_to_seas_dask` method.
    ValueError
        If 'Y' dimension not found in `da` or `smyle_time`.
    """
    if not hasattr(cal, "mon_to_seas_dask"):
        raise TypeError(
            f"cal must have a 'mon_to_seas_dask' method, "
            f"got {type(cal).__name__}."
        )
    if "Y" not in da.dims and "Y" not in da.coords:
        raise ValueError("'Y' dimension not found in input DataArray.")
    if "Y" not in smyle_time.dims and "Y" not in smyle_time.coords:
        raise ValueError("'Y' dimension not found in smyle_time.")

    ds = da.to_dataset(name=field_name)
    ds["time"] = smyle_time.sel(Y=da.Y)
    seas = cal.mon_to_seas_dask(ds)[field_name]
    return ds, seas


def preprocess_nmme_by_init_month(
    nmme_regsst: xr.DataArray,
    smyle_time_dict: Dict[int, xr.DataArray],
    smyle_seas_time_dict: Dict[int, xr.DataArray],
    cal: Any,
    field_name: str = "sst",
    init_months: Optional[List[int]] = None,
) -> Dict[str, Dict[int, Any]]:
    """
    Process NMME data into monthly and seasonal outputs for each init month.

    Parameters
    ----------
    nmme_regsst : xr.DataArray
        Multi-model regional SST anomaly DataArray.
    smyle_time_dict : dict
        Mapping from init month (int) to SMYLE verification time DataArray.
    smyle_seas_time_dict : dict
        Mapping from init month (int) to SMYLE seasonal verification time DataArray.
    cal : object
        Calendar utility object with a `mon_to_seas_dask` method.
    field_name : str, optional
        Variable name, default 'sst'.
    init_months : list of int, optional
        Init months to process. Defaults to [2, 5, 8, 11].

    Returns
    -------
    dict with keys:
        - 'monthly'       : dict {month: DataArray}
        - 'seasonal'      : dict {month: DataArray}
        - 'monthly_time'  : dict {month: DataArray}
        - 'seasonal_time' : dict {month: DataArray}

    Raises
    ------
    ValueError
        If any required init month key is missing from the time dicts.
    """
    if init_months is None:
        init_months = _DEFAULT_INIT_MONTHS

    missing_monthly = [m for m in init_months if m not in smyle_time_dict]
    missing_seasonal = [m for m in init_months if m not in smyle_seas_time_dict]
    if missing_monthly:
        raise ValueError(
            f"smyle_time_dict is missing keys for init months: {missing_monthly}."
        )
    if missing_seasonal:
        raise ValueError(
            f"smyle_seas_time_dict is missing keys for init months: {missing_seasonal}."
        )

    monthly: Dict[int, xr.DataArray] = {}
    seasonal: Dict[int, xr.DataArray] = {}
    monthly_time: Dict[int, xr.DataArray] = {}
    seasonal_time: Dict[int, xr.DataArray] = {}

    for m in init_months:
        da = subset_init_month(nmme_regsst, init_month=m)
        ds, seas = attach_time_and_seasonalize(
            da,
            smyle_time=smyle_time_dict[m],
            cal=cal,
            field_name=field_name,
        )

        monthly[m] = da
        seasonal[m] = seas
        monthly_time[m] = smyle_time_dict[m].sel(Y=da.Y)
        seasonal_time[m] = smyle_seas_time_dict[m].sel(Y=seas.Y)

    return {
        "monthly": monthly,
        "seasonal": seasonal,
        "monthly_time": monthly_time,
        "seasonal_time": seasonal_time,
    }
