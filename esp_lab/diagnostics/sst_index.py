"""Shared SST-index definitions and transformations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import cftime
import numpy as np
import xarray as xr

from esp_lab import stats
from esp_lab.utils.netcdf_utils import atomic_to_netcdf


BASIC_REGIONS: dict[str, dict[str, object]] = {
    "Nino12": {"lonlat": [270.0, 280.0, -10.0, 0.0], "long_name": "Nino 1+2 regional mean SST"},
    "Nino3": {"lonlat": [210.0, 270.0, -5.0, 5.0], "long_name": "Nino 3 regional mean SST"},
    "Nino3.4": {"lonlat": [190.0, 240.0, -5.0, 5.0], "long_name": "Nino 3.4 regional mean SST"},
    "Nino4": {"lonlat": [160.0, 210.0, -5.0, 5.0], "long_name": "Nino 4 regional mean SST"},
    "TNA": {"lonlat": [305.0, 345.0, 5.0, 25.0], "long_name": "TNA regional mean SST"},
    "TSA": {"lonlat": [330.0, 10.0, -20.0, 0.0], "long_name": "TSA regional mean SST"},
    "PACWRAMPOOL": {"lonlat": [60.0, 170.0, -15.0, 15.0], "long_name": "PACWRAMPOOL regional mean SST"},
    "AtlNino": {"lonlat": [340.0, 360.0, -3.0, 3.0], "long_name": "Atlantic Nino regional mean SST"},
    "AtlMDR": {"lonlat": [280.0, 350.0, 10.0, 20.0], "long_name": "Atlantic MDR regional mean SST"},
}
HELPER_REGIONS: dict[str, dict[str, object]] = {
    "IOD_West": {"lonlat": [50.0, 70.0, -10.0, 10.0], "long_name": "IOD West regional mean SST"},
    "IOD_East": {"lonlat": [90.0, 110.0, -10.0, 0.0], "long_name": "IOD East regional mean SST"},
    "TropicalMean": {"lonlat": [0.0, 360.0, -20.0, 20.0], "long_name": "Tropical Mean regional mean SST"},
}
DERIVED_REGIONS = ["IOD", "TNI", "ONI", "RONI", "ELI"]
REGIONS: dict[str, dict[str, object]] = {**BASIC_REGIONS, **HELPER_REGIONS}
VALID_REGIONS = DERIVED_REGIONS + list(BASIC_REGIONS)

ELI_LAT_MIN = -5.0
ELI_LAT_MAX = 5.0
ELI_LON_MIN = 120.0
ELI_LON_MAX = 290.0
TC_LAT_HALF = 5.0
ELI_ALGORITHM_VERSION = 2


def period_tag(data_start: int, data_end: int) -> str:
    """Return the canonical SST-index year-period token."""
    return f"{data_start:04d}_{data_end:04d}"


def region_token(region: str) -> str:
    """Return a filesystem-safe region token used in SST-index filenames."""
    return region.replace(".", "")


def index_file_label(region: str) -> str:
    """Return the canonical SST-index filename label for one region."""
    token = region_token(region)
    return "ELI" if region == "ELI" else f"{token}SST"


def required_base_regions(
    regions: Sequence[str],
    available_regions: Mapping[str, object] | None = None,
) -> list[str]:
    """Return base regional means needed to derive requested SST indices."""
    available = REGIONS if available_regions is None else available_regions
    required: set[str] = set()
    for region in regions:
        if region == "IOD":
            required.update(["IOD_West", "IOD_East"])
        elif region == "TNI":
            required.update(["Nino12", "Nino4"])
        elif region == "ONI":
            required.add("Nino3.4")
        elif region == "RONI":
            required.update(["Nino3.4", "TropicalMean"])
        elif region == "ELI":
            continue
        elif region in available:
            required.add(region)
    return sorted(required)


def write_netcdf_replace(
    dataset: xr.Dataset,
    path: Path | str,
    *,
    encoding: Mapping | None = None,
) -> Path:
    """Write NetCDF through a temporary file, then replace the destination."""
    return atomic_to_netcdf(dataset, path, encoding=encoding)


def latlon_mask(
    lat_coord: xr.DataArray,
    lon_coord: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    lat_dim = lat_coord.dims[0]
    lon_dim = lon_coord.dims[0]
    lat_da = xr.DataArray(lat_coord.values, dims=(lat_dim,), coords={lat_dim: lat_coord.values})
    lon_da = xr.DataArray(np.mod(lon_coord.values, 360.0), dims=(lon_dim,), coords={lon_dim: lon_coord.values})
    lon2d, lat2d = xr.broadcast(lon_da, lat_da)
    mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    return mask.transpose(lat_dim, lon_dim)


def lon2d(lat_coord: xr.DataArray, lon_coord: xr.DataArray) -> xr.DataArray:
    lat_dim = lat_coord.dims[0]
    lon_dim = lon_coord.dims[0]
    lon_da = xr.DataArray(np.mod(lon_coord.values, 360.0), dims=(lon_dim,), coords={lon_dim: lon_coord.values})
    longitude, _ = xr.broadcast(lon_da, lat_coord)
    return longitude.transpose(lat_dim, lon_dim)


def with_lon_0_360(data: xr.DataArray, lon_name: str) -> xr.DataArray:
    """Normalize a regular-grid longitude coordinate to 0-360 and sort it."""
    lon = (data[lon_name] % 360.0).astype(data[lon_name].dtype)
    return data.assign_coords({lon_name: lon}).sortby(lon_name)


def compute_eli_latlon_sst(
    sst: xr.DataArray,
    *,
    lat_name: str = "lat",
    lon_name: str = "lon",
    oceanmask: xr.DataArray | None = None,
    output_name: str = "eli",
) -> xr.DataArray:
    """Compute ELI from a regular latitude-longitude SST field."""
    sst = with_lon_0_360(sst, lon_name).where(np.isfinite(sst))
    lat_coord = sst[lat_name]
    lon_coord = sst[lon_name]
    lat_dim = lat_coord.dims[0]
    lon_dim = lon_coord.dims[0]

    eq_mask = latlon_mask(lat_coord, lon_coord, ELI_LAT_MIN, ELI_LAT_MAX, ELI_LON_MIN, ELI_LON_MAX)
    tropics_mask = latlon_mask(lat_coord, lon_coord, -TC_LAT_HALF, TC_LAT_HALF, 0.0, 360.0)
    if oceanmask is not None:
        oceanmask = oceanmask.transpose(lat_dim, lon_dim)
        eq_mask = eq_mask & oceanmask
        tropics_mask = tropics_mask & oceanmask

    weights = xr.DataArray(
        np.cos(np.deg2rad(lat_coord.values)),
        dims=(lat_dim,),
        coords={lat_dim: lat_coord.values},
    )
    weights2d, _ = xr.broadcast(weights, sst[lon_name])
    weights2d = weights2d.transpose(lat_dim, lon_dim)
    longitude = lon2d(lat_coord, lon_coord)

    tropics_weights = weights2d.where(tropics_mask, 0.0)
    tropical_mean = sst.weighted(tropics_weights).mean((lat_dim, lon_dim), skipna=True)

    warm_weights = weights2d.where(eq_mask & (sst > tropical_mean) & sst.notnull(), 0.0)
    denominator = warm_weights.sum((lat_dim, lon_dim), skipna=True)
    numerator = (warm_weights * longitude).sum((lat_dim, lon_dim), skipna=True)
    eli = (numerator / denominator).astype("float32").where(denominator > 0).rename(output_name)
    eli.attrs.update(
        {
            "long_name": "Equatorial Longitude Index",
            "units": "degrees_east",
            "region": "ELI",
            "description": (
                "Area-weighted centroid longitude of warm SST cells "
                "(SST > tropical-mean SST) in the equatorial Pacific "
                f"(lat {ELI_LAT_MIN}-{ELI_LAT_MAX} deg, lon {ELI_LON_MIN}-{ELI_LON_MAX} deg)."
            ),
            "eli_input_grid": "regridded_latlon",
            "eli_lat_min": ELI_LAT_MIN,
            "eli_lat_max": ELI_LAT_MAX,
            "eli_lon_min": ELI_LON_MIN,
            "eli_lon_max": ELI_LON_MAX,
            "tc_lat_half": TC_LAT_HALF,
            "eli_algorithm_version": ELI_ALGORITHM_VERSION,
        }
    )
    return eli


def compute_model_anom(
    data: xr.DataArray,
    valid_time: xr.DataArray,
    clim_start: int,
    clim_end: int,
) -> xr.DataArray:
    anomaly, _ = stats.remove_drift(data, valid_time, clim_start, clim_end)
    return anomaly


def compute_model_std(
    anomaly: xr.DataArray,
    valid_time: xr.DataArray,
    clim_start: int,
    clim_end: int,
) -> xr.DataArray:
    start = cftime.DatetimeNoLeap(clim_start, 1, 1, 0, 0, 0)
    end = cftime.DatetimeNoLeap(clim_end, 12, 31, 23, 59, 59)
    climatology = anomaly.where((valid_time >= start) & (valid_time <= end))
    dims_to_reduce = [dim for dim in ["Y", "M"] if dim in anomaly.dims]
    sigma = climatology.std(dim=dims_to_reduce)
    sigma = sigma.where(sigma > 0, 1.0)
    return anomaly / sigma


def compute_obs_anom(data: xr.DataArray, clim_start: int, clim_end: int) -> xr.DataArray:
    start = cftime.DatetimeNoLeap(clim_start, 1, 1, 0, 0, 0)
    end = cftime.DatetimeNoLeap(clim_end, 12, 31, 23, 59, 59)
    climatology = data.sel(time=slice(start, end))
    monthly_mean = climatology.groupby("time.month").mean("time")
    return data.groupby("time.month") - monthly_mean


def compute_obs_std(anomaly: xr.DataArray, clim_start: int, clim_end: int) -> xr.DataArray:
    start = cftime.DatetimeNoLeap(clim_start, 1, 1, 0, 0, 0)
    end = cftime.DatetimeNoLeap(clim_end, 12, 31, 23, 59, 59)
    climatology = anomaly.sel(time=slice(start, end))
    sigma = climatology.std("time")
    sigma = sigma.where(sigma > 0, 1.0)
    return anomaly / sigma


def standardize_index_anomaly(
    anomaly: xr.DataArray,
    clim_start: int,
    clim_end: int,
    *,
    time_dim: str,
    ensemble_dim: str | None = None,
) -> xr.DataArray:
    years = anomaly[time_dim].dt.year
    climatology = anomaly.where((years >= clim_start) & (years <= clim_end))
    dims = [dim for dim in (time_dim, ensemble_dim) if dim and dim in anomaly.dims]
    sigma = climatology.std(dim=dims, skipna=True)
    sigma = sigma.where(sigma > 0, 1.0)
    return anomaly / sigma


def derive_indices_from_anomalies(
    base_anom: Mapping[str, xr.DataArray],
    regions: Sequence[str],
    clim_start: int,
    clim_end: int,
    *,
    rolling_dim: str,
    time_dim: str,
    ensemble_dim: str | None = "M",
    time_values: xr.DataArray | None = None,
    standardize: Callable[[xr.DataArray], xr.DataArray] | None = None,
) -> dict[str, xr.DataArray]:
    """Derive SST indices from pre-anomalized base regional means."""
    requested = set(regions)
    derived: dict[str, xr.DataArray] = {}

    if "IOD" in requested and {"IOD_West", "IOD_East"}.issubset(base_anom):
        derived["IOD"] = (base_anom["IOD_West"] - base_anom["IOD_East"]).assign_attrs(
            long_name="Dipole Mode Index (IOD West SST anomaly minus IOD East SST anomaly)",
            region="IOD",
            units="degC",
            index_name="DMI",
            definition="anomaly(IOD_West SST) - anomaly(IOD_East SST)",
            climatology_start_year=clim_start,
            climatology_end_year=clim_end,
        )

    if "ONI" in requested and "Nino3.4" in base_anom:
        nino34 = base_anom["Nino3.4"]
        oni = nino34.rolling({rolling_dim: 3}, center=True, min_periods=1).mean()
        derived["ONI"] = oni.where(nino34.notnull()).assign_attrs(
            long_name="Oceanic Nino Index (3-month running mean of Nino3.4 anomalies)",
            region="Nino3.4",
            units="degC",
        )

    if "TNI" in requested and {"Nino12", "Nino4"}.issubset(base_anom):
        if standardize is None:
            standardize = lambda da: standardize_index_anomaly(
                da, clim_start, clim_end, time_dim=time_dim, ensemble_dim=ensemble_dim
            )
        nino12 = base_anom["Nino12"]
        nino4 = base_anom["Nino4"]
        diff = standardize(nino12) - standardize(nino4)
        tni = diff.rolling({rolling_dim: 5}, center=True, min_periods=1).mean()
        derived["TNI"] = tni.where(nino12.notnull() & nino4.notnull()).assign_attrs(
            long_name="Trans-Niño Index (standardized Nino12 minus standardized Nino4 with 5-month running mean)",
            region="TNI",
            units="1",
        )

    if "RONI" in requested and {"Nino3.4", "TropicalMean"}.issubset(base_anom):
        nino34 = base_anom["Nino3.4"]
        tropical = base_anom["TropicalMean"]
        valid_input = nino34.notnull() & tropical.notnull()
        nino34_3m = nino34.rolling({rolling_dim: 3}, center=True, min_periods=1).mean().where(nino34.notnull())
        diff_3m = (nino34 - tropical).rolling({rolling_dim: 3}, center=True, min_periods=1).mean().where(valid_input)
        if time_values is not None:
            start = cftime.DatetimeNoLeap(clim_start, 1, 1, 0, 0, 0)
            end = cftime.DatetimeNoLeap(clim_end, 12, 31, 23, 59, 59)
            climatology_mask = (time_values >= start) & (time_values <= end)
        else:
            climatology_mask = (nino34[time_dim].dt.year >= clim_start) & (nino34[time_dim].dt.year <= clim_end)
        dims = [dim for dim in (time_dim, ensemble_dim) if dim and dim in diff_3m.dims]
        std_nino = nino34_3m.where(climatology_mask).std(dim=dims, skipna=True)
        std_diff = diff_3m.where(climatology_mask).std(dim=dims, skipna=True)
        ratio = (std_nino / std_diff).fillna(1.0).where(std_diff > 0, 1.0)
        derived["RONI"] = (diff_3m * ratio).where(valid_input).assign_attrs(
            long_name="Relative Oceanic Nino Index",
            region="RONI",
            units="degC",
        )

    return derived


def derive_indices(
    computed_vals: Mapping[str, xr.DataArray],
    time_coords: xr.DataArray,
    clim_start: int,
    clim_end: int,
    *,
    is_model: bool = True,
) -> dict[str, xr.DataArray]:
    """Derive SST indices from base regional means after consistent anomaly removal."""
    if is_model:
        base_anom = {
            name: compute_model_anom(data, time_coords, clim_start, clim_end)
            for name, data in computed_vals.items()
        }

        def standardize(data: xr.DataArray) -> xr.DataArray:
            return compute_model_std(data, time_coords, clim_start, clim_end)

        return derive_indices_from_anomalies(
            base_anom,
            DERIVED_REGIONS,
            clim_start,
            clim_end,
            rolling_dim="L",
            time_dim="Y",
            time_values=time_coords,
            standardize=standardize,
        )

    base_anom = {
        name: compute_obs_anom(data, clim_start, clim_end)
        for name, data in computed_vals.items()
    }

    def standardize(data: xr.DataArray) -> xr.DataArray:
        return compute_obs_std(data, clim_start, clim_end)

    return derive_indices_from_anomalies(
        base_anom,
        DERIVED_REGIONS,
        clim_start,
        clim_end,
        rolling_dim="time",
        time_dim="time",
        time_values=time_coords,
        standardize=standardize,
    )