"""
s2s_io.py
=========
Subseasonal-to-Seasonal (S2S) Filesystem I/O and Data Loader Bridge.

Handles directory discovery, hindcast loading, and daily-to-weekly aggregation:
- Ingests high-frequency atmospheric data (ts/6hourly/2yr/ or ts/daily/)
- Ingests daily land surface data (ts/daily/2yr/)
- Aggregates daily/6-hourly series into non-overlapping 7-day weekly blocks (Weeks 1 to 8)
- Aligns daily observational references (ERA5_Daily, IMERG_Daily, NOAA-OLR_Daily, etc.)
- Caches analysis-ready weekly anomaly bundles
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from esp_lab.data_access_e3sm import drop_feb29
from esp_lab.diagnostics.s2s_core import (
    S2S_WEEKLY_WINDOWS,
    aggregate_daily_to_weekly,
    align_obs_to_weekly_leads,
    compute_weekly_anomalies,
    compute_weekly_climatology,
    get_weekly_window,
)
from esp_lab.paths import leadtime_acc_dir


DEFAULT_DATA_DIR = Path("/global/cfs/cdirs/e3sm/S2S2D/post_process")
DEFAULT_OBS_DIR = Path("/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series")


def s2s_hindcast_dir(
    data_root: str | Path,
    case_prefix: str,
    init_year: int,
    init_month: int,
    member: str,
    component: str = "atm",
    grid: str = "180x360_aave",
    freq: str = "6hourly",
    time_split: str = "2yr",
) -> Path:
    """Return directory containing hindcast time series for a specific initialization."""
    init_stamp = f"{init_year:04d}{init_month:02d}0100"
    case_name = f"{case_prefix}_{init_stamp}"
    return (
        Path(data_root)
        / case_name
        / member
        / "post"
        / component
        / grid
        / "ts"
        / freq
        / time_split
    )


def discover_s2s_files(directory: Path, field: str) -> List[Path]:
    """Find NetCDF files for a variable in directory."""
    if not directory.is_dir():
        return []
    patterns = [
        f"{field}_*.nc",
        f"{field}_*.nc4",
        f"*{field}*.nc",
    ]
    matches: set[Path] = set()
    for pat in patterns:
        matches.update(directory.glob(pat))
    return sorted(matches)


def load_s2s_member_weekly(
    data_root: str | Path,
    case_prefix: str,
    init_year: int,
    init_month: int,
    member: str,
    field: str,
    component: str = "atm",
    grid: str = "180x360_aave",
    weeks: Sequence[int] = range(1, 9),
    chunks: Optional[dict] = None,
) -> Optional[xr.DataArray]:
    """Load and aggregate a single member's hindcast into 8 weekly leads.

    Searches across candidate frequency streams:
    1. ts/daily/2yr or ts/daily/0yr
    2. ts/6hourly/2yr (resampled to daily then weekly)
    """
    data_root = Path(data_root)
    # Check possible frequency locations
    candidate_freqs = [("daily", "2yr"), ("daily", "0yr"), ("6hourly", "2yr")]
    if component == "lnd":
        candidate_freqs = [("daily", "2yr"), ("daily", "0yr")]

    target_files: List[Path] = []
    freq_used = None

    for freq, split in candidate_freqs:
        p = s2s_hindcast_dir(
            data_root=data_root,
            case_prefix=case_prefix,
            init_year=init_year,
            init_month=init_month,
            member=member,
            component=component,
            grid=grid,
            freq=freq,
            time_split=split,
        )
        files = discover_s2s_files(p, field)
        if files:
            target_files = files
            freq_used = freq
            break

    if not target_files:
        return None

    open_chunks = chunks or {"time": -1}
    ds = xr.open_mfdataset(
        target_files,
        combine="by_coords",
        chunks=open_chunks,
        use_cftime=True,
    )
    ds = drop_feb29(ds, "time")

    if field not in ds.data_vars:
        # Check case-insensitive
        matches = [v for v in ds.data_vars if v.lower() == field.lower()]
        if matches:
            field = matches[0]
        else:
            warnings.warn(f"Field {field} not found in {target_files[0]}")
            return None

    da = ds[field]

    # Convert time to 1-based lead days if needed
    import cftime
    init_dt = cftime.DatetimeNoLeap(init_year, init_month, 1)

    time_vals = da["time"].values
    lead_days = []
    for tv in time_vals:
        try:
            cur_dt = cftime.DatetimeNoLeap(tv.year, tv.month, tv.day)
            delta = (cur_dt - init_dt).days
            lead_days.append(int(delta))
        except Exception:
            lead_days.append(-999)

    da = da.assign_coords(lead_day=("time", lead_days))
    # Filter to forecast window 1..56 (or 1..84)
    valid_mask = (da.lead_day >= 1) & (da.lead_day <= 56)
    da = da.where(valid_mask, drop=True)

    # If 6-hourly, average over each lead day first
    if freq_used == "6hourly":
        da = da.groupby("lead_day").mean(dim="time", skipna=True)
        da = da.rename({"lead_day": "d"})
    else:
        # If daily, group or rename to d
        if "d" not in da.dims:
            da = da.swap_dims({"time": "lead_day"}).rename({"lead_day": "d"})

    # Aggregate along d to 7-day weekly blocks for weeks 1..8
    weekly = aggregate_daily_to_weekly(da, day_dim="d", weeks=weeks, output_dim="L")
    return weekly


def load_s2s_campaign_weekly(
    data_root: str | Path,
    case_prefix: str,
    years: Sequence[int],
    init_month: int,
    members: Sequence[str],
    field: str,
    component: str = "atm",
    grid: str = "180x360_aave",
    weeks: Sequence[int] = range(1, 9),
    chunks: Optional[dict] = None,
    verbose: bool = True,
) -> xr.DataArray:
    """Load entire campaign cohort into a unified (Y, M, L, lat, lon) weekly array."""
    year_arrays = []
    valid_years = []

    for yr in years:
        member_arrays = []
        valid_members = []
        for mem in members:
            w_da = load_s2s_member_weekly(
                data_root=data_root,
                case_prefix=case_prefix,
                init_year=yr,
                init_month=init_month,
                member=mem,
                field=field,
                component=component,
                grid=grid,
                weeks=weeks,
                chunks=chunks,
            )
            if w_da is not None:
                member_arrays.append(w_da)
                valid_members.append(mem)

        if member_arrays:
            da_y = xr.concat(member_arrays, dim=pd.Index(valid_members, name="M"))
            year_arrays.append(da_y)
            valid_years.append(yr)
            if verbose:
                print(f"  Loaded {case_prefix} {yr}-{init_month:02d} ({len(valid_members)} members)")

    if not year_arrays:
        raise ValueError(
            f"No data could be loaded for {case_prefix}/{field} in years {years} month {init_month}."
        )

    result = xr.concat(year_arrays, dim=pd.Index(valid_years, name="Y"))
    result.name = field
    return result


def load_s2s_obs_weekly(
    obs_path: str | Path,
    field: str,
    init_years: Sequence[int],
    init_month: int,
    weeks: Sequence[int] = range(1, 9),
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """Load daily observation time series and aggregate to weekly verification leads."""
    obs_path = Path(obs_path)
    if not obs_path.is_file():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    open_chunks = chunks or {"time": -1}
    ds = xr.open_dataset(obs_path, chunks=open_chunks, use_cftime=True)
    ds = drop_feb29(ds, "time")

    if field not in ds.data_vars:
        matches = [v for v in ds.data_vars if v.lower() == field.lower()]
        if matches:
            field = matches[0]
        else:
            raise KeyError(f"Field {field} not found in {obs_path}")

    obs_da = ds[field]

    # Align observation dates with hindcast weekly verification windows
    weekly_obs = align_obs_to_weekly_leads(
        obs_da=obs_da,
        init_years=init_years,
        init_month=init_month,
        weeks=weeks,
        time_dim="time",
        output_lead_dim="L",
        output_year_dim="Y",
    )
    return weekly_obs


def cache_s2s_weekly_bundle(
    anom_da: xr.DataArray,
    case_label: str,
    field: str,
    init_month: int,
    out_dir: str | Path,
) -> Path:
    """Cache analysis-ready weekly anomalies to NetCDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{case_label}_{field}_init{init_month:02d}_weekly_anom.nc"
    out_path = out_dir / filename
    anom_da.to_netcdf(out_path)
    return out_path


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_OBS_DIR",
    "s2s_hindcast_dir",
    "discover_s2s_files",
    "load_s2s_member_weekly",
    "load_s2s_campaign_weekly",
    "load_s2s_obs_weekly",
    "cache_s2s_weekly_bundle",
]

