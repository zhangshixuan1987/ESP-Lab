"""
s2s_io.py
=========
Subseasonal-to-Seasonal (S2S) Filesystem I/O and Data Loader Bridge.

Handles directory discovery, hindcast loading, and daily-to-weekly aggregation:
- Ingests high-frequency atmospheric data (ts/6hourly/2yr/ or ts/daily/)
- Ingests daily land surface data (ts/daily/2yr/)
- Aggregates daily/6-hourly series into non-overlapping 7-day weekly blocks (Weeks 1 to 8)
- Aligns daily observational references (ERA5_Daily, IMERG_Daily, NOAA-OLR_Daily, etc.)
- Caches analysis-ready weekly hindcasts and anomaly bundles

Layout follows the s2d_skill conventions (see ``s2d_diag/README.md``): roots
come from :mod:`esp_lab.env_paths`, per-source products live under
``<s2d_diag>/<source>/s2s_weekly/<stage>/<component>/<FIELD>/`` with
``<source>_init<MM>_<FIELD>_…`` names, and figures are written flat to
``env_paths.s2s_figure_root()`` as ``fig_s2s_<component>_<diagnostic>_…png``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from esp_lab import env_paths
from esp_lab.data_access_e3sm import drop_feb29
from esp_lab.diagnostics.s2s_core import (
    S2S_WEEKLY_WINDOWS,
    aggregate_daily_to_weekly,
    align_obs_to_weekly_leads,
    compute_weekly_anomalies,
    compute_weekly_climatology,
    get_weekly_window,
)
from esp_lab.paths import diagnostic_dir
from esp_lab.utils.filename_utils import figure_filename, source_init_prefix, year_span_token


DEFAULT_DATA_DIR = env_paths.raw_model_root()
DEFAULT_OBS_DIR = env_paths.obs_root()
S2S_DIAGNOSTIC = "s2s_weekly"
WEEKLY_CACHE_VERSION = "s2s_weekly_input_v1"


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


def s2s_weekly_dir(
    source: str, stage: str, component: str, field: str, *, root: str | Path | None = None
) -> Path:
    """Return ``<root>/<source>/s2s_weekly/<stage>/<component>/<FIELD>``."""
    root = env_paths.s2d_diag_root() if root is None else root
    return diagnostic_dir(source, S2S_DIAGNOSTIC, stage, component, field, root=root)


def s2s_weekly_input_path(
    source: str,
    component: str,
    field: str,
    init_month: int,
    years: Sequence[int],
    n_members: int,
    n_weeks: int,
    *,
    root: str | Path | None = None,
) -> Path:
    """Return the weekly hindcast cache path, mirroring the s2d_skill input names.

    ``<source>_init<MM>_<FIELD>_y<a>-<b>_ny<n>_N<members>_W<weeks>_week.nc``, e.g.
    ``JRA55_FOSIRL_init05_TREFHT_y1980-1986_ny7_N10_W08_week.nc``.
    """
    years = [int(year) for year in years]
    name = (
        f"{source_init_prefix(source, init_month)}_{field}_"
        f"{year_span_token((min(years), max(years)))}_ny{len(years)}_"
        f"N{int(n_members):02d}_W{int(n_weeks):02d}_week.nc"
    )
    return s2s_weekly_dir(source, "inputs", component, field, root=root) / name


def _weekly_cache_attrs(case_prefix, years, init_month, members, field, component, grid, weeks):
    return {
        "s2s_weekly_cache_version": WEEKLY_CACHE_VERSION,
        "case_prefix": str(case_prefix),
        "initialization_years": ",".join(str(int(y)) for y in years),
        "initialization_month": int(init_month),
        "requested_members": ",".join(str(m) for m in members),
        "field": str(field),
        "component": str(component),
        "grid": str(grid),
        "lead_weeks": ",".join(str(int(w)) for w in weeks),
    }


def ensure_s2s_campaign_weekly(
    source: str,
    case_prefix: str,
    years: Sequence[int],
    init_month: int,
    members: Sequence[str],
    field: str,
    component: str = "atm",
    grid: str = "180x360_aave",
    weeks: Sequence[int] = range(1, 9),
    *,
    mode: str = "auto",
    root: str | Path | None = None,
    data_root: str | Path | None = None,
    verbose: bool = True,
) -> xr.DataArray:
    """Reuse or build the cached ``(Y, M, L, lat, lon)`` weekly hindcast for one source.

    ``mode`` follows the s2d_skill ``derivation_mode`` contract: ``"auto"``
    reuses a compatible cache and builds a missing or stale one in place,
    ``"require"`` fails instead of building, and ``"rebuild"`` always rebuilds.
    Compatibility is judged from file attributes, never from the file name.
    """
    if mode not in {"auto", "require", "rebuild"}:
        raise ValueError("mode must be 'auto', 'require', or 'rebuild'")
    years, weeks, members = list(years), list(weeks), list(members)
    path = s2s_weekly_input_path(
        source, component, field, init_month, years, len(members), len(weeks), root=root
    )
    expected = _weekly_cache_attrs(case_prefix, years, init_month, members, field, component, grid, weeks)
    reason = "forced rebuild"
    if mode != "rebuild":
        if path.is_file():
            with xr.open_dataarray(path) as cached:
                stale = {k: v for k, v in expected.items() if cached.attrs.get(k) != v}
                if not stale:
                    if verbose:
                        print(f"  [reuse] {path}")
                    return cached.load()
            reason = f"stale attributes {sorted(stale)}"
        else:
            reason = "missing cache"
        if mode == "require":
            raise FileNotFoundError(f"S2S weekly cache unavailable ({reason}): {path}")

    if verbose:
        print(f"  [build] {source} init{int(init_month):02d} {field}: {reason}")
    weekly = load_s2s_campaign_weekly(
        data_root=DEFAULT_DATA_DIR if data_root is None else data_root,
        case_prefix=case_prefix, years=years, init_month=init_month, members=members,
        field=field, component=component, grid=grid, weeks=weeks, verbose=verbose,
    ).load()
    weekly.attrs.update(expected, source=str(source))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".nc.tmp")
    weekly.to_netcdf(temporary)
    temporary.replace(path)
    return weekly


def cache_s2s_weekly_bundle(
    anom_da: xr.DataArray,
    source: str,
    field: str,
    init_month: int,
    component: str = "atm",
    *,
    root: str | Path | None = None,
) -> Path:
    """Cache analysis-ready weekly anomalies as ``<source>_init<MM>_<FIELD>_weekly_anom.nc``."""
    out_dir = s2s_weekly_dir(source, "anomalies", component, field, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_init_prefix(source, init_month)}_{field}_weekly_anom.nc"
    anom_da.to_netcdf(out_path)
    return out_path


def s2s_figure_filename(component: str, diagnostic: str, field: str, *parts, ext: str = "png") -> str:
    """Return ``fig_s2s_<component>_<diagnostic>_<field>_<parts>.png`` (s2d_skill style)."""
    return figure_filename("s2s", component, diagnostic, field, *parts, ext=ext)


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_OBS_DIR",
    "S2S_DIAGNOSTIC",
    "s2s_weekly_dir",
    "s2s_weekly_input_path",
    "ensure_s2s_campaign_weekly",
    "s2s_figure_filename",
    "s2s_hindcast_dir",
    "discover_s2s_files",
    "load_s2s_member_weekly",
    "load_s2s_campaign_weekly",
    "load_s2s_obs_weekly",
    "cache_s2s_weekly_bundle",
]

