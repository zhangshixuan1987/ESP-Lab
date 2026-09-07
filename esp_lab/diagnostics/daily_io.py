"""
daily_io.py
===========
Daily S2D spatial analysis — filesystem I/O bridge.

Handles directory navigation, daily file discovery, metadata probing,
and reading daily spatial fields for the 1–84 day analysis window.

Wraps `esp_lab.data_access_e3sm` and standard NetCDF/cftime loaders.

Functions
---------
daily_data_dir               — build expected path to .../daily/0yr/
discover_daily_files        — glob matching *{FIELD}*.nc files in directory
probe_daily_file             — probe metadata & day coverage -> DailyInventoryRecord
discover_all_daily_files     — loop across experiment x init x member -> inventory DF
load_daily_spatial_field     — open single case/member daily field -> (d, lat, lon) DataArray
load_daily_campaign_field    — load approved campaign cohort -> (Y, M, d, lat, lon) DataArray
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from esp_lab.data_access_e3sm import drop_feb29, time_set_midmonth
from esp_lab.diagnostics.daily_core import (
    DailyDriftConfig,
    DailyGateStatus,
    DailyInventoryRecord,
    DailyInventoryStatus,
    DailyLeadCoverageResult,
    ExperimentSpec,
    check_daily_lead_coverage,
    classify_daily_analysis_status,
    classify_daily_archive_status,
    classify_daily_window_status,
    daily_inventory_records_to_df,
)


def daily_data_dir(
    data_root: str | Path,
    experiment: ExperimentSpec,
    init_year: int,
    init_month: int,
    member: str,
    forecast_year: int = 0,
    component: str = "atm",
    grid: str = "180x360_aave",
) -> Path:
    """Construct deterministic daily postprocessed directory path.

    Layout::

        {data_root}/
          {case_prefix}_{YYYYMMDDHH}/
            {member}/post/{component}/{grid}/ts/daily/{forecast_year}yr/

    Example::

        /global/cfs/cdirs/e3sm/S2S2D/post_process/
          WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_BruteForce_2004050100/
            EN00/post/atm/180x360_aave/ts/daily/0yr/
    """
    init_stamp = f"{init_year:04d}{init_month:02d}0100"
    case_name  = f"{experiment.case_prefix}_{init_stamp}"

    return (
        Path(data_root)
        / case_name
        / member
        / "post"
        / component
        / grid
        / "ts"
        / "daily"
        / f"{forecast_year}yr"
    )


def discover_daily_files(
    directory: Path,
    native_field: str,
) -> List[Path]:
    """Find files containing native_field in directory."""
    if not directory.is_dir():
        return []

    patterns = [
        f"*{native_field}*.nc",
        f"*{native_field}*.nc4",
    ]

    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(directory.glob(pattern))

    return sorted(matches)


def _extract_daily_lead_days(ds: xr.Dataset, native_field: str, init_year: int, init_month: int) -> Tuple[List[int], str, str]:
    """Derive 1-based lead days (d=1, 2, ...) from dataset time coordinate.

    Initialization date timestamp = lead day 0.
    First complete forecast day = lead day 1.
    """
    if "time" not in ds.coords:
        return [], "", ""

    time_vals = ds["time"].values
    if len(time_vals) == 0:
        return [], "", ""

    first_ts = str(time_vals[0])
    last_ts  = str(time_vals[-1])

    import cftime
    init_dt = cftime.DatetimeNoLeap(init_year, init_month, 1)

    lead_days = []
    for tv in time_vals:
        try:
            cur_dt = cftime.DatetimeNoLeap(tv.year, tv.month, tv.day)
            # Days difference from init date
            delta_days = (cur_dt - init_dt).days
            if 1 <= delta_days <= 365:
                lead_days.append(int(delta_days))
        except Exception:
            pass

    return sorted(lead_days), first_ts, last_ts


def probe_daily_file(
    ts_dir: Path,
    native_field: str,
    init_year: int,
    init_month: int,
    expected_days: Sequence[int],
    experiment: str,
    case_prefix: str,
    member: str,
    window_defs: dict,
    chunks: Optional[dict] = None,
) -> DailyInventoryRecord:
    """Probe daily NetCDF files for metadata, timestamp consistency, and 1-84 day coverage."""
    init_date_str = f"{init_year:04d}-{init_month:02d}-01"
    init_stamp    = f"{init_year:04d}{init_month:02d}0100"

    record = DailyInventoryRecord(
        experiment=experiment,
        case_prefix=case_prefix,
        init_date=init_date_str,
        init_year=init_year,
        init_month=init_month,
        member=member,
        variable=native_field,
        stream="daily/0yr",
        expected_days=f"{min(expected_days)}..{max(expected_days)}",
    )

    files = discover_daily_files(ts_dir, native_field)
    record.file_count = len(files)

    if not files:
        record.daily_status = DailyGateStatus.BLOCKED.value
        record.notes = f"No files matching *{native_field}*.nc in {ts_dir}"
        return record

    try:
        open_chunks = chunks or {"time": -1}
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks=open_chunks,
            use_cftime=True,
        )
    except Exception as exc:
        record.daily_status = DailyGateStatus.BLOCKED.value
        record.notes = f"Cannot open dataset: {exc}"
        return record

    try:
        record.variable_found = native_field in ds.data_vars

        if native_field in ds.data_vars:
            record.units = str(ds[native_field].attrs.get("units", "UNKNOWN"))

        # Calendar & Grid
        if "time" in ds.coords:
            cal = ds["time"].encoding.get("calendar", "") or ds["time"].attrs.get("calendar", "")
            record.calendar = str(cal) or "noleap"

        grid_parts = [f"{dim}={ds.sizes[dim]}" for dim in ("lat", "lon", "ncol", "nCells") if dim in ds.dims]
        record.grid_signature = ",".join(grid_parts) or f"dims={dict(ds.sizes)}"

        # Lead days
        avail_days, first_ts, last_ts = _extract_daily_lead_days(ds, native_field, init_year, init_month)
        record.first_timestamp = first_ts
        record.last_timestamp  = last_ts

        coverage = check_daily_lead_coverage(avail_days, expected_days)
        record.available_days = ",".join(str(d) for d in coverage.available_days)
        record.missing_days   = ",".join(str(d) for d in coverage.missing_days)
        record.duplicate_days = ",".join(str(d) for d in coverage.duplicate_days)

        # Window status classification
        win_statuses = classify_daily_window_status(coverage, window_defs)
        record.week_1_status     = win_statuses.get("week_1", DailyGateStatus.BLOCKED).value
        record.weeks_2_3_status  = win_statuses.get("weeks_2_3", DailyGateStatus.BLOCKED).value
        record.weeks_4_6_status  = win_statuses.get("weeks_4_6", DailyGateStatus.BLOCKED).value
        record.weeks_7_12_status = win_statuses.get("weeks_7_12", DailyGateStatus.BLOCKED).value

        record.daily_status = classify_daily_analysis_status(coverage, window_defs).value

        if not record.variable_found:
            record.notes = f"Field '{native_field}' missing from file."
            record.daily_status = DailyGateStatus.BLOCKED.value

    except Exception as exc:
        record.daily_status = DailyGateStatus.BLOCKED.value
        record.notes = f"Probing exception: {exc}"
    finally:
        try:
            ds.close()
        except Exception:
            pass

    return record


def discover_all_daily_files(
    config: DailyDriftConfig,
    variable: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Loop across experiments x init_years x init_months x members and probe files."""
    vars_to_probe = (
        [config.get_variable(variable)] if variable
        else config.variables
    )

    records: List[DailyInventoryRecord] = []

    for exp_label, exp_spec in config.experiments.items():
        for var_spec in vars_to_probe:
            field = var_spec.native_field
            for yr in config.active_years:
                for mo in config.active_months:
                    for member in config.members:
                        ts_dir = daily_data_dir(
                            data_root=config.data_dir,
                            experiment=exp_spec,
                            init_year=yr,
                            init_month=mo,
                            member=member,
                        )
                        rec = probe_file(
                            ts_dir=ts_dir,
                            native_field=field,
                            init_year=yr,
                            init_month=mo,
                            expected_days=config.lead_days,
                            experiment=exp_label,
                            case_prefix=exp_spec.case_prefix,
                            member=member,
                            window_defs=config.window_defs,
                        )
                        records.append(rec)
                        if verbose:
                            icon = "✓" if rec.daily_status == "ANALYSIS_READY" else "✗"
                            print(
                                f"  [{icon}] {exp_label:<15s} {rec.init_date} {member} {field}"
                            )

    return daily_inventory_records_to_df(records)


def load_daily_spatial_field(
    config: DailyDriftConfig,
    experiment: str,
    init_year: int,
    init_month: int,
    member: str,
    field: str,
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """Open postprocessed daily files and return DataArray with dim ``d`` (1..84)."""
    exp_spec = config.experiments[experiment]
    ts_dir = daily_data_dir(
        data_root=config.data_dir,
        experiment=exp_spec,
        init_year=init_year,
        init_month=init_month,
        member=member,
    )

    files = discover_daily_files(ts_dir, field)
    if not files:
        raise FileNotFoundError(f"No files for {field} in {ts_dir}")

    open_chunks = chunks or {"time": -1}
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks=open_chunks,
        use_cftime=True,
    )

    if field not in ds.data_vars:
        raise KeyError(f"Field '{field}' not in {files[0]}")

    da = ds[field]

    # Assign lead day coordinate 'd'
    import cftime
    init_dt = cftime.DatetimeNoLeap(init_year, init_month, 1)
    lead_days = []
    for tv in da.time.values:
        try:
            cur_dt = cftime.DatetimeNoLeap(tv.year, tv.month, tv.day)
            delta = (cur_dt - init_dt).days
            lead_days.append(int(delta))
        except Exception:
            lead_days.append(-1)

    da = da.assign_coords(d=("time", lead_days))
    da = da.swap_dims({"time": "d"})
    da = da.reset_coords(["time"], drop=True)

    # Filter to requested lead days (1..84)
    sel_days = [d for d in config.lead_days if d in da.d.values]
    da = da.sel(d=sel_days)

    return da


def load_daily_campaign_field(
    config: DailyDriftConfig,
    experiment: str,
    field: str,
    init_month: int,
    approved_years: Optional[List[int]] = None,
    approved_members: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    verbose: bool = True,
) -> xr.DataArray:
    """Load approved daily campaign data into (Y, M, d, lat, lon) DataArray."""
    years   = approved_years  or config.active_years
    members = approved_members or config.members

    year_das = []
    for yr in years:
        member_das = []
        for member in members:
            try:
                da_m = load_daily_spatial_field(
                    config, experiment, yr, init_month, member, field, chunks
                )
                member_das.append(da_m)
            except Exception as exc:
                warnings.warn(f"Cannot load {experiment}/{yr}/{member}/{field}: {exc}", stacklevel=2)

        if member_das:
            da_y = xr.concat(member_das, dim=pd.Index(members[:len(member_das)], name="M"))
            year_das.append(da_y)
            if verbose:
                print(f"  Loaded {experiment} yr={yr} init_month={init_month} shape={da_y.shape}")

    if not year_das:
        raise ValueError(f"No daily data loaded for {experiment}/{field}/init_month={init_month}.")

    result = xr.concat(year_das, dim=pd.Index(years[:len(year_das)], name="Y"))
    return result


def load_daily_obs_spatial(
    obs_path: str | Path,
    field: str,
    years_range: Tuple[int, int],
    chunks: Optional[dict] = None,
) -> Optional[xr.DataArray]:
    """Load spatial daily observations dataset (e.g., ERA5_daily)."""
    obs_path = Path(obs_path)
    if not obs_path.is_file():
        warnings.warn(f"load_daily_obs_spatial: file not found: {obs_path}", stacklevel=2)
        return None

    try:
        open_chunks = chunks or {"time": 60}
        ds = xr.open_dataset(obs_path, chunks=open_chunks, use_cftime=True)
        ds = drop_feb29(ds, "time")

        if field not in ds.data_vars:
            warnings.warn(f"load_daily_obs_spatial: field '{field}' not in {obs_path}", stacklevel=2)
            return None

        da = ds[field]
        sy, ey = years_range
        da = da.sel(time=slice(str(sy), str(ey)))
        return da
    except Exception as exc:
        warnings.warn(f"load_daily_obs_spatial error: {exc}", stacklevel=2)
        return None


__all__ = [
    "daily_data_dir",
    "discover_daily_files",
    "probe_daily_file",
    "discover_all_daily_files",
    "load_daily_spatial_field",
    "load_daily_campaign_field",
    "load_daily_obs_spatial",
]
