"""
monthly_io.py
=============
Monthly S2D spatial analysis — filesystem I/O bridge.

This module is the only place that touches the filesystem for the monthly
spatial workflow.  All pure analytical logic lives in monthly_core.py.

Wraps ``esp_lab.data_access_e3sm`` for file discovery and loading.

Functions
---------
build_expected_file_path  — canonical file path for one (experiment, init_tag, member, field)
probe_file                — open a file, check var presence, lead coverage, units, calendar
discover_all_files        — loop over config, call probe_file, return inventory DataFrame
load_spatial_field        — open approved file(s) and return (L, lat, lon) DataArray
load_obs_spatial          — load observation dataset for bias comparison
write_audit_csv           — write a DataFrame to CSV (shared helper)
load_audit_csv            — read an inventory CSV back to DataFrame
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from esp_lab.data_access_e3sm import (
    build_init_tags,
    preprocessor_monthly,
    time_set_midmonth,
)
from esp_lab.diagnostics.monthly_core import (
    GateStatus,
    InventoryRecord,
    InventoryStatus,
    LeadCoverageResult,
    MonthlyConfig,
    check_lead_coverage,
    classify_analysis_status,
    classify_archive_status,
    init_tag_from_year_month,
    inventory_records_to_df,
)


# ===========================================================================
# File path construction
# ===========================================================================


def build_expected_file_path(
    data_dir: str,
    case_prefix: str,
    init_tag: str,
    member: str,
    field: str,
    realm: str = "atm",
    grid: str = "180x360_aave",
    freq: str = "monthly",
    ts_split: str = "2yr",
) -> Path:
    """Construct the canonical file path for one (experiment, init_tag, member, field).

    Expected layout (from data_access_e3sm documentation)::

        {data_dir}/
          {case_prefix}_{init_tag}/
            {member}/
              post/{realm}/{grid}/ts/{freq}/{ts_split}/
                {FIELD}_{start_yyyymm}_{end_yyyymm}.nc

    Parameters
    ----------
    data_dir:
        Root directory containing case directories.
    case_prefix:
        Common prefix of case directories, excluding init tag.
    init_tag:
        Initialization tag in YYYYMMDDHH format (e.g. ``"1980050100"``).
    member:
        Ensemble member tag (e.g. ``"EN00"``).
    field:
        Variable name / filename field prefix (e.g. ``"PRECT"``).

    Returns
    -------
    Path to the expected directory containing the NetCDF file.
    Note: the filename itself includes start/end YYYYMM so we return the
    directory; use ``glob_field_files`` to find the actual file.
    """
    case_dir = f"{case_prefix}_{init_tag}"
    ts_dir = (
        Path(data_dir)
        / case_dir
        / member
        / "post"
        / realm
        / grid
        / "ts"
        / freq
        / ts_split
    )
    return ts_dir


def glob_field_files(ts_dir: Path, field: str) -> List[Path]:
    """Return all NetCDF files matching ``{field}_*.nc`` in ts_dir."""
    if not ts_dir.is_dir():
        return []
    return sorted(ts_dir.glob(f"{field}_*.nc"))


# ===========================================================================
# File probing
# ===========================================================================


def _extract_grid_signature(ds: xr.Dataset) -> str:
    """Produce a compact grid identifier from a dataset."""
    parts = []
    for dim in ("lat", "lon", "ncol", "nCells"):
        if dim in ds.dims:
            parts.append(f"{dim}={ds.sizes[dim]}")
    if not parts:
        parts = [f"dims={dict(ds.sizes)}"]
    return ",".join(parts)


def _extract_calendar(ds: xr.Dataset) -> str:
    """Return calendar string from time coordinate attributes."""
    if "time" in ds.coords:
        cal = ds["time"].encoding.get("calendar", "") or ds["time"].attrs.get("calendar", "")
        if cal:
            return str(cal)
        # Try cftime
        try:
            tv = ds["time"].values[0]
            if hasattr(tv, "calendar"):
                return str(tv.calendar)
        except Exception:
            pass
    return "UNKNOWN"


def _lead_numbers_from_ds(ds: xr.Dataset, field: str, init_tag: str) -> List[int]:
    """Extract lead numbers by year+month matching, not by timestamp day."""
    init_year  = int(init_tag[:4])
    init_month = int(init_tag[4:6])

    if "L" in ds.dims or "L" in ds.coords:
        return sorted(int(x) for x in ds.coords.get("L", ds["L"]).values)

    if "time" not in ds.coords:
        return []

    # Map each time step to a lead number by (year, month)
    leads = []
    for tv in ds["time"].values:
        try:
            ty = int(tv.year); tm = int(tv.month)
        except AttributeError:
            import pandas as _pd
            ts = _pd.Timestamp(tv)
            ty, tm = ts.year, ts.month

        # Compute 1-based lead: months since init
        month_index = (ty - init_year) * 12 + (tm - init_month) + 1
        if 1 <= month_index <= 240:   # sanity: up to 20-year lead
            leads.append(month_index)

    return sorted(leads)


def probe_file(
    ts_dir: Path,
    field: str,
    init_tag: str,
    expected_leads: Sequence[int],
    experiment: str,
    case_prefix: str,
    member: str,
    realm: str = "atm",
    freq: str = "monthly",
    ts_split: str = "2yr",
    chunks: Optional[dict] = None,
) -> InventoryRecord:
    """Open a post-processed file and produce an InventoryRecord.

    Validation checks:
    - File exists and can be opened.
    - Variable exists in the dataset.
    - All expected forecast leads are present (validated by year+month, not timestamp).
    - No lead appears twice.
    - Units and calendar recorded.
    - Grid signature recorded.

    Parameters
    ----------
    ts_dir:
        Directory returned by ``build_expected_file_path``.
    field:
        Variable name.
    init_tag:
        Initialization tag YYYYMMDDHH.
    expected_leads:
        1-based lead list from config.
    Other parameters are recorded directly into the InventoryRecord.

    Returns
    -------
    InventoryRecord with all fields populated.
    """
    record = InventoryRecord(
        experiment=experiment,
        case_prefix=case_prefix,
        init_tag=init_tag,
        member=member,
        variable=field,
        stream=f"{realm}/{freq}/{ts_split}",
    )

    files = glob_field_files(ts_dir, field)
    record.file_count = len(files)

    if not files:
        record.analysis_status = InventoryStatus.MISSING.value
        record.archive_status  = InventoryStatus.MISSING.value
        record.notes = f"No files matching {field}_*.nc in {ts_dir}"
        return record

    try:
        open_chunks = chunks or {"time": -1}
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks=open_chunks,
            use_cftime=True,
        )
        ds = time_set_midmonth(ds, "time")
    except Exception as exc:
        record.analysis_status = InventoryStatus.ERROR.value
        record.archive_status  = InventoryStatus.ERROR.value
        record.notes = f"Cannot open: {exc}"
        return record

    try:
        # Variable presence
        record.variable_found = field in ds.data_vars

        # Units
        if field in ds.data_vars:
            record.units = str(ds[field].attrs.get("units", "UNKNOWN"))

        # Calendar
        record.calendar = _extract_calendar(ds)

        # Grid signature
        record.grid_signature = _extract_grid_signature(ds)

        # Lead numbers by year+month
        available = _lead_numbers_from_ds(ds, field, init_tag)
        coverage  = check_lead_coverage(available, list(expected_leads))

        record.available_leads = ",".join(str(x) for x in coverage.available_leads)
        record.missing_leads   = ",".join(str(x) for x in coverage.missing_leads)
        record.duplicate_leads = ",".join(str(x) for x in coverage.duplicate_leads)

        # Placeholder for window_defs; full classification done in inventory.py
        record.analysis_status = GateStatus.ANALYSIS_READY.value \
            if coverage.is_complete else GateStatus.BLOCKED.value
        record.archive_status  = classify_archive_status(coverage).value

        if not record.variable_found:
            record.notes = f"Variable '{field}' not found; available: {list(ds.data_vars)}"
            record.analysis_status = InventoryStatus.MISSING.value

    except Exception as exc:
        record.analysis_status = InventoryStatus.ERROR.value
        record.archive_status  = InventoryStatus.ERROR.value
        record.notes = f"Probe error: {exc}"
    finally:
        try:
            ds.close()
        except Exception:
            pass

    return record


# ===========================================================================
# Full campaign discovery
# ===========================================================================


def discover_all_files(
    config: MonthlyConfig,
    variable: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Loop over experiment × init_tag × member and call probe_file.

    Parameters
    ----------
    config:
        MonthlyConfig with active_years, active_months, members, experiments.
    variable:
        Limit to this variable; defaults to all variables in config.
    verbose:
        Print progress.

    Returns
    -------
    inventory DataFrame with all InventoryRecord columns.
    """
    variables = (
        [config.get_variable(variable)] if variable
        else config.variables
    )

    init_tags = build_init_tags(config.active_years, config.active_months)

    records: List[InventoryRecord] = []

    for exp_label, exp_spec in config.experiments.items():
        for var_spec in variables:
            field = var_spec.native_name
            for tag in init_tags:
                for member in config.members:
                    ts_dir = build_expected_file_path(
                        data_dir=config.data_dir,
                        case_prefix=exp_spec.case_prefix,
                        init_tag=tag,
                        member=member,
                        field=field,
                        realm=config.realm,
                        grid=config.grid,
                        freq=config.freq,
                        ts_split=config.ts_split,
                    )
                    rec = probe_file(
                        ts_dir=ts_dir,
                        field=field,
                        init_tag=tag,
                        expected_leads=config.leads,
                        experiment=exp_label,
                        case_prefix=exp_spec.case_prefix,
                        member=member,
                        realm=config.realm,
                        freq=config.freq,
                        ts_split=config.ts_split,
                    )
                    records.append(rec)

                    if verbose:
                        status_icon = {
                            "ANALYSIS_READY":   "✓",
                            "ARCHIVE_COMPLETE":  "✓",
                            "BLOCKED":           "✗",
                            "MISSING":           "?",
                            "ERROR":             "!",
                        }.get(rec.analysis_status, "?")
                        print(
                            f"  [{status_icon}] {exp_label:<15s} {tag} {member} {field}"
                        )

    return inventory_records_to_df(records)


# ===========================================================================
# Load approved data
# ===========================================================================


def load_spatial_field(
    config: MonthlyConfig,
    experiment: str,
    init_tag: str,
    member: str,
    field: str,
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """Open post-processed files and return a (L, lat, lon) DataArray.

    Uses ``preprocessor_monthly`` for timestamp normalisation and lead assignment.

    Parameters
    ----------
    config:
        MonthlyConfig.
    experiment:
        Experiment label (must be in config.experiments).
    init_tag:
        YYYYMMDDHH init tag.
    member:
        Ensemble member tag.
    field:
        Variable name.
    chunks:
        Dask chunks.  Default: ``{"time": -1}``.

    Returns
    -------
    DataArray with dims (L, lat, lon) and 1-based L coordinate.
    """
    exp_spec = config.experiments[experiment]
    ts_dir = build_expected_file_path(
        data_dir=config.data_dir,
        case_prefix=exp_spec.case_prefix,
        init_tag=init_tag,
        member=member,
        field=field,
        realm=config.realm,
        grid=config.grid,
        freq=config.freq,
        ts_split=config.ts_split,
    )

    files = glob_field_files(ts_dir, field)
    if not files:
        raise FileNotFoundError(
            f"No {field}_*.nc files found in {ts_dir}. "
            f"Check data_dir and case_prefix."
        )

    open_chunks = chunks or {"time": -1}
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks=open_chunks,
        use_cftime=True,
        preprocess=lambda d: preprocessor_monthly(d, nlead=max(config.leads), field=field),
    )

    da = ds[field]

    # Ensure L coordinate is correct type
    if "L" in da.coords:
        da = da.assign_coords(L=("L", [int(x) for x in da.L.values]))

    # Select only requested leads
    available_leads = [int(x) for x in da.L.values]
    sel_leads = [L for L in config.leads if L in available_leads]
    da = da.sel(L=sel_leads)

    return da


def load_all_members(
    config: MonthlyConfig,
    experiment: str,
    init_tag: str,
    field: str,
    approved_members: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """Load all members for one (experiment, init_tag, field) and concatenate.

    Returns DataArray with dims (M, L, lat, lon).
    """
    members = approved_members or config.members
    member_das = []
    for member in members:
        try:
            da = load_spatial_field(config, experiment, init_tag, member, field, chunks)
            member_das.append(da)
        except Exception as exc:
            warnings.warn(
                f"load_all_members: cannot load {experiment}/{init_tag}/{member}/{field}: "
                f"{exc}",
                stacklevel=2,
            )

    if not member_das:
        raise ValueError(
            f"load_all_members: no members could be loaded for "
            f"{experiment}/{init_tag}/{field}."
        )

    result = xr.concat(member_das, dim=pd.Index(members[:len(member_das)], name="M"))
    return result


def load_campaign_field(
    config: MonthlyConfig,
    experiment: str,
    field: str,
    init_month: int,
    approved_years: Optional[List[int]] = None,
    approved_members: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    verbose: bool = True,
) -> xr.DataArray:
    """Load the full (Y, M, L, lat, lon) DataArray for one season.

    Parameters
    ----------
    config:
        MonthlyConfig.
    experiment:
        Experiment label.
    field:
        Variable name.
    init_month:
        Calendar month for initialization (e.g. 5 for May).
    approved_years:
        Init years to load.  Defaults to config.active_years.
    approved_members:
        Members to load.  Defaults to config.members.
    chunks:
        Dask chunks.
    verbose:
        Print progress.

    Returns
    -------
    DataArray with dims (Y, M, L, lat, lon).  Y = init years, M = members.
    """
    years   = approved_years  or config.active_years
    members = approved_members or config.members
    init_tags = build_init_tags(years, init_month)

    year_das = []
    for year, tag in zip(years, init_tags):
        try:
            da_ym = load_all_members(config, experiment, tag, field, members, chunks)
            year_das.append(da_ym)
            if verbose:
                print(f"  Loaded {experiment}/{tag}  shape={da_ym.shape}")
        except Exception as exc:
            warnings.warn(
                f"load_campaign_field: cannot load {experiment}/{tag}: {exc}",
                stacklevel=2,
            )

    if not year_das:
        raise ValueError(
            f"load_campaign_field: nothing loaded for "
            f"{experiment}/{field}/init_month={init_month}."
        )

    result = xr.concat(year_das, dim=pd.Index(years[:len(year_das)], name="Y"))
    if "M" not in result.dims:
        result = result.expand_dims("M")

    return result


# ===========================================================================
# Observation loading
# ===========================================================================


def load_obs_spatial(
    obs_path: str,
    field: str,
    years_range: Tuple[int, int],
    chunks: Optional[dict] = None,
) -> Optional[xr.DataArray]:
    """Load a spatial observation dataset.

    This is a thin wrapper that opens a local NetCDF file.  For GPCP or
    other obs products, point this at the post-processed file.

    Parameters
    ----------
    obs_path:
        Full path to the observation NetCDF file.
    field:
        Variable name inside the file.
    years_range:
        (start_year, end_year) for time subsetting.
    chunks:
        Dask chunks.

    Returns
    -------
    DataArray with a ``time`` coordinate, or None if file not found.
    """
    obs_path = Path(obs_path)
    if not obs_path.is_file():
        warnings.warn(
            f"load_obs_spatial: observation file not found: {obs_path}. "
            "Bias maps will not be computed (paired-difference only).",
            stacklevel=2,
        )
        return None

    try:
        open_chunks = chunks or {"time": 60}
        ds = xr.open_dataset(obs_path, chunks=open_chunks, use_cftime=True)
        ds = time_set_midmonth(ds, "time")

        if field not in ds.data_vars:
            warnings.warn(
                f"load_obs_spatial: variable '{field}' not in {obs_path}. "
                f"Available: {list(ds.data_vars)}",
                stacklevel=2,
            )
            return None

        da = ds[field]
        sy, ey = years_range
        da = da.sel(time=slice(str(sy), str(ey)))
        return da

    except Exception as exc:
        warnings.warn(f"load_obs_spatial: error opening {obs_path}: {exc}", stacklevel=2)
        return None


# ===========================================================================
# Inventory I/O helpers
# ===========================================================================


def write_audit_csv(df: pd.DataFrame, outdir: Path, prefix: str = "monthly") -> Path:
    """Write a DataFrame to ``{outdir}/{prefix}_inventory.csv``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{prefix}_inventory.csv"
    df.to_csv(path, index=False)
    return path


def load_audit_csv(path: Path) -> pd.DataFrame:
    """Load an inventory CSV back to a DataFrame."""
    return pd.read_csv(path, dtype=str)


__all__ = [
    "build_expected_file_path",
    "glob_field_files",
    "probe_file",
    "discover_all_files",
    "load_spatial_field",
    "load_all_members",
    "load_campaign_field",
    "load_obs_spatial",
    "write_audit_csv",
    "load_audit_csv",
]
