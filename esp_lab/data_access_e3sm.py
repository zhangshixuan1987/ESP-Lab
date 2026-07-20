"""
Utilities for accessing post-processed E3SM S2D hindcast data.

This module mirrors the overall workflow style of the CESM/SMYLE
data_access utility, but is adapted to E3SM directory-driven data
organization, where data are stored under top-level initialization
directories and ensemble subdirectories.

Expected directory structure
----------------------------
<data_dir>/
  <case_prefix>_<init_tag>/
    EN00/
      post/atm/180x360_aave/ts/monthly/2yr/{FIELD}_{start_yyyymm}_{end_yyyymm}.nc
    EN01/
      post/atm/180x360_aave/ts/monthly/2yr/{FIELD}_{start_yyyymm}_{end_yyyymm}.nc
    ...

Example top-level case directory
--------------------------------
WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL_1980050100

Example file
------------
TREFHT_198005_198204.nc

Notes
-----
1. Monthly timestamp convention
   E3SM monthly means are often written with timestamps at the beginning
   of the following month. For example, for a simulation initialized on

       1980-05-01-00000

   the first monthly mean may be stored with timestamp

       1980-06-01-00000

   even though it represents the mean over May 1980.

   This module accounts for that convention when it can be detected from
   time bounds or from the filename's represented YYYYMM range. If those
   are unavailable, model-output semantics are used: timestamps on day 1
   are treated as the beginning of the following month, while other
   timestamps are normalized within their existing month.

2. Calendar convention
   E3SM commonly uses a noleap (365_day) calendar. This module preserves
   the native E3SM calendar and typically returns cftime.DatetimeNoLeap
   coordinates. Calendar harmonization with Gregorian observational
   datasets should be handled downstream, depending on the analysis.

3. Recommended workflow
   For initialized hindcast analysis, lead-based alignment is usually
   safer than relying on exact absolute timestamps when comparing with
   observational datasets.

Typical use
-----------
from esp_lab import data_access_e3sm as data_access

field = "TREFHT"
data_dir = "/pscratch/sd/z/zhan391/e3sm_project/E3SMv3_S2D"
case_prefix = "WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL"
members = [f"EN{i:02d}" for i in range(10)]
init_tags = data_access.build_init_tags(np.arange(1980, 2015), [5, 11])

ds = data_access.get_monthly_data(
    data_dir=data_dir,
    case_prefix=case_prefix,
    members=members,
    init_tags=init_tags,
    field=field,
    nlead=24,
    realm="atm",
    grid="180x360_aave",
    freq="monthly",
    ts_split="2yr",
)
"""

import re
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr


def _validate_path_arg(name: str, value: str) -> None:
    """
    Validate that a path-related argument is a string.

    This mainly guards against accidental tuple creation from notebook code like:
        realm = "atm",
    """
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string, got {type(value).__name__}: {value!r}"
        )


def build_init_tags(
    years: Iterable[int],
    month: Union[int, Iterable[int]],
    init_day: int = 1,
    init_hour: int = 0,
) -> List[str]:
    """
    Build E3SM initialization tags like YYYYMMDDHH.

    Parameters
    ----------
    years : iterable of int
        Initialization years.
    month : int or iterable of int
        Initialization month(s), e.g. 5 or [5, 11].
    init_day : int, optional
        Initialization day, default 1.
    init_hour : int, optional
        Initialization hour, default 0.

    Returns
    -------
    init_tags : list of str
        List of initialization tags like '1980050100'.
    """
    if isinstance(month, int):
        months = [month]
    else:
        months = list(month)

    for m in months:
        if not (1 <= m <= 12):
            raise ValueError(f"month must be in 1-12, got {m}.")
    if not (1 <= init_day <= 31):
        raise ValueError(f"init_day must be in 1-31, got {init_day}.")
    if not (0 <= init_hour <= 23):
        raise ValueError(f"init_hour must be in 0-23, got {init_hour}.")

    init_tags = []
    for year in years:
        for m in months:
            init_tags.append(f"{year:04d}{m:02d}{init_day:02d}{init_hour:02d}")
    return init_tags


FILENAME_PATTERN = re.compile(
    r"^(?P<field>.+)_(?P<start_yyyymm>\d{6})_(?P<end_yyyymm>\d{6})\.nc$"
)


def parse_ts_filename(filename: str) -> Dict[str, str]:
    """
    Parse a post-processed E3SM filename of the form:

        {FIELD}_{start_yyyymm}_{end_yyyymm}.nc

    Parameters
    ----------
    filename : str
        Filename or full path.

    Returns
    -------
    info : dict
        Dictionary with keys:
            - field
            - start_yyyymm
            - end_yyyymm

    Raises
    ------
    ValueError
        If filename does not match the expected format.
    """
    fname = Path(filename).name
    match = FILENAME_PATTERN.match(fname)
    if match is None:
        raise ValueError(f"Unrecognized filename format: {filename}")
    return match.groupdict()


def expected_yyyymm_from_init_tag(init_tag: str, nlead: int) -> Tuple[str, str]:
    """
    Compute expected start and end YYYYMM strings from an init tag and nlead.

    This refers to the represented monthly means, not the raw file timestamps.
    For example, if init_tag='1980050100' and nlead=24, the expected file range is

        198005 -> 198204

    even if the first raw timestamp in the NetCDF file is 1980-06-01.

    Parameters
    ----------
    init_tag : str
        Initialization tag like '1980050100'.
    nlead : int
        Number of lead months.

    Returns
    -------
    start_yyyymm : str
        Start YYYYMM string.
    end_yyyymm : str
        End YYYYMM string.
    """
    year = int(init_tag[:4])
    month = int(init_tag[4:6])

    start_yyyymm = f"{year:04d}{month:02d}"

    end_index = month - 1 + (nlead - 1)
    end_year = year + end_index // 12
    end_month = end_index % 12 + 1
    end_yyyymm = f"{end_year:04d}{end_month:02d}"

    return start_yyyymm, end_yyyymm


def _yyyymm_add_months(yyyymm: str, months: int) -> str:
    year = int(yyyymm[:4])
    month = int(yyyymm[4:6])
    index = (year * 12 + month - 1) + months
    return f"{index // 12:04d}{index % 12 + 1:02d}"


def _datetime_parts(value) -> Tuple[int, int, int]:
    """Return (year, month, day) from common decoded datetime-like values."""
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return int(value.year), int(value.month), int(value.day)

    if np.issubdtype(np.asarray(value).dtype, np.datetime64):
        ts = pd.Timestamp(value)
        return int(ts.year), int(ts.month), int(ts.day)

    raise TypeError(f"Unsupported time value type: {type(value)!r}")


def _is_probably_monthly(time: xr.DataArray) -> bool:
    """Return True when each record appears to represent one consecutive month."""
    try:
        values = time.values
        if len(values) < 2:
            return False

        month_indexes = []
        for value in values:
            year, month, _ = _datetime_parts(value)
            month_indexes.append(year * 12 + month)

        diffs = np.diff(month_indexes)
        return bool(np.all(diffs == 1))
    except Exception:
        return False


def drop_feb29(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    """Drop leap-day records from decoded daily/submonthly time coordinates."""
    if time_name not in ds.coords:
        return ds

    try:
        time = ds[time_name]
        if not hasattr(time, "dt"):
            return ds
        if _is_probably_monthly(time):
            return ds

        keep = ~((time.dt.month == 2) & (time.dt.day == 29))
        return ds.sel({time_name: keep})
    except Exception as e:
        warnings.warn(
            f"drop_feb29: could not drop leap-day records: {e}. "
            "Dataset left unchanged."
        )
        return ds


def _midmonth_time(year: int, month: int, template=None):
    """
    Build a mid-month timestamp using the same calendar family as template.

    Most E3SM/CESM output is noleap, but some model setups can use Gregorian
    or standard calendars. Preserve that decoded calendar instead of forcing
    noleap unconditionally.
    """
    year = int(year)
    month = int(month)

    if template is None:
        return cftime.DatetimeNoLeap(year, month, 15)

    if isinstance(template, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(year=year, month=month, day=15)

    if hasattr(template, "year") and hasattr(template, "month"):
        try:
            return type(template)(year, month, 15)
        except TypeError:
            return cftime.DatetimeNoLeap(year, month, 15)

    return cftime.DatetimeNoLeap(year, month, 15)


def _previous_month(year: int, month: int) -> Tuple[int, int]:
    month -= 1
    if month == 0:
        month = 12
        year -= 1
    return year, month


def _time_bounds_name(ds: xr.Dataset, time_name: str) -> Optional[str]:
    """Return the likely bounds variable for a time coordinate, if present."""
    bounds_name = ds[time_name].attrs.get("bounds")
    if bounds_name in ds:
        return bounds_name

    for candidate in ("time_bnds", "time_bounds", "bounds_time", "time_bound"):
        if candidate in ds:
            return candidate

    return None


def _midmonth_from_bounds(
    ds: xr.Dataset,
    time_name: str,
) -> Optional[List[cftime.DatetimeNoLeap]]:
    """
    Build represented-month timestamps from time bounds.

    For monthly products with bounds [month start, next month start], the
    represented month is unambiguous from the lower bound.
    """
    bounds_name = _time_bounds_name(ds, time_name)
    if bounds_name is None:
        return None

    tb = ds[bounds_name]
    if time_name not in tb.dims:
        return None

    bounds_dim = next((dim for dim in tb.dims if dim != time_name), None)
    if bounds_dim is None or tb.sizes.get(bounds_dim, 0) < 1:
        return None

    new_times = []
    for lower in tb.isel({bounds_dim: 0}).values:
        year, month, _ = _datetime_parts(lower)
        new_times.append(_midmonth_time(year, month, template=lower))

    return new_times


def _represented_start_yyyymm_from_source(ds: xr.Dataset) -> Optional[str]:
    source = ds.encoding.get("source")
    if not source:
        return None

    try:
        return parse_ts_filename(str(source))["start_yyyymm"]
    except ValueError:
        return None


def _infer_time_month_shift(ds: xr.Dataset, time_name: str) -> int:
    """
    Return the month shift to apply before setting day 15.

    0 means timestamps already name the represented month.
    -1 means timestamps are one month after the represented month.
    """
    if ds.sizes.get(time_name, 0) == 0:
        return 0

    first_year, first_month, first_day = _datetime_parts(ds[time_name].values[0])
    start_yyyymm = _represented_start_yyyymm_from_source(ds)
    if start_yyyymm is None:
        return -1 if first_day == 1 else 0

    first_yyyymm = f"{first_year:04d}{first_month:02d}"

    if first_yyyymm == start_yyyymm:
        return 0
    if first_yyyymm == _yyyymm_add_months(start_yyyymm, 1):
        return -1

    warnings.warn(
        "time_set_midmonth: could not infer model timestamp convention from "
        f"source={ds.encoding.get('source')!r}; first timestamp month "
        f"{first_yyyymm} does not match represented start {start_yyyymm} or "
        "the following month. Falling back to model timestamp semantics."
    )
    return -1 if first_day == 1 else 0


def time_set_midmonth(ds: xr.Dataset, time_name: str) -> xr.Dataset:
    """
    Return a copy of ds with monthly time on the represented month midpoint.

    The represented month is detected in this order:
    1. Time bounds lower edge, when bounds are present.
    2. The filename's represented start YYYYMM compared with the first
       timestamp month.
    3. Model-output fallback: day-1 timestamps represent the previous month;
       other timestamps represent their own month.

    This handles both common model conventions: first day of following month
    and last day of represented month.

    Notes
    -----
    - This function preserves the noleap calendar convention used by E3SM.
    - Returned timestamps are cftime.DatetimeNoLeap.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    time_name : str
        Name of time coordinate.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with adjusted time coordinate.
    """
    if time_name not in ds.coords and time_name not in ds.dims:
        raise ValueError(f"time coordinate '{time_name}' not found in dataset.")

    ds = ds.copy()
    ds = drop_feb29(ds, time_name=time_name)

    bounded_times = _midmonth_from_bounds(ds, time_name)
    if bounded_times is not None:
        ds[time_name] = bounded_times
        return ds

    month_shift = _infer_time_month_shift(ds, time_name)
    newtime = []
    for value in ds[time_name].values:
        year, month, _ = _datetime_parts(value)
        if month_shift == -1:
            year, month = _previous_month(year, month)
        newtime.append(_midmonth_time(year, month, template=value))

    ds[time_name] = newtime

    return ds


def preprocessor_monthly(ds0: xr.Dataset, nlead: int, field: str) -> xr.Dataset:
    """
    Standard E3SM monthly preprocessor.

    This function:
    - normalizes time to the represented month midpoint
    - selects the first nlead months
    - defines a lead coordinate
    - preserves the requested field and time coordinate

    Notes
    -----
    - E3SM monthly means may use either represented-month or following-month
      timestamps. The timestamp convention is inferred where possible.
    - This function preserves the native E3SM noleap calendar.
    - Calendar harmonization with Gregorian observational datasets should
      be handled downstream.

    Parameters
    ----------
    ds0 : xarray.Dataset
        Input dataset for one file.
    nlead : int
        Number of lead months to retain.
    field : str
        Requested field name.

    Returns
    -------
    d0 : xarray.Dataset
        Preprocessed dataset with dimensions using L instead of time.
    """
    if field not in ds0:
        raise ValueError(f"Field '{field}' not found in dataset. Available: {list(ds0.data_vars)}")

    ds0 = time_set_midmonth(ds0, 'time')

    available = ds0.sizes.get('time', 0)
    if available < nlead:
        warnings.warn(
            f"File has only {available} time steps but nlead={nlead} requested. "
            "Returning all available time steps."
        )

    d0 = ds0[field].isel(time=slice(0, nlead))
    
    if 'lon' in ds0.coords and 'lat' in ds0.coords:
        d0 = d0.assign_coords({"lon": ds0.lon, "lat": ds0.lat})
        
    d0 = d0.assign_coords(L=("time", np.arange(d0.sizes["time"]) + 1))
    d0 = d0.swap_dims({"time": "L"})
    d0 = d0.to_dataset(name=field)
    d0 = d0.reset_coords(["time"])
    d0["time"] = d0.time.expand_dims("Y")
    #d0 = d0.chunk({"L": -1})
    d0 = d0.chunk({"L": min(12, d0.sizes["L"])}) 
    return d0


def _verification_time_from_init_tags(
    init_tags: Iterable[object],
    leads: Iterable[object],
) -> xr.DataArray:
    """Construct deterministic no-leap verification dates for E3SM hindcasts."""
    init_values = [str(value) for value in init_tags]
    lead_values = [int(value) for value in leads]
    values = []
    for init_tag in init_values:
        if len(init_tag) < 6 or not init_tag[:6].isdigit():
            raise ValueError(f"Invalid E3SM initialization tag: {init_tag!r}")
        init_year = int(init_tag[:4])
        init_month = int(init_tag[4:6])
        row = []
        for lead in lead_values:
            if lead < 1:
                raise ValueError(f"E3SM lead values must be >= 1, got {lead}")
            month_index = init_month - 1 + lead - 1
            row.append(
                cftime.DatetimeNoLeap(
                    init_year + month_index // 12,
                    month_index % 12 + 1,
                    15,
                )
            )
        values.append(row)
    return xr.DataArray(
        np.asarray(values, dtype=object),
        dims=("Y", "L"),
        coords={"Y": init_values, "L": lead_values},
        name="time",
        attrs={
            "long_name": "forecast verification time",
            "construction": "init_tag_plus_lead_v1",
        },
    )

# Backward-compatible alias
preprocessor = preprocessor_monthly


def _split_open_and_post_chunks(
    chunks: Optional[Dict[str, int]],
    post_chunk_dims: Tuple[str, ...] = ("Y", "M"),
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Keep concat-dimension chunks out of open_mfdataset."""
    if chunks is None:
        return {}, {}

    open_chunks = {
        dim: chunk for dim, chunk in chunks.items()
        if dim not in post_chunk_dims
    }
    post_chunks = {
        dim: chunk for dim, chunk in chunks.items()
        if dim in post_chunk_dims
    }
    return open_chunks, post_chunks


def file_dict(
    data_dir: str,
    case_prefix: str,
    member: str,
    field: str,
    realm: str = "atm",
    grid: str = "180x360_aave",
    freq: str = "monthly",
    ts_split: str = "2yr",
    verify_field_name: bool = True,
) -> Dict[str, str]:
    """
    Return a dictionary of filepaths keyed by initialization tag for one member.

    Expected filename pattern:
        {FIELD}_{start_yyyymm}_{end_yyyymm}.nc

    Parameters
    ----------
    data_dir : str
        Base directory containing E3SM S2D cases.
    case_prefix : str
        Common prefix of case directories, excluding init tag.
    member : str
        Ensemble member, e.g. 'EN00'.
    field : str
        Variable name to search for, e.g. 'TREFHT'.
    realm : str, optional
        Realm name under post/, default 'atm'.
    grid : str, optional
        Post-processed grid name, default '180x360_aave'.
    freq : str, optional
        Frequency directory, default 'monthly'.
    ts_split : str, optional
        Post-processing split directory, e.g. '1yr', '2yr', '5yr'.
    verify_field_name : bool, optional
        If True, verify parsed field name exactly matches `field`.

        - True (recommended):
            Prevents accidental mismatches such as loading TREFHTMX when
            TREFHT was requested.

        - False:
            Accepts any file matching the filename glob pattern. Faster but less safe.

    Returns
    -------
    filepaths : dict
        Dictionary keyed by init_tag with filepath values.

    Raises
    ------
    ValueError
        If multiple matching files are found for one init/member/field.
    """
    _validate_path_arg("data_dir", data_dir)
    _validate_path_arg("case_prefix", case_prefix)
    _validate_path_arg("member", member)
    _validate_path_arg("field", field)
    _validate_path_arg("realm", realm)
    _validate_path_arg("grid", grid)
    _validate_path_arg("freq", freq)
    _validate_path_arg("ts_split", ts_split)

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    case_dirs = sorted(data_path.glob(f"{case_prefix}_*"))
    if not case_dirs:
        warnings.warn(f"No case directories matching '{case_prefix}_*' found in {data_dir}.")
    filepaths: Dict[str, str] = {}

    for case_dir in case_dirs:
        init_tag = case_dir.name.split("_")[-1]
        matches = sorted(
            (case_dir / member / "post" / realm / grid / "ts" / freq / ts_split).glob(
                f"{field}_*.nc"
            )
        )
        if not matches:
            continue

        valid_matches = []
        for path in matches:
            try:
                info = parse_ts_filename(str(path))
            except ValueError:
                continue

            if verify_field_name and info["field"] != field:
                continue

            valid_matches.append(str(path))

        if len(valid_matches) == 1:
            filepaths[init_tag] = valid_matches[0]
        elif len(valid_matches) > 1:
            raise ValueError(
                f"Multiple matching files found for field={field}, "
                f"init={init_tag}, member={member}: {valid_matches}"
            )

    return filepaths


def nested_file_list_by_init(
    data_dir: str,
    case_prefix: str,
    members: List[str],
    init_tags: List[str],
    field: str,
    realm: str = "atm",
    grid: str = "180x360_aave",
    freq: str = "monthly",
    ts_split: str = "2yr",
    require_all_members: bool = True,
    verify_field_name: bool = True,
    verify_coverage: bool = False,
    nlead: Optional[int] = None,
) -> Tuple[List[List[str]], List[str]]:
    """
    Retrieve a nested list of files for requested init tags and members.

    Parameters
    ----------
    data_dir : str
        Base directory containing E3SM S2D cases.
    case_prefix : str
        Common prefix of case directories, excluding init tag.
    members : list of str
        Ensemble members, e.g. ['EN00', 'EN01', ...].
    init_tags : list of str
        Initialization tags like ['1980050100', '1980110100', ...].
    field : str
        Variable name to search for.
    realm : str, optional
        Realm name under post/, default 'atm'.
    grid : str, optional
        Post-processed grid name, default '180x360_aave'.
    freq : str, optional
        Frequency directory, default 'monthly'.
    ts_split : str, optional
        Post-processing split directory, e.g. '1yr', '2yr', '5yr'.
    require_all_members : bool, optional
        Controls whether all ensemble members must be present for an init.

        - True (recommended for production):
            Only keep init tags where all requested members exist.
            This ensures a consistent (init, member, ...) structure and avoids
            member misalignment across initializations.

        - False:
            Keep init tags even if some members are missing.
            Useful for quick inspection of incomplete datasets, but downstream
            code must be careful about consistency assumptions.

    verify_field_name : bool, optional
        If True, verify parsed field name exactly matches `field`.

    verify_coverage : bool, optional
        If True, verify that filename date coverage matches init_tag and nlead.

        Example:
            init_tag = '1980050100'
            nlead    = 24

        Expected represented range:
            198005 -> 198204

        Notes
        -----
        This checks the represented monthly coverage encoded in the filename,
        not the raw file timestamps. Raw monthly timestamps in E3SM may begin
        one month later, e.g. 1980-06-01 for the May 1980 mean.

        - True (recommended for strict reproducibility):
            Enforce exact coverage match.

        - False:
            Allow mismatched longer files and let the preprocessor slice the
            first nlead months.

    nlead : int, optional
        Number of lead months. Required if verify_coverage=True.

    Returns
    -------
    nested_files : list of list of str
        Nested list ordered as [init][member].
    valid_inits : list of str
        Initialization tags retained in output.

    Raises
    ------
    ValueError
        If verify_coverage=True and nlead is not provided.
    """
    if not members:
        raise ValueError("members list must not be empty.")
    if not init_tags:
        raise ValueError("init_tags list must not be empty.")
    if verify_coverage and nlead is None:
        raise ValueError("nlead must be provided when verify_coverage=True")

    member_file_dicts = {
        member: file_dict(
            data_dir=data_dir,
            case_prefix=case_prefix,
            member=member,
            field=field,
            realm=realm,
            grid=grid,
            freq=freq,
            ts_split=ts_split,
            verify_field_name=verify_field_name,
        )
        for member in members
    }

    nested_files: List[List[str]] = []
    valid_inits: List[str] = []

    for init_tag in init_tags:
        files_this_init: List[str] = []

        for member in members:
            member_files = member_file_dicts[member]
            if init_tag not in member_files:
                continue

            path = member_files[init_tag]

            if verify_coverage:
                info = parse_ts_filename(path)
                exp_start, exp_end = expected_yyyymm_from_init_tag(init_tag, nlead)
                if info["start_yyyymm"] != exp_start or info["end_yyyymm"] != exp_end:
                    continue

            files_this_init.append(path)

        if require_all_members:
            if len(files_this_init) == len(members):
                nested_files.append(files_this_init)
                valid_inits.append(init_tag)
            elif len(files_this_init) > 0:
                warnings.warn(
                    f"init_tag={init_tag}: found {len(files_this_init)}/{len(members)} members; "
                    "skipping because require_all_members=True."
                )
            else:
                warnings.warn(f"init_tag={init_tag}: no files found for any member; skipping.")
        else:
            if len(files_this_init) > 0:
                nested_files.append(files_this_init)
                valid_inits.append(init_tag)
            else:
                warnings.warn(f"init_tag={init_tag}: no files found for any member; skipping.")

    return nested_files, valid_inits


def get_monthly_data(
    data_dir: str,
    case_prefix: str,
    members: List[str],
    init_tags: List[str],
    field: str,
    nlead: int,
    preproc: Union[str, Callable] = "default",
    chunks: Optional[Dict[str, int]] = None,
    realm: str = "atm",
    grid: str = "180x360_aave",
    freq: str = "monthly",
    ts_split: str = "2yr",
    require_all_members: bool = True,
    verify_field_name: bool = True,
    verify_coverage: bool = False,
    engine: str = "netcdf4",
) -> xr.Dataset:
    """
    Return a dask-backed xarray dataset arranged as (init, lead, member, ...).

    Parameters
    ----------
    data_dir : str
        Base directory containing E3SM S2D cases.
    case_prefix : str
        Common prefix of case directories, excluding init tag.
    members : list of str
        Ensemble members, e.g. ['EN00', 'EN01', ...].
    init_tags : list of str
        Initialization tags like ['1980050100', '1980110100', ...].
    field : str
        Variable name to be loaded.
    nlead : int
        Number of lead months to retain.
    preproc : {"default"} or callable, optional
        - "default": use built-in monthly E3SM preprocessor
        - callable: custom preprocessing function with signature
          preproc(ds0, nlead, field)
    chunks : dict, optional
        Chunk settings for lazy loading. Chunks for concat dimensions
        ``Y`` and ``M`` are applied after ``open_mfdataset`` so xarray can
        first use the native on-disk chunking for each source file.
    realm : str, optional
        Realm name under post/, default 'atm'.
    grid : str, optional
        Post-processed grid name, default '180x360_aave'.
    freq : str, optional
        Frequency directory, default 'monthly'.
    ts_split : str, optional
        Post-processing split directory, e.g. '1yr', '2yr', '5yr'.
    require_all_members : bool, optional
        See nested_file_list_by_init().
    verify_field_name : bool, optional
        See nested_file_list_by_init().
    verify_coverage : bool, optional
        See nested_file_list_by_init().
    engine : str, optional
        Backend to use for xarray (default is "netcdf4").

    Returns
    -------
    ds : xarray.Dataset
        Dataset arranged as (init, lead, member, ...).

    Raises
    ------
    ValueError
        If no matching files are found.
    TypeError
        If `preproc` is neither "default" nor a callable.
    """
    _validate_path_arg("data_dir", data_dir)
    _validate_path_arg("case_prefix", case_prefix)
    _validate_path_arg("realm", realm)
    _validate_path_arg("grid", grid)
    _validate_path_arg("freq", freq)
    _validate_path_arg("ts_split", ts_split)
    _validate_path_arg("field", field)

    open_chunks, post_chunks = _split_open_and_post_chunks(chunks)

    if preproc == "default":
        preproc_func = preprocessor_monthly
    elif callable(preproc):
        preproc_func = preproc
    else:
        raise TypeError(
            "preproc must be 'default' or a callable with signature "
            "preproc(ds0, nlead, field)"
        )

    file_list, valid_inits = nested_file_list_by_init(
        data_dir=data_dir,
        case_prefix=case_prefix,
        members=members,
        init_tags=init_tags,
        field=field,
        realm=realm,
        grid=grid,
        freq=freq,
        ts_split=ts_split,
        require_all_members=require_all_members,
        verify_field_name=verify_field_name,
        verify_coverage=verify_coverage,
        nlead=nlead,
    )

    if not file_list:
        raise ValueError(
            f"No files found for field={field}, case_prefix={case_prefix}, "
            f"realm={realm}, grid={grid}, freq={freq}, ts_split={ts_split}"
        )

    open_kwargs = dict(
        combine="nested",
        concat_dim=["Y", "M"],
        data_vars=[field],
        coords="minimal",
        compat="override",
        engine=engine,
        preprocess=partial(preproc_func, nlead=nlead, field=field),
        chunks=open_chunks,
    )
    try:
        ds = xr.open_mfdataset(file_list, parallel=False, **open_kwargs)
    except Exception as e:
        # Parallel HDF5 reads can collide on some file systems; retry serially.
        if any(tok in str(e) for tok in ("HDF error", "Unable to open file", "Unknown file format", "errno = -")):
            warnings.warn(
                f"open_mfdataset with parallel=True failed ({e}). "
                "Retrying with parallel=False."
            )
            ds = xr.open_mfdataset(file_list, parallel=False, **open_kwargs)
        else:
            raise

    ds = ds.assign_coords(Y=("Y", valid_inits))
    n_members_loaded = ds.sizes["M"]
    if n_members_loaded != len(members):
        warnings.warn(
            f"Loaded {n_members_loaded} member(s) but {len(members)} were requested. "
            "This can happen when require_all_members=False and some members are missing."
        )
    ds = ds.assign_coords(M=("M", members[:n_members_loaded]))
    ds = ds.transpose("Y", "L", "M", ...)
    # Y and L are authoritative.  Reconstruct time after multi-file concat so
    # a lazily combined source time variable cannot become detached from its
    # initialization year (the failure previously seen in SMYLE benchmarks).
    valid_time = _verification_time_from_init_tags(ds.Y.values, ds.L.values)
    ds["time"] = valid_time.assign_coords(Y=ds.Y, L=ds.L)
    ds.attrs["verification_time_construction"] = "init_tag_plus_lead_v1"
    if post_chunks:
        ds = ds.chunk(post_chunks)

    return ds

def e3sm_region_mask(da, lonlat, lat_name="lat", lon_name="lon"):
    """
    Boolean regional mask for rectilinear lat/lon grid.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input object containing lat/lon coordinates.
    lonlat : sequence
        [lon_w, lon_e, lat_s, lat_n]
    lat_name, lon_name : str
        Coordinate names.

    Returns
    -------
    xr.DataArray
        Boolean mask on (lat, lon), True inside the region.
    """
    if len(lonlat) != 4:
        raise ValueError(f"lonlat must have exactly 4 elements [lon_w, lon_e, lat_s, lat_n], got {len(lonlat)}.")
    lon_w, lon_e, lat_s, lat_n = lonlat
    if lat_s > lat_n:
        raise ValueError(f"lat_s ({lat_s}) must be <= lat_n ({lat_n}).")

    if lat_name not in da.coords or lon_name not in da.coords:
        raise ValueError(f"Coordinates {lat_name} and/or {lon_name} not found in input data.")

    lat = da[lat_name]
    lon = da[lon_name]

    # Normalize longitudes in the data and the bounds to [0, 360)
    lon_360 = lon % 360
    lon_w_360 = lon_w % 360
    lon_e_360 = lon_e % 360

    # Build 2D lon/lat
    lon2d, lat2d = xr.broadcast(lon_360, lat)
    
    # Safely transpose if both dimensions exist in the broadcasted result
    if lat_name in lon2d.dims and lon_name in lon2d.dims:
        lon2d = lon2d.transpose(lat_name, lon_name)
        lat2d = lat2d.transpose(lat_name, lon_name)

    lat_mask = (lat2d >= lat_s) & (lat2d <= lat_n)
    
    if lon_w_360 <= lon_e_360:
        lon_mask = (lon2d >= lon_w_360) & (lon2d <= lon_e_360)
    else:
        # Handles regions crossing the prime meridian (0 degrees) when using [0, 360) representation
        lon_mask = (lon2d >= lon_w_360) | (lon2d <= lon_e_360)

    region = lat_mask & lon_mask
    return region


def e3sm_area_weights(da, lat_name="lat", lon_name="lon", area=None):
    """
    Return 2D area weights for a rectilinear lat/lon grid.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input object containing lat/lon coordinates.
    area : xr.DataArray, optional
        True grid-cell area weights on (lat, lon). If None, uses cos(lat).

    Returns
    -------
    xr.DataArray
        2D weights on (lat, lon)
    """
    if area is not None:
        return area

    if lat_name not in da.coords:
        raise ValueError(f"Coordinate '{lat_name}' not found in input data.")

    lat = da[lat_name]
    lon = da[lon_name] if lon_name in da.coords else None

    if lon is None and area is None:
        warnings.warn(
            f"Coordinate '{lon_name}' not found; returning 1D cos(lat) weights only."
        )

    wlat = np.cos(np.deg2rad(lat))
    
    if lon is not None:
        w2d, _ = xr.broadcast(wlat, lon)
        if lat_name in w2d.dims and lon_name in w2d.dims:
            w2d = w2d.transpose(lat_name, lon_name)
    else:
        w2d = wlat
        
    return w2d


def e3sm_regional_weights(
    da,
    lonlat,
    lat_name="lat",
    lon_name="lon",
    area=None,
    mask=None,
):
    """
    Build regional weights like the POP example:
    weights inside region, 0 outside.
    """
    region = e3sm_region_mask(da, lonlat, lat_name=lat_name, lon_name=lon_name)
    weights = e3sm_area_weights(da, lat_name=lat_name, lon_name=lon_name, area=area)

    if mask is not None:
        # Ensure mask is broadcasted/aligned correctly
        region = region & mask

    # Compute masked weights, filling NaNs with 0
    return weights.where(region, 0).fillna(0)


def e3sm_regional_mean(
    da,
    lonlat,
    lat_name="lat",
    lon_name="lon",
    area=None,
    mask=None,
):
    """
    Area-weighted regional mean for E3SM lat/lon data.
    """
    reg_weights = e3sm_regional_weights(
        da,
        lonlat,
        lat_name=lat_name,
        lon_name=lon_name,
        area=area,
        mask=mask,
    )
    
    # Calculate weighted mean over spatial dimensions
    # For datasets with lat/lon named dimensions
    spatial_dims = [dim for dim in [lat_name, lon_name] if dim in da.dims]
    
    if not spatial_dims:
        raise ValueError(f"DataArray does not contain spatial dimensions: {lat_name}, {lon_name}")
        
    return da.weighted(reg_weights).mean(dim=spatial_dims, skipna=True)
