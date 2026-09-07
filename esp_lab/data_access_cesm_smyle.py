"""
Utilities for accessing post-processed CESM-SMYLE hindcast data.

This module provides data-access functions for the CESM-SMYLE hindcast
archive stored under a per-initialization-date directory tree. The
interface mirrors ``data_access_e3sm`` so that the two datasets can be
loaded with near-identical calling code.

Expected directory structure
----------------------------
<data_dir>/
  b.e21.BSMYLE.f09_g17_<YYYYMM>0100/
    EN01/
      post/atm/f09_g17/ts/monthly/
        b.e21.BSMYLE.f09_g17.<YYYY>-<MM>.<EEE>.cam.h0.<FIELD>.<start_YYYYMM>-<end_YYYYMM>.nc
    EN02/
      post/atm/f09_g17/ts/monthly/
        ...
    ...

Example top-level case directory
---------------------------------
b.e21.BSMYLE.f09_g17_1980020100

Example file (member EN20, init 2018-08)
-----------------------------------------
b.e21.BSMYLE.f09_g17.2018-08.020.cam.h0.TREFHT.201808-202007.nc

Notes
-----
1. Initialization tag format
   Tags match the E3SM convention:  YYYYMMDDHH  (e.g. '1980020100').
   The last 4 characters of the case-directory suffix encode day and hour
   and are always '0100' for this archive.

2. Member naming
   Ensemble directories are named EN01 … EN40.  The embedded three-digit
   member index in the filename is the decimal integer with zero padding
   (EN01 → '001', EN40 → '040').

3. Grid
   The f09_g17 grid is the standard FV 0.9° × 1.25° lat–lon grid used by
   CAM.  Files contain regular (lat, lon) coordinates and do not require a
   pre-processing regrid step from an unstructured grid.

4. Monthly timestamp convention
   CAM monthly means are typically written with end-of-period or
   beginning-of-following-period timestamps.  This module applies the same
   ``time_set_midmonth`` logic used by ``data_access_e3sm`` to normalize
   timestamps to the 15th of the represented month.

Typical use
-----------
from esp_lab import data_access_cesm_smyle as smyle_access

field = "TREFHT"
data_dir = "/global/cfs/cdirs/e3sm/S2S2D/CESM-SMYLE"
members = [f"EN{i:02d}" for i in range(1, 21)]   # EN01 … EN20
init_tags = smyle_access.build_init_tags(range(1980, 2019), [2, 5, 8, 11])

ds = smyle_access.get_monthly_data(
    data_dir=data_dir,
    members=members,
    init_tags=init_tags,
    field=field,
    nlead=24,
)
# ds has dimensions (Y, L, M, lat, lon)
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



# ---------------------------------------------------------------------------
# Internal helpers (shared with data_access_e3sm but kept local to avoid
# a cross-module import dependency)
# ---------------------------------------------------------------------------

# Filename pattern:
#   b.e21.BSMYLE.f09_g17.<YYYY>-<MM>.<EEE>.cam.h0.<FIELD>.<start>-<end>.nc
_FILENAME_RE = re.compile(
    r"^b\.e21\.BSMYLE\.f09_g17\."
    r"(?P<init_year>\d{4})-(?P<init_month>\d{2})\."
    r"(?P<member_idx>\d{3})\.cam\.h0\."
    r"(?P<field>[^.]+)\."
    r"(?P<start_yyyymm>\d{6})-(?P<end_yyyymm>\d{6})\.nc$"
)

# Case directory suffix pattern:
#   b.e21.BSMYLE.f09_g17_<YYYYMM>0100
_CASEDIR_RE = re.compile(
    r"^b\.e21\.BSMYLE\.f09_g17_(?P<init_tag>\d{10})$"
)


def _parse_smyle_filename(filename: str) -> Dict[str, str]:
    """
    Parse a CESM-SMYLE timeseries filename.

    Returns a dict with keys: init_year, init_month, member_idx,
    field, start_yyyymm, end_yyyymm.

    Raises ValueError for unrecognised filenames.
    """
    fname = Path(filename).name
    m = _FILENAME_RE.match(fname)
    if m is None:
        raise ValueError(f"Unrecognised CESM-SMYLE filename: {filename!r}")
    return m.groupdict()


def _member_dir_to_idx(member: str) -> str:
    """Convert 'EN20' -> '020'."""
    num = int(member[2:])
    return f"{num:03d}"


def _datetime_parts(value) -> Tuple[int, int, int]:
    """Return (year, month, day) from common decoded datetime-like values."""
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return int(value.year), int(value.month), int(value.day)
    if np.issubdtype(np.asarray(value).dtype, np.datetime64):
        ts = pd.Timestamp(value)
        return int(ts.year), int(ts.month), int(ts.day)
    raise TypeError(f"Unsupported time value type: {type(value)!r}")


def _is_probably_monthly(time: xr.DataArray) -> bool:
    """Return True when the time axis looks like consecutive monthly means."""
    try:
        values = time.values
        if len(values) < 2:
            return False
        idxs = []
        for v in values:
            y, m, _ = _datetime_parts(v)
            idxs.append(y * 12 + m)
        return bool(np.all(np.diff(idxs) == 1))
    except Exception:
        return False


def _midmonth_time(year: int, month: int, template=None):
    """Build a mid-month timestamp, preserving the decoded calendar."""
    year, month = int(year), int(month)
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
        month, year = 12, year - 1
    return year, month


def _time_bounds_name(ds: xr.Dataset, time_name: str) -> Optional[str]:
    bounds_name = ds[time_name].attrs.get("bounds")
    if bounds_name in ds:
        return bounds_name
    for candidate in ("time_bnds", "time_bounds", "bounds_time", "time_bound"):
        if candidate in ds:
            return candidate
    return None


def _midmonth_from_bounds(ds: xr.Dataset, time_name: str) -> Optional[List]:
    bounds_name = _time_bounds_name(ds, time_name)
    if bounds_name is None:
        return None
    tb = ds[bounds_name]
    if time_name not in tb.dims:
        return None
    bounds_dim = next((d for d in tb.dims if d != time_name), None)
    if bounds_dim is None or tb.sizes.get(bounds_dim, 0) < 1:
        return None
    new_times = []
    for lower in tb.isel({bounds_dim: 0}).values:
        y, mo, _ = _datetime_parts(lower)
        new_times.append(_midmonth_time(y, mo, template=lower))
    return new_times


def drop_feb29(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    """Drop leap-day records from decoded daily/submonthly time axes."""
    if time_name not in ds.coords:
        return ds
    try:
        time = ds[time_name]
        if not hasattr(time, "dt") or _is_probably_monthly(time):
            return ds
        keep = ~((time.dt.month == 2) & (time.dt.day == 29))
        return ds.sel({time_name: keep})
    except Exception:
        return ds


def time_set_midmonth(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    """
    Return a copy of ds with monthly time normalized to the 15th of the
    represented month.

    The represented month is detected from time bounds when available;
    otherwise a day-1 heuristic is used (day-1 timestamps are treated as
    the beginning of the following month and shifted back one month).
    """
    if time_name not in ds.coords and time_name not in ds.dims:
        raise ValueError(f"time coordinate '{time_name}' not found in dataset.")

    ds = ds.copy()
    ds = drop_feb29(ds, time_name)

    bounded = _midmonth_from_bounds(ds, time_name)
    if bounded is not None:
        ds[time_name] = bounded
        return ds

    newtime = []
    for value in ds[time_name].values:
        y, mo, day = _datetime_parts(value)
        if day == 1:
            y, mo = _previous_month(y, mo)
        newtime.append(_midmonth_time(y, mo, template=value))
    ds[time_name] = newtime
    return ds


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_init_tags(
    years: Iterable[int],
    month: Union[int, Iterable[int]],
    init_day: int = 1,
    init_hour: int = 0,
) -> List[str]:
    """
    Build CESM-SMYLE initialization tags in YYYYMMDDHH format.

    Parameters
    ----------
    years : iterable of int
        Initialization years (e.g. range(1980, 2019)).
    month : int or iterable of int
        Initialization month(s), e.g. 2 or [2, 5, 8, 11].
    init_day : int, optional
        Day component of the tag (always 1 for this archive).
    init_hour : int, optional
        Hour component of the tag (always 0 for this archive).

    Returns
    -------
    init_tags : list of str
        E.g. ['1980020100', '1980050100', ...].
    """
    months = [month] if isinstance(month, int) else list(month)
    for m in months:
        if not (1 <= m <= 12):
            raise ValueError(f"month must be in 1–12, got {m}.")
    tags = []
    for y in years:
        for m in months:
            tags.append(f"{y:04d}{m:02d}{init_day:02d}{init_hour:02d}")
    return tags


VERIFICATION_TIME_VERSION = "init_tag_plus_lead_v1"


def expected_verification_time(
    init_tags: Iterable[object],
    leads: Iterable[object],
) -> xr.DataArray:
    """Build deterministic mid-month verification dates from init tags and leads."""
    init_values = [str(value) for value in init_tags]
    lead_values = [int(value) for value in leads]
    values = []
    for init_tag in init_values:
        if len(init_tag) < 6 or not init_tag[:6].isdigit():
            raise ValueError(f"Invalid CESM-SMYLE initialization tag: {init_tag!r}")
        init_year = int(init_tag[:4])
        init_month = int(init_tag[4:6])
        row = []
        for lead in lead_values:
            if lead < 1:
                raise ValueError(f"CESM-SMYLE lead values must be >= 1, got {lead}")
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
            "construction": VERIFICATION_TIME_VERSION,
        },
    )


def verification_time_mismatch_count(
    ds: xr.Dataset,
    *,
    init_month: Optional[int] = None,
) -> int:
    """Count dates inconsistent with the dataset's initialization tags and leads."""
    if "Y" not in ds.coords or "L" not in ds.coords:
        raise ValueError("CESM-SMYLE dataset must have Y and L coordinates.")
    if init_month is not None:
        bad_init = [str(value) for value in ds.Y.values if int(str(value)[4:6]) != init_month]
        if bad_init:
            raise ValueError(
                f"Dataset contains initialization tags inconsistent with month {init_month:02d}: "
                f"{bad_init[:3]}"
            )
    expected = expected_verification_time(ds.Y.values, ds.L.values)
    if "time" not in ds or ds["time"].dims != ("Y", "L"):
        return expected.size
    actual = ds["time"]
    mismatch = (
        (actual.dt.year != expected.dt.year)
        | (actual.dt.month != expected.dt.month)
        | (actual.dt.day != expected.dt.day)
    )
    return int(mismatch.sum().compute())


def ensure_verification_time(
    ds: xr.Dataset,
    *,
    init_month: Optional[int] = None,
    warn_on_repair: bool = True,
) -> xr.Dataset:
    """Return *ds* with a valid deterministic ``time(Y, L)`` coordinate."""
    mismatch_count = verification_time_mismatch_count(ds, init_month=init_month)
    if mismatch_count and warn_on_repair:
        warnings.warn(
            f"Repairing {mismatch_count} inconsistent CESM-SMYLE verification "
            "timestamp(s) from initialization tags and lead values.",
            stacklevel=2,
        )
    expected = expected_verification_time(ds.Y.values, ds.L.values)
    expected = expected.assign_coords(Y=ds.Y, L=ds.L)
    out = ds.copy()
    out["time"] = expected
    out.attrs["verification_time_construction"] = VERIFICATION_TIME_VERSION
    out.attrs["verification_time_repaired_count"] = mismatch_count
    return out


def expected_yyyymm_range(init_tag: str, nlead: int) -> Tuple[str, str]:
    """
    Return expected (start_yyyymm, end_yyyymm) for a given init tag and nlead.

    Parameters
    ----------
    init_tag : str
        Initialization tag like '1980020100'.
    nlead : int
        Number of lead months.

    Returns
    -------
    start_yyyymm, end_yyyymm : str
        E.g. ('198002', '198201') for nlead=24, init_tag='1980020100'.
    """
    year, month = int(init_tag[:4]), int(init_tag[4:6])
    start = f"{year:04d}{month:02d}"
    end_idx = (year * 12 + month - 1) + (nlead - 1)
    end = f"{end_idx // 12:04d}{end_idx % 12 + 1:02d}"
    return start, end


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def file_dict(
    data_dir: str,
    member: str,
    field: str,
    verify_field_name: bool = True,
) -> Dict[str, str]:
    """
    Return a dict of filepaths keyed by init_tag for one ensemble member.

    Parameters
    ----------
    data_dir : str
        Root CESM-SMYLE directory (contains the per-init case subdirs).
    member : str
        Ensemble member, e.g. 'EN01'.
    field : str
        Variable name, e.g. 'TREFHT'.
    verify_field_name : bool, optional
        When True (default), reject files whose parsed field does not exactly
        match ``field`` (guards against partial-name globbing collisions).

    Returns
    -------
    filepaths : dict
        Keys are init_tag strings; values are absolute file paths.

    Raises
    ------
    ValueError
        If multiple matching files exist for the same init/member/field.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    member_idx = _member_dir_to_idx(member)
    filepaths: Dict[str, str] = {}

    for case_dir in sorted(data_path.glob("b.e21.BSMYLE.f09_g17_*")):
        m = _CASEDIR_RE.match(case_dir.name)
        if m is None:
            continue
        init_tag = m.group("init_tag")

        ts_dir = case_dir / member / "post" / "atm" / "f09_g17" / "ts" / "monthly"
        if not ts_dir.exists():
            continue

        # Glob by field and member index to narrow the candidate set
        pattern = f"b.e21.BSMYLE.f09_g17.*.{member_idx}.cam.h0.{field}.*.nc"
        matches = sorted(ts_dir.glob(pattern))

        valid: List[str] = []
        for path in matches:
            try:
                info = _parse_smyle_filename(str(path))
            except ValueError:
                continue
            if verify_field_name and info["field"] != field:
                continue
            valid.append(str(path))

        if len(valid) == 1:
            filepaths[init_tag] = valid[0]
        elif len(valid) > 1:
            raise ValueError(
                f"Multiple files found for field={field}, member={member}, "
                f"init={init_tag}: {valid}"
            )
        # len == 0 → silently skip (no file for this init/member)

    return filepaths


def nested_file_list_by_init(
    data_dir: str,
    members: List[str],
    init_tags: List[str],
    field: str,
    require_all_members: bool = True,
    verify_field_name: bool = True,
    verify_coverage: bool = False,
    nlead: Optional[int] = None,
) -> Tuple[List[List[str]], List[str]]:
    """
    Retrieve a nested list of files organised as [init][member].

    Parameters
    ----------
    data_dir : str
        Root CESM-SMYLE directory.
    members : list of str
        Ensemble members, e.g. ['EN01', 'EN02', ...].
    init_tags : list of str
        Requested initialization tags.
    field : str
        Variable name.
    require_all_members : bool, optional
        When True (default), only keep init tags where every requested member
        has a file.  When False, keep partial inits (downstream code must
        handle ragged member counts).
    verify_field_name : bool, optional
        Passed through to ``file_dict``.
    verify_coverage : bool, optional
        When True, reject files whose encoded date range does not match the
        expected (init_tag, nlead) coverage.
    nlead : int, optional
        Required when verify_coverage=True.

    Returns
    -------
    nested_files : list of list of str
        Outer list = valid inits, inner list = per-member files.
    valid_inits : list of str
        Init tags retained in the output.
    """
    if not members:
        raise ValueError("members must not be empty.")
    if not init_tags:
        raise ValueError("init_tags must not be empty.")
    if verify_coverage and nlead is None:
        raise ValueError("nlead is required when verify_coverage=True.")

    # Pre-build per-member file dicts to avoid re-scanning the directory
    member_dicts = {
        mem: file_dict(
            data_dir=data_dir,
            member=mem,
            field=field,
            verify_field_name=verify_field_name,
        )
        for mem in members
    }

    nested_files: List[List[str]] = []
    valid_inits: List[str] = []

    for init_tag in init_tags:
        files_this_init: List[str] = []

        for mem in members:
            if init_tag not in member_dicts[mem]:
                continue
            path = member_dicts[mem][init_tag]

            if verify_coverage:
                try:
                    info = _parse_smyle_filename(path)
                except ValueError:
                    continue
                exp_start, exp_end = expected_yyyymm_range(init_tag, nlead)
                if info["start_yyyymm"] != exp_start or info["end_yyyymm"] != exp_end:
                    continue

            files_this_init.append(path)

        n_found = len(files_this_init)
        n_req = len(members)

        if require_all_members:
            if n_found == n_req:
                nested_files.append(files_this_init)
                valid_inits.append(init_tag)
            elif n_found > 0:
                warnings.warn(
                    f"init_tag={init_tag}: found {n_found}/{n_req} members; "
                    "skipping (require_all_members=True)."
                )
            else:
                warnings.warn(f"init_tag={init_tag}: no files found; skipping.")
        else:
            if n_found > 0:
                nested_files.append(files_this_init)
                valid_inits.append(init_tag)
            else:
                warnings.warn(f"init_tag={init_tag}: no files found; skipping.")

    return nested_files, valid_inits


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def preprocessor_monthly(ds0: xr.Dataset, nlead: int, field: str) -> xr.Dataset:
    """
    Standard CESM-SMYLE monthly preprocessor applied per file.

    Steps
    -----
    1. Normalize time to the 15th of the represented month.
    2. Select the first ``nlead`` months.
    3. Attach a 1-based lead coordinate ``L``.
    4. Return a dataset with ``L`` as the primary time-like dimension and
       the original ``time`` array retained as a coordinate.

    Parameters
    ----------
    ds0 : xarray.Dataset
        Single-file dataset as opened by xarray.
    nlead : int
        Number of lead months to retain.
    field : str
        Variable to keep.

    Returns
    -------
    d0 : xarray.Dataset
        Preprocessed dataset with dimensions (L, lat, lon).
    """
    if field not in ds0:
        raise ValueError(
            f"Field '{field}' not found. Available: {list(ds0.data_vars)}"
        )

    ds0 = time_set_midmonth(ds0, "time")

    available = ds0.sizes.get("time", 0)
    if available < nlead:
        warnings.warn(
            f"File has only {available} time step(s) but nlead={nlead} "
            "requested; returning all available steps."
        )

    d0 = ds0[field].isel(time=slice(0, nlead))

    if "lon" in ds0.coords and "lat" in ds0.coords:
        d0 = d0.assign_coords(lon=ds0.lon, lat=ds0.lat)

    d0 = d0.assign_coords(L=("time", np.arange(d0.sizes["time"]) + 1))
    d0 = d0.swap_dims({"time": "L"})
    d0 = d0.to_dataset(name=field)
    d0 = d0.reset_coords(["time"])
    d0["time"] = d0.time.expand_dims("Y")
    d0 = d0.chunk({"L": min(12, d0.sizes["L"])})
    return d0


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_monthly_data(
    data_dir: str,
    members: List[str],
    init_tags: List[str],
    field: str,
    nlead: int,
    preproc: Union[str, Callable] = "default",
    chunks: Optional[Dict[str, int]] = None,
    require_all_members: bool = True,
    verify_field_name: bool = True,
    verify_coverage: bool = False,
    engine: str = "netcdf4",
    open_parallel: bool = True,
) -> xr.Dataset:
    """
    Return a dask-backed xarray dataset arranged as (Y, L, M, lat, lon).

    Parameters
    ----------
    data_dir : str
        Root CESM-SMYLE directory.
    members : list of str
        Ensemble members, e.g. ['EN01', ..., 'EN40'].
    init_tags : list of str
        Initialization tags like ['1980020100', '1980050100', ...].
        Use :func:`build_init_tags` to generate these.
    field : str
        CAM variable name, e.g. 'TREFHT', 'TS', 'PRECT', 'PSL'.
    nlead : int
        Number of lead months to load.
    preproc : {"default"} or callable, optional
        Preprocessing function.  "default" uses the built-in
        :func:`preprocessor_monthly`.  A custom callable must accept
        ``(ds0, nlead, field)`` keyword arguments.
    chunks : dict, optional
        Chunk settings for lazy loading. Chunks for concat dimensions
        ``Y`` and ``M`` are applied after ``open_mfdataset`` so xarray can
        first use the native on-disk chunking for each source file.
    require_all_members : bool, optional
        See :func:`nested_file_list_by_init`.
    verify_field_name : bool, optional
        See :func:`file_dict`.
    verify_coverage : bool, optional
        See :func:`nested_file_list_by_init`.
    engine : str, optional
        xarray backend engine (default 'netcdf4').
    open_parallel : bool, optional
        Passed to :func:`xarray.open_mfdataset`.  Set to ``False`` on
        filesystems where parallel netCDF metadata opens are noisy or unstable.

    Returns
    -------
    ds : xarray.Dataset
        Dimensions: (Y, L, M, lat, lon).

        - Y : init_tag strings (e.g. '1980020100').
        - L : 1-based integer lead month.
        - M : member name string (e.g. 'EN01').
        - lat, lon : spatial coordinates from the f09_g17 grid.

    Raises
    ------
    ValueError
        If no matching files are found.
    TypeError
        If ``preproc`` is neither "default" nor callable.

    Examples
    --------
    >>> import numpy as np
    >>> from esp_lab import data_access_cesm_smyle as smyle
    >>> data_dir = "/global/cfs/cdirs/e3sm/S2S2D/CESM-SMYLE"
    >>> members = [f"EN{i:02d}" for i in range(1, 11)]   # first 10 members
    >>> init_tags = smyle.build_init_tags(range(1980, 2019), [2, 5, 8, 11])
    >>> ds = smyle.get_monthly_data(
    ...     data_dir=data_dir,
    ...     members=members,
    ...     init_tags=init_tags,
    ...     field="TREFHT",
    ...     nlead=24,
    ... )
    """
    open_chunks, post_chunks = _split_open_and_post_chunks(chunks)

    if preproc == "default":
        preproc_func = preprocessor_monthly
    elif callable(preproc):
        preproc_func = preproc
    else:
        raise TypeError(
            "preproc must be 'default' or a callable with signature "
            "preproc(ds0, nlead, field)."
        )

    file_list, valid_inits = nested_file_list_by_init(
        data_dir=data_dir,
        members=members,
        init_tags=init_tags,
        field=field,
        require_all_members=require_all_members,
        verify_field_name=verify_field_name,
        verify_coverage=verify_coverage,
        nlead=nlead,
    )

    if not file_list:
        raise ValueError(
            f"No files found for field={field!r} in {data_dir!r}."
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

    if open_parallel:
        try:
            ds = xr.open_mfdataset(file_list, parallel=True, **open_kwargs)
        except Exception as e:
            if any(tok in str(e) for tok in (
                "HDF error",
                "Unable to open file",
                "Unknown file format",
                "NetCDF: Not a valid ID",
                "errno = -",
            )):
                warnings.warn(
                    f"open_mfdataset parallel=True failed ({e}). "
                    "Retrying with parallel=False."
                )
                # NetCDF4 metadata opens on CFS can fail intermittently when many
                # files are opened in parallel; the serial path is slower but safer.
                ds = xr.open_mfdataset(file_list, parallel=False, **open_kwargs)
            else:
                raise
    else:
        ds = xr.open_mfdataset(file_list, parallel=False, **open_kwargs)

    ds = ds.assign_coords(Y=("Y", valid_inits))
    n_loaded = ds.sizes["M"]
    ds = ds.assign_coords(M=("M", members[:n_loaded]))
    ds = ds.transpose("Y", "L", "M", ...)
    # Do not trust a lazily concatenated source ``time`` variable here.  Older
    # benchmark files demonstrated that it can become detached from Y during
    # multi-file combination even while the field data remain correctly
    # ordered.  Y and L fully determine verification time for this archive.
    ds = ensure_verification_time(ds, warn_on_repair=False)
    if post_chunks:
        ds = ds.chunk(post_chunks)

    return ds


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_filename(
    field: str,
    init_month: int,
    nens: int = 20,
    nlead: int = 24,
    freq: str = "seas",
) -> str:
    """
    Return the standard benchmark filename for one (field, init_month).

    Parameters
    ----------
    field : str
        Variable name, e.g. 'TREFHT'.
    init_month : int
        Initialization month (1–12).
    nens : int, optional
        Number of ensemble members (default 20).
    nlead : int, optional
        Number of lead months (default 24).
    freq : str, optional
        Temporal frequency suffix, e.g. 'seas' or 'mon' (default 'seas').

    Returns
    -------
    str
        Filename like 'BSMYLE05_TREFHT_N20_M24_seas.nc'.
    """
    return f"BSMYLE{init_month:02d}_{field}_N{nens:02d}_M{nlead:02d}_{freq}.nc"


def benchmark_path(
    field: str,
    init_month: int,
    benchmark_dir: str | Path,
    nens: int = 20,
    nlead: int = 24,
    freq: str = "seas",
) -> Path:
    """Resolve the benchmark file that :func:`load_benchmark` will open."""
    fname = benchmark_filename(field, init_month, nens=nens, nlead=nlead, freq=freq)
    benchmark_root = Path(benchmark_dir)
    candidates = [
        benchmark_root / "leadtime_acc" / "inputs" / "atm" / field / fname,
        benchmark_root / "leadtime_acc" / "inputs" / field / fname,
        benchmark_root / fname,
    ]
    fpath = next((path for path in candidates if path.exists()), candidates[0])
    if not fpath.exists():
        raise FileNotFoundError(
            "Benchmark file not found. Checked:\n"
            + "\n".join(f"  {path}" for path in candidates)
            + "\n"
            "Run scripts/run_process_cesm_smyle_benchmark.py to generate it."
        )
    return fpath


def load_benchmark(
    field: str,
    init_month: int,
    benchmark_dir: str | Path,
    nens: int = 20,
    nlead: int = 24,
    freq: str = "seas",
    chunks: Optional[Dict[str, int]] = None,
) -> xr.Dataset:
    """
    Load a pre-processed CESM-SMYLE benchmark file.

    Benchmark files are created by
    ``scripts/run_process_cesm_smyle_benchmark.py``.  They contain seasonal
    (or monthly) means with dimensions (Y, L, M, lat, lon) on the native
    f09_g17 grid.

    Parameters
    ----------
    field : str
        Variable name, e.g. 'TREFHT', 'TS', 'PRECT', 'PSL'.
    init_month : int
        Initialization month (2, 5, 8, or 11 for this archive).
    benchmark_dir : path-like
        CESM-SMYLE diagnostic directory or direct directory containing benchmark
        files.
    nens : int, optional
        Number of ensemble members encoded in the filename (default 20).
    nlead : int, optional
        Number of lead months encoded in the filename (default 24).
    freq : str, optional
        Temporal frequency suffix to look for (default 'seas').
    chunks : dict, optional
        Chunk specification for lazy loading.  Defaults to a sensible
        spatial chunking when not provided.

    Returns
    -------
    ds : xr.Dataset
        Lazy-loaded dataset with dimensions (Y, L, M, lat, lon).

    Raises
    ------
    FileNotFoundError
        If the benchmark file does not exist.

    Examples
    --------
    >>> from esp_lab import data_access_cesm_smyle as smyle
    >>> ds = smyle.load_benchmark(
    ...     "TREFHT", init_month=5, benchmark_dir="/path/to/CESM-SMYLE"
    ... )
    >>> print(ds)
    """
    fpath = benchmark_path(
        field, init_month, benchmark_dir, nens=nens, nlead=nlead, freq=freq
    )
    if chunks is None:
        # f09_g17 grid is 192 lat × 288 lon; chunk at half-grid
        chunks = {"Y": 3, "L": -1, "M": 2, "lat": 96, "lon": 144}
    ds = xr.open_dataset(str(fpath), chunks=chunks)
    return ensure_verification_time(ds, init_month=init_month)
