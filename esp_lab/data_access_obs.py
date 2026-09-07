from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import warnings

import cftime
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_OBS_DIR = "/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series/"


# Default mapping from common E3SM model variable names to CMIP6 CMOR names
# used by the observational archive.
#
# Notes
# -----
# - Observational files under obs_for_e3sm_diags/time-series generally follow
#   CMIP6 CMOR naming conventions, e.g. tas, pr, psl, ua, va.
# - Some flux variables may require definition/sign checks depending on the
#   exact diagnostic use case.
DEFAULT_E3SM_TO_CMOR = {
    "TREFHT": "tas",
    "TS": "ts",
    "PRECT": "pr",
    "PSL": "psl",
    "PS": "ps",
    "TMQ": "prw",
    "U": "ua",
    "V": "va",
    "UBOT": "uas",
    "VBOT": "vas",
    "Q": "hus",
    "T": "ta",
    "OMEGA": "wap",
    "Z3": "zg",
    "LHFLX": "hfls",
    "SHFLX": "hfss",
    "FLDS": "rlds",
    "FLNS": "rlus",
    "FLUT": "rlut",
    "FLUTC": "rlutcs",
    "FSDS": "rsds",
    "FSNS": "rsus",
    "FSNTOA": "rsut",
    "SOLIN": "rsdt",
    "PRECL": "prl",
    "PRECC": "prc",
    "PRECSL": "prsn",
}


def _validate_path_arg(name: str, value: str) -> None:
    """
    Validate that a path-related argument is a string.

    This mainly guards against accidental tuple creation from notebook code like:
        product = "HadISST2",
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}: {value!r}")


def resolve_obs_dir(obs_dir: Optional[str] = None) -> Path:
    """
    Resolve the observation directory to a valid Path object.
    """
    if obs_dir is None:
        obs_dir = DEFAULT_OBS_DIR

    _validate_path_arg("obs_dir", obs_dir)
    obs_path = Path(obs_dir).expanduser().resolve()

    if not obs_path.exists():
        raise FileNotFoundError(f"Observation directory does not exist: {obs_path}")

    return obs_path


def list_products(obs_dir: Optional[str] = None) -> List[str]:
    """
    List available observational product subdirectories.

    Parameters
    ----------
    obs_dir : str, optional
        Base observation directory. If None, use DEFAULT_OBS_DIR.

    Returns
    -------
    products : list of str
        Sorted list of available product names.
    """
    obs_path = resolve_obs_dir(obs_dir)
    return sorted([p.name for p in obs_path.iterdir() if p.is_dir()])


def resolve_obs_field(
    field: Optional[str],
    use_cmor_map: bool = True,
    field_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Resolve an input field name to the CMOR variable name used by the
    observational archive.

    Parameters
    ----------
    field : str or None
        Input field name. This may be either:
        - a native E3SM model variable name, e.g. 'TREFHT'
        - a CMIP6 CMOR variable name, e.g. 'tas'
    use_cmor_map : bool, optional
        If True, map E3SM variable names to CMOR names when possible.
    field_map : dict or None, optional
        Custom mapping dictionary. If None, use DEFAULT_E3SM_TO_CMOR.

    Returns
    -------
    field_out : str or None
        Resolved CMOR-compatible field name, or None if field is None.
    """
    if field is None:
        return None

    if not isinstance(field, str):
        raise TypeError(f"field must be a string or None, got {type(field).__name__}: {field!r}")

    if not use_cmor_map:
        return field

    mapping = DEFAULT_E3SM_TO_CMOR if field_map is None else field_map
    return mapping.get(field, field)


def _unique_fields(*fields: Optional[str]) -> List[str]:
    """Return field names in order, dropping None values and duplicates."""
    out = []
    for field in fields:
        if field is None or field in out:
            continue
        out.append(field)
    return out


def _select_existing_field(
    ds: xr.Dataset,
    field_resolved: Optional[str],
    candidates: List[str],
) -> Optional[str]:
    """
    Select the variable name present in a dataset.

    Some observational archives mix CMOR-style aliases used by diagnostics
    with legacy E3SM-style names used on disk, e.g. PRECT instead of pr.
    """
    if field_resolved is None:
        return None

    for field in candidates:
        if field in ds.data_vars:
            return field

    return field_resolved


def _rename_field_if_needed(
    ds: xr.Dataset,
    field_in: Optional[str],
    field_out: Optional[str],
) -> xr.Dataset:
    """Rename the selected field to the requested resolved name when needed."""
    if (
        field_in is not None
        and field_out is not None
        and field_in != field_out
        and field_in in ds.data_vars
        and field_out not in ds.data_vars
    ):
        return ds.rename({field_in: field_out})

    return ds


def parse_obs_filename(filename: str) -> Dict[str, str]:
    """
    Parse an observational filename of the form:

        {field}_{start_yyyymm}_{end_yyyymm}.nc

    where `field` follows CMIP6 CMOR naming conventions.
    """
    fname = Path(filename).name

    if not fname.endswith(".nc"):
        raise ValueError(f"Not a netCDF filename: {filename}")

    stem = fname[:-3]
    parts = stem.rsplit("_", 2)

    if len(parts) != 3:
        raise ValueError(f"Unrecognized observational filename format: {filename}")

    field, start_yyyymm, end_yyyymm = parts

    if not (start_yyyymm.isdigit() and len(start_yyyymm) == 6):
        raise ValueError(f"Invalid start YYYYMM in filename: {filename}")
    if not (end_yyyymm.isdigit() and len(end_yyyymm) == 6):
        raise ValueError(f"Invalid end YYYYMM in filename: {filename}")

    return {
        "field": field,
        "start_yyyymm": start_yyyymm,
        "end_yyyymm": end_yyyymm,
    }


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize latitude/longitude coordinate names to lat/lon and
    enforce monotonic coordinate ordering.

    Notes
    -----
    - Renames latitude -> lat and longitude -> lon when needed.
    - Sorts lat and lon if present.
    - Converts longitude to [0, 360) if negative longitudes are detected.
    """
    rename_dict = {}

    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"

    if rename_dict:
        ds = ds.rename(rename_dict)

    if "lat" in ds.coords:
        ds = ds.sortby("lat")

    if "lon" in ds.coords:
        try:
            if (ds["lon"] < 0).any():
                ds = ds.assign_coords(lon=(ds["lon"] % 360))
        except Exception as e:
            warnings.warn(
                f"Could not normalize longitude to [0, 360): {e}. "
                "Longitude coordinate left unchanged."
            )
        ds = ds.sortby("lon")

    return ds


def _build_monthly_noleap_time(base_year: int, ntime: int):
    """
    Construct monthly cftime.DatetimeNoLeap timestamps at mid-month.

    Parameters
    ----------
    base_year : int
        Starting year corresponding to the first monthly record.
    ntime : int
        Number of time steps. Must be divisible by 12.

    Returns
    -------
    time_vals : list of cftime.DatetimeNoLeap
        Mid-month monthly noleap timestamps.
    """
    if not isinstance(base_year, int) or base_year < 1:
        raise ValueError(f"base_year must be a positive integer, got {base_year!r}.")
    if not isinstance(ntime, int) or ntime < 1:
        raise ValueError(f"ntime must be a positive integer, got {ntime!r}.")
    if ntime % 12 != 0:
        raise ValueError(
            f"time dimension size ({ntime}) is not divisible by 12; "
            "cannot safely build monthly noleap timestamps."
        )

    nyears = ntime // 12
    return [
        cftime.DatetimeNoLeap(base_year + y, m + 1, 15)
        for y in range(nyears)
        for m in range(12)
    ]


def _time_bounds_name(ds: xr.Dataset, time_name: str) -> Optional[str]:
    """Return the likely bounds variable for a time coordinate, if present."""
    bounds_name = ds[time_name].attrs.get("bounds")
    if bounds_name in ds:
        return bounds_name

    for candidate in ("time_bnds", "time_bounds", "bounds_time", "time_bound"):
        if candidate in ds:
            return candidate

    return None


def _datetime_parts(value):
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

    If the dataset has already been converted to noleap upstream, template
    will be a noleap cftime object and the returned timestamp follows that.
    Gregorian/standard observations decoded as numpy/pandas datetimes remain
    datetime64-compatible.
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


def _midmonth_from_bounds(ds: xr.Dataset, time_name: str):
    """
    Build represented-month timestamps from time bounds.

    For monthly products with bounds [month start, next month start], the
    represented month is unambiguous from the lower bound. This avoids guessing
    whether a day-1 coordinate is a month-start timestamp or a following-month
    timestamp.
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

    lower_bounds = tb.isel({bounds_dim: 0}).values
    new_times = []
    for lower in lower_bounds:
        year, month, _ = _datetime_parts(lower)
        new_times.append(_midmonth_time(year, month, template=lower))

    return new_times


def transform_to_mid_month(ds: xr.Dataset, time_name: str = "time") -> xr.Dataset:
    """
    Safely normalize monthly time values to the 15th of the represented month.
    
    Notes
    -----
    The previous implementation treated every day-1 timestamp as the first day
    of the following month and shifted it backward. That is unsafe for many
    observational products, where monthly data are stored at true month start.

    This function is intentionally conservative:
    - if time bounds are available, use their lower bound as the represented
      month;
    - if timestamps are already on day 15, leave the represented month as-is;
    - otherwise, set the day to 15 in the same timestamp month.
    """
    if time_name not in ds.coords:
        return ds

    try:
        bounded_times = _midmonth_from_bounds(ds, time_name)
        if bounded_times is not None:
            return ds.assign_coords({time_name: bounded_times})

        times = ds[time_name].values
        if len(times) == 0:
            return ds

        new_times = []
        for t in times:
            year, month, _ = _datetime_parts(t)
            new_times.append(_midmonth_time(year, month, template=t))

        ds = ds.assign_coords({time_name: new_times})
    except Exception as e:
        warnings.warn(
            f"transform_to_mid_month: could not shift time to mid-month: {e}. "
            "Time coordinate left unchanged."
        )

    return ds

def ensure_time_coordinate(
    ds: xr.Dataset,
    *,
    decode_times: bool = True,
    force_noleap: bool = False,
    base_year: Optional[int] = None,
    time_name: str = "time",
) -> xr.Dataset:
    """
    Robustly ensure dataset has a usable time coordinate.

    Strategy
    --------
    1. If time is already decoded, keep it.
    2. If time is numeric with CF metadata, try xr.decode_cf().
    3. If requested, convert decoded time to noleap calendar.
    4. If still unresolved and base_year is provided, rebuild as monthly
       DatetimeNoLeap timestamps.
    """
    if time_name not in ds.coords:
        return ds

    out = ds

    # Case 1: already decoded
    try:
        if hasattr(out[time_name], "dt"):
            out = transform_to_mid_month(out, time_name=time_name)
            out = drop_feb29(out, time_name=time_name)
            if force_noleap:
                try:
                    out = out.convert_calendar("noleap")
                except Exception as e:
                    warnings.warn(
                        f"ensure_time_coordinate: calendar conversion to noleap failed: {e}."
                    )
            return out
    except Exception as e:
        warnings.warn(
            f"ensure_time_coordinate: unexpected error inspecting time coordinate: {e}."
        )

    # Case 2: try CF decoding from units/calendar metadata
    if decode_times:
        try:
            out = xr.decode_cf(out)
            if hasattr(out[time_name], "dt"):
                out = transform_to_mid_month(out, time_name=time_name)
                out = drop_feb29(out, time_name=time_name)
                if force_noleap:
                    try:
                        out = out.convert_calendar("noleap")
                    except Exception as e:
                        warnings.warn(
                            f"ensure_time_coordinate: calendar conversion to noleap failed "
                            f"after CF decoding: {e}."
                        )
                return out
        except Exception as e:
            warnings.warn(
                f"ensure_time_coordinate: xr.decode_cf failed: {e}. "
                "Falling back to manual time reconstruction if base_year is provided."
            )

    # Case 3: fallback manual reconstruction
    if base_year is not None:
        ntime = out.sizes[time_name]
        out = out.copy()
        # _build_monthly_noleap_time already generates mid-month (day=15)
        out = out.assign_coords(
            {time_name: _build_monthly_noleap_time(base_year, ntime)}
        )
        return out

    raise ValueError(
        f"Could not decode time coordinate '{time_name}'. "
        "Provide a valid base_year for manual monthly reconstruction."
    )


def crop_time(
    ds: xr.Dataset,
    start_year: Optional[str] = None,
    end_year: Optional[str] = None,
) -> xr.Dataset:
    """
    Crop dataset by time using robust year-based selection.
    Uses string time slicing or boolean masking to robustly 
    handle both cftime and numpy datetime64.
    """
    if "time" not in ds.coords:
        return ds

    if start_year is None and end_year is None:
        return ds

    if start_year is not None and end_year is not None:
        if int(start_year) > int(end_year):
            raise ValueError(
                f"start_year ({start_year}) must be <= end_year ({end_year})."
            )

    # Method 1: Try native string-based slicing first (fastest and most standard)
    time_slice = slice(
        str(start_year) if start_year is not None else None,
        str(end_year) if end_year is not None else None
    )

    try:
        return ds.sel(time=time_slice)
    except (KeyError, TypeError, ValueError):
        # Method 2: Fall back to dt.year boolean masking if slicing fails
        time = ds["time"]
        mask = xr.ones_like(time, dtype=bool)

        if start_year is not None:
            mask = mask & (time.dt.year >= int(start_year))

        if end_year is not None:
            mask = mask & (time.dt.year <= int(end_year))

        return ds.where(mask, drop=True)


def preprocessor_monthly(
    ds: xr.Dataset,
    field: Optional[str] = None,
    start_year: Optional[str] = None,
    end_year: Optional[str] = None,
    harmonize_time: bool = True,
    calendar: str = "noleap",
    decode_times: bool = True,
    base_year: Optional[int] = None,
) -> xr.Dataset:
    """
    Standard preprocessing for observational monthly time-series data.

    This function:
    - standardizes lat/lon names
    - robustly ensures a usable time coordinate
    - optionally harmonizes to noleap calendar
    - crops to the requested time range
    - optionally retains only the requested field
    """
    ds = standardize_latlon(ds)

    ds = ensure_time_coordinate(
        ds,
        decode_times=decode_times,
        force_noleap=harmonize_time and calendar == "noleap",
        base_year=base_year,
        time_name="time",
    )

    ds = crop_time(ds, start_year=start_year, end_year=end_year)

    if field is not None:
        if field not in ds.data_vars:
            raise ValueError(
                f"Field {field!r} not found in dataset. "
                f"Available variables: {list(ds.data_vars)}"
            )
        ds = ds[[field]]

    return ds


def find_obs_file(
    obs_dir: Optional[str] = None,
    product: Optional[str] = None,
    field: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Find an observational file inside a product subdirectory.

    Parameters
    ----------
    obs_dir : str, optional
        Base observation directory. If None, use DEFAULT_OBS_DIR.
    product : str, optional
        Product subdirectory name.
    field : str, optional
        CMIP6 CMOR variable name used to identify the file, e.g.
        'tas', 'pr', 'psl', 'ua', 'va'.
    filename : str, optional
        Exact filename to load.

    Returns
    -------
    filepath : str
        Path to the matched file.
    """
    obs_root = resolve_obs_dir(obs_dir)

    if product is None:
        raise ValueError("product must be provided")

    _validate_path_arg("product", product)

    obs_path = obs_root / product
    if not obs_path.exists():
        raise FileNotFoundError(f"Product directory does not exist: {obs_path}")

    if filename is not None:
        _validate_path_arg("filename", filename)
        path = obs_path / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return str(path)

    matches = sorted(obs_path.glob("*.nc"))

    if field is not None:
        _validate_path_arg("field", field)
        field_lower = field.lower()

        parsed_matches = []
        for p in matches:
            try:
                info = parse_obs_filename(p.name)
            except ValueError:
                continue
            if info["field"].lower() == field_lower:
                parsed_matches.append(p)

        matches = parsed_matches

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No matching files found in {obs_path} "
            f"for field={field!r}, filename={filename!r}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple matching files found in {obs_path}: "
            f"{[m.name for m in matches]}. "
            "Please specify filename explicitly."
        )

    return str(matches[0])


def get_monthly_data(
    obs_dir: Optional[str] = None,
    product: Optional[str] = None,
    field: Optional[str] = None,
    filename: Optional[str] = None,
    chunks: Optional[Dict[str, int]] = None,
    preproc: Union[str, Callable] = "default",
    start_year: Optional[str] = None,
    end_year: Optional[str] = None,
    harmonize_time: bool = True,
    calendar: str = "noleap",
    decode_times: bool = True,
    base_year: Optional[int] = None,
    use_cmor_map: bool = True,
    field_map: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> xr.Dataset:
    """
    Load a preprocessed observational monthly time-series dataset.

    Parameters
    ----------
    obs_dir : str, optional
        Base observation directory. If None, use DEFAULT_OBS_DIR.
    product : str, optional
        Product subdirectory name.
    field : str, optional
        Requested variable name. This may be either:
        - an E3SM native variable name, e.g. 'TREFHT'
        - a CMIP6 CMOR variable name, e.g. 'tas'

        If `use_cmor_map=True`, known E3SM names are mapped internally to
        CMOR variable names for file lookup and variable selection.
    filename : str, optional
        Exact filename to load.
    chunks : dict, optional
        Chunk settings for xarray open_dataset.
    preproc : {"default"} or callable, optional
        - "default": use built-in monthly preprocessor
        - callable: custom preprocessing function with signature
          preproc(ds, field=None, start_year=None, end_year=None, **kwargs)
    start_year : str, optional
        Start year for cropping.
    end_year : str, optional
        End year for cropping.
    harmonize_time : bool, optional
        If True, attempt to harmonize to the requested calendar.
    calendar : str, optional
        Target calendar for time harmonization.
    decode_times : bool, optional
        Passed to xarray.open_dataset and also used in robust time handling.
    base_year : int, optional
        Fallback base year for manual monthly noleap time reconstruction when
        time cannot be decoded automatically. Use only for raw datasets that
        lack a usable decoded time coordinate.
    use_cmor_map : bool, optional
        If True, resolve native model field names to CMOR names.
    field_map : dict or None, optional
        Custom field mapping dictionary. If None, use DEFAULT_E3SM_TO_CMOR.
    verbose : bool, optional
        If True, print the filepath being loaded and any field mapping used.

    Returns
    -------
    ds : xr.Dataset
        Loaded and preprocessed observational dataset.
    """
    field_resolved = resolve_obs_field(
        field,
        use_cmor_map=use_cmor_map,
        field_map=field_map,
    )

    valid_calendars = {"noleap", "365_day", "standard", "gregorian", "proleptic_gregorian", "all_leap", "366_day", "julian", "360_day"}
    if calendar not in valid_calendars:
        raise ValueError(
            f"calendar={calendar!r} is not a recognized CF calendar. "
            f"Expected one of: {sorted(valid_calendars)}."
        )

    if verbose and field is not None and field_resolved != field:
        print(f"[OBS] Mapping field '{field}' -> '{field_resolved}'")

    field_candidates = _unique_fields(field_resolved, field)
    lookup_errors = []
    filepath = None
    filepath_field = None
    for candidate in field_candidates or [None]:
        try:
            filepath = find_obs_file(
                obs_dir=obs_dir,
                product=product,
                field=candidate,
                filename=filename,
            )
            filepath_field = candidate
            break
        except FileNotFoundError as exc:
            lookup_errors.append(str(exc))

    if filepath is None:
        if len(lookup_errors) == 1:
            raise FileNotFoundError(lookup_errors[0])
        raise FileNotFoundError(
            "No matching observational file found for any field alias "
            f"{field_candidates!r}, filename={filename!r}. "
            f"Errors: {lookup_errors}"
        )

    if chunks is None:
        chunks = {}

    if verbose:
        print(f"[OBS] Loading: {filepath}")
        if filepath_field is not None and filepath_field != field_resolved:
            print(
                f"[OBS] File matched field '{filepath_field}' "
                f"for requested field '{field_resolved}'"
            )

    ds = xr.open_dataset(filepath, chunks=chunks, decode_times=decode_times)
    preproc_field = _select_existing_field(ds, field_resolved, field_candidates)

    if preproc == "default":
        out = preprocessor_monthly(
            ds,
            field=preproc_field,
            start_year=start_year,
            end_year=end_year,
            harmonize_time=harmonize_time,
            calendar=calendar,
            decode_times=decode_times,
            base_year=base_year,
        )
        return _rename_field_if_needed(out, preproc_field, field_resolved)

    if callable(preproc):
        out = preproc(
            ds,
            field=preproc_field,
            start_year=start_year,
            end_year=end_year,
            harmonize_time=harmonize_time,
            calendar=calendar,
            decode_times=decode_times,
            base_year=base_year,
        )
        return _rename_field_if_needed(out, preproc_field, field_resolved)

    raise TypeError(
        "preproc must be 'default' or a callable with signature "
        "preproc(ds, field=None, start_year=None, end_year=None, **kwargs)"
    )


def merge_obs(primary: xr.DataArray, secondary: xr.DataArray) -> xr.DataArray:
    """
    Fill missing values in the primary observational field using values
    from the secondary field.
    """
    if not isinstance(primary, xr.DataArray) or not isinstance(secondary, xr.DataArray):
        raise TypeError("Inputs to merge_obs must be xarray.DataArray instances")

    if "time" in primary.dims and "time" in secondary.dims:
        if primary.sizes.get("time") != secondary.sizes.get("time"):
            raise ValueError(
                f"Cannot merge! Time sizes do not match. "
                f"primary ({primary.name}) time length: {primary.sizes.get('time')}, "
                f"secondary ({secondary.name}) time length: {secondary.sizes.get('time')}"
            )

    return primary.fillna(secondary)


def mon_to_seas_obs(
    ds: xr.Dataset,
    var: str,
    time_name: str = "time",
    field_map: Optional[Dict[str, str]] = None,
) -> xr.DataArray:
    """
    Apply CRU-style centered 3-month rolling mean and return DataArray
    with optional renaming back to E3SM convention.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    var : str
        Variable name in dataset (CMOR name, e.g. "tas").
    time_name : str, optional
        Time dimension name.
    field_map : dict, optional
        Mapping from E3SM -> CMOR (e.g., {"TREFHT": "tas"}).
        Used here to rename output back to E3SM naming.

    Returns
    -------
    xr.DataArray
    """
    if var not in ds:
        raise ValueError(f"{var!r} not found in dataset")
    if time_name not in ds.dims and time_name not in ds.coords:
        raise ValueError(f"time dimension '{time_name}' not found in dataset.")

    da = (
        ds[var]
        .rolling({time_name: 3}, min_periods=3, center=True)
        .mean()
        .dropna(time_name, how="all")
    )

    # Reverse mapping: CMOR -> E3SM
    if field_map is not None:
        reverse_map = {v: k for k, v in field_map.items()}
        if var in reverse_map:
            da = da.rename(reverse_map[var])

    return da


def obs_region_mask(
    da: xr.DataArray,
    lonlat,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """
    Boolean regional mask for a rectilinear observational lat/lon grid.

    Handles both [-180, 180] and [0, 360) longitude conventions automatically,
    and correctly masks regions that cross the dateline.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input object containing lat/lon coordinates.
    lonlat : sequence of length 4
        [lon_w, lon_e, lat_s, lat_n]. Longitudes may be in either
        [-180, 180] or [0, 360); they are normalized internally to [0, 360).
    lat_name : str, optional
        Latitude coordinate name, default 'lat'.
    lon_name : str, optional
        Longitude coordinate name, default 'lon'.

    Returns
    -------
    xr.DataArray
        Boolean mask, True inside the region.

    Raises
    ------
    ValueError
        If lonlat does not have 4 elements, lat_s > lat_n, or coordinates
        are not found.
    """
    lonlat = list(lonlat)
    if len(lonlat) != 4:
        raise ValueError(
            f"lonlat must have exactly 4 elements [lon_w, lon_e, lat_s, lat_n], "
            f"got {len(lonlat)}."
        )
    lon_w, lon_e, lat_s, lat_n = lonlat
    if lat_s > lat_n:
        raise ValueError(f"lat_s ({lat_s}) must be <= lat_n ({lat_n}).")

    if lat_name not in da.coords:
        raise ValueError(f"Coordinate '{lat_name}' not found in input data.")
    if lon_name not in da.coords:
        raise ValueError(f"Coordinate '{lon_name}' not found in input data.")

    lat = da[lat_name]
    lon = da[lon_name]

    # Normalize data longitudes and region bounds to [0, 360)
    lon_360 = lon % 360
    lon_w_360 = lon_w % 360
    lon_e_360 = lon_e % 360

    # Broadcast to 2D and enforce (lat, lon) ordering
    lon2d, lat2d = xr.broadcast(lon_360, lat)
    if lat_name in lon2d.dims and lon_name in lon2d.dims:
        lon2d = lon2d.transpose(lat_name, lon_name)
        lat2d = lat2d.transpose(lat_name, lon_name)

    lat_mask = (lat2d >= lat_s) & (lat2d <= lat_n)

    # A full-longitude region such as [0, 360] has coincident normalized
    # endpoints. Treat it as the full circle rather than a zero-width strip.
    if abs(lon_e - lon_w) >= 360:
        lon_mask = xr.ones_like(lon2d, dtype=bool)
    elif lon_w_360 <= lon_e_360:
        lon_mask = (lon2d >= lon_w_360) & (lon2d <= lon_e_360)
    else:
        # Region crosses the dateline in [0, 360) convention
        lon_mask = (lon2d >= lon_w_360) | (lon2d <= lon_e_360)

    return lat_mask & lon_mask


def obs_regional_weights(
    da: xr.DataArray,
    lonlat,
    lat_name: str = "lat",
    lon_name: str = "lon",
    area: xr.DataArray = None,
    mask: xr.DataArray = None,
) -> xr.DataArray:
    """
    Build 2D area weights for a regional mean over observational data.

    Weights are cos(lat) inside the region (or `area` if provided), and
    zero outside. An additional boolean `mask` (e.g. land/ocean) can be
    combined with the region.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input object containing lat/lon coordinates.
    lonlat : sequence of length 4
        [lon_w, lon_e, lat_s, lat_n].
    lat_name : str, optional
        Latitude coordinate name, default 'lat'.
    lon_name : str, optional
        Longitude coordinate name, default 'lon'.
    area : xr.DataArray, optional
        True grid-cell area weights. If None, cos(lat) is used.
    mask : xr.DataArray, optional
        Additional boolean mask (True = keep). Applied on top of the
        regional mask, e.g. to restrict to land or ocean only.

    Returns
    -------
    xr.DataArray
        2D weights on (lat, lon), zero outside the region.
    """
    region = obs_region_mask(da, lonlat, lat_name=lat_name, lon_name=lon_name)

    if mask is not None:
        region = region & mask.astype(bool)

    if area is not None:
        weights = area
    else:
        if lat_name not in da.coords:
            raise ValueError(f"Coordinate '{lat_name}' not found in input data.")
        wlat = xr.DataArray(
            np.cos(np.deg2rad(da[lat_name])), dims=[lat_name]
        )
        if lon_name in da.coords:
            weights, _ = xr.broadcast(wlat, da[lon_name])
            if lat_name in weights.dims and lon_name in weights.dims:
                weights = weights.transpose(lat_name, lon_name)
        else:
            weights = wlat

    return weights.where(region, 0).fillna(0)


def obs_regional_mean(
    da: xr.DataArray,
    lonlat,
    lat_name: str = "lat",
    lon_name: str = "lon",
    area: xr.DataArray = None,
    mask: xr.DataArray = None,
) -> xr.DataArray:
    """
    Area-weighted regional mean for observational lat/lon data.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray on a rectilinear lat/lon grid.
    lonlat : sequence of length 4
        [lon_w, lon_e, lat_s, lat_n].
    lat_name : str, optional
        Latitude coordinate name, default 'lat'.
    lon_name : str, optional
        Longitude coordinate name, default 'lon'.
    area : xr.DataArray, optional
        True grid-cell area weights. If None, cos(lat) is used.
    mask : xr.DataArray, optional
        Additional boolean mask (True = keep).

    Returns
    -------
    xr.DataArray
        Weighted regional mean with spatial dimensions reduced.

    Raises
    ------
    ValueError
        If neither lat nor lon dimension is found in `da`.
    """
    reg_weights = obs_regional_weights(
        da,
        lonlat,
        lat_name=lat_name,
        lon_name=lon_name,
        area=area,
        mask=mask,
    )

    spatial_dims = [dim for dim in [lat_name, lon_name] if dim in da.dims]
    if not spatial_dims:
        raise ValueError(
            f"DataArray does not contain spatial dimensions '{lat_name}' or '{lon_name}'."
        )

    return da.weighted(reg_weights).mean(dim=spatial_dims, skipna=True)
