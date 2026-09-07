"""I/O runner for start-specific, two-reference monthly drift diagnostics.

Logical diagnostic names (for example ``SST``) may map to different archive
variable names.  Physical fields are normalized to a declared output unit
before references or departures are constructed.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from esp_lab import data_access_e3sm
from esp_lab.land_skill import depth_integrated_soil_water_mm
from esp_lab.diagnostics.two_reference_drift import (
    compute_diagnostics,
    match_model_climatology_to_valid_time,
    match_observation_to_valid_time,
    validate_compatible_fields,
)
from esp_lab.utils.calendar_utils import time_set_midmonth
from esp_lab.utils import regrid_utils as regrid


EXPECTED_MONTHLY_LEADS = np.arange(1, 25)
PATH_SETTINGS = {
    "s2d_diag_root": Path("/global/cfs/cdirs/e3sm/S2S2D/s2d_diag"),
}
S2D_DIAG_ROOT = PATH_SETTINGS["s2d_diag_root"]
E3SMLE_ROOT = Path("/global/cfs/cdirs/e3sm/S2S2D/E3SMLE")
E3SMLE_ENSMEAN_ROOT = E3SMLE_ROOT / "ensmean" / "post"
E3SMLE_ENSSPREAD_ROOT = E3SMLE_ROOT / "ensspread" / "post"
E3SMLE_GRID = "180x360_aave"
SOIL_MOISTURE_DEPTH_RANGE_M = (0.0, 1.6)
DEFAULT_LEAD_WINDOWS = {
    "initial_month": [1],
    "early_adjustment": [1, 2, 3],
    "seasonal": [4, 5, 6],
    "extended_seasonal": [7, 8, 9],
    "late_year1": [10, 11, 12],
    "year2": list(range(13, 25)),
}
UNIT_TRANSFORMS = {
    ("atm", "TREFHT"): {
        "scale": 1.0,
        "offset": -273.15,
        "input_units": ("k", "kelvin"),
        "units": "degC",
    },
    ("ocn", "SST"): {
        "scale": 1.0,
        "offset": 0.0,
        "input_units": ("c", "degc", "degree_celsius", "degrees_celsius"),
        "units": "degC",
    },
    ("atm", "PSL"): {
        "scale": 1.0e-2,
        "offset": 0.0,
        "input_units": ("pa", "pascal", "pascals"),
        "units": "hPa",
    },
    ("atm", "PRECT"): {
        # Precipitation observations and E3SM fields have different native
        # units, so the conversion is selected from the input-units metadata.
        "input_conversions": {
            "m/s": (1000.0 * 86400.0, 0.0),
            "m s-1": (1000.0 * 86400.0, 0.0),
            "m s^-1": (1000.0 * 86400.0, 0.0),
            "mm/d": (1.0, 0.0),
            "mm day-1": (1.0, 0.0),
            "mm day^-1": (1.0, 0.0),
        },
        "units": "mm/day",
    },
}

PathLike = str | Path
HistoricalPaths = PathLike | Sequence[PathLike]


def initialization_year_values(values: xr.DataArray | Sequence[object]) -> np.ndarray:
    """Return four-digit initialization years from integer years or init tags."""
    raw = values.values if isinstance(values, xr.DataArray) else np.asarray(values)
    years = []
    for value in np.asarray(raw).reshape(-1):
        text = str(value)
        if len(text) < 4 or not text[:4].isdigit():
            raise ValueError(f"Cannot derive initialization year from Y={value!r}")
        years.append(int(text[:4]))
    return np.asarray(years, dtype=int)


def select_initialization_years(
    data: xr.Dataset | xr.DataArray,
    years: tuple[int, int],
    *,
    year_dim: str = "Y",
) -> xr.Dataset | xr.DataArray:
    """Select an inclusive year range for integer-Y or ``YYYYMMDDHH``-tagged Y."""
    if year_dim not in data.coords:
        raise ValueError(f"Input lacks initialization coordinate {year_dim!r}")
    start, end = map(int, years)
    if start > end:
        raise ValueError("Initialization-year range must be increasing")
    parsed = initialization_year_values(data[year_dim])
    keep = (parsed >= start) & (parsed <= end)
    if not keep.any():
        raise ValueError(f"No initialization years fall within {start}-{end}")
    return data.isel({year_dim: np.flatnonzero(keep)})


@dataclass(frozen=True)
class AttractorConfig:
    """Location and climatology settings for an E3SMLE reference."""

    root: Path = E3SMLE_ROOT
    ensmean_root: Path = E3SMLE_ENSMEAN_ROOT
    ensspread_root: Path = E3SMLE_ENSSPREAD_ROOT
    grid: str = E3SMLE_GRID
    reference_start_year: int = 1981
    reference_end_year: int = 2010
    frequency: str = "monthly"

    def __post_init__(self):
        if self.frequency not in {"monthly", "daily"}:
            raise ValueError("frequency must be 'monthly' or 'daily'")
        if self.reference_start_year > self.reference_end_year:
            raise ValueError("reference_start_year must not exceed reference_end_year")


def discover_e3smle_files(
    variable: str,
    component: str,
    *,
    root: PathLike = E3SMLE_ENSMEAN_ROOT,
    frequency: str = "monthly",
    grid: str = E3SMLE_GRID,
    chunk: str | None = None,
) -> list[Path]:
    """Discover E3SMLE ensemble-statistic files without assuming a year range."""
    base = Path(root) / component / "ts" / frequency / grid
    search_root = base / chunk if chunk else base
    if not search_root.is_dir():
        raise FileNotFoundError(f"E3SMLE archive directory does not exist: {search_root}")
    paths = sorted(search_root.rglob(f"{variable}_*.nc"))
    if not paths:
        raise FileNotFoundError(
            f"No E3SMLE {frequency} files for {component}/{variable} under {search_root}"
        )
    return paths


def load_e3smle_timeseries(
    variable: str,
    component: str,
    *,
    root: PathLike = E3SMLE_ENSMEAN_ROOT,
    frequency: str = "monthly",
    grid: str = E3SMLE_GRID,
    chunk: str | None = None,
    chunks: Mapping[str, int] | None = None,
) -> xr.DataArray:
    """Open the already-computed E3SMLE ensemble mean as one time series.

    This function never reads individual historical members.  Call ``.close()``
    on the returned DataArray after use to release its file handles.
    """
    paths = discover_e3smle_files(
        variable, component, root=root, frequency=frequency, grid=grid, chunk=chunk
    )
    dataset = _open_historical_dataset(paths, chunks=chunks)
    try:
        data = _select_variable(dataset, [variable], "E3SMLE attractor variable")
        if "time" not in data.dims:
            raise ValueError(f"E3SMLE {component}/{variable} lacks a time dimension")
        data.attrs.update(
            e3smle_statistic="historical ensemble mean",
            e3smle_archive_root=str(root),
            e3smle_file_count=len(paths),
            e3smle_frequency=frequency,
            e3smle_grid=grid,
        )
        return data
    except Exception:
        dataset.close()
        raise


def expand_requests(requests: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Expand request blocks into source/component/variable/month jobs."""
    jobs: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for request in requests:
        missing = {"sources", "component", "variables", "init_months"} - set(request)
        if missing:
            raise ValueError(f"Request is missing keys: {sorted(missing)}")
        for source in request["sources"]:
            for variable in request["variables"]:
                for init_month in request["init_months"]:
                    key = (str(source), str(request["component"]), str(variable), int(init_month))
                    if key in seen:
                        raise ValueError(f"Duplicate request: {key}")
                    if not 1 <= key[3] <= 12:
                        raise ValueError(f"Invalid initialization month: {key[3]}")
                    seen.add(key)
                    jobs.append(
                        {
                            "source": key[0],
                            "component": key[1],
                            "variable": key[2],
                            "init_month": key[3],
                            "regrid": bool(request.get("regrid", False)),
                            **(
                                {"hindcast_variable": str(request["hindcast_variable"])}
                                if request.get("hindcast_variable") else {}
                            ),
                        }
                    )
    return jobs


def discover_monthly_hindcast(
    job: Mapping[str, object], *, root: str | Path = S2D_DIAG_ROOT
) -> Path:
    """Find exactly one eligible ``Y,M,L=1..24`` monthly hindcast cache."""
    directory = (
        Path(root)
        / str(job["source"])
        / "leadtime_acc"
        / "inputs"
        / str(job["component"])
        / str(job["variable"])
    )
    token = f"{int(job['init_month']):02d}_{job['variable']}"
    candidates = sorted(directory.glob(f"*{token}*mon*.nc"))
    eligible = []
    for path in candidates:
        try:
            with xr.open_dataset(path, decode_times=False) as dataset:
                variable = str(job["variable"])
                if (
                    variable in dataset
                    and {"Y", "M", "L"} <= set(dataset[variable].dims)
                    and "time" in dataset
                    and tuple(dataset["time"].dims) == ("Y", "L")
                    and np.array_equal(dataset["L"].values, EXPECTED_MONTHLY_LEADS)
                ):
                    eligible.append(path)
        except Exception:
            continue
    if len(eligible) != 1:
        raise FileNotFoundError(
            f"Expected exactly one monthly Y/M/L=1..24 cache for {dict(job)} in {directory}; "
            f"found {len(eligible)}. Seasonal *_seas.nc caches are not valid substitutes."
        )
    return eligible[0]


def monthly_hindcast_output_path(
    job: Mapping[str, object],
    *,
    member_count: int,
    lead_count: int = 24,
    root: str | Path = S2D_DIAG_ROOT,
) -> Path:
    """Return the canonical monthly cache path consumed by both drift notebooks."""
    directory = (
        Path(root) / str(job["source"]) / "leadtime_acc" / "inputs"
        / str(job["component"]) / str(job["variable"])
    )
    filename = (
        f"{job['source']}_{int(job['init_month']):02d}_{job['variable']}"
        f"_N{member_count:02d}_M{lead_count:02d}_mon.nc"
    )
    return directory / filename


def prepare_monthly_hindcast_cache(
    job: Mapping[str, object],
    *,
    case_prefix: str,
    init_years: Sequence[int],
    members: Sequence[str],
    data_root: str | Path,
    output_root: str | Path = S2D_DIAG_ROOT,
    lead_count: int = 24,
    grid: str = E3SMLE_GRID,
    frequency: str = "monthly",
    chunk: str = "2yr",
    engine: str | None = "netcdf4",
    chunks: Mapping[str, int] | None = None,
    force: bool = False,
) -> Path:
    """Materialize the missing monthly Y/M/L cache from post-processed hindcasts.

    This is intentionally separate from seasonal aggregation.  It uses the
    project's established E3SM loader, which corrects monthly timestamps to
    the represented mid-month and verifies member/lead coverage.
    """
    destination = monthly_hindcast_output_path(
        job, member_count=len(members), lead_count=lead_count, root=output_root
    )
    if destination.exists() and not force:
        return destination
    init_tags = data_access_e3sm.build_init_tags(init_years, int(job["init_month"]))
    archive_variable = str(job.get("hindcast_variable", job["variable"]))
    kwargs = dict(
        data_dir=str(data_root), case_prefix=case_prefix, members=list(members),
        init_tags=init_tags, field=archive_variable, nlead=int(lead_count),
        chunks={} if chunks is None else dict(chunks), realm=str(job["component"]),
        grid=grid, freq=frequency, ts_split=chunk, require_all_members=True,
        verify_field_name=True, verify_coverage=True,
    )
    if engine is not None:
        kwargs["engine"] = engine
    dataset = data_access_e3sm.get_monthly_data(**kwargs)
    variable = str(job["variable"])
    try:
        if archive_variable not in dataset or "time" not in dataset:
            raise ValueError(f"Monthly hindcast must contain {archive_variable!r} and 'time'")
        archive_field = dataset[archive_variable].rename(variable)
        if not {"Y", "M", "L"} <= set(archive_field.dims):
            raise ValueError(
                f"Monthly hindcast {archive_variable!r} lacks Y/M/L: {archive_field.dims}"
            )
        if tuple(dataset["time"].dims) != ("Y", "L"):
            raise ValueError(f"Monthly hindcast time must have Y,L dimensions: {dataset.time.dims}")
        if not np.array_equal(dataset.L.values, np.arange(1, lead_count + 1)):
            raise ValueError(f"Monthly hindcast leads are not contiguous 1..{lead_count}")
        if list(map(str, dataset.Y.values)) != init_tags:
            raise ValueError(
                "Monthly hindcast initialization coverage is incomplete: "
                f"requested {init_tags[0]}..{init_tags[-1]}, got "
                f"{list(map(str, dataset.Y.values))[:3]}..{list(map(str, dataset.Y.values))[-3:]}"
            )
        if list(map(str, dataset.M.values)) != list(members):
            raise ValueError("Monthly hindcast member coordinates do not match the request")
        if (str(job["component"]), variable) == ("lnd", "H2OSOI"):
            archive_field = apply_unit_transform(archive_field, "lnd", "H2OSOI")
        product = xr.Dataset({variable: archive_field, "time": dataset["time"]})
        product.attrs.update(
            source=str(job["source"]), component=str(job["component"]),
            variable=variable, initialization_month=int(job["init_month"]),
            frequency="monthly", lead_convention="L=1 is initialization month",
            source_data_root=str(data_root), case_prefix=case_prefix,
            source_variable=archive_variable,
        )
        atomic_to_netcdf(product, destination)
    finally:
        dataset.close()
    return destination


def _select_variable(dataset: xr.Dataset, names: Iterable[str], label: str) -> xr.DataArray:
    for name in names:
        if name and name in dataset:
            return dataset[name]
    raise KeyError(f"Cannot find {label}; tried {list(names)}, available={list(dataset.data_vars)}")


def apply_unit_transform(data: xr.DataArray, component: str, variable: str) -> xr.DataArray:
    """Apply the same configurable scale/offset convention as the climatology notebook."""
    if (component, variable) == ("lnd", "H2OSOI"):
        current_units = str(data.attrs.get("units", "")).strip().lower().replace(" ", "")
        if "levgrnd" in data.dims:
            if current_units not in {"mm3/mm3", "m3/m3", "1", "fraction"}:
                raise ValueError(
                    f"Cannot safely integrate lnd/H2OSOI from units "
                    f"{data.attrs.get('units')!r}"
                )
            return depth_integrated_soil_water_mm(
                data, SOIL_MOISTURE_DEPTH_RANGE_M, vertical_dim="levgrnd"
            )
        if current_units != "mm":
            raise ValueError(
                "Depth-integrated lnd/H2OSOI must use millimetres; got "
                f"{data.attrs.get('units')!r}"
            )
        converted = data.copy(deep=False)
        converted.attrs.update(data.attrs)
        converted.attrs.update(
            units="mm", depth_top_m=SOIL_MOISTURE_DEPTH_RANGE_M[0],
            depth_bottom_m=SOIL_MOISTURE_DEPTH_RANGE_M[1],
            vertical_aggregation="provided depth-integrated water equivalent",
        )
        return converted
    transform = UNIT_TRANSFORMS.get((component, variable))
    if transform is None:
        if not data.attrs.get("units"):
            raise ValueError(f"{component}/{variable} has no units and no configured transform")
        return data
    current_units = str(data.attrs.get("units", "")).strip().lower().replace("°", "deg")
    target_units = str(transform["units"])
    if current_units == target_units.lower():
        return data
    conversions = transform.get("input_conversions")
    if conversions is not None:
        parameters = conversions.get(current_units)
    elif current_units in transform["input_units"]:
        parameters = (float(transform["scale"]), float(transform["offset"]))
    else:
        parameters = None
    if parameters is None:
        raise ValueError(
            f"Cannot safely transform {component}/{variable} from units "
            f"{data.attrs.get('units')!r} to {target_units!r}"
        )
    scale, offset = map(float, parameters)
    converted = data * scale + offset
    converted.attrs.update(data.attrs)
    converted.attrs.update(
        units=str(transform["units"]),
        unit_transform=f"output = input * {scale} + {offset}",
    )
    return converted


def apply_spread_unit_transform(
    data: xr.DataArray, component: str, variable: str
) -> xr.DataArray:
    """Normalize spread units without applying an additive state-field offset."""
    transform = UNIT_TRANSFORMS.get((component, variable))
    if transform is None:
        if not data.attrs.get("units"):
            raise ValueError(f"{component}/{variable} spread has no units")
        return data
    current = str(data.attrs.get("units", "")).strip().lower().replace("°", "deg")
    target = str(transform["units"]).lower()
    conversions = transform.get("input_conversions")
    if current == target:
        scale = 1.0
    elif conversions is not None and current in conversions:
        scale = float(conversions[current][0])
    elif current in set(transform.get("input_units", ())) | {"k", "kelvin"}:
        scale = float(transform["scale"])
    else:
        raise ValueError(f"Cannot normalize {component}/{variable} spread units {current!r}")
    converted = data * scale
    converted.attrs.update(data.attrs)
    converted.attrs.update(
        units=str(transform["units"]),
        unit_transform="multiplicative spread conversion; no additive offset",
    )
    return converted


def _grid_signature(data: xr.DataArray) -> str:
    if "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError("Regridding requires lat and lon coordinates")
    digest = hashlib.sha1()
    digest.update(np.asarray(data.lat.values).tobytes())
    digest.update(np.asarray(data.lon.values).tobytes())
    return digest.hexdigest()


def _regrid_field(
    data: xr.DataArray,
    target_grid: xr.Dataset,
    regridders: dict[str, object],
    *,
    method: str = "conservative",
    periodic: bool = True,
) -> xr.DataArray:
    if (
        np.array_equal(data["lat"].values, target_grid["lat"].values)
        and np.array_equal(data["lon"].values, target_grid["lon"].values)
    ):
        result = data.copy(deep=False)
        result.attrs.update(data.attrs)
        result.attrs["regrid_method"] = "identity; source and target coordinates match"
        return result
    signature = _grid_signature(data)
    if signature not in regridders:
        regridders[signature] = regrid.make_regridder(
            data.to_dataset(name=data.name or "field"),
            target_grid,
            method=method,
            periodic=periodic,
        )
    result = regridders[signature](data)
    # xESMF supplies its own regridding metadata but does not consistently
    # propagate physical metadata from the source DataArray.
    regrid_attrs = dict(result.attrs)
    result.attrs.update(data.attrs)
    result.attrs.update(regrid_attrs)
    return result


def build_monthly_attractor_climatology(
    historical: xr.DataArray, *, start_year: int, end_year: int
) -> xr.DataArray:
    """Build a 12-month E3SM free-running climatology over an explicit period."""
    if "time" not in historical.dims:
        raise ValueError("Historical attractor input must contain time")
    selected = historical.where(
        (historical.time.dt.year >= start_year) & (historical.time.dt.year <= end_year),
        drop=True,
    )
    if selected.sizes.get("time", 0) == 0:
        raise ValueError(f"Historical input has no dates in {start_year}-{end_year}")
    actual_year_months = [
        (int(year), int(month))
        for year, month in zip(
            np.asarray(selected.time.dt.year.values).reshape(-1),
            np.asarray(selected.time.dt.month.values).reshape(-1),
        )
    ]
    duplicate_year_months = sorted(
        {key for key in actual_year_months if actual_year_months.count(key) > 1}
    )
    if duplicate_year_months:
        raise ValueError(
            "Historical input contains duplicate monthly records: "
            f"{duplicate_year_months[:8]}"
        )
    expected_year_months = {
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    }
    missing_year_months = sorted(expected_year_months - set(actual_year_months))
    if missing_year_months:
        labels = ", ".join(
            f"{year:04d}-{month:02d}" for year, month in missing_year_months[:12]
        )
        suffix = " ..." if len(missing_year_months) > 12 else ""
        raise ValueError(
            f"Historical input is missing {len(missing_year_months)} monthly record(s) "
            f"required for {start_year}-{end_year}: {labels}{suffix}"
        )
    climatology = selected.groupby("time.month").mean("time", skipna=True)
    months = np.asarray(climatology.month.values, dtype=int)
    if not np.array_equal(months, np.arange(1, 13)):
        raise ValueError(
            f"Historical climatology is missing calendar months: got {months.tolist()}"
        )
    climatology.attrs.update(historical.attrs)
    climatology.attrs.update(
        reference_type="E3SM historical/free-running monthly climatology",
        climatology_start_year=int(start_year),
        climatology_end_year=int(end_year),
    )
    return climatology


def build_attractor_climatology(
    historical: xr.DataArray,
    *,
    start_year: int,
    end_year: int,
    frequency: str = "monthly",
) -> xr.DataArray:
    """Build a calendar climatology for monthly or future daily workflows."""
    if frequency == "monthly":
        return build_monthly_attractor_climatology(
            historical, start_year=start_year, end_year=end_year
        )
    if frequency != "daily":
        raise ValueError("frequency must be 'monthly' or 'daily'")
    if "time" not in historical.dims:
        raise ValueError("Historical attractor input must contain time")
    selected = historical.where(
        (historical.time.dt.year >= start_year) & (historical.time.dt.year <= end_year),
        drop=True,
    )
    if selected.sizes.get("time", 0) == 0:
        raise ValueError(f"Historical input has no dates in {start_year}-{end_year}")
    climatology = selected.groupby("time.dayofyear").mean("time", skipna=True)
    climatology.attrs.update(historical.attrs)
    climatology.attrs.update(
        reference_type="E3SM historical/free-running day-of-year climatology",
        climatology_start_year=int(start_year),
        climatology_end_year=int(end_year),
    )
    return climatology


def _normalize_historical_paths(paths: HistoricalPaths) -> list[Path]:
    if isinstance(paths, (str, Path)):
        normalized = [Path(paths)]
    else:
        normalized = [Path(path) for path in paths]
    if not normalized:
        raise ValueError("At least one historical reference file is required")
    missing = [path for path in normalized if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Historical reference file does not exist: {missing[0]}")
    return normalized


def _open_historical_dataset(
    paths: Sequence[Path], *, chunks: Mapping[str, int] | None = None
) -> xr.Dataset:
    """Open E3SM monthly means and standardize them to represented mid-month.

    E3SM uses a no-leap calendar and writes monthly means at the right endpoint
    of each averaging interval.  Thus ``TREFHT_198001_198012.nc`` decodes as
    1980-02-01 through 1981-01-01 although the records represent January
    through December 1980.  Apply the same ``time_set_midmonth`` treatment as
    the lead-time hindcast loader, yielding January 15 through December 15
    without changing the field values or no-leap calendar.
    """
    if len(paths) == 1:
        dataset = xr.open_dataset(paths[0], chunks={} if chunks is None else chunks)
    else:
        dataset = xr.open_mfdataset(
            paths, combine="by_coords", chunks={} if chunks is None else chunks,
            data_vars="minimal", coords="minimal", compat="override",
        )
    bounds_name = str(dataset["time"].attrs.get("bounds", ""))
    if bounds_name and bounds_name in dataset:
        bounds = dataset[bounds_name]
        bounds_dims = [dim for dim in bounds.dims if dim != "time"]
        if len(bounds_dims) != 1 or bounds.sizes[bounds_dims[0]] < 2:
            dataset.close()
            raise ValueError(f"Unsupported time bounds dimensions: {bounds.dims}")
        original_time_attrs = dict(dataset["time"].attrs)
        dataset = time_set_midmonth(dataset, "time")
        dataset["time"].attrs.update(
            original_time_attrs,
            original_timestamp_convention="right interval endpoint",
            diagnostic_timestamp_convention="represented month, day 15",
            diagnostic_timestamp_reason=(
                "same time_set_midmonth treatment used by the E3SM lead-time "
                "hindcast loader"
            ),
        )
    return dataset


def _open_observation_dataset(
    paths: Sequence[Path], *, chunks: Mapping[str, int] | None = None
) -> xr.Dataset:
    """Open one or more observation files without altering their timestamps."""
    if len(paths) == 1:
        return xr.open_dataset(paths[0], chunks={} if chunks is None else chunks)
    return xr.open_mfdataset(
        paths, combine="by_coords", chunks={} if chunks is None else chunks,
        data_vars="minimal", coords="minimal", compat="override",
    )


def _historical_reference_label(paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        return str(paths[0])
    common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    return f"{len(paths)} files under {common_parent}"


def output_path(
    job: Mapping[str, object], *, root: str | Path = S2D_DIAG_ROOT
) -> Path:
    """Return the component-aware reusable diagnostic path."""
    filename = (
        f"{job['source']}_{int(job['init_month']):02d}_{job['variable']}"
        "_two_reference_monthly_drift.nc"
    )
    return (
        Path(root)
        / "drift_diagnostics"
        / str(job["variable"])
        / str(job["source"])
        / str(job["component"])
        / filename
    )


def reference_output_path(
    job: Mapping[str, object], *, root: str | Path = S2D_DIAG_ROOT
) -> Path:
    """Return the path for references matched to one source's exact Y/L valid times."""
    filename = (
        f"{job['source']}_{int(job['init_month']):02d}_{job['variable']}"
        "_monthly_references.nc"
    )
    return Path(root) / "drift_diagnostics" / "references" / str(job["variable"]) / filename


def load_drift_references(
    source: str,
    init_month: int,
    variable: str,
    *,
    component: str = "atm",
    root: str | Path = S2D_DIAG_ROOT,
    path: str | Path | None = None,
    hindcast: xr.Dataset | xr.DataArray | None = None,
    require_matching_masks: bool = True,
    require_attractor_spread: bool = False,
) -> xr.Dataset:
    """Open and validate the prepared output of ``0_run_drift_references``.

    The loader inspects known schema aliases and exposes stable canonical names
    ``X_obs`` and ``X_att``.  Set ``require_attractor_spread`` when the consumer
    also needs the canonical ``sigma_att`` field.  The loader performs no
    observation or E3SMLE preparation.  The caller owns the returned dataset
    and should close it after use.
    """
    job = {
        "source": source, "init_month": int(init_month),
        "variable": variable, "component": component,
    }
    reference_path = Path(path) if path is not None else reference_output_path(job, root=root)
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Prepared drift reference does not exist: {reference_path}. "
            "Run jupyter/0_run_drift_references.ipynb first."
        )
    raw = xr.open_dataset(reference_path, chunks={})
    try:
        obs_name = next(
            (name for name in ("X_obs", "obs_ref", "observation_reference") if name in raw),
            None,
        )
        att_name = next(
            (name for name in ("X_att", "att_ref", "attractor_reference") if name in raw),
            None,
        )
        if obs_name is None or att_name is None:
            raise ValueError(
                f"Unrecognized reference schema in {reference_path}; "
                f"data variables are {list(raw.data_vars)}"
            )
        rename_map = {
            old: new
            for old, new in ((obs_name, "X_obs"), (att_name, "X_att"))
            if old != new
        }
        selected_names = [obs_name, att_name]
        spread_name = next(
            (name for name in ("sigma_att", "att_spread", "attractor_spread") if name in raw),
            None,
        )
        if require_attractor_spread and spread_name is None:
            raise ValueError(
                f"Prepared drift reference lacks the required historical ensemble "
                f"spread (sigma_att): {reference_path}. Rebuild it with "
                "jupyter/0_run_drift_references.ipynb."
            )
        if spread_name is not None:
            selected_names.append(spread_name)
            if spread_name != "sigma_att":
                rename_map[spread_name] = "sigma_att"
        result = raw[selected_names].rename(rename_map)
        if "valid_time" in raw.coords:
            result = result.assign_coords(valid_time=raw["valid_time"])
        elif "time" in raw and tuple(raw["time"].dims) == ("Y", "L"):
            result = result.assign_coords(valid_time=raw["time"])
        else:
            raise ValueError("Prepared reference product lacks a Y,L valid-time coordinate")
        if not {"Y", "L"} <= set(result.dims):
            raise ValueError("Prepared references must retain Y and L dimensions")
        validate_compatible_fields(result["X_obs"], result["X_att"])
        if "sigma_att" in result:
            validate_compatible_fields(result["X_att"], result["sigma_att"])
            if bool((result.sigma_att < 0).any()):
                raise ValueError("Historical ensemble spread contains negative values")
        if require_matching_masks:
            masks_equal = bool((result.X_obs.notnull() == result.X_att.notnull()).all())
            if not masks_equal:
                raise ValueError("Observation and attractor reference masks are incompatible")
            if "sigma_att" in result:
                spread_mask_equal = bool(
                    (result.X_att.notnull() == result.sigma_att.notnull()).all()
                )
                if not spread_mask_equal:
                    raise ValueError(
                        "Attractor reference and historical-spread masks are incompatible"
                    )

        if hindcast is not None:
            hindcast_field = hindcast[variable] if isinstance(hindcast, xr.Dataset) else hindcast
            hindcast_mean = (
                hindcast_field.mean("M", skipna=True) if "M" in hindcast_field.dims
                else hindcast_field
            )
            hindcast_mean = hindcast_mean.assign_attrs(hindcast_field.attrs)
            validate_compatible_fields(hindcast_mean, result.X_obs, result.X_att)
            hindcast_time = None
            if isinstance(hindcast, xr.Dataset) and "time" in hindcast:
                hindcast_time = hindcast["time"]
            elif "valid_time" in hindcast_field.coords:
                hindcast_time = hindcast_field["valid_time"]
            if hindcast_time is None or tuple(hindcast_time.dims) != ("Y", "L"):
                raise ValueError("Hindcast lacks a Y,L valid-time coordinate")
            if not np.array_equal(hindcast_time.values, result.valid_time.values):
                raise ValueError("Prepared-reference valid times do not match hindcast valid times")
        result.attrs.update(raw.attrs)
        schema = "X_obs,X_att,sigma_att" if "sigma_att" in result else "X_obs,X_att"
        result.attrs.update(reference_product=str(reference_path), schema=schema)
        return result
    except Exception:
        raw.close()
        raise


def atomic_to_netcdf(dataset: xr.Dataset, path: str | Path) -> None:
    """Write a compressed product and atomically replace its destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    encoding = {
        name: {"zlib": True, "complevel": 1}
        for name in dataset.data_vars
        if dataset[name].ndim > 0
    }
    try:
        dataset.to_netcdf(temporary, encoding=encoding)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_reference_product(
    job: Mapping[str, object],
    *,
    observation_path: HistoricalPaths,
    historical_path: HistoricalPaths,
    observation_variable: str | None = None,
    historical_variable: str | None = None,
    spread_path: HistoricalPaths | None = None,
    spread_variable: str | None = None,
    climatology_years: tuple[int, int] = (1981, 2010),
    analysis_years: tuple[int, int] | None = None,
) -> xr.Dataset:
    """Prepare observation and attractor fields on a hindcast's exact Y/L grid."""
    hindcast_path = discover_monthly_hindcast(job)
    variable = str(job["variable"])
    component = str(job["component"])
    historical_paths = _normalize_historical_paths(historical_path)
    observation_paths = _normalize_historical_paths(observation_path)
    spread_paths = _normalize_historical_paths(spread_path) if spread_path is not None else None
    regridders: dict[str, object] = {}
    # Nested context managers retain compatibility with the project's Python 3.8
    # environment (parenthesized context-manager lists require Python 3.10).
    with xr.open_dataset(hindcast_path, chunks={}) as hindcast_dataset:
        if "lat" not in hindcast_dataset.coords or "lon" not in hindcast_dataset.coords:
            raise ValueError("Monthly hindcast must provide lat/lon target-grid coordinates")
        target_grid = xr.Dataset(
            coords={"lat": hindcast_dataset.lat, "lon": hindcast_dataset.lon}
        )
        with _open_observation_dataset(observation_paths) as observation_dataset:
            with _open_historical_dataset(historical_paths) as historical_dataset:
                valid_time = hindcast_dataset["time"]
                if analysis_years is not None:
                    valid_time = select_initialization_years(valid_time, analysis_years)
                valid_time = valid_time.load()
                observation = apply_unit_transform(
                    _select_variable(
                        observation_dataset,
                        [
                            observation_variable,
                            variable,
                            "tas" if variable == "TREFHT" else "",
                        ],
                        "observational reference variable",
                    ),
                    component,
                    variable,
                )
                historical = apply_unit_transform(
                    _select_variable(
                        historical_dataset,
                        [
                            historical_variable,
                            variable,
                            "tas" if variable == "TREFHT" else "",
                        ],
                        "historical reference variable",
                    ),
                    component,
                    variable,
                )
                if bool(job.get("regrid", False)):
                    observation = _regrid_field(observation, target_grid, regridders)
                    historical = _regrid_field(historical, target_grid, regridders)
                obs_units = str(observation.attrs.get("units", "")).strip()
                att_units = str(historical.attrs.get("units", "")).strip()
                if not obs_units or not att_units:
                    raise ValueError("Prepared reference inputs must declare physical units")
                if obs_units.lower() != att_units.lower():
                    raise ValueError(
                        f"Reference units differ after conversion: {obs_units!r} vs {att_units!r}"
                    )
                climatology = build_monthly_attractor_climatology(
                    historical,
                    start_year=climatology_years[0],
                    end_year=climatology_years[1],
                )
                obs_ref = match_observation_to_valid_time(observation, valid_time)
                att_ref = match_model_climatology_to_valid_time(climatology, valid_time)
                product_fields = {"obs_ref": obs_ref, "att_ref": att_ref}
                if spread_paths is not None:
                    with _open_historical_dataset(spread_paths) as spread_dataset:
                        spread = apply_spread_unit_transform(
                            _select_variable(
                                spread_dataset,
                                [spread_variable, historical_variable, variable],
                                "historical ensemble-spread variable",
                            ),
                            component,
                            variable,
                        )
                        if bool(job.get("regrid", False)):
                            spread = _regrid_field(spread, target_grid, regridders)
                        spread_climatology = build_monthly_attractor_climatology(
                            spread,
                            start_year=climatology_years[0],
                            end_year=climatology_years[1],
                        )
                        sigma_att = match_model_climatology_to_valid_time(
                            spread_climatology, valid_time
                        ).rename("sigma_att")
                        sigma_att.attrs.update(
                            units=att_units,
                            reference_type="E3SM historical ensemble-spread monthly climatology",
                        )
                        product_fields["sigma_att"] = sigma_att
                product = xr.Dataset(product_fields).assign_coords(valid_time=valid_time)
                common_valid = product.obs_ref.notnull() & product.att_ref.notnull()
                if "sigma_att" in product:
                    common_valid = common_valid & product.sigma_att.notnull()
                for name in product.data_vars:
                    product[name] = product[name].where(common_valid)
                # Enforce units at the persisted interface consumed by 5a.
                product["obs_ref"].attrs["units"] = obs_units
                product["att_ref"].attrs["units"] = att_units
                product.attrs.update(
                    source=str(job["source"]),
                    component=component,
                    variable=variable,
                    initialization_month=int(job["init_month"]),
                    observational_reference=_historical_reference_label(observation_paths),
                    model_attractor_reference=_historical_reference_label(historical_paths),
                    model_attractor_spread_reference=(
                        _historical_reference_label(spread_paths)
                        if spread_paths is not None else "not requested"
                    ),
                    attractor_climatology_period=(
                        f"{climatology_years[0]}-{climatology_years[1]}"
                    ),
                    analysis_start_year=int(initialization_year_values(valid_time.Y).min()),
                    analysis_end_year=int(initialization_year_values(valid_time.Y).max()),
                    source_valid_time_input=str(hindcast_path),
                    reference_matching=(
                        "observations by exact year-month; attractor by calendar month"
                    ),
                )
                return product.compute()


def run_job(
    job: Mapping[str, object],
    *,
    observation_path: HistoricalPaths | None = None,
    historical_path: HistoricalPaths | None = None,
    prepared_reference_path: str | Path | None = None,
    observation_variable: str | None = None,
    historical_variable: str | None = None,
    spread_path: HistoricalPaths | None = None,
    spread_variable: str | None = None,
    climatology_years: tuple[int, int] = (1981, 2010),
    analysis_years: tuple[int, int] | None = None,
    distance_tolerance: float = 1.0e-6,
    output_root: str | Path = S2D_DIAG_ROOT,
    force: bool = False,
) -> Path:
    """Compute and save one monthly start-specific two-reference product."""
    destination = output_path(job, root=output_root)
    if destination.exists() and not force:
        return destination
    hindcast_path = discover_monthly_hindcast(job)
    variable = str(job["variable"])
    component = str(job["component"])
    target_grid = regrid.make_latlon_grid(dlat=1.0, dlon=1.0)
    regridders: dict[str, object] = {}

    if prepared_reference_path is None and (observation_path is None or historical_path is None):
        raise ValueError(
            "Supply prepared_reference_path or both observation_path and historical_path"
        )
    with xr.open_dataset(hindcast_path, chunks={}) as hindcast_dataset:
        values = apply_unit_transform(hindcast_dataset[variable], component, variable)
        valid_time = hindcast_dataset["time"]
        if analysis_years is not None:
            values = select_initialization_years(values, analysis_years)
            valid_time = valid_time.sel(Y=values.Y)
        valid_time = valid_time.load()
        if bool(job.get("regrid", False)):
            values = _regrid_field(values, target_grid, regridders)
        if prepared_reference_path is not None:
            with xr.open_dataset(prepared_reference_path, chunks={}) as references:
                obs_ref = references["obs_ref"].load()
                att_ref = references["att_ref"].load()
                sigma_att = references["sigma_att"].load() if "sigma_att" in references else None
                if "valid_time" not in references.coords:
                    raise ValueError("Prepared reference product lacks valid_time")
                for dim in ("Y", "L"):
                    if not np.array_equal(references[dim].values, valid_time[dim].values):
                        raise ValueError(f"Prepared references have incompatible {dim} coordinates")
                if not np.array_equal(references["valid_time"].values, valid_time.values):
                    raise ValueError("Prepared references do not match hindcast valid times")
                obs_reference_label = str(
                    references.attrs.get("observational_reference", "unknown")
                )
                att_reference_label = str(
                    references.attrs.get("model_attractor_reference", "unknown")
                )
        else:
            references = prepare_reference_product(
                job,
                observation_path=observation_path,
                historical_path=historical_path,
                observation_variable=observation_variable,
                historical_variable=historical_variable,
                spread_path=spread_path,
                spread_variable=spread_variable,
                climatology_years=climatology_years,
                analysis_years=analysis_years,
            )
            obs_ref = references["obs_ref"]
            att_ref = references["att_ref"]
            sigma_att = references.get("sigma_att")
            obs_reference_label = str(observation_path)
            att_reference_label = _historical_reference_label(
                _normalize_historical_paths(historical_path)
            )
        product = compute_diagnostics(
            values,
            obs_ref,
            att_ref,
            baseline_lead=1,
            distance_tolerance=distance_tolerance,
        )
        if sigma_att is not None:
            validate_compatible_fields(product["e_att"], sigma_att)
            product["z_att"] = (
                product["e_att"] / sigma_att.where(sigma_att > 0)
            ).assign_attrs(
                units="1",
                diagnostic=(
                    "departure from model attractor normalized by historical "
                    "ensemble spread"
                ),
            )
        product = product.assign_coords(valid_time=valid_time)
        product.attrs.update(
            title=f"{job['source']} {component} {variable} two-reference monthly drift",
            source=str(job["source"]),
            component=component,
            variable=variable,
            initialization_month=int(job["init_month"]),
            lead_convention="L=1 is initialization-month mean; L=24 is final forecast month",
            observational_reference=obs_reference_label,
            model_attractor_reference=att_reference_label,
            attractor_climatology_period=f"{climatology_years[0]}-{climatology_years[1]}",
            source_input=str(hindcast_path),
            ensemble_handling="mean over M only; Y retained",
            analysis_start_year=int(initialization_year_values(valid_time.Y).min()),
            analysis_end_year=int(initialization_year_values(valid_time.Y).max()),
            regridded=str(bool(job.get("regrid", False))).lower(),
        )
        output_chunks = {"Y": 1, "L": 1, "lat": 90, "lon": 180}
        product = product.chunk(
            {name: size for name, size in output_chunks.items() if name in product.dims}
        ).compute()

    if not bool(np.isfinite(product.hindcast_mean).any()):
        raise ValueError("Computed hindcast_mean contains no finite values")
    atomic_to_netcdf(product, destination)
    return destination


__all__ = [
    "AttractorConfig",
    "DEFAULT_LEAD_WINDOWS",
    "E3SMLE_ENSMEAN_ROOT",
    "E3SMLE_ENSSPREAD_ROOT",
    "E3SMLE_GRID",
    "E3SMLE_ROOT",
    "EXPECTED_MONTHLY_LEADS",
    "UNIT_TRANSFORMS",
    "apply_spread_unit_transform",
    "apply_unit_transform",
    "atomic_to_netcdf",
    "build_attractor_climatology",
    "build_monthly_attractor_climatology",
    "discover_monthly_hindcast",
    "discover_e3smle_files",
    "expand_requests",
    "output_path",
    "load_e3smle_timeseries",
    "load_drift_references",
    "initialization_year_values",
    "monthly_hindcast_output_path",
    "prepare_monthly_hindcast_cache",
    "prepare_reference_product",
    "reference_output_path",
    "run_job",
    "select_initialization_years",
]
