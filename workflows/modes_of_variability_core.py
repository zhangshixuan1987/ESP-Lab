"""Reusable data and PCMDI EOF functions for modes-of-variability workflows."""

from __future__ import annotations

import argparse
import copy
import contextlib
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

_prefix = Path(sys.prefix)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")


def _ensure_native_data_path(env_name: str, path: Path, sentinel: str) -> None:
    current = Path(os.environ.get(env_name, ""))
    if not (current / sentinel).is_file() and (path / sentinel).is_file():
        os.environ[env_name] = str(path)


_gdal_data = _prefix / "share" / "gdal"
_proj_data = _prefix / "share" / "proj"
_ensure_native_data_path("GDAL_DATA", _gdal_data, "header.dxf")
_ensure_native_data_path("PROJ_DATA", _proj_data, "proj.db")
_ensure_native_data_path("PROJ_LIB", _proj_data, "proj.db")

import numpy as np
import xarray as xr

from esp_lab.diagnostics import (
    DEFAULT_CLIMATOLOGY_END_YEAR,
    DEFAULT_CLIMATOLOGY_START_YEAR,
)
import cftime

from pcmdi_metrics.io import load_regions_specs
from pcmdi_metrics.utils import calculate_area_weights, calculate_grid_area
from pcmdi_metrics.variability_mode.lib import (
    eof_analysis_get_variance_mode,
    gain_pseudo_pcs,
)

from esp_lab import data_access_cesm_smyle as smyle_access
from esp_lab import data_access_e3sm as e3sm_access
from esp_lab import data_access_nmme as nmme_access
from esp_lab import data_access_obs as obs_access
from esp_lab import stats
from esp_lab.paths import CESM_SMYLE_DIAG_DIR, NMME_FIXED_DIR
from esp_lab.utils import calendar_utils as cal
from esp_lab.utils import regrid_utils as regrid
from esp_lab.utils import sst_utils


LOG = logging.getLogger(__name__)
_NUMPY_SVD = np.linalg.svd


def _svd_with_gesvd_fallback(
    array,
    full_matrices=True,
    compute_uv=True,
    hermitian=False,
):
    """Retry rare NumPy SVD non-convergence with LAPACK's robust gesvd driver."""
    try:
        return _NUMPY_SVD(
            array,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
        )
    except np.linalg.LinAlgError:
        from scipy.linalg import svd

        return svd(
            np.asarray(array),
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            lapack_driver="gesvd",
            check_finite=False,
        )


@contextlib.contextmanager
def robust_svd_retry():
    """Temporarily give eofs a gesvd fallback for a failed calculation."""
    original = np.linalg.svd
    np.linalg.svd = _svd_with_gesvd_fallback
    try:
        yield
    finally:
        np.linalg.svd = original


def eof_analysis_with_svd_fallback(*args, **kwargs):
    """Run PCMDI EOF analysis, retrying only numerical SVD non-convergence."""
    try:
        return eof_analysis_get_variance_mode(*args, **kwargs)
    except ValueError as error:
        if "error encountered in SVD" not in str(error):
            raise
        LOG.warning("NumPy SVD did not converge; retrying with LAPACK gesvd.")
        with robust_svd_retry():
            return eof_analysis_get_variance_mode(*args, **kwargs)
MODEL_SOURCES = ("e3sm", "smyle", "nmme")
SUPPORTED_MODES = (
    "NAM", "NAO", "SAM", "PSA1", "PSA2", "PNA", "NPO",
    "EA", "SCA", "PDO", "NPGO", "AMO",
)
MODE_EOF_NUMBER = {
    "NAM": 1, "NAO": 1, "SAM": 1, "PNA": 1, "PDO": 1, "AMO": 1,
    "NPO": 2, "NPGO": 2, "PSA1": 2, "EA": 2, "PSA2": 3, "SCA": 3,
}
MODE_ORIGIN_DOMAIN = {
    "NPO": "PNA", "NPGO": "PDO", "PSA1": "SAM", "PSA2": "SAM",
    "EA": "NAO", "SCA": "NAO",
}
PRESSURE_MODES = {
    "NAM", "NAO", "SAM", "PSA1", "PSA2", "PNA", "NPO", "EA", "SCA",
}
TEMPERATURE_MODES = {"PDO", "NPGO", "AMO"}
WRITE_GLOBAL_TELECONNECTIONS_IN_INDEX_PRODUCTS = False


@dataclass(frozen=True)
class StationNaoDefinition:
    south_name: str = "Lisbon"
    south_lon: float = -9.1393
    south_lat: float = 38.72
    north_name: str = "Reykjavik"
    north_lon: float = -21.9426
    north_lat: float = 64.1466


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--modes", nargs="+", type=str.upper, choices=SUPPORTED_MODES, default=["NAO"]
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("e3sm", "smyle", "nmme", "obs", "era5"),
        default=["e3sm", "smyle", "obs"],
    )
    parser.add_argument("--init-months", nargs="+", type=int, default=[5, 11])
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument(
        "--clim-start", type=int, default=DEFAULT_CLIMATOLOGY_START_YEAR
    )
    parser.add_argument("--clim-end", type=int, default=DEFAULT_CLIMATOLOGY_END_YEAR)
    parser.add_argument("--monthly-nlead", type=int, default=24)
    parser.add_argument("--target-dlat", type=float, default=2.5)
    parser.add_argument("--target-dlon", type=float, default=2.5)
    parser.add_argument("--regrid-method", default="conservative")
    parser.add_argument("--no-periodic", action="store_true")
    parser.add_argument(
        "--eof-scaling", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--remove-domain-mean", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--eof-bootstrap-iterations",
        type=int,
        default=0,
        help=(
            "Hierarchical EOF bootstrap iterations. Zero disables bootstrap "
            "diagnostics while retaining North eigenvalue diagnostics."
        ),
    )
    parser.add_argument("--eof-bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--eof-bootstrap-confidence",
        type=float,
        default=0.90,
        help="Central confidence level for EOF pattern and metric intervals.",
    )
    parser.add_argument(
        "--eof-strategy",
        "--eof_strategy",
        choices=("conventional", "fixed_obs_projection"),
        default="fixed_obs_projection",
        help=(
            "Model EOF strategy. 'fixed_obs_projection' projects model anomalies "
            "onto the observed EOF basis and computes regression patterns; "
            "'conventional' also fits model EOFs separately. Use conventional only "
            "when the model sample is long enough for stable EOF estimation."
        ),
    )
    parser.add_argument(
        "--eof-reference-source",
        "--eof_reference_source",
        choices=("obs", "era5", "hadisst2"),
        default="obs",
        help=(
            "Source used to define the fixed EOF basis. Currently this maps to "
            "the observation products selected by --psl-obs-product/--ts-obs-product."
        ),
    )
    parser.add_argument(
        "--eof-reference-start-year",
        "--eof_reference_start_year",
        type=int,
        default=None,
        help="Start year for the observation EOF training period. Defaults to --start-year when omitted.",
    )
    parser.add_argument(
        "--eof-reference-end-year",
        "--eof_reference_end_year",
        type=int,
        default=None,
        help="End year for the observation EOF training period. Defaults to --end-year when omitted.",
    )
    parser.add_argument(
        "--regression-confidence",
        "--regression_confidence",
        type=float,
        default=0.95,
        help="Confidence level for gridpoint regression significance masks.",
    )
    parser.add_argument(
        "--sst-ocean-mask-min-valid-fraction",
        "--sst_ocean_mask_min_valid_fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--sst-ocean-mask-resolution",
        "--sst_ocean_mask_resolution",
        choices=("110m", "50m", "10m"),
        default="110m",
    )
    parser.add_argument(
        "--model-sst-land-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply a reusable source-grid land mask before regridding E3SM/SMYLE "
            "SST (default: enabled). Pressure fields are unaffected."
        ),
    )
    parser.add_argument(
        "--e3sm-data-dir", default="/global/cfs/cdirs/e3sm/S2S2D/post_process"
    )
    parser.add_argument(
        "--e3sm-case-prefix",
        default="WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL",
    )
    parser.add_argument("--e3sm-cache-tag", default="e3sm")
    parser.add_argument("--e3sm-display-name", default="E3SM")
    parser.add_argument("--e3sm-nens", type=int, default=10)
    parser.add_argument("--e3sm-engine", default="netcdf4")
    parser.add_argument("--e3sm-grid", default="180x360_aave")
    parser.add_argument(
        "--smyle-benchmark-dir",
        default=str(CESM_SMYLE_DIAG_DIR),
    )
    parser.add_argument("--smyle-nens", type=int, default=20)
    parser.add_argument(
        "--nmme-root",
        default="/global/cfs/cdirs/e3sm/S2S2D/NMME/data_hindcast_by_member",
    )
    parser.add_argument(
        "--nmme-models",
        nargs="+",
        default=[],
        help=(
            "NMME model directory names to include. Required when --sources "
            "includes nmme."
        ),
    )
    parser.add_argument(
        "--nmme-field",
        default="auto",
        help="Raw NMME variable to read: 'auto', 'sst', or 'prmsl'.",
    )
    parser.add_argument(
        "--nmme-chunks",
        default="",
        help=(
            "Comma-separated chunks for raw NMME member files, e.g. "
            "'S:12,L:-1,Y:181,X:360'. Empty uses the archive/native chunks."
        ),
    )
    parser.add_argument(
        "--nmme-sst-land-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply the Natural Earth land mask to NMME SST immediately after "
            "reading (default: enabled)."
        ),
    )
    parser.add_argument(
        "--nmme-fixed-dir",
        type=Path,
        default=NMME_FIXED_DIR,
        help="Directory for reusable per-model NMME fixed fields.",
    )
    parser.add_argument(
        "--obs-dir",
        default="/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series",
    )
    parser.add_argument("--psl-obs-product", default="ERA5")
    parser.add_argument("--psl-obs-var", default="psl")
    parser.add_argument("--ts-obs-product", default="HadISST2")
    parser.add_argument("--ts-obs-var", default="sst")
    parser.add_argument("--obs-start-year", type=int, default=1979)
    parser.add_argument("--obs-end-year", type=int, default=2019)
    parser.add_argument("--dask-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--merge-manifest",
        action="store_true",
        help="Merge this run's products into a compatible manifest under a file lock.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--legacy-nao-layout", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # By default, train the observed EOF basis on the same years used for the
    # model hindcast verification.  This avoids an accidental mismatch between
    # the EOF reference period and the model period.  Users can still override
    # the reference period explicitly with --eof-reference-start-year and/or
    # --eof-reference-end-year for sensitivity tests using a longer observed
    # record.
    if args.eof_reference_start_year is None:
        args.eof_reference_start_year = args.start_year
    if args.eof_reference_end_year is None:
        args.eof_reference_end_year = args.end_year

    return args


def grid_token(dlat: float, dlon: float) -> str:
    token = lambda value: f"{value:g}".replace(".", "p")
    return f"{token(dlat)}x{token(dlon)}deg"


def atomic_to_netcdf(
    dataset: xr.Dataset,
    path: Path,
    encoding: dict[str, dict[str, object]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        dataset.to_netcdf(temporary, encoding=encoding or {})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert_units(data: xr.DataArray, field: str) -> xr.DataArray:
    units = str(data.attrs.get("units", "")).strip().lower()
    indexer = {dim: 0 for dim in data.dims if data.sizes[dim]}
    sample = float(data.isel(indexer).mean(skipna=True).load())
    if field == "PSL":
        result = data * 1.0e-2 if units in {"pa", "pascal", "pascals"} or sample > 2000 else data
        target_units = "hPa"
    else:
        result = data - 273.15 if units in {"k", "kelvin", "degk"} or sample > 150 else data
        target_units = "degC"
    result.attrs.update(data.attrs)
    result.attrs["units"] = target_units
    return result


def select_field(dataset: xr.Dataset, field: str) -> xr.DataArray:
    aliases = {
        "PSL": ("PSL", "psl", "slp"),
        "SST": ("SST", "TS", "sst", "ts", "tos"),
    }
    for candidate in aliases[field]:
        if candidate in dataset:
            return dataset[candidate]
    raise KeyError(f"Cannot find {field!r}; variables: {list(dataset.data_vars)}")


def common_year_subset(obj: xr.Dataset | xr.DataArray, years: list[int]):
    requested = set(years)
    selected_labels = [
        value for value in obj.Y.values if int(str(value)[:4]) in requested
    ]
    if not selected_labels:
        raise ValueError("No requested initialization years are available.")
    return obj.sel(Y=selected_labels)


def drop_empty_leads(
    dataset: xr.Dataset, variable: xr.DataArray
) -> tuple[xr.Dataset, xr.DataArray]:
    if "L" not in variable.dims:
        return dataset, variable
    reduction_dims = [dim for dim in variable.dims if dim != "L"]
    valid = variable.notnull().any(reduction_dims).compute()
    valid_leads = variable.L.where(valid, drop=True)
    if not valid_leads.size:
        raise ValueError(f"{variable.name or 'Field'} has no valid lead times.")
    return dataset.sel(L=valid_leads), variable.sel(L=valid_leads)


def parse_chunk_spec(spec: str | None) -> dict[str, int]:
    """Parse a compact xarray chunk spec such as ``"Y:45,X:90"``."""
    if not spec:
        return {}
    chunks: dict[str, int] = {}
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid chunk spec {part!r}; expected NAME:SIZE.")
        name, value = (item.strip() for item in part.split(":", 1))
        if not name:
            raise ValueError(f"Invalid chunk spec {part!r}; empty dimension name.")
        chunks[name] = int(value)
    return chunks


def add_months_noleap(year: int, month: int, offset: int) -> cftime.DatetimeNoLeap:
    month_index = month - 1 + offset
    return cftime.DatetimeNoLeap(year + month_index // 12, month_index % 12 + 1, 15)


def nmme_member_dataset(
    member_dir: Path,
    *,
    model: str,
    member: str,
    field: str,
    chunks: dict[str, int],
    apply_sst_land_mask: bool = True,
    fixed_dir: Path = NMME_FIXED_DIR,
) -> xr.Dataset:
    files = sorted(member_dir.glob(f"{field}_{model}_{member}_S*.nc"))
    if not files:
        raise FileNotFoundError(f"No NMME {field!r} files found under {member_dir}")
    dataset = xr.open_mfdataset(
        [str(path) for path in files],
        combine="by_coords",
        decode_times=False,
        chunks=chunks,
    )
    if "S" not in dataset:
        raise ValueError(f"NMME member dataset lacks forecast-start coordinate S: {member_dir}")
    dataset = dataset.drop_vars("M", errors="ignore")
    dataset = nmme_access.decode_cf_time(dataset, time_var="S")
    if field == "sst":
        dataset[field] = nmme_access.mask_invalid_sst(
            dataset[field],
            apply_land_mask=apply_sst_land_mask,
            lon_name="X",
            lat_name="Y",
            model=model,
            fixed_dir=fixed_dir,
        )
    dataset = dataset.expand_dims(M=[f"{model}:{member}"])
    return dataset


def nmme_field_dataset(
    init_month: int,
    settings: dict[str, object],
    args: argparse.Namespace,
) -> xr.Dataset:
    """Load gridded NMME hindcasts into the common Y/L/M/lat/lon layout."""
    field = str(settings["field"])
    requested_field = str(getattr(args, "nmme_field", "auto")).lower()
    if requested_field == "auto":
        archive_field = "prmsl" if field == "PSL" else "sst"
    else:
        archive_field = requested_field
    expected_archive_field = "prmsl" if field == "PSL" else "sst"
    if archive_field != expected_archive_field:
        raise ValueError(
            f"NMME source for {field} modes requires {expected_archive_field!r}, "
            f"got --nmme-field={archive_field!r}."
        )

    models = [str(model).strip() for model in getattr(args, "nmme_models", []) if str(model).strip()]
    if not models:
        raise ValueError("--nmme-models is required when --sources includes nmme.")

    nmme_root = Path(args.nmme_root)
    chunks = parse_chunk_spec(getattr(args, "nmme_chunks", ""))
    member_datasets: list[xr.Dataset] = []
    failures: list[str] = []

    for model in models:
        field_dir = nmme_root / model / archive_field
        if not field_dir.is_dir():
            failures.append(f"{model}: missing {field_dir}")
            continue
        member_dirs = sorted(path for path in field_dir.glob("M*") if path.is_dir())
        if not member_dirs:
            failures.append(f"{model}: no member directories under {field_dir}")
            continue
        for member_dir in member_dirs:
            try:
                member_datasets.append(
                    nmme_member_dataset(
                        member_dir,
                        model=model,
                        member=member_dir.name,
                        field=archive_field,
                        chunks=chunks,
                        apply_sst_land_mask=bool(args.nmme_sst_land_mask),
                        fixed_dir=Path(args.nmme_fixed_dir),
                    )
                )
            except Exception as error:
                failures.append(f"{model}/{member_dir.name}: {error}")

    if not member_datasets:
        raise ValueError("No NMME member datasets could be loaded: " + "; ".join(failures))
    if failures:
        LOG.warning("Some NMME members were skipped: %s", "; ".join(failures[:20]))

    dataset = xr.concat(
        member_datasets,
        dim="M",
        join="outer",
        combine_attrs="drop_conflicts",
    )
    rename = {}
    if "Y" in dataset.dims or "Y" in dataset.coords:
        rename["Y"] = "lat"
    if "X" in dataset.dims or "X" in dataset.coords:
        rename["X"] = "lon"
    dataset = dataset.rename(rename)
    if "lat" not in dataset.coords or "lon" not in dataset.coords:
        raise ValueError("NMME dataset must provide Y/X or lat/lon coordinates.")
    if dataset["lat"][0] > dataset["lat"][-1]:
        dataset = dataset.reindex(lat=dataset.lat[::-1])
    dataset = dataset.assign_coords(lon=(dataset.lon % 360)).sortby("lon")

    init_times = dataset["S"].where(dataset["S"].dt.month == init_month, drop=True)
    init_times = init_times.where(
        (init_times.dt.year >= args.start_year)
        & (init_times.dt.year <= args.end_year),
        drop=True,
    )
    if init_times.size == 0:
        raise ValueError(
            f"No NMME initialization times found for month {init_month:02d} "
            f"during {args.start_year}-{args.end_year}."
        )
    dataset = dataset.sel(S=init_times)
    years = dataset["S"].dt.year.astype(int).data
    dataset = dataset.rename({"S": "Y"}).assign_coords(Y=years)
    dataset = dataset.assign_coords(L=np.arange(dataset.sizes["L"]) + 1)
    if dataset.sizes["L"] > args.monthly_nlead:
        dataset = dataset.isel(L=slice(0, args.monthly_nlead))

    years_array = dataset["Y"].values.astype(int)
    leads = dataset["L"].values.astype(int)
    valid_time = xr.DataArray(
        np.array(
            [
                [add_months_noleap(int(year), init_month, int(lead) - 1) for lead in leads]
                for year in years_array
            ],
            dtype=object,
        ),
        dims=("Y", "L"),
        coords={"Y": dataset["Y"], "L": dataset["L"]},
        name="time",
    )
    dataset = dataset.rename({archive_field: field})
    dataset["time"] = valid_time
    dataset[field].attrs.setdefault("units", "Pa" if field == "PSL" else "degC")
    dataset[field].attrs.setdefault(
        "long_name",
        "NMME sea level pressure" if field == "PSL" else "NMME sea surface temperature",
    )
    return dataset


def mode_settings(mode: str) -> dict[str, object]:
    mode = mode.upper()
    domain_mode = MODE_ORIGIN_DOMAIN.get(mode, mode)
    domain = load_regions_specs()[domain_mode]["domain"]
    lat_s, lat_n = sorted(domain["latitude"])
    field = "PSL" if mode in PRESSURE_MODES else "SST"
    return {
        "mode": mode,
        "eof_number": MODE_EOF_NUMBER[mode],
        "domain_mode": domain_mode,
        "lon_w": float(domain["longitude"][0]),
        "lon_e": float(domain["longitude"][1]),
        "lat_s": float(lat_s),
        "lat_n": float(lat_n),
        "field": field,
        "archive_field": "PSL" if field == "PSL" else "TS",
        "frequency": "seasonal" if field == "PSL" else "monthly",
        "obs_product": "ERA5" if field == "PSL" else "HadISST2",
        "obs_var": "psl" if field == "PSL" else "sst",
        "units": "hPa" if field == "PSL" else "degC",
    }


def select_mode_domain(data: xr.DataArray, settings: dict[str, object]) -> xr.DataArray:
    lon360 = data.lon % 360
    raw_west, raw_east = float(settings["lon_w"]), float(settings["lon_e"])
    domain_width = raw_east - raw_west
    if domain_width <= 0:
        domain_width += 360
    continuous_lon = ((lon360 - raw_west) % 360) + raw_west
    if domain_width >= 359:
        lon_mask = xr.ones_like(lon360, dtype=bool)
    else:
        lon_mask = (
            (continuous_lon >= raw_west)
            & (continuous_lon <= raw_west + domain_width)
        )
    region = data.where(
        lon_mask
        & (data.lat >= float(settings["lat_s"]))
        & (data.lat <= float(settings["lat_n"])),
        drop=True,
    )
    return region.assign_coords(
        lon=continuous_lon.where(lon_mask, drop=True)
    ).sortby("lon")


def global_sst_for_regression(
    data: xr.DataArray,
    *,
    ocean_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return a full-map SST field for teleconnection regression maps."""
    field = data.where(np.isfinite(data)).sortby("lat").sortby("lon")
    if ocean_mask is not None:
        aligned_mask = align_spatial_mask(
            ocean_mask,
            field,
            mask_name="global_ocean_mask",
        )
        field = field.where(aligned_mask)
    return bounded_dataset(field, "mode_field")["mode_field"]


def validate_spatial_coordinates(
    dataset: xr.Dataset | xr.DataArray,
    context: str,
) -> None:
    """Validate the one-dimensional spatial coordinate contract for products."""
    for name in ("lat", "lon"):
        if name not in dataset.coords:
            raise ValueError(f"{context} is missing the {name!r} coordinate.")
        coordinate = dataset[name]
        values = np.asarray(coordinate, dtype=float)
        if coordinate.ndim != 1 or values.size < 2:
            raise ValueError(f"{context} {name} must be a nontrivial 1D coordinate.")
        if not np.isfinite(values).all():
            raise ValueError(f"{context} {name} contains non-finite values.")
        differences = np.diff(values)
        if not np.all(differences > 0):
            raise ValueError(f"{context} {name} must be strictly increasing.")
        typical_spacing = float(np.median(differences))
        if not np.allclose(
            differences,
            typical_spacing,
            rtol=1.0e-6,
            atol=max(1.0e-8, abs(typical_spacing) * 1.0e-6),
        ):
            raise ValueError(
                f"{context} {name} spacing is discontinuous: "
                f"min={differences.min():g}, max={differences.max():g}."
            )
    latitude = np.asarray(dataset["lat"], dtype=float)
    longitude = np.asarray(dataset["lon"], dtype=float)
    if latitude.min() < -90 or latitude.max() > 90:
        raise ValueError(f"{context} latitude lies outside [-90, 90].")
    if longitude[-1] - longitude[0] >= 360:
        raise ValueError(f"{context} longitude span must be less than 360 degrees.")


def bounded_dataset(data: xr.DataArray, name: str) -> xr.Dataset:
    validate_spatial_coordinates(data, f"{name} domain")
    dataset = data.rename(name).to_dataset()
    for coordinate, axis, units, bounds_name in (
        ("lat", "Y", "degrees_north", "lat_bnds"),
        ("lon", "X", "degrees_east", "lon_bnds"),
    ):
        dataset[coordinate].attrs.update(axis=axis, units=units, bounds=bounds_name)
        values = np.asarray(dataset[coordinate], dtype=float)
        diffs = np.abs(np.diff(values))
        spacing = float(np.median(diffs[diffs > 0]))
        dataset[bounds_name] = xr.DataArray(
            np.column_stack((values - spacing / 2, values + spacing / 2)),
            dims=(coordinate, "bnds"),
            coords={coordinate: dataset[coordinate], "bnds": [0, 1]},
        )
    return dataset


def spatial_mean(dataset: xr.Dataset, variable: str) -> xr.DataArray:
    weights = calculate_area_weights(calculate_grid_area(dataset))
    valid = weights.where(dataset[variable].notnull())
    return (dataset[variable] * valid).sum(("lat", "lon"), skipna=True) / valid.sum(
        ("lat", "lon"), skipna=True
    )


def eof_ready_dataset(dataset: xr.Dataset, variable: str) -> xr.Dataset:
    """Apply a time-invariant finite spatial mask required by EOF solvers."""
    field = dataset[variable].where(np.isfinite(dataset[variable]))
    if "time" not in field.dims:
        raise ValueError(f"{variable!r} must have a time dimension for EOF analysis.")

    valid_mask = field.notnull().all("time")
    valid_points = int(valid_mask.sum().compute())
    if valid_points < 2:
        raise ValueError(
            f"{variable!r} has only {valid_points} spatial points with complete "
            "finite coverage across time."
        )

    result = dataset.copy()
    result[variable] = field.where(valid_mask)
    return result


def north_eigenvalue_diagnostics(solver, eof_number: int) -> xr.Dataset:
    """Return North et al. eigenvalue errors and target-mode separation."""
    n_modes = eof_number + 1
    eigenvalues = solver.eigenvalues(neigs=n_modes).rename(
        mode="diagnosed_mode"
    )
    errors = solver.northTest(neigs=n_modes).rename(mode="diagnosed_mode")
    diagnosed_mode = np.arange(1, eigenvalues.sizes["diagnosed_mode"] + 1)
    eigenvalues = eigenvalues.assign_coords(diagnosed_mode=diagnosed_mode)
    errors = errors.assign_coords(diagnosed_mode=diagnosed_mode)

    target = eigenvalues.sel(diagnosed_mode=eof_number)
    target_error = errors.sel(diagnosed_mode=eof_number)
    neighbor = eigenvalues.sel(diagnosed_mode=eof_number + 1)
    neighbor_error = errors.sel(diagnosed_mode=eof_number + 1)
    gap = target - neighbor
    separated = gap > (target_error + neighbor_error)

    return xr.Dataset(
        {
            "mode_eigenvalue": eigenvalues,
            "mode_eigenvalue_north_error": errors,
            "mode_eigenvalue_gap_to_next": gap,
            "mode_north_separated_from_next": separated.astype("int8"),
        }
    )


def cosine_weighted_pattern_metrics(
    pattern: xr.DataArray,
    reference: xr.DataArray,
) -> tuple[float, float]:
    """Return cosine-latitude-weighted RMSE and centered pattern correlation."""
    pattern, reference = xr.align(pattern, reference, join="inner")
    valid = np.isfinite(pattern) & np.isfinite(reference)
    weights = np.cos(np.deg2rad(reference.lat)).broadcast_like(reference).where(valid)
    weight_sum = weights.sum(("lat", "lon"))
    difference = pattern - reference
    rmse = np.sqrt(
        (weights * difference**2).sum(("lat", "lon")) / weight_sum
    )
    pattern_mean = (weights * pattern).sum(("lat", "lon")) / weight_sum
    reference_mean = (weights * reference).sum(("lat", "lon")) / weight_sum
    pattern_anomaly = pattern - pattern_mean
    reference_anomaly = reference - reference_mean
    covariance = (
        weights * pattern_anomaly * reference_anomaly
    ).sum(("lat", "lon"))
    pattern_variance = (weights * pattern_anomaly**2).sum(("lat", "lon"))
    reference_variance = (
        weights * reference_anomaly**2
    ).sum(("lat", "lon"))
    pcc = covariance / np.sqrt(pattern_variance * reference_variance)
    return float(rmse), float(pcc)


def enforce_mode_sign(
    mode: str,
    pattern: xr.DataArray,
    pc: xr.DataArray | None = None,
    reverse_sign: bool = False,
) -> tuple[xr.DataArray, xr.DataArray | None, bool]:
    """Enforce PCMDI sign conventions independently of longitude encoding."""
    checks = {
        "PDO": (30, 40, 150, 180, -1),
        "PNA": (80, 90, None, None, 1),
        "NAM": (60, 80, None, None, -1),
        "NAO": (60, 80, None, None, -1),
        "EA": (50, 60, 330, 360, 1),
        "SCA": (50, 70, 10, 40, 1),
        "SAM": (-90, -60, None, None, -1),
        "PSA1": (-64.5, -59.5, 207.5, 212.5, -1),
        "PSA2": (-62.5, -57.5, 277.5, 282.5, -1),
    }
    if mode not in checks:
        return pattern, pc, reverse_sign

    south, north, west, east, desired_sign = checks[mode]
    mask = (pattern.lat >= south) & (pattern.lat <= north)
    if west is not None and east is not None:
        longitude = pattern.lon % 360
        mask = mask & (longitude >= west) & (longitude <= east)
    regional_mean = pattern.where(mask, drop=True).mean(skipna=True)
    value = float(regional_mean)
    if np.isfinite(value) and value * desired_sign < 0:
        pattern = -pattern
        if pc is not None:
            pc = -pc
        reverse_sign = not reverse_sign
    return pattern, pc, reverse_sign


def metric_interval(
    values: xr.DataArray,
    confidence: float,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return lower, median, and upper bootstrap summaries."""
    tail = (1.0 - confidence) / 2.0
    quantiles = values.quantile([tail, 0.5, 1.0 - tail], dim="bootstrap")
    return (
        quantiles.sel(quantile=tail, drop=True),
        quantiles.sel(quantile=0.5, drop=True),
        quantiles.sel(quantile=1.0 - tail, drop=True),
    )


def eof_bootstrap_diagnostics(
    prepared: xr.Dataset,
    baseline_pattern: xr.DataArray,
    settings: dict[str, object],
    args: argparse.Namespace,
    bootstrap_indices: np.ndarray,
    reference_patterns: xr.DataArray | None = None,
) -> tuple[xr.Dataset, xr.DataArray]:
    """Recompute sign-aligned EOFs for supplied bootstrap sample indices."""
    patterns, fractions, rmses, pccs = [], [], [], []
    eof_number = int(settings["eof_number"])
    for bootstrap, indices in enumerate(bootstrap_indices):
        sample = prepared["mode_field"].isel(time=indices)
        sample = sample.assign_coords(time=np.arange(sample.sizes["time"]))
        bootstrap_prepared = eof_ready_dataset(
            bounded_dataset(sample, "mode_field"), "mode_field"
        )
        pattern, _, fraction, _, _ = eof_analysis_with_svd_fallback(
            str(settings["mode"]),
            bootstrap_prepared,
            "mode_field",
            eofn=eof_number,
            eofn_max=eof_number,
            EofScaling=args.eof_scaling,
        )
        pattern, _, _ = enforce_mode_sign(str(settings["mode"]), pattern)
        _, alignment = cosine_weighted_pattern_metrics(pattern, baseline_pattern)
        if np.isfinite(alignment) and alignment < 0:
            pattern = -pattern
        patterns.append(pattern.expand_dims(bootstrap=[bootstrap]))
        fractions.append(fraction.expand_dims(bootstrap=[bootstrap]))
        if reference_patterns is not None:
            reference = reference_patterns.sel(bootstrap=bootstrap)
            rmse, pcc = cosine_weighted_pattern_metrics(pattern, reference)
            rmses.append(rmse)
            pccs.append(pcc)

    bootstrap_patterns = xr.concat(patterns, "bootstrap")
    bootstrap_fractions = xr.concat(fractions, "bootstrap")
    confidence = float(args.eof_bootstrap_confidence)
    lower_probability = (1.0 - confidence) / 2.0
    upper_probability = 1.0 - lower_probability
    quantiles = [lower_probability, 0.5, upper_probability]
    pattern_quantiles = bootstrap_patterns.quantile(quantiles, dim="bootstrap")
    fraction_quantiles = bootstrap_fractions.quantile(quantiles, dim="bootstrap")
    baseline_nonzero = abs(baseline_pattern) > 1.0e-12
    sign_agreement = (
        ((bootstrap_patterns * baseline_pattern) > 0)
        .where(baseline_nonzero)
        .mean("bootstrap")
    )
    diagnostics = xr.Dataset(
        {
            "mode_pattern_bootstrap_lower": pattern_quantiles.sel(
                quantile=lower_probability, drop=True
            ),
            "mode_pattern_bootstrap_median": pattern_quantiles.sel(
                quantile=0.5, drop=True
            ),
            "mode_pattern_bootstrap_upper": pattern_quantiles.sel(
                quantile=upper_probability, drop=True
            ),
            "mode_pattern_sign_agreement": sign_agreement,
            "mode_pattern_sign_stable": (
                sign_agreement >= confidence
            ).astype("int8"),
            "mode_variance_fraction_bootstrap_lower": fraction_quantiles.sel(
                quantile=lower_probability, drop=True
            ),
            "mode_variance_fraction_bootstrap_median": fraction_quantiles.sel(
                quantile=0.5, drop=True
            ),
            "mode_variance_fraction_bootstrap_upper": fraction_quantiles.sel(
                quantile=upper_probability, drop=True
            ),
        }
    )
    if reference_patterns is not None:
        rmse_bootstrap = xr.DataArray(
            rmses, dims="bootstrap", coords={"bootstrap": bootstrap_patterns.bootstrap}
        )
        pcc_bootstrap = xr.DataArray(
            pccs, dims="bootstrap", coords={"bootstrap": bootstrap_patterns.bootstrap}
        )
        diagnostics["mode_pattern_rmse_bootstrap"] = rmse_bootstrap
        diagnostics["mode_pattern_pcc_bootstrap"] = pcc_bootstrap
        rmse_lower, rmse_median, rmse_upper = metric_interval(
            rmse_bootstrap, confidence
        )
        pcc_lower, pcc_median, pcc_upper = metric_interval(
            pcc_bootstrap, confidence
        )
        diagnostics.update(
            {
                "mode_pattern_rmse_bootstrap_lower": rmse_lower,
                "mode_pattern_rmse_bootstrap_median": rmse_median,
                "mode_pattern_rmse_bootstrap_upper": rmse_upper,
                "mode_pattern_pcc_bootstrap_lower": pcc_lower,
                "mode_pattern_pcc_bootstrap_median": pcc_median,
                "mode_pattern_pcc_bootstrap_upper": pcc_upper,
            }
        )
    diagnostics.attrs.update(
        eof_bootstrap_iterations=int(bootstrap_patterns.sizes["bootstrap"]),
        eof_bootstrap_confidence=confidence,
        eof_bootstrap_seed=int(args.eof_bootstrap_seed),
        eof_bootstrap_sign_alignment=(
            "Each bootstrap EOF is aligned to the full-sample EOF using "
            "cosine-latitude-weighted pattern correlation."
        ),
    )
    return diagnostics, bootstrap_patterns


# --------------------------------------------------------
# SST/ocean masking helpers
# --------------------------------------------------------
def get_lat_lon_names(
    obj: xr.Dataset | xr.DataArray,
) -> tuple[str, str]:
    """Return latitude and longitude coordinate names."""

    lat_candidates = (
        "lat",
        "latitude",
        "LAT",
        "Latitude",
        "nav_lat",
        "y",
    )

    lon_candidates = (
        "lon",
        "longitude",
        "LON",
        "Longitude",
        "nav_lon",
        "x",
    )

    lat_name = next(
        (name for name in lat_candidates if name in obj.coords),
        None,
    )

    lon_name = next(
        (name for name in lon_candidates if name in obj.coords),
        None,
    )

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not identify latitude/longitude coordinates. "
            f"Available coordinates: {list(obj.coords)}"
        )

    return lat_name, lon_name

def _as_bool_mask(mask: xr.DataArray, name: str = "mask") -> xr.DataArray:
    """Return a clean boolean mask with missing values treated as False."""
    result = mask.fillna(False).astype(bool)
    result.name = name
    return result


def align_spatial_mask(
    mask: xr.DataArray,
    target: xr.DataArray,
    *,
    mask_name: str = "mask",
) -> xr.DataArray:
    """Align a 2D lat/lon mask to a target field.

    The workflow normally builds all products on the same regular target grid,
    so this should be an exact coordinate alignment.  The nearest-neighbor
    fallback makes the mask robust to tiny coordinate-rounding differences after
    regridding or NetCDF read/write cycles while still failing for genuinely
    incompatible grids.
    """
    mask = _as_bool_mask(mask, mask_name)
    lat_name, lon_name = get_lat_lon_names(mask)
    target_lat_name, target_lon_name = get_lat_lon_names(target)

    if lat_name != "lat" or lon_name != "lon":
        mask = mask.rename({lat_name: "lat", lon_name: "lon"})
    if target_lat_name != "lat" or target_lon_name != "lon":
        target = target.rename({target_lat_name: "lat", target_lon_name: "lon"})

    mask = mask.sortby("lat").sortby("lon")
    target_lat = target["lat"]
    target_lon = target["lon"]

    try:
        aligned = mask.reindex(lat=target_lat, lon=target_lon)
        if bool(aligned.notnull().all().compute()):
            return _as_bool_mask(aligned, mask_name)
    except Exception:
        pass

    # Coordinate values may differ by roundoff.  Use a conservative tolerance:
    # smaller than half a grid interval, but never smaller than 1e-8 degrees.
    def _tolerance(values: xr.DataArray) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            return 1.0e-8
        spacing = float(np.nanmedian(np.abs(np.diff(np.sort(arr)))))
        return max(1.0e-8, spacing * 1.0e-3)

    lat_tol = _tolerance(target_lat)
    lon_tol = _tolerance(target_lon)
    aligned = mask.reindex(
        lat=target_lat,
        lon=target_lon,
        method="nearest",
        tolerance={"lat": lat_tol, "lon": lon_tol},
    )
    if not bool(aligned.notnull().all().compute()):
        missing = int(aligned.isnull().sum().compute())
        raise ValueError(
            f"Could not align {mask_name!r} to target grid; {missing} mask "
            "points are missing after nearest-neighbor alignment."
        )
    return _as_bool_mask(aligned, mask_name)


def generate_reference_ocean_mask(
    data: xr.DataArray,
    min_valid_fraction: float = 0.5,
    natural_earth_resolution: str = "110m",
) -> xr.DataArray:
    """Generate a robust ocean mask on the analysis grid for SST modes.

    A grid cell is retained only when it is frequently finite in the reference
    SST product and it is not classified as land by Natural Earth.  This avoids
    contaminating PDO/NPGO/AMO EOFs with finite land/ice-fill values introduced
    by preprocessing or regridding.
    """
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie between 0 and 1.")

    import regionmask

    lat_name, lon_name = get_lat_lon_names(data)

    non_spatial_dims = [
        dim for dim in data.dims
        if dim not in (lat_name, lon_name)
    ]

    if non_spatial_dims:
        valid_fraction = data.notnull().mean(non_spatial_dims)
    else:
        valid_fraction = data.notnull()

    obs_ocean = valid_fraction >= min_valid_fraction

    if natural_earth_resolution not in {"110m", "50m", "10m"}:
        raise ValueError(
            "natural_earth_resolution must be one of '110m', '50m', or '10m'."
        )
    # v5.0.0 is already available in the analysis environment.  Resolve only
    # the requested layer so a 110m run does not eagerly instantiate land_50
    # and emit its unrelated polar-coverage warning.
    natural_earth = regionmask.defined_regions.natural_earth_v5_0_0
    resolution_token = natural_earth_resolution.removesuffix("m")
    land = getattr(natural_earth, f"land_{resolution_token}")
    land_mask = land.mask(data[lon_name], data[lat_name])
    geometry_ocean = land_mask.isnull()

    ocean_mask = _as_bool_mask(obs_ocean & geometry_ocean, "ocean_mask")
    if lat_name != "lat" or lon_name != "lon":
        ocean_mask = ocean_mask.rename({lat_name: "lat", lon_name: "lon"})
    ocean_mask = ocean_mask.sortby("lat").sortby("lon")

    n_ocean = int(ocean_mask.sum().compute())
    n_total = int(ocean_mask.size)
    if n_ocean < 2:
        raise ValueError(
            "Reference SST ocean mask retained fewer than two grid cells. "
            "Check the input SST missing-value convention, coordinate names, "
            "and --sst-ocean-mask-min-valid-fraction."
        )
    LOG.info(
        "Reference SST ocean mask retained %s/%s grid cells (%.1f%%).",
        n_ocean,
        n_total,
        100.0 * n_ocean / max(n_total, 1),
    )
    return ocean_mask


def prepare_domain(
    data: xr.DataArray,
    settings: dict[str, object],
    remove_domain_mean: bool,
    valid_mask: xr.DataArray | None = None,
) -> xr.Dataset:
    working = data.where(np.isfinite(data))

    if not remove_domain_mean and str(settings["mode"]) in TEMPERATURE_MODES:
        belt = working.where((working.lat >= -60) & (working.lat <= 70), drop=True)
        belt_ds = bounded_dataset(belt, "mode_field")
        working = working - spatial_mean(belt_ds, "mode_field")

    dataset = bounded_dataset(select_mode_domain(working, settings), "mode_field")

    if valid_mask is not None:
        aligned_mask = align_spatial_mask(
            valid_mask,
            dataset["mode_field"],
            mask_name="valid_mask",
        )
        retained = int(aligned_mask.sum().compute())
        if retained < 2:
            raise ValueError(
                f"{settings['mode']} valid_mask retained fewer than two grid cells "
                "inside the mode domain."
            )
        dataset["mode_field"] = dataset["mode_field"].where(aligned_mask)

    if remove_domain_mean:
        dataset["mode_field"] = dataset["mode_field"] - spatial_mean(
            dataset, "mode_field"
        )

    return dataset


def prepare_reference_domain(
    sample: xr.DataArray,
    settings: dict[str, object],
    args: argparse.Namespace,
) -> tuple[xr.Dataset, xr.DataArray | None]:
    """Prepare a reference EOF domain and apply the SST ocean mask before EOFs."""
    prepared = prepare_domain(sample, settings, args.remove_domain_mean)
    ocean_mask = None

    if str(settings["mode"]) in TEMPERATURE_MODES:
        ocean_mask = generate_reference_ocean_mask(
            prepared["mode_field"],
            min_valid_fraction=float(
                getattr(args, "sst_ocean_mask_min_valid_fraction", 0.5)
            ),
            natural_earth_resolution=str(
                getattr(args, "sst_ocean_mask_resolution", "110m")
            ),
        )
        ocean_mask = align_spatial_mask(
            ocean_mask,
            prepared["mode_field"],
            mask_name="ocean_mask",
        )
        prepared["mode_field"] = prepared["mode_field"].where(ocean_mask)
        retained = int(prepared["mode_field"].notnull().any("time").sum().compute())
        if retained < 2:
            raise ValueError(
                f"{settings['mode']} reference ocean mask retained fewer than two "
                "valid grid cells after applying it to the SST anomalies."
            )

    prepared = eof_ready_dataset(prepared, "mode_field")
    return prepared, ocean_mask


def pcmdi_mode_reference(
    anomalies: xr.DataArray,
    settings: dict[str, object],
    args: argparse.Namespace,
) -> tuple[xr.Dataset, dict[int, dict[str, object]]]:
    pcs, patterns, fractions, north_diagnostics, bootstrap_diagnostics = (
        [], [], [], [], []
    )
    global_regression_patterns = []
    global_regression_pvalues = []
    global_regression_significant = []
    global_regression_r_values = []
    global_ocean_masks = []
    ocean_masks = []
    references: dict[int, dict[str, object]] = {}
    eof_number = int(settings["eof_number"])

    for month in range(1, 13):
        sample = anomalies.where(anomalies.time.dt.month == month, drop=True).load()
        if sample.sizes.get("time", 0) < 3:
            raise ValueError(
                f"{settings['mode']} reference EOF month {month:02d} has fewer "
                "than three samples. Check EOF reference years and input data."
            )

        prepared, ocean_mask = prepare_reference_domain(sample, settings, args)

        pattern, pc, fraction, reverse_sign, solver = eof_analysis_with_svd_fallback(
            str(settings["mode"]),
            prepared,
            "mode_field",
            eofn=eof_number,
            eofn_max=eof_number,
            EofScaling=args.eof_scaling,
        )

        pattern, pc, reverse_sign = enforce_mode_sign(
            str(settings["mode"]), pattern, pc, reverse_sign
        )

        pcs.append(pc)
        patterns.append(pattern.expand_dims(target_month=[month]))
        fractions.append(fraction.expand_dims(target_month=[month]))

        global_pattern = None
        global_ocean_mask = None
        if (
            WRITE_GLOBAL_TELECONNECTIONS_IN_INDEX_PRODUCTS
            and str(settings["mode"]) in TEMPERATURE_MODES
        ):
            global_field = global_sst_for_regression(sample)
            global_ocean_mask = generate_reference_ocean_mask(
                global_field,
                min_valid_fraction=float(
                    getattr(args, "sst_ocean_mask_min_valid_fraction", 0.5)
                ),
                natural_earth_resolution=str(
                    getattr(args, "sst_ocean_mask_resolution", "110m")
                ),
            )
            global_field = global_sst_for_regression(
                sample,
                ocean_mask=global_ocean_mask,
            )
            global_regression_ds = regression_pattern_from_projected_pc(
                global_field,
                pc,
                confidence_level=float(getattr(args, "regression_confidence", 0.95)),
            )
            global_pattern = global_regression_ds["mode_regression_pattern"]
            global_regression_patterns.append(
                global_pattern.expand_dims(target_month=[month])
            )
            global_regression_pvalues.append(
                global_regression_ds["mode_regression_pvalue"].expand_dims(
                    target_month=[month]
                )
            )
            global_regression_significant.append(
                global_regression_ds["mode_regression_significant"].expand_dims(
                    target_month=[month]
                )
            )
            global_regression_r_values.append(
                global_regression_ds["mode_regression_r"].expand_dims(
                    target_month=[month]
                )
            )
            global_ocean_masks.append(
                global_ocean_mask.expand_dims(target_month=[month]).astype("int8")
            )

        north_diagnostics.append(
            north_eigenvalue_diagnostics(solver, eof_number).expand_dims(
                target_month=[month]
            )
        )
        if ocean_mask is not None:
            ocean_masks.append(ocean_mask.expand_dims(target_month=[month]))

        bootstrap_patterns = None
        if args.eof_bootstrap_iterations:
            rng = np.random.default_rng(
                np.random.SeedSequence([args.eof_bootstrap_seed, month])
            )
            bootstrap_indices = rng.integers(
                0,
                prepared.sizes["time"],
                size=(args.eof_bootstrap_iterations, prepared.sizes["time"]),
            )
            bootstrap_ds, bootstrap_patterns = eof_bootstrap_diagnostics(
                prepared,
                pattern,
                settings,
                args,
                bootstrap_indices,
            )
            bootstrap_diagnostics.append(
                bootstrap_ds.expand_dims(target_month=[month])
            )

        valid_mask = prepared["mode_field"].notnull().all("time")
        if ocean_mask is not None:
            valid_mask = valid_mask & ocean_mask
        valid_mask = _as_bool_mask(valid_mask, "valid_mask")

        references[month] = {
            "solver": solver,
            "reverse_sign": reverse_sign,
            "valid_mask": valid_mask,
            "ocean_mask": ocean_mask,
            "global_ocean_mask": global_ocean_mask,
            "pattern": pattern,
            "global_pattern": global_pattern,
            "bootstrap_patterns": bootstrap_patterns,
        }

    result = xr.Dataset(
        {
            "mode_index": xr.concat(pcs, "time").sortby("time"),
            "mode_pattern": xr.concat(patterns, "target_month"),
            "mode_variance_fraction": xr.concat(fractions, "target_month"),
        }
    )

    result = xr.merge([result, xr.concat(north_diagnostics, "target_month")])

    if ocean_masks:
        result["mode_ocean_mask"] = xr.concat(ocean_masks, "target_month").astype("int8")
        result["mode_ocean_mask"].attrs.update(
            long_name="Reference ocean mask applied before SST EOF analysis",
            description=(
                "1=ocean grid cell retained for SST-mode EOF/reference projection; "
                "0=land or insufficient valid observed SST samples"
            ),
            min_valid_fraction=float(
                getattr(args, "sst_ocean_mask_min_valid_fraction", 0.5)
            ),
            natural_earth_resolution=str(
                getattr(args, "sst_ocean_mask_resolution", "110m")
            ),
        )

    if global_regression_patterns:
        result["mode_global_regression_pattern"] = xr.concat(
            global_regression_patterns, "target_month"
        )
        result["mode_global_regression_pvalue"] = xr.concat(
            global_regression_pvalues, "target_month"
        )
        result["mode_global_regression_significant"] = xr.concat(
            global_regression_significant, "target_month"
        ).astype("int8")
        result["mode_global_regression_r"] = xr.concat(
            global_regression_r_values, "target_month"
        )
        result["mode_global_ocean_mask"] = xr.concat(
            global_ocean_masks, "target_month"
        ).astype("int8")
        result["mode_global_regression_pattern"].attrs.update(
            long_name=(
                "Full-map SST regression pattern onto the observed EOF principal "
                "component"
            ),
            description=(
                "Computed for SST modes so PDO/AMO diagnostics can display global "
                "teleconnection maps while retaining regional EOF-domain PCs."
            ),
        )

    if bootstrap_diagnostics:
        result = xr.merge(
            [result, xr.concat(bootstrap_diagnostics, "target_month")]
        )

    return result, references

def target_month(valid_time: xr.DataArray, lead: int) -> int:
    values = np.asarray(valid_time.sel(L=lead).dt.month).ravel().astype(int)
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def regression_pattern_from_projected_pc(
    field: xr.DataArray,
    pc: xr.DataArray,
    *,
    confidence_level: float = 0.95,
) -> xr.Dataset:
    """Regress a gridded field onto a projected PC and return slope/p values.

    Parameters
    ----------
    field
        Prepared model anomalies with dimensions ``time``, ``lat``, ``lon``.
    pc
        Projected principal component with dimension ``time``. This should come
        from projecting the model anomalies onto the fixed observed EOF basis.
    confidence_level
        Confidence level used to create the integer significance mask.

    Notes
    -----
    The returned slope has physical field units per one PC unit. This is not a
    separately fitted model EOF. It is a model regression pattern associated
    with the observed-mode projected index, which is more stable for short S2D
    hindcasts than estimating EOFs from each lead/sample.
    """
    if "time" not in field.dims or "time" not in pc.dims:
        raise ValueError("field and pc must both have a 'time' dimension.")
    field, pc = xr.align(field, pc, join="inner")
    finite_time = np.isfinite(pc)
    if int(finite_time.sum()) < 3:
        empty = field.isel(time=0, drop=True) * np.nan
        return xr.Dataset(
            {
                "mode_regression_pattern": empty,
                "mode_regression_intercept": empty,
                "mode_regression_r": empty,
                "mode_regression_pvalue": empty,
                "mode_regression_significant": empty.astype("int8"),
            }
        )

    field = field.sel(time=pc.time.where(finite_time, drop=True))
    pc = pc.sel(time=field.time)
    n = int(pc.sizes["time"])
    pc_anom = pc - pc.mean("time")
    pc_var = (pc_anom**2).sum("time")
    if float(pc_var) <= 0 or not np.isfinite(float(pc_var)):
        empty = field.isel(time=0, drop=True) * np.nan
        return xr.Dataset(
            {
                "mode_regression_pattern": empty,
                "mode_regression_intercept": empty,
                "mode_regression_r": empty,
                "mode_regression_pvalue": empty,
                "mode_regression_significant": empty.astype("int8"),
            }
        )

    field_mean = field.mean("time", skipna=True)
    field_anom = field - field_mean
    slope = (field_anom * pc_anom).sum("time", skipna=True) / pc_var
    intercept = field_mean - slope * pc.mean("time")

    field_var = (field_anom**2).sum("time", skipna=True)
    covariance = (field_anom * pc_anom).sum("time", skipna=True)
    r = covariance / np.sqrt(field_var * pc_var)
    r = r.clip(min=-1.0, max=1.0)

    def _pvalue_from_r(r_values):
        from scipy import stats as scipy_stats

        r_values = np.asarray(r_values, dtype=float)
        out = np.full_like(r_values, np.nan, dtype=float)
        valid = np.isfinite(r_values) & (np.abs(r_values) < 1.0) & (n > 2)
        if np.any(valid):
            t_values = r_values[valid] * np.sqrt((n - 2) / (1.0 - r_values[valid] ** 2))
            out[valid] = 2.0 * scipy_stats.t.sf(np.abs(t_values), df=n - 2)
        out[np.isfinite(r_values) & (np.abs(r_values) >= 1.0) & (n > 2)] = 0.0
        return out

    pvalue = xr.apply_ufunc(
        _pvalue_from_r,
        r,
        input_core_dims=[[]],
        output_core_dims=[[]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    )
    alpha = 1.0 - float(confidence_level)
    significant = (pvalue < alpha).where(np.isfinite(pvalue), False).astype("int8")

    slope.name = "mode_regression_pattern"
    intercept.name = "mode_regression_intercept"
    r.name = "mode_regression_r"
    pvalue.name = "mode_regression_pvalue"
    significant.name = "mode_regression_significant"
    return xr.Dataset(
        {
            "mode_regression_pattern": slope,
            "mode_regression_intercept": intercept,
            "mode_regression_r": r,
            "mode_regression_pvalue": pvalue,
            "mode_regression_significant": significant,
        }
    )


def pcmdi_mode_model(
    anomalies: xr.DataArray,
    valid_time: xr.DataArray,
    references: dict[int, dict[str, object]],
    settings: dict[str, object],
    args: argparse.Namespace,
) -> xr.Dataset:
    """Compute model mode diagnostics.

    ``args.eof_strategy == 'fixed_obs_projection'`` avoids estimating EOFs from
    short model hindcast samples. It projects model anomalies onto the observed
    EOF basis and computes regression/significance patterns from that projected
    index. ``'conventional'`` preserves the original behavior and additionally
    fits model EOFs for each lead.
    """
    eof_strategy = getattr(args, "eof_strategy", "fixed_obs_projection")
    if eof_strategy not in {"fixed_obs_projection", "conventional"}:
        raise ValueError(f"Unsupported eof_strategy={eof_strategy!r}")

    cbf_pcs = []
    conventional_pcs = []
    patterns = []
    fractions = []
    months = []
    north_diagnostics = []
    bootstrap_diagnostics = []
    reference_rmses = []
    reference_pccs = []
    regression_patterns = []
    regression_pvalues = []
    regression_significant = []
    regression_r_values = []
    global_regression_patterns = []
    global_regression_pvalues = []
    global_regression_significant = []
    global_regression_r_values = []
    eof_number = int(settings["eof_number"])
    for lead_value in anomalies.L.values:
        lead = int(lead_value)
        month = target_month(valid_time, lead)
        months.append(month)
        lead_data = anomalies.sel(L=lead_value)
        sample = lead_data.stack(sample=("Y", "M")).transpose("sample", "lat", "lon")
        sample = sample.reset_index("sample", drop=True).rename(sample="time")
        sample = sample.assign_coords(time=np.arange(sample.sizes["time"])).load()
        reference = references[month]
        valid_mask = reference["valid_mask"]

        if settings["mode"] in TEMPERATURE_MODES:
            ocean_mask = reference.get("ocean_mask")
            if ocean_mask is not None:
                valid_mask = valid_mask & ocean_mask

        prepared = eof_ready_dataset(
            prepare_domain(
                sample,
                settings,
                args.remove_domain_mean,
                valid_mask=valid_mask,
            ),
            "mode_field",
        )
        cbf = gain_pseudo_pcs(
            reference["solver"],
            prepared["mode_field"],
            eofn=eof_number,
            reverse_sign=bool(reference["reverse_sign"]),
            EofScaling=args.eof_scaling,
        )
        shape, coords = (lead_data.sizes["Y"], lead_data.sizes["M"]), {
            "Y": lead_data.Y, "M": lead_data.M,
        }
        cbf_values = np.asarray(cbf.values)
        cbf_pcs.append(
            xr.DataArray(
                cbf_values.reshape(shape), dims=("Y", "M"), coords=coords
            ).expand_dims(L=[lead])
        )

        pc_time = xr.DataArray(
            cbf_values,
            dims="time",
            coords={"time": prepared["mode_field"].time},
            name="projected_pc",
        )
        regression_ds = regression_pattern_from_projected_pc(
            prepared["mode_field"],
            pc_time,
            confidence_level=float(getattr(args, "regression_confidence", 0.95)),
        )
        regression_pattern = regression_ds["mode_regression_pattern"]
        regression_patterns.append(regression_pattern.expand_dims(L=[lead]))
        regression_pvalues.append(regression_ds["mode_regression_pvalue"].expand_dims(L=[lead]))
        regression_significant.append(regression_ds["mode_regression_significant"].expand_dims(L=[lead]))
        regression_r_values.append(regression_ds["mode_regression_r"].expand_dims(L=[lead]))

        metric_pattern = regression_pattern
        metric_reference = reference["pattern"]
        if (
            WRITE_GLOBAL_TELECONNECTIONS_IN_INDEX_PRODUCTS
            and str(settings["mode"]) in TEMPERATURE_MODES
        ):
            global_field = global_sst_for_regression(
                sample,
                ocean_mask=reference.get("global_ocean_mask"),
            )
            global_regression_ds = regression_pattern_from_projected_pc(
                global_field,
                pc_time,
                confidence_level=float(getattr(args, "regression_confidence", 0.95)),
            )
            global_regression_pattern = global_regression_ds[
                "mode_regression_pattern"
            ]
            global_regression_patterns.append(
                global_regression_pattern.expand_dims(L=[lead])
            )
            global_regression_pvalues.append(
                global_regression_ds["mode_regression_pvalue"].expand_dims(L=[lead])
            )
            global_regression_significant.append(
                global_regression_ds["mode_regression_significant"].expand_dims(
                    L=[lead]
                )
            )
            global_regression_r_values.append(
                global_regression_ds["mode_regression_r"].expand_dims(L=[lead])
            )
            if reference.get("global_pattern") is not None:
                metric_pattern = global_regression_pattern
                metric_reference = reference["global_pattern"]

        reference_rmse, reference_pcc = cosine_weighted_pattern_metrics(
            metric_pattern, metric_reference
        )
        reference_rmses.append(reference_rmse)
        reference_pccs.append(reference_pcc)

        if eof_strategy == "conventional":
            pattern, pc, fraction, _, solver = eof_analysis_with_svd_fallback(
                str(settings["mode"]),
                prepared,
                "mode_field",
                eofn=eof_number,
                eofn_max=eof_number,
                EofScaling=args.eof_scaling,
            )
            pattern, pc, _ = enforce_mode_sign(str(settings["mode"]), pattern, pc)
            conventional_pcs.append(
                xr.DataArray(
                    pc.values.reshape(shape), dims=("Y", "M"), coords=coords
                ).expand_dims(L=[lead])
            )
            patterns.append(pattern.expand_dims(L=[lead]))
            fractions.append(fraction.expand_dims(L=[lead]))
            north_diagnostics.append(
                north_eigenvalue_diagnostics(solver, eof_number).expand_dims(L=[lead])
            )
            if args.eof_bootstrap_iterations:
                n_years = lead_data.sizes["Y"]
                n_members = lead_data.sizes["M"]
                rng = np.random.default_rng(
                    np.random.SeedSequence([args.eof_bootstrap_seed, lead])
                )
                bootstrap_indices = np.empty(
                    (
                        args.eof_bootstrap_iterations,
                        n_years * n_members,
                    ),
                    dtype=int,
                )
                for bootstrap in range(args.eof_bootstrap_iterations):
                    year_indices = rng.integers(0, n_years, size=n_years)
                    member_indices = rng.integers(
                        0, n_members, size=(n_years, n_members)
                    )
                    bootstrap_indices[bootstrap] = (
                        year_indices[:, None] * n_members + member_indices
                    ).ravel()
                reference_bootstrap_patterns = reference["bootstrap_patterns"]
                bootstrap_ds, _ = eof_bootstrap_diagnostics(
                    prepared,
                    pattern,
                    settings,
                    args,
                    bootstrap_indices,
                    reference_patterns=reference_bootstrap_patterns,
                )
                bootstrap_diagnostics.append(bootstrap_ds.expand_dims(L=[lead]))

    result_data = {
        "mode_index": xr.concat(cbf_pcs, "L").transpose("Y", "L", "M"),
        "mode_regression_pattern": xr.concat(regression_patterns, "L"),
        "mode_regression_pvalue": xr.concat(regression_pvalues, "L"),
        "mode_regression_significant": xr.concat(regression_significant, "L"),
        "mode_regression_r": xr.concat(regression_r_values, "L"),
        "target_month": xr.DataArray(months, dims="L", coords={"L": anomalies.L}),
        "mode_pattern_rmse_reference": xr.DataArray(
            reference_rmses, dims="L", coords={"L": anomalies.L}
        ),
        "mode_pattern_pcc_reference": xr.DataArray(
            reference_pccs, dims="L", coords={"L": anomalies.L}
        ),
    }

    if global_regression_patterns:
        result_data.update(
            {
                "mode_global_regression_pattern": xr.concat(
                    global_regression_patterns, "L"
                ),
                "mode_global_regression_pvalue": xr.concat(
                    global_regression_pvalues, "L"
                ),
                "mode_global_regression_significant": xr.concat(
                    global_regression_significant, "L"
                ).astype("int8"),
                "mode_global_regression_r": xr.concat(
                    global_regression_r_values, "L"
                ),
            }
        )

    if eof_strategy == "fixed_obs_projection":
        # Backward-compatible aliases for plotting scripts that expect mode_pattern.
        # These are regression patterns, not model EOFs.
        result_data["mode_pattern"] = result_data["mode_regression_pattern"]
        result_data["mode_variance_fraction"] = xr.full_like(
            result_data["target_month"].astype(float), np.nan
        )
    else:
        result_data.update(
            {
                "mode_index_conventional": xr.concat(conventional_pcs, "L").transpose("Y", "L", "M"),
                "mode_pattern_conventional": xr.concat(patterns, "L"),
                "mode_pattern": xr.concat(patterns, "L"),
                "mode_variance_fraction": xr.concat(fractions, "L"),
            }
        )

    result = xr.Dataset(result_data)
    if eof_strategy == "conventional" and north_diagnostics:
        result = xr.merge([result, xr.concat(north_diagnostics, "L")], compat="override")
    if bootstrap_diagnostics:
        result = xr.merge([result, xr.concat(bootstrap_diagnostics, "L")], compat="override")

    result.attrs.update(
        eof_strategy=eof_strategy,
        model_pattern_definition=(
            "regression of model anomalies onto projected observed-EOF PC"
            if eof_strategy == "fixed_obs_projection"
            else "conventional model EOF; regression pattern is also provided"
        ),
    )
    return result



def station_nao_model(data: xr.DataArray, definition: StationNaoDefinition) -> xr.DataArray:
    south = data.sel(lon=definition.south_lon % 360, lat=definition.south_lat, method="nearest")
    north = data.sel(lon=definition.north_lon % 360, lat=definition.north_lat, method="nearest")
    return (south / south.std(("Y", "M")) - north / north.std(("Y", "M"))).rename("nao_station")


def station_nao_obs(data: xr.DataArray, definition: StationNaoDefinition) -> xr.DataArray:
    south = data.sel(lon=definition.south_lon % 360, lat=definition.south_lat, method="nearest")
    north = data.sel(lon=definition.north_lon % 360, lat=definition.north_lat, method="nearest")
    return (
        south.groupby("time.month") / south.groupby("time.month").std("time")
        - north.groupby("time.month") / north.groupby("time.month").std("time")
    ).rename("nao_station")


def product_paths(
    outdir: Path,
    mode: str,
    source: str,
    init_month: int | None,
    grid_name: str,
    settings: dict[str, object],
    legacy: bool,
) -> tuple[Path, Path]:
    init_token = "" if init_month is None else f"_init{init_month:02d}"
    if source == "obs":
        source_dir = str(settings["obs_product"])
        source_name = source_dir.lower()
    elif source == "smyle":
        source_dir = "CESM-SMYLE"
        source_name = "smyle"
    elif source == "nmme":
        source_dir = "NMME"
        source_name = "nmme"
    else:
        source_dir = source
        source_name = source
    source_root = outdir / source_dir / "modes_variability"
    field_name = f"{source_name}{init_token}_{str(settings['field']).lower()}_{settings['frequency']}_{grid_name}.nc"
    field_path = source_root / "fields" / field_name
    if legacy:
        index_name = "era5_nao_reference.nc" if source == "obs" else f"{source}{init_token}_nao.nc"
        return field_path, source_root / "indices" / index_name
    index_name = (
        f"{source_name}_{mode.lower()}_reference.nc"
        if source == "obs"
        else f"{source}{init_token}_{mode.lower()}.nc"
    )
    return field_path, source_root / "modes" / mode.lower() / "indices" / index_name


def model_field_dataset(
    source: str,
    init_month: int,
    settings: dict[str, object],
    args: argparse.Namespace,
    destination_grid: xr.Dataset,
) -> xr.Dataset:
    years = list(range(args.start_year, args.end_year + 1))
    archive_field, field = str(settings["archive_field"]), str(settings["field"])
    if source == "nmme":
        model_fields: list[xr.DataArray] = []
        model_failures: list[str] = []
        valid_time = None
        models = [
            str(model).strip()
            for model in getattr(args, "nmme_models", [])
            if str(model).strip()
        ]
        for model_number, model in enumerate(models, start=1):
            LOG.info(
                "NMME init %02d: loading model %d/%d: %s",
                init_month,
                model_number,
                len(models),
                model,
            )
            model_args = copy.copy(args)
            model_args.nmme_models = [model]
            processed = None
            try:
                processed = nmme_field_dataset(init_month, settings, model_args)
                if settings["frequency"] == "seasonal":
                    processed_for_field = cal.mon_to_seas_dask(processed)
                else:
                    processed_for_field = processed
                selected = select_field(processed_for_field, field)
                processed_for_field, selected = drop_empty_leads(
                    processed_for_field,
                    selected,
                )
                data = convert_units(selected, field).rename(field)
                source_dataset = data.to_dataset()
                regridder = regrid.make_regridder(
                    source_dataset,
                    destination_grid,
                    method=args.regrid_method,
                    periodic=not args.no_periodic,
                )
                model_regridded = regridder(source_dataset)[field]
                model_regridded.attrs.update(data.attrs)
                model_regridded = common_year_subset(
                    model_regridded,
                    years,
                ).astype("float32")
                LOG.info(
                    "NMME init %02d: materializing %s on the target grid",
                    init_month,
                    model,
                )
                model_regridded = model_regridded.compute().chunk(
                    {
                        "Y": min(5, model_regridded.sizes.get("Y", 1)),
                        "L": -1,
                        "M": 1,
                        "lat": min(36, model_regridded.sizes.get("lat", 1)),
                        "lon": min(72, model_regridded.sizes.get("lon", 1)),
                    }
                )
                if valid_time is None:
                    valid_time = common_year_subset(
                        processed_for_field["time"],
                        years,
                    ).load()
                model_fields.append(model_regridded)
            except Exception as error:
                model_failures.append(f"{model}: {error}")
                LOG.warning(
                    "NMME init %02d: skipping %s after processing failure: %s",
                    init_month,
                    model,
                    error,
                )
            finally:
                if processed is not None:
                    processed.close()

        if not model_fields or valid_time is None:
            raise ValueError(
                "No NMME model fields could be prepared: "
                + "; ".join(model_failures)
            )
        if model_failures:
            LOG.warning(
                "NMME init %02d completed with %d skipped model(s): %s",
                init_month,
                len(model_failures),
                "; ".join(model_failures),
            )
        data = xr.concat(
            model_fields,
            dim="M",
            join="outer",
            combine_attrs="drop_conflicts",
        )
        data.attrs.update(model_fields[0].attrs)
        anomalies, climatology = stats.remove_drift(
            data,
            valid_time,
            args.clim_start,
            args.clim_end,
        )
        return xr.Dataset(
            {
                field: data,
                f"{field}_anom": anomalies.rename(f"{field}_anom"),
                f"{field}_drift_climatology": climatology.rename(
                    f"{field}_drift_climatology"
                ),
                "valid_time": valid_time,
            }
        )

    if source == "e3sm":
        raw = e3sm_access.get_monthly_data(
            data_dir=args.e3sm_data_dir,
            case_prefix=args.e3sm_case_prefix,
            members=[f"EN{i:02d}" for i in range(args.e3sm_nens)],
            init_tags=e3sm_access.build_init_tags(years, init_month),
            field=archive_field,
            nlead=args.monthly_nlead,
            chunks={},
            realm="atm",
            grid=args.e3sm_grid,
            freq="monthly",
            ts_split="2yr",
            require_all_members=True,
            verify_field_name=True,
            verify_coverage=True,
            engine=args.e3sm_engine,
        )
        processed = cal.mon_to_seas_dask(raw) if settings["frequency"] == "seasonal" else raw
    elif source == "smyle":
        processed = smyle_access.load_benchmark(
            field=archive_field,
            init_month=init_month,
            benchmark_dir=args.smyle_benchmark_dir,
            nens=args.smyle_nens,
            nlead=args.monthly_nlead,
            freq="seas" if settings["frequency"] == "seasonal" else "mon",
            chunks={"Y": 3, "L": -1, "M": 2, "lat": 96, "lon": 144},
        )
    else:
        raise ValueError(f"Unsupported model source: {source}")
    selected = select_field(processed, field)
    processed, selected = drop_empty_leads(processed, selected)
    if field == "SST" and source in {"e3sm", "smyle"}:
        if source == "e3sm":
            source_id = f"E3SM:{args.e3sm_cache_tag or args.e3sm_case_prefix}"
            land_mask_path = (
                Path(args.outdir)
                / str(args.e3sm_cache_tag or "e3sm")
                / "fixed"
                / "sftlf.E3SM.nc"
            )
        else:
            source_id = "CESM-SMYLE"
            land_mask_path = (
                Path(args.outdir)
                / "CESM-SMYLE"
                / "fixed"
                / "sftlf.CESM-SMYLE.nc"
            )
        data, _ = sst_utils.prepare_sst(
            selected,
            apply_land_mask=bool(args.model_sst_land_mask),
            land_mask_path=land_mask_path,
            source=source_id,
        )
        data = data.rename(field)
    else:
        data = convert_units(selected, field).rename(field)
    source_dataset = data.to_dataset()
    regridder = regrid.make_regridder(
        source_dataset,
        destination_grid,
        method=args.regrid_method,
        periodic=not args.no_periodic,
    )
    regridded = regridder(source_dataset)[field]
    regridded.attrs.update(data.attrs)
    data = common_year_subset(regridded, years)
    valid_time = common_year_subset(processed["time"], years).load()
    anomalies, climatology = stats.remove_drift(data, valid_time, args.clim_start, args.clim_end)
    return xr.Dataset(
        {
            field: data,
            f"{field}_anom": anomalies.rename(f"{field}_anom"),
            f"{field}_drift_climatology": climatology.rename(f"{field}_drift_climatology"),
            "valid_time": valid_time,
        }
    )


def obs_field_dataset(
    settings: dict[str, object],
    args: argparse.Namespace,
    destination_grid: xr.Dataset,
) -> xr.Dataset:
    field = str(settings["field"])
    product, obs_var = (
        (args.psl_obs_product, args.psl_obs_var)
        if field == "PSL"
        else (args.ts_obs_product, args.ts_obs_var)
    )
    monthly = obs_access.get_monthly_data(
        obs_dir=args.obs_dir,
        field=field,
        field_map={field: obs_var},
        product=product,
        start_year=args.obs_start_year,
        end_year=args.obs_end_year,
        chunks="auto",
        verbose=True,
    )
    data = convert_units(monthly[obs_var], field)
    if field == "SST":
        data = xr.where(data.notnull() & (data < -2), -1.8, data)
        data.attrs.update(monthly[obs_var].attrs)
        data.attrs["units"] = "degC"
    if settings["frequency"] == "seasonal":
        data = obs_access.mon_to_seas_obs(
            data.to_dataset(name=obs_var), var=obs_var, field_map={field: obs_var}
        )
    data = data.rename(field)
    source_dataset = data.to_dataset()
    regridder = regrid.make_regridder(
        source_dataset,
        destination_grid,
        method=args.regrid_method,
        periodic=not args.no_periodic,
    )
    regridded = regridder(source_dataset)[field]
    regridded.attrs.update(data.attrs)
    data = regridded
    if field == "SST":
        data = xr.where(data.notnull() & (data < -2), -1.8, data)
        data.attrs.update(regridded.attrs)
    climatology = data.sel(time=slice(str(args.clim_start), str(args.clim_end))).groupby("time.month").mean("time")
    anomalies = (data.groupby("time.month") - climatology).rename(f"{field}_anom")
    return xr.Dataset({field: data, f"{field}_anom": anomalies})
