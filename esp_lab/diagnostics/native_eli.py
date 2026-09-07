"""Native MPAS-Ocean Equatorial Longitude Index (ELI) calculation and workflows."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

import cftime
import numpy as np
import xarray as xr

from esp_lab.diagnostics.sst_index import (
    ELI_ALGORITHM_VERSION,
    ELI_LAT_MAX,
    ELI_LAT_MIN,
    ELI_LON_MAX,
    ELI_LON_MIN,
    TC_LAT_HALF,
)
from esp_lab.utils.netcdf_utils import atomic_to_netcdf
from esp_lab.utils.sst_utils import (
    SST_PREPROCESSING_VERSION,
    mask_nonphysical_sst,
    normalize_sst_to_degc,
)

LOG = logging.getLogger(__name__)

DEFAULT_MPAS_MESH_FILE = Path(
    "/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/IcoswISC30E3r5/"
    "mpaso.IcoswISC30E3r5.rstFromG-chrysalis.20231121.nc"
)
DEFAULT_RAW_SIMULATION_DIR = Path("/global/cfs/cdirs/e3smdata/simulations/S2S2D")
OCN_HIST_PATTERN = "*mpaso.hist.am.timeSeriesStatsMonthly.*.nc"
OCN_SST_VAR = "timeMonthly_avg_activeTracers_temperature"


def load_mpas_mesh_geometry(
    mesh_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cell latitude (deg N), longitude (deg E in [0, 360)), and area (m^2) from an MPAS mesh file."""
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"MPAS mesh file not found: {mesh_path}")

    with xr.open_dataset(mesh_path) as mesh:
        for mesh_var in ["latCell", "lonCell", "areaCell"]:
            if mesh_var not in mesh:
                raise KeyError(f"Required mesh variable {mesh_var!r} is missing from {mesh_path}")
        lat = mesh["latCell"].values * 180.0 / np.pi
        lon = mesh["lonCell"].values * 180.0 / np.pi
        area = mesh["areaCell"].values

    lon = np.mod(lon, 360.0)

    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("Mesh latitude/longitude contains non-finite values.")
    if not (np.isfinite(area).all() and np.all(area > 0.0)):
        raise ValueError("Mesh cell areas must be finite and strictly positive.")

    return lat, lon, area


def build_eli_mesh_subsets(
    lat: np.ndarray,
    lon: np.ndarray,
    area: np.ndarray,
    *,
    eli_lat_min: float = ELI_LAT_MIN,
    eli_lat_max: float = ELI_LAT_MAX,
    eli_lon_min: float = ELI_LON_MIN,
    eli_lon_max: float = ELI_LON_MAX,
    tc_lat_half: float = TC_LAT_HALF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build cell index arrays for the ELI region and the tropical reference band."""
    region_eq = (
        (lat >= eli_lat_min)
        & (lat <= eli_lat_max)
        & (lon >= eli_lon_min)
        & (lon <= eli_lon_max)
    )
    region_tropics = (lat >= -tc_lat_half) & (lat <= tc_lat_half)

    if not np.any(region_eq):
        raise ValueError("Equatorial Pacific mask selected zero mesh cells; check ELI bounds.")
    if not np.any(region_tropics):
        raise ValueError("Tropical reference mask selected zero mesh cells; check TC_LAT_HALF.")

    idx_union = np.flatnonzero(region_eq | region_tropics)
    eq_on_union = region_eq[idx_union]
    tropics_on_union = region_tropics[idx_union]

    lon_eq = lon[idx_union][eq_on_union].astype(np.float32)
    area_eq = area[idx_union][eq_on_union].astype(np.float64)
    area_tr = area[idx_union][tropics_on_union].astype(np.float64)

    return idx_union, eq_on_union, tropics_on_union, lon_eq, area_eq, area_tr


def weighted_mean_skipna_1d(values: np.ndarray, weights: np.ndarray) -> float:
    """Area-weighted mean for a 1D slice ignoring non-finite values."""
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.nan
    denom = np.sum(weights[finite])
    if denom <= 0.0:
        return np.nan
    return float(np.sum(values[finite] * weights[finite]) / denom)


def read_surface_sst_on_cells(
    path: Path | str,
    idx_union: np.ndarray,
    *,
    var_name: str = OCN_SST_VAR,
) -> np.ndarray:
    """Read one monthly MPAS-Ocean file on the specified cell indices only."""
    path = Path(path)
    with xr.open_dataset(path, decode_times=False) as ds:
        if var_name not in ds:
            raise KeyError(f"{var_name!r} not found in {path}")
        sst_da = ds[var_name]
        if "nVertLevels" in sst_da.dims:
            sst_da = sst_da.isel(nVertLevels=0)
        if "Time" in sst_da.dims:
            sst_da = sst_da.isel(Time=0)
        if "nCells" not in sst_da.dims:
            raise ValueError(f"{var_name!r} must have nCells dimension; found {sst_da.dims}")

        sst_da = normalize_sst_to_degc(sst_da)
        sst_da = mask_nonphysical_sst(sst_da)
        sst = sst_da.isel(nCells=idx_union).astype("float32").load().values

    if sst.ndim != 1:
        raise ValueError(f"Expected 1-D SST after slicing {path}, got shape {sst.shape}")
    return sst


def eli_from_sst_on_union(
    sst: np.ndarray,
    eq_on_union: np.ndarray,
    tropics_on_union: np.ndarray,
    lon_eq: np.ndarray,
    area_eq: np.ndarray,
    area_tr: np.ndarray,
) -> np.float32:
    """Compute one ELI centroid longitude from SST subsetted to ``idx_union``."""
    sst_eq = sst[eq_on_union]
    sst_tr = sst[tropics_on_union]
    tc = weighted_mean_skipna_1d(sst_tr, area_tr)
    if not np.isfinite(tc):
        return np.float32(np.nan)
    warm = np.isfinite(sst_eq) & (sst_eq > tc)
    denominator = np.sum(area_eq[warm])
    if denominator <= 0.0:
        return np.float32(np.nan)
    return np.float32(np.sum(area_eq[warm] * lon_eq[warm]) / denominator)


def compute_native_eli_member(
    hist_dir: Path | str,
    pattern: str,
    idx_union: np.ndarray,
    eq_on_union: np.ndarray,
    tropics_on_union: np.ndarray,
    lon_eq: np.ndarray,
    area_eq: np.ndarray,
    area_tr: np.ndarray,
    nlead: int,
    seasonal_lead_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute monthly and seasonal ELI for one ensemble member across lead months."""
    hist_dir = Path(hist_dir)
    files = sorted(hist_dir.glob(pattern))
    if not files:
        return (
            np.full(nlead, np.nan, dtype=np.float32),
            np.full(len(seasonal_lead_indices), np.nan, dtype=np.float32),
        )

    sst_by_lead = np.full((nlead, len(idx_union)), np.nan, dtype=np.float32)
    for lead_idx, path in enumerate(files[:nlead]):
        sst_by_lead[lead_idx] = read_surface_sst_on_cells(path, idx_union)

    monthly = np.array(
        [
            eli_from_sst_on_union(
                sst,
                eq_on_union=eq_on_union,
                tropics_on_union=tropics_on_union,
                lon_eq=lon_eq,
                area_eq=area_eq,
                area_tr=area_tr,
            )
            for sst in sst_by_lead
        ],
        dtype=np.float32,
    )

    seasonal = np.full(len(seasonal_lead_indices), np.nan, dtype=np.float32)
    for season_idx, center_idx in enumerate(seasonal_lead_indices):
        if center_idx == 0 or center_idx + 1 >= nlead:
            continue
        window = sst_by_lead[center_idx - 1 : center_idx + 2]
        complete = np.all(np.isfinite(window), axis=0)
        seasonal_sst = np.full(window.shape[1], np.nan, dtype=np.float32)
        seasonal_sst[complete] = window[:, complete].mean(axis=0)
        seasonal[season_idx] = eli_from_sst_on_union(
            seasonal_sst,
            eq_on_union=eq_on_union,
            tropics_on_union=tropics_on_union,
            lon_eq=lon_eq,
            area_eq=area_eq,
            area_tr=area_tr,
        )

    return monthly, seasonal


def _compute_case_member_worker(task: tuple) -> tuple:
    (
        yi,
        mi,
        case,
        member,
        hist_dir,
        pattern,
        idx_union,
        eq_on_union,
        tropics_on_union,
        lon_eq,
        area_eq,
        area_tr,
        nlead,
        seasonal_lead_indices,
    ) = task
    try:
        monthly, seasonal = compute_native_eli_member(
            hist_dir,
            pattern,
            idx_union,
            eq_on_union,
            tropics_on_union,
            lon_eq,
            area_eq,
            area_tr,
            nlead,
            seasonal_lead_indices,
        )
        return yi, mi, case, member, monthly, seasonal, None
    except Exception as exc:
        return yi, mi, case, member, None, None, exc


def native_eli_output_is_current(path: Path | str, expected: Mapping[str, object]) -> bool:
    """Check whether a cached native ELI NetCDF file has matching attributes."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path, decode_times=False) as cached:
            return all(cached.attrs.get(key) == value for key, value in expected.items())
    except Exception:
        return False


def build_native_eli_datasets(
    eli_all: np.ndarray,
    eli_all_seas: np.ndarray,
    lead_years: Sequence[int],
    init_month: int,
    members: Sequence[str],
    nlead: int,
    seasonal_lead_indices: np.ndarray,
    attrs_common: Mapping[str, object],
) -> tuple[xr.Dataset, xr.Dataset]:
    """Construct monthly and seasonal xarray Datasets with standard metadata and coordinates."""
    y_coords = np.array(lead_years, dtype=np.int32)
    l_coords = np.arange(1, nlead + 1, dtype=np.int32)
    m_coords = np.arange(len(members), dtype=np.int32)

    ds_mon = xr.Dataset(
        {
            "eli": xr.DataArray(
                eli_all,
                dims=("Y", "L", "M"),
                coords={"Y": y_coords, "L": l_coords, "M": m_coords},
                attrs={
                    "long_name": "Equatorial Longitude Index",
                    "units": "degrees_east",
                    "description": (
                        "Area-weighted centroid longitude of warm SST cells "
                        "(SST > tropical-mean SST) in the equatorial Pacific "
                        f"(lat {attrs_common.get('eli_lat_min', ELI_LAT_MIN)}-{attrs_common.get('eli_lat_max', ELI_LAT_MAX)} deg, "
                        f"lon {attrs_common.get('eli_lon_min', ELI_LON_MIN)}-{attrs_common.get('eli_lon_max', ELI_LON_MAX)} deg).  "
                        f"Tc reference band: +/-{attrs_common.get('tc_lat_half', TC_LAT_HALF)} deg."
                    ),
                },
            )
        }
    )

    ds_mon["Y"].attrs = {"long_name": "initialization year", "units": "year"}
    ds_mon["L"].attrs = {"long_name": "forecast lead month", "units": "months"}
    ds_mon["M"].attrs = {"long_name": "ensemble member index"}

    time_values = np.array(
        [
            [
                cftime.DatetimeNoLeap(
                    year + (init_month - 1 + lead_idx) // 12,
                    (init_month - 1 + lead_idx) % 12 + 1,
                    15,
                )
                for lead_idx in range(nlead)
            ]
            for year in lead_years
        ],
        dtype=object,
    )
    ds_mon["time"] = xr.DataArray(time_values, dims=("Y", "L"))
    ds_mon["time"].attrs = {"long_name": "forecast valid time"}
    ds_mon["member_id"] = xr.DataArray(
        np.array(members, dtype="U5"),
        dims="M",
        attrs={"long_name": "ensemble member label"},
    )
    ds_mon.attrs = {**attrs_common, "frequency": "monthly"}

    seasonal_leads = seasonal_lead_indices.astype(np.int32) + 1
    ds_seas = ds_mon.isel(L=seasonal_lead_indices).copy()
    ds_seas["eli"] = xr.DataArray(
        eli_all_seas,
        dims=("Y", "L", "M"),
        coords={
            "Y": ds_mon["Y"],
            "L": seasonal_leads,
            "M": ds_mon["M"],
        },
        attrs=ds_mon["eli"].attrs,
    )
    ds_seas.attrs = {
        **attrs_common,
        "frequency": "seasonal",
        "seasonal_aggregation": (
            "Three-month mean native SST centered on Jan/Apr/Jul/Oct, "
            "followed by ELI centroid calculation"
        ),
    }

    return ds_mon, ds_seas


def process_native_eli_case(
    *,
    case_prefix: str,
    data_dir: Path | str,
    outdir: Path | str,
    mesh_path: Path | str = DEFAULT_MPAS_MESH_FILE,
    cache_tag: str | None = None,
    display_name: str | None = None,
    init_months: Sequence[int] = (5, 11),
    year_start: int = 1980,
    year_end: int = 2018,
    nlead: int = 24,
    nens: int = 10,
    workers: int = 1,
    force: bool = False,
    eli_lat_min: float = ELI_LAT_MIN,
    eli_lat_max: float = ELI_LAT_MAX,
    eli_lon_min: float = ELI_LON_MIN,
    eli_lon_max: float = ELI_LON_MAX,
    tc_lat_half: float = TC_LAT_HALF,
) -> list[Path]:
    """Compute and save monthly and seasonal native MPAS-Ocean ELI NetCDF files for one case."""
    data_dir = Path(data_dir)
    outdir = Path(outdir)
    mesh_path = Path(mesh_path)
    outdir.mkdir(parents=True, exist_ok=True)

    lat, lon, area = load_mpas_mesh_geometry(mesh_path)
    (
        idx_union,
        eq_on_union,
        tropics_on_union,
        lon_eq,
        area_eq,
        area_tr,
    ) = build_eli_mesh_subsets(
        lat,
        lon,
        area,
        eli_lat_min=eli_lat_min,
        eli_lat_max=eli_lat_max,
        eli_lon_min=eli_lon_min,
        eli_lon_max=eli_lon_max,
        tc_lat_half=tc_lat_half,
    )

    lead_years = list(range(year_start, year_end + 1))
    members = [f"EN{i:02d}" for i in range(nens)]
    written_paths: list[Path] = []

    for init_month in init_months:
        outfile_mon = outdir / f"E3SMLE{init_month:02d}_ELI_native_N{nens:02d}_M{nlead:02d}.nc"
        outfile_seas = outdir / f"E3SMLE{init_month:02d}_ELI_native_N{nens:02d}_M{nlead:02d}_seas.nc"

        target_months = (init_month - 1 + np.arange(nlead)) % 12 + 1
        seasonal_lead_indices = np.flatnonzero(np.isin(target_months, [1, 4, 7, 10]))

        expected_common = {
            "case_prefix": case_prefix,
            "init_month": int(init_month),
            "start_year": int(year_start),
            "end_year": int(year_end),
            "run_nens": int(nens),
            "eli_lat_min": eli_lat_min,
            "eli_lat_max": eli_lat_max,
            "eli_lon_min": eli_lon_min,
            "eli_lon_max": eli_lon_max,
            "tc_lat_half": tc_lat_half,
            "sst_variable": OCN_SST_VAR,
            "sst_preprocessing_version": int(SST_PREPROCESSING_VERSION),
            "eli_native_algorithm_version": int(ELI_ALGORITHM_VERSION),
        }

        if not force and (
            native_eli_output_is_current(outfile_mon, {**expected_common, "frequency": "monthly"})
            and native_eli_output_is_current(outfile_seas, {**expected_common, "frequency": "seasonal"})
        ):
            LOG.info("Skipping %s init %02d: current native ELI files exist", case_prefix, init_month)
            written_paths.extend([outfile_mon, outfile_seas])
            continue

        LOG.info("Computing native ELI for %s init %02d", case_prefix, init_month)

        eli_all = np.full((len(lead_years), nlead, nens), np.nan, dtype=np.float32)
        eli_all_seas = np.full((len(lead_years), len(seasonal_lead_indices), nens), np.nan, dtype=np.float32)

        cases_for_month = [f"{case_prefix}_{year}{init_month:02d}0100" for year in lead_years]
        n_workers = max(1, min(int(workers), nens))

        for yi, (year, case) in enumerate(zip(lead_years, cases_for_month)):
            case_dir = data_dir / case
            if not case_dir.is_dir():
                LOG.warning("Case directory not found: %s", case_dir)
                continue

            tasks = [
                (
                    yi,
                    mi,
                    case,
                    member,
                    case_dir / member / "archive" / "ocn" / "hist",
                    OCN_HIST_PATTERN,
                    idx_union,
                    eq_on_union,
                    tropics_on_union,
                    lon_eq,
                    area_eq,
                    area_tr,
                    nlead,
                    seasonal_lead_indices,
                )
                for mi, member in enumerate(members)
            ]

            if n_workers == 1:
                results_iter = map(_compute_case_member_worker, tasks)
                pool = None
            else:
                pool = ThreadPoolExecutor(max_workers=n_workers)
                futures = [pool.submit(_compute_case_member_worker, task) for task in tasks]
                results_iter = (future.result() for future in as_completed(futures))

            try:
                for _, mi, case_name, member, monthly, seasonal, exc in results_iter:
                    if exc is not None:
                        LOG.error("Error in %s/%s: %s", case_name, member, exc)
                        continue
                    eli_all[yi, :, mi] = monthly
                    eli_all_seas[yi, :, mi] = seasonal
            finally:
                if pool is not None:
                    pool.shutdown(wait=True)

        attrs_common = {
            **expected_common,
            "case_key": display_name or case_prefix,
            "display_name": display_name or case_prefix,
            "cache_tag": cache_tag or "",
            "data_dir": str(data_dir),
            "case_nens": int(nens),
            "nworkers": int(workers),
            "sst_units": "degC",
            "source_grid": "native MPAS-Ocean",
        }

        ds_mon, ds_seas = build_native_eli_datasets(
            eli_all,
            eli_all_seas,
            lead_years=lead_years,
            init_month=init_month,
            members=members,
            nlead=nlead,
            seasonal_lead_indices=seasonal_lead_indices,
            attrs_common=attrs_common,
        )

        encoding = {"eli": {"zlib": True, "complevel": 1, "dtype": "float32"}}
        atomic_to_netcdf(ds_mon, outfile_mon, encoding=encoding)
        atomic_to_netcdf(ds_seas, outfile_seas, encoding=encoding)
        written_paths.extend([outfile_mon, outfile_seas])

    return written_paths
