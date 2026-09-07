#!/usr/bin/env python3
"""Generate and save regional SST indices for E3SM, CESM-SMYLE, and observations."""

from __future__ import annotations

import argparse
import os

# Fix GDAL/PROJ library paths for regionmask/pyogrio
conda_prefix = os.environ.get("CONDA_PREFIX", "/global/homes/z/zhan391/.conda/envs/e3sm_analysis")
if conda_prefix:
    gdal_path = os.path.join(conda_prefix, "share", "gdal")
    proj_path = os.path.join(conda_prefix, "share", "proj")
    if os.path.exists(gdal_path):
        os.environ["GDAL_DATA"] = gdal_path
    if os.path.exists(proj_path):
        os.environ["PROJ_LIB"] = proj_path
import logging
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import xarray as xr
import dask

# Insert repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esp_lab import data_access_e3sm as data_access
from esp_lab import data_access_obs as obs_access
from esp_lab import data_access_cesm_smyle as smyle_access
from esp_lab import stats
from esp_lab.utils import spatial_utils as spatial
from esp_lab.utils import calendar_utils as cal
from esp_lab.utils.sst_utils import SST_PREPROCESSING_VERSION, prepare_sst
from esp_lab.diagnostics import (
    DEFAULT_CLIMATOLOGY_END_YEAR,
    DEFAULT_CLIMATOLOGY_START_YEAR,
    S2DDiagnostics,
    S2DConfig,
)
from esp_lab.diagnostics.regional import compute_weights, compute_regional_mean
from esp_lab.diagnostics.sst_index import (
    REGIONS,
    VALID_REGIONS,
    compute_eli_latlon_sst,
    compute_model_anom,
    compute_model_std,
    compute_obs_anom,
    compute_obs_std,
    derive_indices,
    required_base_regions,
)

LOG = logging.getLogger(__name__)
SST_INDEX_OUTPUT_VERSION = 2
S2D_DIAG_ROOT = Path("/global/cfs/cdirs/e3sm/S2S2D/s2d_diag")
E3SMLE_DIAG_DIR = S2D_DIAG_ROOT
CESM_SMYLE_DIAG_DIR = S2D_DIAG_ROOT / "CESM-SMYLE"
HADISST2_DIAG_DIR = S2D_DIAG_ROOT / "HadISST2"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=["obs", "e3sm", "smyle"],
        choices=["obs", "e3sm", "smyle"],
        help="Data sources to process. Default: obs e3sm smyle",
    )
    p.add_argument(
        "--regions",
        nargs="+",
        default=VALID_REGIONS,
        help="List of regions/indices to compute. Default is all 13.",
    )
    p.add_argument(
        "--outdir",
        default=str(E3SMLE_DIAG_DIR),
        help=f"Output root for E3SM case outputs. Default: {E3SMLE_DIAG_DIR}",
    )
    p.add_argument(
        "--e3sm-data-dir",
        default="/global/cfs/cdirs/e3sm/S2S2D/post_process",
        help="Root directory containing post-processed E3SM hindcast case directories.",
    )
    p.add_argument(
        "--e3sm-case-prefix",
        default="WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL",
        help="E3SM case prefix before the initialization timestamp.",
    )
    p.add_argument(
        "--e3sm-cache-tag",
        default=None,
        help="Optional E3SM case tag used as a subdirectory under --outdir.",
    )
    p.add_argument(
        "--e3sm-display-name",
        default=None,
        help="Optional display name stored in E3SM output attributes.",
    )
    p.add_argument(
        "--smyle-outdir",
        default=str(CESM_SMYLE_DIAG_DIR),
        help=f"Output directory for CESM-SMYLE. Default: {CESM_SMYLE_DIAG_DIR}",
    )
    p.add_argument(
        "--obs-outdir",
        default=str(HADISST2_DIAG_DIR / "sst_index" / "timeseries"),
        help="Output directory for HadISST2 SST index files.",
    )
    p.add_argument(
        "--init-months",
        nargs="+",
        type=int,
        default=[5, 11],
        help="Initialization months to process. Default: 5 11",
    )
    p.add_argument(
        "--year-start",
        type=int,
        default=1980,
        help="Start year. Default: 1980",
    )
    p.add_argument(
        "--year-end",
        type=int,
        default=2018,
        help="End year. Default: 2018",
    )
    p.add_argument(
        "--climy0",
        type=int,
        default=DEFAULT_CLIMATOLOGY_START_YEAR,
        help=f"Climatology start year. Default: {DEFAULT_CLIMATOLOGY_START_YEAR}",
    )
    p.add_argument(
        "--climy1",
        type=int,
        default=DEFAULT_CLIMATOLOGY_END_YEAR,
        help=f"Climatology end year. Default: {DEFAULT_CLIMATOLOGY_END_YEAR}",
    )
    p.add_argument(
        "--nlead",
        type=int,
        default=24,
        help="Number of lead months. Default: 24",
    )
    p.add_argument(
        "--e3sm-nens",
        type=int,
        default=10,
        help="Ensemble members for E3SM. Default: 10",
    )
    p.add_argument(
        "--smyle-nens",
        type=int,
        default=20,
        help="Ensemble members for CESM-SMYLE. Default: 20",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of Dask workers. Default: 8 (set to 0 for no dask client)",
    )
    p.add_argument(
        "--custom-regions",
        type=str,
        default=None,
        help="JSON string defining custom regions/indices.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite of existing files.",
    )
    p.add_argument(
        "--sst-land-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply reusable source-grid land masks to SST (default: enabled).",
    )
    return p.parse_args()


def _safe_to_netcdf(
    ds: xr.Dataset,
    path: Path | str,
    encoding: dict | None = None,
    *,
    sst_land_mask: bool,
) -> None:
    """Write NetCDF through a temporary file to avoid corrupted cache files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.copy()
    ds.attrs.update(
        sst_index_output_version=SST_INDEX_OUTPUT_VERSION,
        sst_preprocessing_version=SST_PREPROCESSING_VERSION,
        sst_land_mask=str(bool(sst_land_mask)).lower(),
    )
    tmp_path = path.with_suffix(".nc.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    ds.to_netcdf(tmp_path, encoding=encoding)
    os.replace(tmp_path, path)


def _output_is_current(path: Path, args: argparse.Namespace) -> bool:
    """Return whether an existing index used the current SST preparation."""
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            metadata_current = (
                dataset.attrs.get("sst_index_output_version")
                == SST_INDEX_OUTPUT_VERSION
                and dataset.attrs.get("sst_land_mask")
                == str(bool(args.sst_land_mask)).lower()
            )
            if not metadata_current:
                return False
            # RONI depends on the full-longitude TropicalMean region. Reject
            # caches produced by the former 0-to-360 zero-width-mask bug.
            if "RONI" in path.name:
                return "sst" in dataset and bool(dataset["sst"].notnull().any())
            return True
    except Exception:
        return False


def _smyle_output_paths(root: str | Path) -> tuple[Path, Path]:
    """Return the canonical SMYLE timeseries directory and fixed mask path."""
    root = Path(root)
    return (
        root / "sst_index" / "timeseries",
        root / "fixed" / "sftlf.CESM-SMYLE.nc",
    )


def get_required_base_regions(regions_list):
    return required_base_regions(regions_list, REGIONS)


def process_e3sm(args: argparse.Namespace) -> None:
    case_label = args.e3sm_display_name or args.e3sm_cache_tag or args.e3sm_case_prefix
    LOG.info(f"Processing E3SM regional SST Indices for {case_label}...")
    case_outdir = Path(args.outdir)
    if args.e3sm_cache_tag:
        case_outdir = case_outdir / args.e3sm_cache_tag
    outdir = case_outdir / "sst_index" / "timeseries"
    fixed_dir = case_outdir / "fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    landmask_file = fixed_dir / "sftlf.E3SM.nc"

    members = [f"EN{i:02d}" for i in range(args.e3sm_nens)]
    init_years = {
        m: list(range(args.year_start, args.year_end + 1))
        for m in args.init_months
    }

    cfg = S2DConfig(
        field="TS",
        data_dir=args.e3sm_data_dir,
        case_prefix=args.e3sm_case_prefix,
        members=members,
        nlead=args.nlead,
        init_months=args.init_months,
        init_years=init_years,
        region=[-180.0, 180.0, -90.0, 90.0],  # Global region
        region_name="Global",
        climy0=args.climy0,
        climy1=args.climy1,
        outdir=str(case_outdir),
        force_rewrite=args.force,
        realm="atm",
        grid="180x360_aave",
        freq="monthly",
        ts_split="2yr",
        engine="netcdf4",
        chunks={},
        require_all_members=True,
        verify_field_name=True,
        verify_coverage=True,
        persist_intermediate=True,
        load_regional_series=False,
        convert_ts_to_degC=True,
        seasonal_nlead=8,
    )

    diag = S2DDiagnostics(
        cfg=cfg,
        data_access=data_access,
        obs_access=obs_access,
        stats=stats,
        spatial=spatial,
        cal=cal,
    )

    first_m = args.init_months[0]
    ds_first = diag.load_model(first_m)
    ds_first["TS"], landmask = prepare_sst(
        ds_first["TS"],
        apply_land_mask=args.sst_land_mask,
        land_mask_path=landmask_file,
        source=f"E3SM:{args.e3sm_cache_tag or args.e3sm_case_prefix}",
        force_land_mask=args.force,
    )
    oceanmask = (
        (~landmask).chunk({"lat": 90, "lon": 180})
        if landmask is not None
        else xr.ones_like(ds_first["TS"].isel(Y=0, L=0, M=0), dtype=bool)
    )
    LOG.info("E3SM SST land mask enabled: %s", args.sst_land_mask)

    required_base = get_required_base_regions(args.regions)

    for m in args.init_months:
        ds = ds_first if m == first_m else diag.load_model(m)
        if m != first_m:
            ds["TS"], _ = prepare_sst(
                ds["TS"],
                apply_land_mask=args.sst_land_mask,
                land_mask_path=landmask_file,
                source=f"E3SM:{args.e3sm_cache_tag or args.e3sm_case_prefix}",
            )
        ds_seas = diag.compute_seasonal_dataset(ds)
        
        # Build Dask graph for all base regional averages
        dask_dict = {}
        for r in required_base:
            lonlat = REGIONS[r]["lonlat"]
            weights = compute_weights(diag.data_access, ds["TS"], lonlat, oceanmask)
            dask_dict[f"{r}_mon"] = compute_regional_mean(ds["TS"], weights)
            dask_dict[f"{r}_seas"] = compute_regional_mean(ds_seas["TS"], weights)
            
        if "ELI" in args.regions:
            dask_dict["ELI_mon"] = compute_eli_latlon_sst(ds["TS"], oceanmask=oceanmask)
            dask_dict["ELI_seas"] = compute_eli_latlon_sst(ds_seas["TS"], oceanmask=oceanmask)

        LOG.info(f"Computing base indices for month {m}...")
        computed_vals = dask.compute(dask_dict)[0]
        
        base_vals_mon = {r: computed_vals[f"{r}_mon"] for r in required_base}
        base_vals_seas = {r: computed_vals[f"{r}_seas"] for r in required_base}
        
        time_mon = ds["time"].load()
        time_seas = ds_seas["time"].load()
        
        # Derive TNI, ONI, RONI, IOD
        derived_mon = derive_indices(base_vals_mon, time_mon, args.climy0, args.climy1, is_model=True)
        derived_seas = derive_indices(base_vals_seas, time_seas, args.climy0, args.climy1, is_model=True)
        
        all_mon = {**base_vals_mon, **derived_mon}
        all_seas = {**base_vals_seas, **derived_seas}
        if "ELI" in args.regions:
            all_mon["ELI"] = computed_vals["ELI_mon"]
            all_seas["ELI"] = computed_vals["ELI_seas"]
        
        # Write only the requested regions to disk
        index_encoding = {"sst": {"zlib": True, "complevel": 1}}
        for r in args.regions:
            if r == "ELI":
                outfile_mon = outdir / f"E3SMLE{m:02d}_ELI_N{args.e3sm_nens:02d}_M{args.nlead:02d}.nc"
                if args.force or not _output_is_current(outfile_mon, args):
                    ds_out_mon = all_mon[r].to_dataset()
                    ds_out_mon["time"] = time_mon
                    ds_out_mon.attrs.update({
                        "case_prefix": args.e3sm_case_prefix,
                        "cache_tag": args.e3sm_cache_tag or "",
                        "display_name": args.e3sm_display_name or "",
                        "source_grid": "regridded 180x360_aave TS",
                    })
                    _safe_to_netcdf(ds_out_mon, outfile_mon, encoding={"eli": {"zlib": True, "complevel": 1, "dtype": "float32"}}, sst_land_mask=args.sst_land_mask)
                    LOG.info(f"Saved E3SM monthly ELI: {outfile_mon}")

                outfile_seas = outdir / f"E3SMLE{m:02d}_ELI_N{args.e3sm_nens:02d}_M{args.nlead:02d}_seas.nc"
                if args.force or not _output_is_current(outfile_seas, args):
                    ds_out_seas = all_seas[r].to_dataset()
                    ds_out_seas["time"] = time_seas
                    ds_out_seas.attrs.update({
                        "case_prefix": args.e3sm_case_prefix,
                        "cache_tag": args.e3sm_cache_tag or "",
                        "display_name": args.e3sm_display_name or "",
                        "source_grid": "regridded 180x360_aave TS",
                    })
                    _safe_to_netcdf(ds_out_seas, outfile_seas, encoding={"eli": {"zlib": True, "complevel": 1, "dtype": "float32"}}, sst_land_mask=args.sst_land_mask)
                    LOG.info(f"Saved E3SM seasonal ELI: {outfile_seas}")
                continue

            outfile_mon = outdir / f"E3SMLE{m:02d}_TS_N{args.e3sm_nens:02d}_M{args.nlead:02d}_{r}SST_mon.nc"
            if args.force or not _output_is_current(outfile_mon, args):
                ds_out_mon = all_mon[r].rename("sst").to_dataset()
                ds_out_mon["time"] = time_mon
                ds_out_mon.attrs.update({
                    "case_prefix": args.e3sm_case_prefix,
                    "cache_tag": args.e3sm_cache_tag or "",
                    "display_name": args.e3sm_display_name or "",
                })
                if "long_name" not in ds_out_mon["sst"].attrs:
                    ds_out_mon["sst"].attrs.update({
                        "long_name": f"{r} regional mean SST",
                        "region": r,
                        "units": "degC",
                    })
                if r in REGIONS:
                    ds_out_mon["sst"].attrs["lonlat"] = str(REGIONS[r]["lonlat"])
                _safe_to_netcdf(ds_out_mon, outfile_mon, encoding=index_encoding, sst_land_mask=args.sst_land_mask)
                LOG.info(f"Saved E3SM monthly index for {r}: {outfile_mon}")

            outfile_seas = outdir / f"E3SMLE{m:02d}_TS_N{args.e3sm_nens:02d}_M{args.nlead:02d}_{r}SST_seas.nc"
            if args.force or not _output_is_current(outfile_seas, args):
                ds_out_seas = all_seas[r].rename("sst").to_dataset()
                ds_out_seas["time"] = time_seas
                ds_out_seas.attrs.update({
                    "case_prefix": args.e3sm_case_prefix,
                    "cache_tag": args.e3sm_cache_tag or "",
                    "display_name": args.e3sm_display_name or "",
                })
                if "long_name" not in ds_out_seas["sst"].attrs:
                    ds_out_seas["sst"].attrs.update({
                        "long_name": f"{r} regional mean SST",
                        "region": r,
                        "units": "degC",
                    })
                if r in REGIONS:
                    ds_out_seas["sst"].attrs["lonlat"] = str(REGIONS[r]["lonlat"])
                _safe_to_netcdf(ds_out_seas, outfile_seas, encoding=index_encoding, sst_land_mask=args.sst_land_mask)
                LOG.info(f"Saved E3SM seasonal index for {r}: {outfile_seas}")


def process_obs(args: argparse.Namespace) -> None:
    LOG.info("Processing Observations (HadISST2) regional SST Indices...")
    outdir = Path(args.obs_outdir)
    fixed_dir = HADISST2_DIAG_DIR / "fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)

    obs_dir = "/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series"
    obs_product = "HadISST2"
    obs_var = "sst"
    obs_yrs = 1869
    obs_yre = 2022

    obs_chunk = {
        "time": 120,
        "lat": 180,
        "lon": 360,
    }

    obs_hadisst = obs_access.get_monthly_data(
        obs_dir=obs_dir,
        field="TS",
        field_map={"TS": obs_var},
        product=obs_product,
        start_year=obs_yrs,
        end_year=obs_yre,
        chunks=obs_chunk,
        verbose=False,
    )

    obs_hadisst = obs_hadisst.sel(time=slice(str(obs_yrs), str(obs_yre)))
    obs_sst_attrs = dict(obs_hadisst[obs_var].attrs)
    obs_hadisst[obs_var] = xr.where(
        obs_hadisst[obs_var] < -2.0,
        -1.8,
        obs_hadisst[obs_var],
    )
    obs_hadisst[obs_var].attrs.update(obs_sst_attrs)
    obs_hadisst = obs_hadisst.chunk(obs_chunk)

    obs_mask_file = fixed_dir / f"sftlf.{obs_product}.nc"
    obs_hadisst[obs_var], landmasko = prepare_sst(
        obs_hadisst[obs_var],
        apply_land_mask=args.sst_land_mask,
        land_mask_path=obs_mask_file,
        source=obs_product,
        force_land_mask=args.force,
    )

    obs_hadisst_seas = obs_access.mon_to_seas_obs(
        obs_hadisst,
        var=obs_var,
        field_map={"TS": obs_var},
    )
    obs_hadisst_seas = obs_hadisst_seas.chunk(obs_chunk)

    oceanmasko = (
        (~landmasko).chunk({"lat": 90, "lon": 180})
        if landmasko is not None
        else xr.ones_like(obs_hadisst[obs_var].isel(time=0), dtype=bool)
    )
    LOG.info("Observed SST land mask enabled: %s", args.sst_land_mask)

    obs_mon_da = obs_hadisst[obs_var] if isinstance(obs_hadisst, xr.Dataset) else obs_hadisst
    obs_seas_da = obs_hadisst_seas[obs_var] if isinstance(obs_hadisst_seas, xr.Dataset) else obs_hadisst_seas

    obs_mon_da = obs_mon_da.rename("TS")
    obs_seas_da = obs_seas_da.rename("TS")

    required_base = get_required_base_regions(args.regions)

    obs_dask = {}
    for r in required_base:
        lonlat = REGIONS[r]["lonlat"]
        obs_reg_ocean_area_mon = obs_access.obs_regional_weights(
            obs_mon_da,
            lonlat,
            lat_name="lat",
            lon_name="lon",
            mask=oceanmasko,
        )
        obs_reg_ocean_area_seas = obs_access.obs_regional_weights(
            obs_seas_da,
            lonlat,
            lat_name="lat",
            lon_name="lon",
            mask=oceanmasko,
        )
        obs_dask[f"{r}_mon"] = obs_mon_da.weighted(obs_reg_ocean_area_mon).mean(("lat", "lon"))
        obs_dask[f"{r}_seas"] = obs_seas_da.weighted(obs_reg_ocean_area_seas).mean(("lat", "lon"))
    if "ELI" in args.regions:
        obs_dask["ELI_mon"] = compute_eli_latlon_sst(obs_mon_da, oceanmask=oceanmasko)
        obs_dask["ELI_seas"] = compute_eli_latlon_sst(obs_seas_da, oceanmask=oceanmasko)

    LOG.info("Computing Observations base indices...")
    computed_vals = dask.compute(obs_dask)[0]

    base_vals_mon = {r: computed_vals[f"{r}_mon"] for r in required_base}
    base_vals_seas = {r: computed_vals[f"{r}_seas"] for r in required_base}

    time_mon = obs_hadisst["time"].load()
    time_seas = obs_hadisst_seas["time"].load()

    # Derive TNI, ONI, RONI, IOD
    derived_mon = derive_indices(base_vals_mon, time_mon, args.climy0, args.climy1, is_model=False)
    derived_seas = derive_indices(base_vals_seas, time_seas, args.climy0, args.climy1, is_model=False)

    all_mon = {**base_vals_mon, **derived_mon}
    all_seas = {**base_vals_seas, **derived_seas}
    if "ELI" in args.regions:
        all_mon["ELI"] = computed_vals["ELI_mon"]
        all_seas["ELI"] = computed_vals["ELI_seas"]

    for r in args.regions:
        if r == "ELI":
            outfile_seas = outdir / "HadISST2_sst_ELI_seas.nc"
            if args.force or not _output_is_current(outfile_seas, args):
                ds_seas = all_seas[r].to_dataset()
                ds_seas["time"] = time_seas
                ds_seas.attrs["source_grid"] = "HadISST2 lat/lon"
                _safe_to_netcdf(ds_seas, outfile_seas, encoding={"eli": {"zlib": True, "complevel": 1, "dtype": "float32"}}, sst_land_mask=args.sst_land_mask)
                LOG.info(f"Saved Obs seasonal ELI: {outfile_seas}")

            outfile_mon = outdir / "HadISST2_sst_ELI_mon.nc"
            if args.force or not _output_is_current(outfile_mon, args):
                ds_mon = all_mon[r].to_dataset()
                ds_mon["time"] = time_mon
                ds_mon.attrs["source_grid"] = "HadISST2 lat/lon"
                _safe_to_netcdf(ds_mon, outfile_mon, encoding={"eli": {"zlib": True, "complevel": 1, "dtype": "float32"}}, sst_land_mask=args.sst_land_mask)
                LOG.info(f"Saved Obs monthly ELI: {outfile_mon}")
            continue

        outfile_seas = outdir / f"HadISST2_sst_{r}SST_seas.nc"
        if args.force or not _output_is_current(outfile_seas, args):
            ds_seas = all_seas[r].rename("sst").to_dataset()
            ds_seas["time"] = time_seas
            if "long_name" not in ds_seas["sst"].attrs:
                ds_seas["sst"].attrs.update({
                    "long_name": f"{r} regional mean SST (Observations)",
                    "region": r,
                    "units": "degC",
                })
            if r in REGIONS:
                ds_seas["sst"].attrs["lonlat"] = str(REGIONS[r]["lonlat"])
            _safe_to_netcdf(ds_seas, outfile_seas, sst_land_mask=args.sst_land_mask)
            LOG.info(f"Saved Obs seasonal index for {r}: {outfile_seas}")

        outfile_mon = outdir / f"HadISST2_sst_{r}SST_mon.nc"
        if args.force or not _output_is_current(outfile_mon, args):
            ds_mon = all_mon[r].rename("sst").to_dataset()
            ds_mon["time"] = time_mon
            if "long_name" not in ds_mon["sst"].attrs:
                ds_mon["sst"].attrs.update({
                    "long_name": f"{r} regional mean SST (Observations)",
                    "region": r,
                    "units": "degC",
                })
            if r in REGIONS:
                ds_mon["sst"].attrs["lonlat"] = str(REGIONS[r]["lonlat"])
            _safe_to_netcdf(ds_mon, outfile_mon, sst_land_mask=args.sst_land_mask)
            LOG.info(f"Saved Obs monthly index for {r}: {outfile_mon}")


def process_smyle(args: argparse.Namespace) -> None:
    LOG.info("Processing CESM-SMYLE regional SST Indices...")
    SMYLE_BENCHMARK_DIR = str(CESM_SMYLE_DIAG_DIR)
    smyle_outdir, smyle_mask_file = _smyle_output_paths(args.smyle_outdir)
    smyle_outdir.mkdir(parents=True, exist_ok=True)

    smyle_chunks = {
        "Y": 3,
        "L": -1,
        "M": 2,
        "lat": 96,
        "lon": 144,
    }

    smyle_index_encoding = {
        "sst": {
            "zlib": True,
            "complevel": 1,
        }
    }

    lat_name = "lat"
    lon_name = "lon"

    required_base = get_required_base_regions(args.regions)

    for m in args.init_months:
        ds_smyle = smyle_access.load_benchmark(
            field="TS",
            init_month=m,
            benchmark_dir=SMYLE_BENCHMARK_DIR,
            nens=args.smyle_nens,
            nlead=args.nlead,
            freq="seas",
            chunks=smyle_chunks,
        )

        smyle_da_seas = ds_smyle["TS"] if "TS" in ds_smyle else (ds_smyle["SST"] if "SST" in ds_smyle else ds_smyle["sst"])

        ds_smyle_mon = smyle_access.load_benchmark(
            field="TS",
            init_month=m,
            benchmark_dir=SMYLE_BENCHMARK_DIR,
            nens=args.smyle_nens,
            nlead=args.nlead,
            freq="mon",
            chunks=smyle_chunks,
        )

        smyle_da_mon = ds_smyle_mon["TS"] if "TS" in ds_smyle_mon else (ds_smyle_mon["SST"] if "SST" in ds_smyle_mon else ds_smyle_mon["sst"])
        smyle_da_mon, smyle_landmask = prepare_sst(
            smyle_da_mon,
            apply_land_mask=args.sst_land_mask,
            land_mask_path=smyle_mask_file,
            source="CESM-SMYLE",
            force_land_mask=args.force and m == args.init_months[0],
        )
        smyle_da_seas, _ = prepare_sst(
            smyle_da_seas,
            apply_land_mask=args.sst_land_mask,
            land_mask_path=smyle_mask_file,
            source="CESM-SMYLE",
        )
        smyle_oceanmask = (
            (~smyle_landmask).chunk(
                {lat_name: smyle_chunks["lat"], lon_name: smyle_chunks["lon"]}
            )
            if smyle_landmask is not None
            else xr.ones_like(smyle_da_mon.isel(Y=0, L=0, M=0), dtype=bool)
        )

        smyle_dask = {}
        for r in required_base:
            lonlat = REGIONS[r]["lonlat"]
            smyle_reg_ocean_area_mon = obs_access.obs_regional_weights(
                smyle_da_mon,
                lonlat,
                lat_name=lat_name,
                lon_name=lon_name,
                mask=smyle_oceanmask,
            )
            smyle_reg_ocean_area_seas = obs_access.obs_regional_weights(
                smyle_da_seas,
                lonlat,
                lat_name=lat_name,
                lon_name=lon_name,
                mask=smyle_oceanmask,
            )
            smyle_dask[f"{r}_mon"] = smyle_da_mon.weighted(smyle_reg_ocean_area_mon).mean((lat_name, lon_name))
            smyle_dask[f"{r}_seas"] = smyle_da_seas.weighted(smyle_reg_ocean_area_seas).mean((lat_name, lon_name))
        if "ELI" in args.regions:
            smyle_dask["ELI_mon"] = compute_eli_latlon_sst(
                smyle_da_mon,
                lat_name=lat_name,
                lon_name=lon_name,
                oceanmask=smyle_oceanmask,
            )
            smyle_dask["ELI_seas"] = compute_eli_latlon_sst(
                smyle_da_seas,
                lat_name=lat_name,
                lon_name=lon_name,
                oceanmask=smyle_oceanmask,
            )

        LOG.info(f"Computing CESM-SMYLE base indices for month {m}...")
        computed_vals = dask.compute(smyle_dask)[0]

        base_vals_mon = {r: computed_vals[f"{r}_mon"] for r in required_base}
        base_vals_seas = {r: computed_vals[f"{r}_seas"] for r in required_base}

        time_mon = ds_smyle_mon["time"].load()
        time_seas = ds_smyle["time"].load()

        # Derive TNI, ONI, RONI, IOD
        derived_mon = derive_indices(base_vals_mon, time_mon, args.climy0, args.climy1, is_model=True)
        derived_seas = derive_indices(base_vals_seas, time_seas, args.climy0, args.climy1, is_model=True)

        all_mon = {**base_vals_mon, **derived_mon}
        all_seas = {**base_vals_seas, **derived_seas}
        if "ELI" in args.regions:
            all_mon["ELI"] = computed_vals["ELI_mon"]
            all_seas["ELI"] = computed_vals["ELI_seas"]

        for r in args.regions:
            if r == "ELI":
                outfile_mon = smyle_outdir / f"BSMYLE{m:02d}_ELI_N{args.smyle_nens:02d}_M{args.nlead:02d}.nc"
                if args.force or not _output_is_current(outfile_mon, args):
                    ds_idx_mon = all_mon[r].to_dataset()
                    ds_idx_mon["time"] = time_mon
                    ds_idx_mon.attrs["source_grid"] = "CESM-SMYLE regridded TS benchmark"
                    _safe_to_netcdf(ds_idx_mon, outfile_mon, encoding={"eli": {"zlib": True, "complevel": 1, "dtype": "float32"}}, sst_land_mask=args.sst_land_mask)
                    LOG.info(f"Saved CESM-SMYLE monthly ELI: {outfile_mon}")

                outfile_seas = smyle_outdir / f"BSMYLE{m:02d}_ELI_N{args.smyle_nens:02d}_M{args.nlead:02d}_seas.nc"
                if args.force or not _output_is_current(outfile_seas, args):
                    ds_idx = all_seas[r].to_dataset()
                    ds_idx["time"] = time_seas
                    ds_idx.attrs["source_grid"] = "CESM-SMYLE regridded TS benchmark"
                    _safe_to_netcdf(ds_idx, outfile_seas, encoding={"eli": {"zlib": True, "complevel": 1, "dtype": "float32"}}, sst_land_mask=args.sst_land_mask)
                    LOG.info(f"Saved CESM-SMYLE seasonal ELI: {outfile_seas}")
                continue

            outfile_seas = smyle_outdir / f"BSMYLE{m:02d}_TS_N{args.smyle_nens:02d}_M{args.nlead:02d}_{r}SST_seas.nc"
            if args.force or not _output_is_current(outfile_seas, args):
                ds_idx = all_seas[r].rename("sst").to_dataset()
                ds_idx["time"] = time_seas
                if "long_name" not in ds_idx["sst"].attrs:
                    ds_idx["sst"].attrs.update({
                        "long_name": f"CESM-SMYLE {r} regional mean SST",
                        "region": r,
                        "units": "degC",
                    })
                if r in REGIONS:
                    ds_idx["sst"].attrs["lonlat"] = str(REGIONS[r]["lonlat"])
                _safe_to_netcdf(ds_idx, outfile_seas, encoding=smyle_index_encoding, sst_land_mask=args.sst_land_mask)
                LOG.info(f"Saved CESM-SMYLE seasonal index for {r}: {outfile_seas}")

            outfile_mon = smyle_outdir / f"BSMYLE{m:02d}_TS_N{args.smyle_nens:02d}_M{args.nlead:02d}_{r}SST_mon.nc"
            if args.force or not _output_is_current(outfile_mon, args):
                ds_idx_mon = all_mon[r].rename("sst").to_dataset()
                ds_idx_mon["time"] = time_mon
                if "long_name" not in ds_idx_mon["sst"].attrs:
                    ds_idx_mon["sst"].attrs.update({
                        "long_name": f"CESM-SMYLE {r} regional mean SST",
                        "region": r,
                        "units": "degC",
                    })
                if r in REGIONS:
                    ds_idx_mon["sst"].attrs["lonlat"] = str(REGIONS[r]["lonlat"])
                _safe_to_netcdf(ds_idx_mon, outfile_mon, encoding=smyle_index_encoding, sst_land_mask=args.sst_land_mask)
                LOG.info(f"Saved CESM-SMYLE monthly index for {r}: {outfile_mon}")


def main() -> None:
    args = parse_args()

    if getattr(args, "custom_regions", None):
        import json
        try:
            custom_regs = json.loads(args.custom_regions)
            for name, info in custom_regs.items():
                REGIONS[name] = info
                if name not in VALID_REGIONS:
                    VALID_REGIONS.append(name)
        except Exception as e:
            LOG.error(f"Failed to parse custom regions: {e}")
            sys.exit(1)

    # Validate regions
    for r in args.regions:
        if r not in VALID_REGIONS:
            LOG.error(f"Invalid region '{r}'. Valid choices are: {VALID_REGIONS}")
            sys.exit(1)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.perf_counter()

    client = None
    cluster = None
    if args.workers > 0:
        import dask
        from dask.distributed import Client, LocalCluster
        dask.config.set({"array.slicing.split_large_chunks": True})
        cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1)
        client = Client(cluster)
        LOG.info(f"Dask Dashboard: {client.dashboard_link}")

    try:
        if "e3sm" in args.sources:
            process_e3sm(args)
        if "obs" in args.sources:
            process_obs(args)
        if "smyle" in args.sources:
            process_smyle(args)
    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()

    elapsed = time.perf_counter() - t0
    LOG.info(f"All processing completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
