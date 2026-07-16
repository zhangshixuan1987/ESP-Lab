#!/usr/bin/env python3
"""Generate and save 13 regional SST indices for E3SM, CESM-SMYLE, and observations."""

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
from typing import Dict, Any, List
import numpy as np
import xarray as xr
import cftime
import pandas as pd
import dask

# Insert repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esp_lab import data_access_e3sm as data_access
from esp_lab import data_access_obs as obs_access
from esp_lab import data_access_cesm_smyle as smyle_access
from esp_lab import stats
from esp_lab.paths import CESM_SMYLE_DIAG_DIR, E3SMLE_DIAG_DIR
from esp_lab.utils import spatial_utils as spatial
from esp_lab.utils import calendar_utils as cal
from esp_lab.diagnostics import S2DDiagnostics, S2DConfig
from esp_lab.diagnostics.regional import build_landmask, compute_weights, compute_regional_mean

LOG = logging.getLogger(__name__)

REGIONS = {}
VALID_REGIONS = ["IOD", "TNI", "ONI", "RONI"]


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
        help=f"Output directory for E3SM and obs. Default: {E3SMLE_DIAG_DIR}",
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
        default=1980,
        help="Climatology start year. Default: 1980",
    )
    p.add_argument(
        "--climy1",
        type=int,
        default=2010,
        help="Climatology end year. Default: 2010",
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
    return p.parse_args()


def _safe_to_netcdf(ds: xr.Dataset, path: Path | str, encoding: dict | None = None) -> None:
    """Write NetCDF through a temporary file to avoid corrupted cache files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".nc.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    ds.to_netcdf(tmp_path, encoding=encoding)
    os.replace(tmp_path, path)


def _convert_smyle_sst_to_degC(da: xr.DataArray, field: str) -> xr.DataArray:
    da = da.copy()
    units = str(da.attrs.get("units", "")).lower()
    if field.upper() == "TS" or units in ["k", "kelvin"]:
        da = da - 273.15
        da.attrs["units"] = "degC"
    elif "c" in units or "degc" in units or "celsius" in units:
        da.attrs["units"] = "degC"
    else:
        da.attrs["units"] = da.attrs.get("units", "unknown")
    return da


def get_required_base_regions(regions_list: List[str]) -> List[str]:
    req = set()
    for r in regions_list:
        if r in ["Nino12", "Nino3", "Nino3.4", "Nino4", "TNA", "TSA", "PACWRAMPOOL", "AtlNino", "AtlMDR"]:
            req.add(r)
        elif r == "IOD":
            req.add("IOD_West")
            req.add("IOD_East")
        elif r == "TNI":
            req.add("Nino12")
            req.add("Nino4")
        elif r == "ONI":
            req.add("Nino3.4")
        elif r == "RONI":
            req.add("Nino3.4")
            req.add("TropicalMean")
        elif r in REGIONS:
            req.add(r)
    return list(req)


def compute_model_anom(da: xr.DataArray, da_time: xr.DataArray, climy0: int, climy1: int) -> xr.DataArray:
    da_anom, _ = stats.remove_drift(da, da_time, climy0, climy1)
    return da_anom


def compute_model_std(da_anom: xr.DataArray, da_time: xr.DataArray, climy0: int, climy1: int) -> xr.DataArray:
    d1 = cftime.DatetimeNoLeap(climy0, 1, 1, 0, 0, 0)
    d2 = cftime.DatetimeNoLeap(climy1, 12, 31, 23, 59, 59)
    da_anom_clim = da_anom.where((da_time >= d1) & (da_time <= d2))
    dims_to_reduce = [dim for dim in ["Y", "M"] if dim in da_anom.dims]
    sigma = da_anom_clim.std(dim=dims_to_reduce)
    sigma = sigma.where(sigma > 0, 1.0)
    return da_anom / sigma


def compute_obs_anom(da: xr.DataArray, climy0: int, climy1: int) -> xr.DataArray:
    d1 = cftime.DatetimeNoLeap(climy0, 1, 1, 0, 0, 0)
    d2 = cftime.DatetimeNoLeap(climy1, 12, 31, 23, 59, 59)
    da_clim_period = da.sel(time=slice(d1, d2))
    climo = da_clim_period.groupby("time.month").mean("time")
    anom = da.groupby("time.month") - climo
    return anom


def compute_obs_std(da_anom: xr.DataArray, climy0: int, climy1: int) -> xr.DataArray:
    d1 = cftime.DatetimeNoLeap(climy0, 1, 1, 0, 0, 0)
    d2 = cftime.DatetimeNoLeap(climy1, 12, 31, 23, 59, 59)
    da_anom_clim = da_anom.sel(time=slice(d1, d2))
    sigma = da_anom_clim.std("time")
    sigma = sigma.where(sigma > 0, 1.0)
    return da_anom / sigma


def derive_indices(computed_vals: Dict[str, xr.DataArray], time_coords: xr.DataArray, climy0: int, climy1: int, is_model: bool = True) -> Dict[str, xr.DataArray]:
    derived = {}
    
    if is_model:
        def get_anom(da):
            return compute_model_anom(da, time_coords, climy0, climy1)
        def get_std(da_anom):
            return compute_model_std(da_anom, time_coords, climy0, climy1)
        rolling_dim = "L"
        time_var = time_coords
    else:
        def get_anom(da):
            return compute_obs_anom(da, climy0, climy1)
        def get_std(da_anom):
            return compute_obs_std(da_anom, climy0, climy1)
        rolling_dim = "time"
        time_var = time_coords
        
    # 1. IOD = IOD_West - IOD_East (absolute difference)
    if "IOD_West" in computed_vals and "IOD_East" in computed_vals:
        derived["IOD"] = computed_vals["IOD_West"] - computed_vals["IOD_East"]
        derived["IOD"].attrs.update({
            "long_name": "Indian Ocean Dipole regional mean SST",
            "region": "IOD",
            "units": "degC",
        })
        
    # 2. ONI = 3-month running mean of Nino3.4 anomalies
    if "Nino3.4" in computed_vals:
        nino34_anom = get_anom(computed_vals["Nino3.4"])
        oni = nino34_anom.rolling({rolling_dim: 3}, center=True, min_periods=1).mean()
        oni.attrs.update({
            "long_name": "Oceanic Nino Index (3-month running mean of Nino3.4 anomalies)",
            "region": "Nino3.4",
            "units": "degC",
        })
        derived["ONI"] = oni
        
    # 3. TNI = standardized Nino 1+2 minus standardized Nino 4 with 5-month running mean
    if "Nino12" in computed_vals and "Nino4" in computed_vals:
        nino12_anom = get_anom(computed_vals["Nino12"])
        nino4_anom = get_anom(computed_vals["Nino4"])
        
        nino12_std = get_std(nino12_anom)
        nino4_std = get_std(nino4_anom)
        
        diff = nino12_std - nino4_std
        tni = diff.rolling({rolling_dim: 5}, center=True, min_periods=1).mean()
        tni.attrs.update({
            "long_name": "Trans-Niño Index (standardized Nino12 minus standardized Nino4 with 5-month running mean)",
            "region": "TNI",
            "units": "1",
        })
        derived["TNI"] = tni
        
    # 4. RONI = 3-month running mean of (Nino3.4 anomalies minus TropicalMean anomalies), scaled to match Nino3.4 variance
    if "Nino3.4" in computed_vals and "TropicalMean" in computed_vals:
        nino34_anom = get_anom(computed_vals["Nino3.4"])
        trop_anom = get_anom(computed_vals["TropicalMean"])
        
        diff = nino34_anom - trop_anom
        diff_3m = diff.rolling({rolling_dim: 3}, center=True, min_periods=1).mean()
        nino34_3m = nino34_anom.rolling({rolling_dim: 3}, center=True, min_periods=1).mean()
        
        d1 = cftime.DatetimeNoLeap(climy0, 1, 1, 0, 0, 0)
        d2 = cftime.DatetimeNoLeap(climy1, 12, 31, 23, 59, 59)
        
        if is_model:
            nino_clim = nino34_3m.where((time_var >= d1) & (time_var <= d2))
            diff_clim = diff_3m.where((time_var >= d1) & (time_var <= d2))
            dims_to_reduce = [dim for dim in ["Y", "M"] if dim in nino_clim.dims]
            std_nino = nino_clim.std(dim=dims_to_reduce)
            std_diff = diff_clim.std(dim=dims_to_reduce)
        else:
            nino_clim = nino34_3m.sel(time=slice(d1, d2))
            diff_clim = diff_3m.sel(time=slice(d1, d2))
            std_nino = nino_clim.std("time")
            std_diff = diff_clim.std("time")
            
        ratio = std_nino / std_diff
        ratio = ratio.fillna(1.0).where(std_diff > 0, 1.0)
        roni = diff_3m * ratio
        roni.attrs.update({
            "long_name": "Relative Oceanic Nino Index",
            "region": "RONI",
            "units": "degC",
        })
        derived["RONI"] = roni
        
    return derived


def process_e3sm(args: argparse.Namespace) -> None:
    case_label = args.e3sm_display_name or args.e3sm_cache_tag or args.e3sm_case_prefix
    LOG.info(f"Processing E3SM regional SST Indices for {case_label}...")
    outdir = Path(args.outdir)
    if args.e3sm_cache_tag:
        outdir = outdir / args.e3sm_cache_tag
    fixed_dir = outdir / "fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)
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
        outdir=str(outdir),
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

    # Save fixed landmask
    if not landmask_file.exists() or args.force:
        first_m = args.init_months[0]
        ds_first = diag.load_model(first_m)
        landmask = build_landmask(ds_first["TS"], diag.spatial)
        landmask.name = "sftlf"
        landmask.attrs["long_name"] = "land fraction mask"
        landmask.attrs["units"] = "1"
        landmask = landmask.chunk({"lat": 90, "lon": 180})
        _safe_to_netcdf(landmask.to_dataset(), landmask_file)
        LOG.info(f"Saved E3SM landmask to {landmask_file}")
    else:
        landmask = xr.open_dataarray(landmask_file, chunks={"lat": 90, "lon": 180})

    oceanmask = (~landmask.astype(bool)).chunk({"lat": 90, "lon": 180})

    required_base = get_required_base_regions(args.regions)

    for m in args.init_months:
        ds = diag.load_model(m)
        ds_seas = diag.compute_seasonal_dataset(ds)
        
        # Build Dask graph for all base regional averages
        dask_dict = {}
        for r in required_base:
            lonlat = REGIONS[r]["lonlat"]
            weights = compute_weights(diag.data_access, ds["TS"], lonlat, oceanmask)
            dask_dict[f"{r}_mon"] = compute_regional_mean(ds["TS"], weights)
            dask_dict[f"{r}_seas"] = compute_regional_mean(ds_seas["TS"], weights)
            
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
        
        # Write only the requested regions to disk
        index_encoding = {"sst": {"zlib": True, "complevel": 1}}
        for r in args.regions:
            outfile_mon = outdir / f"E3SMLE{m:02d}_TS_N{args.e3sm_nens:02d}_M{args.nlead:02d}_{r}SST_mon.nc"
            if not outfile_mon.exists() or args.force:
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
                _safe_to_netcdf(ds_out_mon, outfile_mon, encoding=index_encoding)
                LOG.info(f"Saved E3SM monthly index for {r}: {outfile_mon}")

            outfile_seas = outdir / f"E3SMLE{m:02d}_TS_N{args.e3sm_nens:02d}_M{args.nlead:02d}_{r}SST_seas.nc"
            if not outfile_seas.exists() or args.force:
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
                _safe_to_netcdf(ds_out_seas, outfile_seas, encoding=index_encoding)
                LOG.info(f"Saved E3SM seasonal index for {r}: {outfile_seas}")


def process_obs(args: argparse.Namespace) -> None:
    LOG.info("Processing Observations (HadISST2) regional SST Indices...")
    outdir = Path(args.outdir)
    fixed_dir = outdir / "fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)

    obs_dir = "/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series"
    obs_product = "HadISST2"
    obs_var = "sst"
    obs_yrs = 1869
    obs_yre = 2022

    obs_chunk = {
        "time": 120,
        "lat": 90,
        "lon": 180,
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
    obs_hadisst[obs_var] = xr.where(
        obs_hadisst[obs_var] < -2.0,
        -1.8,
        obs_hadisst[obs_var],
    )
    obs_hadisst = obs_hadisst.chunk(obs_chunk)

    obs_hadisst_seas = obs_access.mon_to_seas_obs(
        obs_hadisst,
        var=obs_var,
        field_map={"TS": obs_var},
    )
    obs_hadisst_seas = obs_hadisst_seas.chunk(obs_chunk)

    # Save obs landmask
    obs_mask_file = fixed_dir / f"sftlf.{obs_product}.nc"
    if not obs_mask_file.exists() or args.force:
        da_obs_mask = obs_hadisst_seas[obs_var] if isinstance(obs_hadisst_seas, xr.Dataset) else obs_hadisst_seas
        landmasko = spatial.create_land_sea_mask(
            da_obs_mask,
            lat_key="lat",
            lon_key="lon",
        )
        landmasko = landmasko.astype("int8")
        landmasko.name = "sftlf"
        landmasko.attrs["long_name"] = "land fraction mask"
        landmasko.attrs["units"] = "1"
        landmasko = landmasko.chunk({"lat": 90, "lon": 180})
        _safe_to_netcdf(landmasko.to_dataset(), obs_mask_file)
        LOG.info(f"Saved Obs landmask to {obs_mask_file}")
    else:
        landmasko = xr.open_dataarray(obs_mask_file, chunks={"lat": 90, "lon": 180})

    oceanmasko = (~landmasko.astype(bool)).chunk({"lat": 90, "lon": 180})

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

    for r in args.regions:
        outfile_seas = outdir / f"HadISST2_sst_{r}SST_seas.nc"
        if not outfile_seas.exists() or args.force:
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
            _safe_to_netcdf(ds_seas, outfile_seas)
            LOG.info(f"Saved Obs seasonal index for {r}: {outfile_seas}")

        outfile_mon = outdir / f"HadISST2_sst_{r}SST_mon.nc"
        if not outfile_mon.exists() or args.force:
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
            _safe_to_netcdf(ds_mon, outfile_mon)
            LOG.info(f"Saved Obs monthly index for {r}: {outfile_mon}")


def process_smyle(args: argparse.Namespace) -> None:
    LOG.info("Processing CESM-SMYLE regional SST Indices...")
    SMYLE_BENCHMARK_DIR = str(CESM_SMYLE_DIAG_DIR)
    smyle_outdir = Path(args.smyle_outdir)

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
        smyle_da_seas = _convert_smyle_sst_to_degC(smyle_da_seas, "TS")

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
        smyle_da_mon = _convert_smyle_sst_to_degC(smyle_da_mon, "TS")

        # Ocean mask
        smyle_landmask = spatial.create_land_sea_mask(
            smyle_da_mon,
            lat_key=lat_name,
            lon_key=lon_name,
        )
        smyle_oceanmask = (~smyle_landmask.astype(bool)).chunk(
            {lat_name: smyle_chunks["lat"], lon_name: smyle_chunks["lon"]}
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

        for r in args.regions:
            r_dir = smyle_outdir / r
            r_dir.mkdir(parents=True, exist_ok=True)
            
            outfile_seas = r_dir / f"BSMYLE{m:02d}_TS_N{args.smyle_nens:02d}_M{args.nlead:02d}_{r}SST_seas.nc"
            if not outfile_seas.exists() or args.force:
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
                _safe_to_netcdf(ds_idx, outfile_seas, encoding=smyle_index_encoding)
                LOG.info(f"Saved CESM-SMYLE seasonal index for {r}: {outfile_seas}")

            outfile_mon = r_dir / f"BSMYLE{m:02d}_TS_N{args.smyle_nens:02d}_M{args.nlead:02d}_{r}SST_mon.nc"
            if not outfile_mon.exists() or args.force:
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
                _safe_to_netcdf(ds_idx_mon, outfile_mon, encoding=smyle_index_encoding)
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
