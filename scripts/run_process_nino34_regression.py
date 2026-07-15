#!/usr/bin/env python3
"""Generate an observed Nino3.4 index and regress TC density diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from esp_lab.paths import E3SMLE_DIAG_DIR


DEFAULT_SST_FILE = (
    "/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series/"
    "HadISST2/sst_186901_202212.nc"
)
DEFAULT_DIAG_FILE = str(
    E3SMLE_DIAG_DIR
    / "tc_lead_track_density_WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL_set2_1980_2018.nc"
)
DEFAULT_OUTDIR = str(E3SMLE_DIAG_DIR)

NINO34_LONLAT = (190.0, 240.0, -5.0, 5.0)
SEASON_NAMES = ("NH_JJASON", "SH_DJFMAM")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sst-file", default=DEFAULT_SST_FILE, help="Monthly observed SST NetCDF.")
    p.add_argument("--sst-var", default="sst", help="SST variable name.")
    p.add_argument("--diag-file", default=DEFAULT_DIAG_FILE, help="TC density diagnostic NetCDF.")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory.")
    p.add_argument("--year-start", type=int, default=1980, help="First regression year.")
    p.add_argument("--year-end", type=int, default=2018, help="Last regression year.")
    p.add_argument("--clim-start", type=int, default=1981, help="First climatology year.")
    p.add_argument("--clim-end", type=int, default=2010, help="Last climatology year.")
    p.add_argument("--nino-out", default=None, help="Optional output path for monthly Nino3.4 data.")
    p.add_argument("--reg-out", default=None, help="Optional output path for regression diagnostics.")
    p.add_argument("--model-t-threshold", type=float, default=2.0)
    p.add_argument("--obs-t-threshold", type=float, default=1.67)
    return p.parse_args()


def _standardize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old in ("latitude", "nav_lat", "yt_ocean"):
        if old in ds.coords or old in ds.dims:
            rename[old] = "lat"
            break
    for old in ("longitude", "nav_lon", "xt_ocean"):
        if old in ds.coords or old in ds.dims:
            rename[old] = "lon"
            break
    if rename:
        ds = ds.rename(rename)
    if "lon" in ds.coords:
        ds = ds.assign_coords(lon=(ds["lon"] % 360)).sortby("lon")
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    return ds


def _region_mask(da: xr.DataArray, lonlat=NINO34_LONLAT) -> xr.DataArray:
    lon_w, lon_e, lat_s, lat_n = lonlat
    lon = da["lon"] % 360
    lat = da["lat"]
    lon2d, lat2d = xr.broadcast(lon, lat)
    lon2d = lon2d.transpose("lat", "lon")
    lat2d = lat2d.transpose("lat", "lon")
    lat_mask = (lat2d >= lat_s) & (lat2d <= lat_n)
    lon_mask = (lon2d >= lon_w) & (lon2d <= lon_e)
    return lat_mask & lon_mask


def compute_monthly_nino34(
    sst_file: Path,
    sst_var: str,
    clim_start: int,
    clim_end: int,
) -> xr.Dataset:
    ds = _standardize_lat_lon(xr.open_dataset(sst_file))
    if sst_var not in ds:
        raise ValueError(f"{sst_var!r} not found in {sst_file}; available: {list(ds.data_vars)}")

    sst = ds[sst_var]
    if not {"time", "lat", "lon"}.issubset(sst.dims):
        raise ValueError(f"{sst_var!r} must have time, lat, lon dimensions; got {sst.dims}")

    mask = _region_mask(sst)
    weights = xr.DataArray(np.cos(np.deg2rad(sst["lat"])), dims=("lat",), coords={"lat": sst["lat"]})
    weights, _ = xr.broadcast(weights, sst["lon"])
    weights = weights.transpose("lat", "lon").where(mask, 0.0)

    nino34_sst = sst.weighted(weights).mean(("lat", "lon"), skipna=True)
    nino34_sst = nino34_sst.rename("nino34_sst")

    clim = nino34_sst.sel(time=slice(f"{clim_start}-01-01", f"{clim_end}-12-31"))
    clim = clim.groupby("time.month").mean("time", skipna=True).rename("nino34_monthly_climatology")
    nino34 = (nino34_sst.groupby("time.month") - clim).rename("nino34")
    nino34.attrs.update(
        {
            "long_name": "Nino3.4 SST anomaly",
            "units": sst.attrs.get("units", "degC"),
            "region": "5S-5N, 170W-120W",
            "climatology": f"{clim_start}-{clim_end} monthly mean",
        }
    )
    nino34_sst.attrs.update(
        {
            "long_name": "Nino3.4 area-weighted mean SST",
            "units": sst.attrs.get("units", "degC"),
            "region": "5S-5N, 170W-120W",
        }
    )

    out = xr.Dataset({"nino34": nino34, "nino34_sst": nino34_sst, "nino34_monthly_climatology": clim})
    out.attrs.update(
        {
            "description": "Observed monthly Nino3.4 index from area-weighted SST anomalies.",
            "source_file": str(sst_file),
            "source_variable": sst_var,
            "climatology": f"{clim_start}-{clim_end}",
            "nino34_lonlat": "190E-240E, 5S-5N",
        }
    )
    return out


def seasonal_nino34(nino34: xr.DataArray) -> xr.DataArray:
    month = nino34["time"].dt.month
    year = nino34["time"].dt.year

    nh_mask = month.isin([6, 7, 8, 9, 10, 11])
    nh = (
        nino34.where(nh_mask, drop=True)
        .assign_coords(year=("time", year.where(nh_mask, drop=True).data.astype(int)))
        .groupby("year")
        .mean("time", skipna=True)
    )

    sh_mask = month.isin([12, 1, 2, 3, 4, 5])
    sh_year = xr.where(month == 12, year, year - 1)
    sh = (
        nino34.where(sh_mask, drop=True)
        .assign_coords(year=("time", sh_year.where(sh_mask, drop=True).data.astype(int)))
        .groupby("year")
        .mean("time", skipna=True)
    )

    season_coord = xr.DataArray(list(SEASON_NAMES), dims="season", name="season")
    out = xr.concat([nh, sh], dim=season_coord)
    out = out.sortby("year").transpose("season", "year").rename("seasonal_nino34_index")
    out.attrs.update(nino34.attrs)
    return out


def regress_y_on_x(y: xr.DataArray, x: xr.DataArray, sample_dim: str = "year") -> xr.Dataset:
    x, y = xr.align(x, y, join="inner")
    valid = np.isfinite(x) & np.isfinite(y)

    n = valid.sum(sample_dim)
    x_valid = x.where(valid)
    y_valid = y.where(valid)

    x_mean = x_valid.mean(sample_dim, skipna=True)
    y_mean = y_valid.mean(sample_dim, skipna=True)
    x_anom = x_valid - x_mean
    y_anom = y_valid - y_mean

    cov_xy = (x_anom * y_anom).sum(sample_dim, skipna=True)
    var_x = (x_anom**2).sum(sample_dim, skipna=True)
    var_y = (y_anom**2).sum(sample_dim, skipna=True)

    slope = (cov_xy / var_x).where(n >= 3)
    intercept = (y_mean - slope * x_mean).where(n >= 3)
    corr = (cov_xy / np.sqrt(var_x * var_y)).where(n >= 3)
    tval = (corr * np.sqrt((n - 2) / (1 - corr**2))).where(n >= 3)

    return xr.Dataset({"slope": slope, "intercept": intercept, "corr": corr, "tval": tval, "n": n})


def build_regression_dataset(
    diag_file: Path,
    nino_index: xr.DataArray,
    year_start: int,
    year_end: int,
    model_t_threshold: float,
    obs_t_threshold: float,
) -> xr.Dataset:
    ds_diag = xr.open_dataset(diag_file)
    required = {"tc_track_density_mean", "obs_track_density_mean"}
    missing = sorted(required.difference(ds_diag.data_vars))
    if missing:
        raise ValueError(f"{diag_file} is missing required variables: {missing}")

    analysis_years = np.arange(year_start, year_end + 1)
    diag_years = ds_diag["year"].values.astype(int)
    nino_years = nino_index["year"].values.astype(int)
    common_years = np.intersect1d(np.intersect1d(analysis_years, diag_years), nino_years)
    if common_years.size < 3:
        raise ValueError(
            "Need at least 3 overlapping years; "
            f"analysis={analysis_years[0]}-{analysis_years[-1]}, "
            f"diag={diag_years.min()}-{diag_years.max()}, "
            f"nino={nino_years.min()}-{nino_years.max()}"
        )

    ds_diag = ds_diag.sel(year=common_years)
    nino_index = nino_index.sel(year=common_years)

    model_density = ds_diag["tc_track_density_mean"]
    if "sample_count" in ds_diag:
        model_density = model_density.where(ds_diag["sample_count"] > 0)

    reg_model = regress_y_on_x(model_density, nino_index)
    reg_obs = regress_y_on_x(ds_diag["obs_track_density_mean"], nino_index)

    ds_reg = xr.Dataset(
        {
            "model_slope": reg_model["slope"],
            "model_intercept": reg_model["intercept"],
            "model_corr": reg_model["corr"],
            "model_tval": reg_model["tval"],
            "model_n": reg_model["n"],
            "obs_slope": reg_obs["slope"],
            "obs_intercept": reg_obs["intercept"],
            "obs_corr": reg_obs["corr"],
            "obs_tval": reg_obs["tval"],
            "obs_n": reg_obs["n"],
            "model_slope_sig": reg_model["slope"].where(np.abs(reg_model["tval"]) >= model_t_threshold, 0.0),
            "obs_slope_sig": reg_obs["slope"].where(np.abs(reg_obs["tval"]) >= obs_t_threshold, 0.0),
            "nino_index": nino_index,
        }
    )
    ds_reg.attrs.update(
        {
            "description": "Regression of TC-season track-density diagnostics against seasonal Nino3.4.",
            "density_file": str(diag_file),
            "regression_years": f"{common_years.min()}-{common_years.max()}",
            "model_significance_t_threshold": model_t_threshold,
            "obs_significance_t_threshold": obs_t_threshold,
        }
    )
    return ds_reg


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    nino_out = Path(args.nino_out) if args.nino_out else outdir / (
        f"nino34_hadisst2_monthly_anom_clim{args.clim_start}_{args.clim_end}.nc"
    )
    if args.reg_out:
        reg_out = Path(args.reg_out)
    else:
        diag_stem = Path(args.diag_file).stem
        if diag_stem.startswith("tc_lead_track_density_"):
            reg_stem = diag_stem.replace("tc_lead_track_density_", "tc_lead_track_density_enso_regression_", 1)
        else:
            reg_stem = f"{diag_stem}_enso_regression"
        reg_out = outdir / f"{reg_stem}_hadisst2.nc"

    nino_ds = compute_monthly_nino34(Path(args.sst_file), args.sst_var, args.clim_start, args.clim_end)
    nino_ds["seasonal_nino34_index"] = seasonal_nino34(nino_ds["nino34"])
    nino_out.parent.mkdir(parents=True, exist_ok=True)
    nino_ds.to_netcdf(nino_out)
    print(f"Saved monthly/seasonal Nino3.4: {nino_out}")

    ds_reg = build_regression_dataset(
        Path(args.diag_file),
        nino_ds["seasonal_nino34_index"],
        args.year_start,
        args.year_end,
        args.model_t_threshold,
        args.obs_t_threshold,
    )
    ds_reg.attrs.update({"nino_file": str(nino_out), "nino_variable": "seasonal_nino34_index"})
    reg_out.parent.mkdir(parents=True, exist_ok=True)
    ds_reg.to_netcdf(reg_out)
    print(f"Saved ENSO regression diagnostics: {reg_out}")
    print(ds_reg)


if __name__ == "__main__":
    main()
