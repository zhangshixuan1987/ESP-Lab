"""Reusable teleconnection diagnostic workflow for SST indices and downstream fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as scipy_stats

DEFAULT_DIAG_ROOT = Path("/global/cfs/cdirs/e3sm/S2S2D/s2d_diag")
DEFAULT_OUTPUT_DIR = Path("/global/cfs/cdirs/e3sm/S2S2D/s2d_diag/teleconnections")
DEFAULT_FIGURE_DIR = Path("/global/cfs/cdirs/e3sm/www/zhan391/esp-lab_diag/teleconnections")

MEMBER_DIMS = ("M", "member", "ensemble")

E3SM_CASES: dict[str, dict[str, Any]] = {
    "E3SM-FOSIRL": {
        "cache_tag": "JRA55_FOSIRL",
        "display_name": "E3SMv3-FOSIRL",
        "color": "black",
        "supports_land": True,
    },
    "E3SM-Reanalysis": {
        "cache_tag": "Reanalysis",
        "display_name": "E3SMv3-Reanalysis",
        "color": "tab:blue",
        "supports_land": True,
    },
    "E3SM-4DEnVarOcn": {
        "cache_tag": "4DEnVarOcn",
        "display_name": "E3SMv3-4DEnVarOcn",
        "color": "tab:purple",
        "supports_land": False,
    },
}

SST_INDEX_REGIONS: dict[str, list[float]] = {
    "Nino12": [270.0, 280.0, -10.0, 0.0],
    "Nino3": [210.0, 270.0, -5.0, 5.0],
    "Nino3.4": [190.0, 240.0, -5.0, 5.0],
    "Nino4": [160.0, 210.0, -5.0, 5.0],
    "TNA": [305.0, 345.0, 5.0, 25.0],
    "TSA": [330.0, 10.0, -20.0, 0.0],
    "PACWRAMPOOL": [60.0, 170.0, -15.0, 15.0],
    "AtlNino": [340.0, 360.0, -3.0, 3.0],
    "AtlMDR": [280.0, 350.0, 10.0, 20.0],
    "IOD": [50.0, 110.0, -10.0, 10.0],
    "TNI": [160.0, 280.0, -10.0, 10.0],
    "ONI": [190.0, 240.0, -5.0, 5.0],
    "RONI": [190.0, 240.0, -20.0, 20.0],
}

DOWNSTREAM_VARIABLES: dict[str, dict[str, Any]] = {
    "TREFHT": {
        "realm": "atm",
        "family": "atmosphere",
        "label": "2-m air temperature",
        "units": "degC",
    },
    "TS": {
        "realm": "atm",
        "family": "atmosphere",
        "label": "surface temperature",
        "units": "degC",
    },
    "PRECT": {
        "realm": "atm",
        "family": "atmosphere",
        "label": "precipitation",
        "units": "mm/day",
    },
    "PSL": {
        "realm": "atm",
        "family": "atmosphere",
        "label": "sea-level pressure",
        "units": "hPa",
    },
    "SST": {
        "realm": "ocn",
        "family": "atmosphere",
        "label": "sea-surface temperature",
        "units": "degC",
    },
    "H2OSNO": {
        "realm": "lnd",
        "family": "land",
        "label": "snow water equivalent",
        "reference": "C3S_SWE",
        "units": "mm",
    },
    "H2OSOI": {
        "realm": "lnd",
        "family": "land",
        "label": "0-1.6 m soil moisture",
        "reference": "CPC_Soil_Moisture_V2",
        "units": "mm",
        "depth_token": "0-1.6m",
    },
}


def parse_init_years(raw_values: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Extract 4-digit calendar years from initialization values (e.g. '1980050100' or 1980)."""
    flat = np.asarray(raw_values).ravel()
    parsed = []
    for val in flat:
        if hasattr(val, "year"):
            parsed.append(int(val.year))
        else:
            s = str(val).strip()
            # If string is at least 4 digits, first 4 characters are the year
            parsed.append(int(s[:4]))
    return np.asarray(parsed, dtype=int).reshape(np.asarray(raw_values).shape)


def time_year_month(values: Sequence[Any] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return year and month arrays from an array of date/timestamp objects."""
    flat = np.asarray(values).ravel()
    years: list[int] = []
    months: list[int] = []
    for value in flat:
        if hasattr(value, "year") and hasattr(value, "month"):
            years.append(int(value.year))
            months.append(int(value.month))
        else:
            stamp = pd.Timestamp(value)
            years.append(int(stamp.year))
            months.append(int(stamp.month))
    shape = np.asarray(values).shape
    return np.asarray(years, dtype=int).reshape(shape), np.asarray(months, dtype=int).reshape(shape)


def lead_signature(
    valid_time: xr.DataArray, lead: int, init_dim: str = "Y"
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the unique lead-offset years and target calendar months for a given lead."""
    selected = valid_time.sel(L=lead)
    years, months = time_year_month(selected.values)
    init_years = parse_init_years(selected[init_dim].values)
    year_offsets = tuple(sorted(set(int(y) - int(iy) for y, iy in zip(years.ravel(), init_years.ravel()))))
    cal_months = tuple(sorted(set(int(m) for m in months.ravel())))
    return year_offsets, cal_months


def match_leads(
    index_time: xr.DataArray, field_time: xr.DataArray, init_dim: str = "Y"
) -> list[tuple[int, int]]:
    """Map SST index lead coordinates to field lead coordinates by matching target dates."""
    field_by_signature: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    for field_lead in field_time.L.values:
        sig = lead_signature(field_time, int(field_lead), init_dim)
        field_by_signature[sig] = int(field_lead)

    pairs: list[tuple[int, int]] = []
    for index_lead in index_time.L.values:
        sig = lead_signature(index_time, int(index_lead), init_dim)
        if sig in field_by_signature:
            pairs.append((int(index_lead), field_by_signature[sig]))
    return pairs


def ensemble_mean(da: xr.DataArray) -> xr.DataArray:
    """Compute ensemble mean over any present member dimension."""
    dims = [d for d in MEMBER_DIMS if d in da.dims]
    return da.mean(dims, skipna=True) if dims else da


def linear_detrend(da: xr.DataArray, dim: str = "sample") -> xr.DataArray:
    """Remove linear trend along dimension."""
    if dim not in da.dims or da.sizes[dim] < 3:
        return da
    x = xr.DataArray(np.arange(da.sizes[dim], dtype=float), dims=dim, coords={dim: da[dim]})
    coef = da.polyfit(dim=dim, deg=1, skipna=True)
    trend = xr.polyval(x, coef.polyfit_coefficients)
    return da - trend


def monthly_anomaly(obs: xr.DataArray, clim_years: tuple[int, int]) -> xr.DataArray:
    """Compute monthly anomalies from observational monthly series using climatology years."""
    if "time" not in obs.dims:
        return obs
    y0, y1 = clim_years
    clim = obs.sel(time=slice(str(y0), str(y1))).groupby("time.month").mean("time", skipna=True)
    return obs.groupby("time.month") - clim


def lead_anomaly(
    fcst: xr.DataArray, valid_time: xr.DataArray, clim_years: tuple[int, int]
) -> xr.DataArray:
    """Remove lead-dependent mean across initialization years within climatology years."""
    if bool(fcst.attrs.get("is_anomaly", False)) or fcst.name == "anomaly":
        return fcst
    y0, y1 = clim_years
    init_dim = "Y" if "Y" in fcst.dims else "year"
    init_years = parse_init_years(fcst[init_dim].values)
    mask = xr.DataArray((init_years >= y0) & (init_years <= y1), dims=init_dim, coords={init_dim: fcst[init_dim]})
    clim = fcst.where(mask).mean(init_dim, skipna=True)
    return fcst - clim


def observed_at_valid_time(obs: xr.DataArray, valid_time_for_lead: xr.DataArray) -> xr.DataArray:
    """Extract 2D observation slices corresponding to the valid dates of a lead DataArray."""
    years, months = time_year_month(valid_time_for_lead.values)
    keys = pd.MultiIndex.from_arrays([obs.time.dt.year.values, obs.time.dt.month.values], names=["year", "month"])
    lookup = {(int(y), int(m)): i for i, (y, m) in enumerate(keys)}
    positions = [lookup.get((int(y), int(m)), -1) for y, m in zip(years.ravel(), months.ravel())]
    pieces = []
    for pos in positions:
        if pos < 0:
            pieces.append(xr.full_like(obs.isel(time=0, drop=True), np.nan))
        else:
            pieces.append(obs.isel(time=pos, drop=True))
    sample = xr.DataArray(np.arange(len(pieces)), dims="sample", name="sample")
    return xr.concat(pieces, dim=sample)


def corr_and_p(
    x: xr.DataArray, y: xr.DataArray, minimum_years: int = 20
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Pairwise Pearson correlation and two-sided p-value along sample dimension."""
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum("sample")
    xv = x.where(valid)
    yv = y.where(valid)
    xa = xv - xv.mean("sample", skipna=True)
    ya = yv - yv.mean("sample", skipna=True)
    denom = np.sqrt((xa ** 2).sum("sample", skipna=True) * (ya ** 2).sum("sample", skipna=True))
    r = xr.where(denom > 0, (xa * ya).sum("sample", skipna=True) / denom, np.nan)
    r = r.clip(-1.0, 1.0).where(n >= minimum_years)
    dof = n - 2
    denom_t = 1.0 - r ** 2
    t = xr.where(
        denom_t > 0,
        r * np.sqrt(dof / denom_t),
        xr.where(np.abs(r) >= 1.0, np.sign(r) * np.inf, np.nan),
    )
    p = xr.apply_ufunc(
        lambda z, df: 2.0 * scipy_stats.t.sf(np.abs(z), df),
        t,
        dof,
        dask="parallelized",
        output_dtypes=[float],
    )
    p = xr.where(np.isinf(t), 0.0, p)
    return r.rename("correlation"), p.rename("pvalue"), n.rename("sample_count")


def lat_name(da: xr.DataArray) -> str:
    """Find latitude coordinate name."""
    for name in ("lat", "latitude"):
        if name in da.coords:
            return name
    raise ValueError("No latitude coordinate found")


def weighted_spatial_metrics(
    model_map: xr.DataArray,
    obs_map: xr.DataArray,
    model_p: xr.DataArray,
    obs_p: xr.DataArray,
    *,
    latitude_bounds: tuple[float, float] = (-80.0, 80.0),
    area_weighted: bool = True,
    sign_agreement_threshold: float = 0.0,
    alpha: float = 0.10,
) -> xr.Dataset:
    """Calculate area-weighted pattern fidelity metrics between model and observed correlation maps."""
    lat = lat_name(model_map)
    lo, hi = latitude_bounds
    domain = (model_map[lat] >= lo) & (model_map[lat] <= hi)
    valid = np.isfinite(model_map) & np.isfinite(obs_map) & domain
    weights = xr.ones_like(model_map)
    if area_weighted:
        weights = np.cos(np.deg2rad(model_map[lat])).broadcast_like(model_map)
    weights = weights.where(valid)

    spatial_dims = [d for d in model_map.dims if d in ("lat", "latitude", "lon", "longitude")]
    wsum = weights.sum(spatial_dims, skipna=True)
    wsum_safe = xr.where(wsum > 0, wsum, np.nan)

    mm = (model_map * weights).sum(spatial_dims, skipna=True) / wsum_safe
    om = (obs_map * weights).sum(spatial_dims, skipna=True) / wsum_safe
    ma, oa = model_map - mm, obs_map - om

    cov = (ma * oa * weights).sum(spatial_dims, skipna=True)
    var_m = (ma ** 2 * weights).sum(spatial_dims, skipna=True)
    var_o = (oa ** 2 * weights).sum(spatial_dims, skipna=True)

    pattern_corr = xr.where(var_m * var_o > 0, cov / np.sqrt(var_m * var_o), np.nan)
    crmse = np.sqrt((((ma - oa) ** 2) * weights).sum(spatial_dims, skipna=True) / wsum_safe)
    slope = xr.where(var_o > 0, cov / var_o, np.nan)
    amplitude_ratio = xr.where(var_o > 0, np.sqrt(var_m / var_o), np.nan)

    active = valid & ((abs(model_map) > sign_agreement_threshold) | (abs(obs_map) > sign_agreement_threshold))
    active_weight = weights.where(active).sum(spatial_dims, skipna=True)
    sign_agree_weight = weights.where(active & (np.sign(model_map) == np.sign(obs_map))).sum(
        spatial_dims, skipna=True
    )
    sign_agreement = xr.where(active_weight > 0, sign_agree_weight / active_weight, np.nan)

    model_sig_fraction = (
        weights.where(valid & (model_p < alpha)).sum(spatial_dims, skipna=True) / wsum_safe
    )
    obs_sig_fraction = (
        weights.where(valid & (obs_p < alpha)).sum(spatial_dims, skipna=True) / wsum_safe
    )
    sig_overlap_fraction = (
        weights.where(valid & (model_p < alpha) & (obs_p < alpha)).sum(spatial_dims, skipna=True)
        / wsum_safe
    )

    return xr.Dataset({
        "pattern_correlation": pattern_corr,
        "centered_rmse": crmse,
        "regression_slope": slope,
        "amplitude_ratio": amplitude_ratio,
        "sign_agreement_fraction": sign_agreement,
        "model_significant_area_fraction": model_sig_fraction,
        "observed_significant_area_fraction": obs_sig_fraction,
        "significant_overlap_area_fraction": sig_overlap_fraction,
    })


def _attrs_match(ds: xr.Dataset, expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        actual = ds.attrs.get(key)
        if actual is None:
            continue
        if str(actual) != str(value):
            return False
    return True


def select_cache(
    candidates: Sequence[Path | str],
    *,
    required_vars: Sequence[str] = (),
    expected_attrs: Mapping[str, Any] | None = None,
    description: str = "cache",
    allow_ambiguous: bool = False,
) -> Path:
    """Select a unique matching NetCDF cache file from a list of candidates."""
    expected_attrs = expected_attrs or {}
    valid: list[Path] = []
    rejected: list[tuple[Path, str]] = []
    for raw in sorted(set(map(Path, candidates))):
        path = Path(raw)
        if not path.is_file():
            continue
        try:
            with xr.open_dataset(path) as ds:
                missing = set(required_vars) - set(ds.variables)
                if missing:
                    rejected.append((path, f"missing variables: {sorted(missing)}"))
                    continue
                if not _attrs_match(ds, expected_attrs):
                    rejected.append((path, "metadata attribute mismatch"))
                    continue
            valid.append(path)
        except Exception as exc:
            rejected.append((path, f"unreadable: {exc}"))

    if not valid:
        detail = "\n".join(f"  - {p}: {why}" for p, why in rejected[-8:])
        raise FileNotFoundError(f"No compatible {description} found.\n{detail}")
    if len(valid) > 1 and not allow_ambiguous:
        raise RuntimeError(
            f"Ambiguous {description}; compatible candidates:\n  - " + "\n  - ".join(map(str, valid))
        )
    return max(valid, key=lambda p: p.stat().st_mtime) if len(valid) > 1 else valid[0]


def upstream_sst_paths(
    system: str, init_month: int, index_name: str, diag_root: Path = DEFAULT_DIAG_ROOT
) -> tuple[Path, Path]:
    """Return paths to forecast and observed SST index seasonal timeseries files."""
    case = E3SM_CASES[system]
    base = diag_root / case["cache_tag"] / "sst_index" / "timeseries"
    forecast = base / f"E3SMLE{init_month:02d}_TS_N10_M24_{index_name}SST_seas.nc"
    observed = (
        diag_root / "HadISST2" / "sst_index" / "timeseries" / f"HadISST2_sst_{index_name}SST_seas.nc"
    )
    return forecast, observed


def atmospheric_candidates(
    system: str, init_month: int, variable: str, diag_root: Path = DEFAULT_DIAG_ROOT
) -> tuple[list[Path], list[Path]]:
    """Find candidate files for atmospheric prepared forecasts and observations."""
    tag = E3SM_CASES[system]["cache_tag"]
    realm = DOWNSTREAM_VARIABLES[variable]["realm"]
    roots = [
        diag_root / tag / "prepared_skill" / realm / variable,
        diag_root / tag / "leadtime_acc" / "prepared_skill" / realm / variable,
    ]
    model = [p for root in roots for p in root.rglob(f"*{init_month:02d}*{variable}*.nc") if root.exists()]
    obs_roots = [
        diag_root / "observations" / "prepared_skill" / realm / variable,
        diag_root / "observations" / "leadtime_acc" / "prepared_skill" / realm / variable,
    ]
    obs = [p for root in obs_roots for p in root.rglob("*.nc") if root.exists()]
    return model, obs


def land_candidates(
    system: str, init_month: int, variable: str, diag_root: Path = DEFAULT_DIAG_ROOT
) -> tuple[list[Path], list[Path]]:
    """Find candidate files for land prepared forecasts and observations without invalid wildcards."""
    tag = E3SM_CASES[system]["cache_tag"]
    depth = DOWNSTREAM_VARIABLES[variable].get("depth_token", "")
    pattern = f"*{init_month:02d}*{depth}*.nc" if depth else f"*{init_month:02d}*.nc"
    obs_pattern = f"*{depth}*.nc" if depth else "*.nc"

    m_root = diag_root / tag / "leadtime_acc" / "inputs" / "land" / variable
    model = list(m_root.glob(pattern)) if m_root.is_dir() else []
    ref = DOWNSTREAM_VARIABLES[variable]["reference"]
    o_root = diag_root / ref / "leadtime_acc" / "inputs" / "land" / variable
    obs = list(o_root.glob(obs_pattern)) if o_root.is_dir() else []
    return model, obs


def resolve_downstream_paths(
    system: str,
    init_month: int,
    variable: str,
    *,
    diag_root: Path = DEFAULT_DIAG_ROOT,
    target_grid: str = "latlon_1.0x1.0_periodic-True",
    allow_ambiguous: bool = False,
) -> tuple[Path, Path]:
    """Resolve downstream prepared forecast and observation paths."""
    spec = DOWNSTREAM_VARIABLES[variable]
    if spec["family"] == "atmosphere":
        model_candidates, obs_candidates = atmospheric_candidates(
            system, init_month, variable, diag_root=diag_root
        )
        model = select_cache(
            model_candidates,
            required_vars=("anomaly", "time"),
            expected_attrs={
                "variable": variable,
                "initialization_month": init_month,
                "target_grid": target_grid,
            },
            description=f"1a {system} {variable} init {init_month:02d} prepared anomaly",
            allow_ambiguous=allow_ambiguous,
        )
        obs = select_cache(
            obs_candidates,
            required_vars=("observation",),
            expected_attrs={"variable": variable, "target_grid": target_grid},
            description=f"1a {variable} prepared observation",
            allow_ambiguous=allow_ambiguous,
        )
    else:
        if not E3SM_CASES[system].get("supports_land", True):
            raise FileNotFoundError(f"{system} has no prepared land products for {variable}")
        model_candidates, obs_candidates = land_candidates(
            system, init_month, variable, diag_root=diag_root
        )
        model = select_cache(
            model_candidates,
            required_vars=(variable, "time"),
            expected_attrs={"field": variable, "initialization_month": init_month},
            description=f"1b {system} {variable} init {init_month:02d} prepared forecast",
            allow_ambiguous=allow_ambiguous,
        )
        obs = select_cache(
            obs_candidates,
            required_vars=(variable,),
            expected_attrs={"field": variable},
            description=f"1b {variable} prepared reference",
            allow_ambiguous=allow_ambiguous,
        )
    return model, obs


def build_teleconnection_inventory(config: Mapping[str, Any]) -> pd.DataFrame:
    """Scan and verify availability of all required upstream products."""
    diag_root = Path(config["paths"].get("diag_root", DEFAULT_DIAG_ROOT))
    index_name = config["selection"]["upstream_index"]
    target_grid = config["selection"].get("target_grid", "latlon_1.0x1.0_periodic-True")
    allow_ambiguous = config["cache"].get("allow_ambiguous_matches", False)

    rows: list[dict[str, Any]] = []
    for system in config["selection"]["systems"]:
        for init_month in config["selection"]["init_months"]:
            idx_fcst, idx_obs = upstream_sst_paths(system, init_month, index_name, diag_root=diag_root)
            for variable in config["selection"]["downstream_variables"]:
                spec = DOWNSTREAM_VARIABLES[variable]
                # Check if system supports this variable realm
                if spec["family"] == "land" and not E3SM_CASES[system].get("supports_land", True):
                    rows.append({
                        "system": system,
                        "init_month": init_month,
                        "index": index_name,
                        "variable": variable,
                        "index_forecast": str(idx_fcst),
                        "index_observed": str(idx_obs),
                        "field_forecast": None,
                        "field_observed": None,
                        "status": "skipped",
                        "detail": f"{system} does not produce land diagnostics",
                    })
                    continue

                try:
                    fld_fcst, fld_obs = resolve_downstream_paths(
                        system,
                        init_month,
                        variable,
                        diag_root=diag_root,
                        target_grid=target_grid,
                        allow_ambiguous=allow_ambiguous,
                    )
                    status = "ready" if idx_fcst.is_file() and idx_obs.is_file() else "missing"
                    detail = ""
                except Exception as exc:
                    fld_fcst = fld_obs = None
                    status = "missing"
                    detail = str(exc).splitlines()[0]

                rows.append({
                    "system": system,
                    "init_month": init_month,
                    "index": index_name,
                    "variable": variable,
                    "index_forecast": str(idx_fcst),
                    "index_observed": str(idx_obs),
                    "field_forecast": str(fld_fcst) if fld_fcst else None,
                    "field_observed": str(fld_obs) if fld_obs else None,
                    "status": status,
                    "detail": detail,
                })
    return pd.DataFrame(rows)


def open_inputs(
    system: str, init_month: int, variable: str, config: Mapping[str, Any]
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, list[Path]]:
    """Open and extract forecast & observed SST index and downstream field DataArrays."""
    diag_root = Path(config["paths"].get("diag_root", DEFAULT_DIAG_ROOT))
    index_name = config["selection"]["upstream_index"]
    target_grid = config["selection"].get("target_grid", "latlon_1.0x1.0_periodic-True")
    allow_ambiguous = config["cache"].get("allow_ambiguous_matches", False)

    idx_fcst_path, idx_obs_path = upstream_sst_paths(system, init_month, index_name, diag_root=diag_root)
    fld_fcst_path, fld_obs_path = resolve_downstream_paths(
        system, init_month, variable, diag_root=diag_root, target_grid=target_grid, allow_ambiguous=allow_ambiguous
    )

    idx_fcst_ds = xr.open_dataset(idx_fcst_path)
    idx_obs_ds = xr.open_dataset(idx_obs_path)
    fld_fcst_ds = xr.open_dataset(fld_fcst_path)
    fld_obs_ds = xr.open_dataset(fld_obs_path)

    idx_fcst = idx_fcst_ds["sst"]
    idx_time = idx_fcst_ds["time"]
    idx_obs = idx_obs_ds["sst"]

    fld_fcst = fld_fcst_ds["anomaly"] if "anomaly" in fld_fcst_ds else fld_fcst_ds[variable]
    fld_time = fld_fcst_ds["time"]
    fld_obs = fld_obs_ds["observation"] if "observation" in fld_obs_ds else fld_obs_ds[variable]

    sources = [idx_fcst_path, idx_obs_path, fld_fcst_path, fld_obs_path]
    return idx_fcst, idx_time, idx_obs, fld_fcst, fld_time, fld_obs, sources


def compute_system_teleconnection(
    system: str, init_month: int, variable: str, config: Mapping[str, Any]
) -> xr.Dataset:
    """Compute lead-dependent teleconnection metrics for one (system, init_month, variable) tuple."""
    idx_fcst, idx_time, idx_obs, fld_fcst, fld_time, fld_obs, sources = open_inputs(
        system, init_month, variable, config
    )
    init_dim = "Y" if "Y" in idx_fcst.dims else "year"

    clim_years = config["selection"]["climatology_years"]
    idx_fcst = lead_anomaly(idx_fcst, idx_time, clim_years)
    fld_fcst = lead_anomaly(fld_fcst, fld_time, clim_years)
    idx_obs = monthly_anomaly(idx_obs, clim_years)
    fld_obs = monthly_anomaly(fld_obs, clim_years)

    lead_pairs = match_leads(idx_time, fld_time, init_dim)
    requested = config["selection"].get("leads")
    if requested is not None:
        requested_set = set(map(int, requested))
        lead_pairs = [(idx_lead, fld_lead) for idx_lead, fld_lead in lead_pairs if idx_lead in requested_set]
    if not lead_pairs:
        raise ValueError(f"No common requested leads for {system}, {variable}, init={init_month}")

    y0, y1 = config["selection"]["verification_years"]
    idx_init_years = parse_init_years(idx_fcst[init_dim].values)
    fld_init_years = parse_init_years(fld_fcst[init_dim].values)

    common_years = sorted(list(set(idx_init_years) & set(fld_init_years) & set(range(y0, y1 + 1))))
    if len(common_years) < config["analysis"].get("minimum_years", 20):
        raise ValueError(
            f"Insufficient common verification years ({len(common_years)}) for {system}, {variable}, init={init_month}"
        )

    idx_pos = [i for i, y in enumerate(idx_init_years) if y in common_years]
    fld_pos = [i for i, y in enumerate(fld_init_years) if y in common_years]

    outputs: list[xr.Dataset] = []
    sample = xr.DataArray(np.arange(len(common_years)), dims="sample", name="sample")

    for index_lead, field_lead in lead_pairs:
        idx_t = idx_time.sel(L=index_lead).isel({init_dim: idx_pos})
        fld_t = fld_time.sel(L=field_lead).isel({init_dim: fld_pos})
        iy, im = time_year_month(idx_t.values)
        fy, fm = time_year_month(fld_t.values)
        if not (np.array_equal(iy, fy) and np.array_equal(im, fm)):
            raise ValueError(
                f"SST index L={index_lead} and {variable} L={field_lead} target dates differ for common starts"
            )

        xmod = (
            ensemble_mean(idx_fcst.sel(L=index_lead).isel({init_dim: idx_pos}))
            .rename({init_dim: "sample"})
            .assign_coords(sample=sample)
        )
        ymod = (
            ensemble_mean(fld_fcst.sel(L=field_lead).isel({init_dim: fld_pos}))
            .rename({init_dim: "sample"})
            .assign_coords(sample=sample)
        )
        xobs = observed_at_valid_time(idx_obs, idx_t)
        yobs = observed_at_valid_time(fld_obs, fld_t)

        xmod, ymod = xr.align(xmod, ymod, join="inner")
        xobs, yobs = xr.align(xobs, yobs, join="inner")

        if config["analysis"].get("detrend", True):
            xmod, ymod, xobs, yobs = map(linear_detrend, (xmod, ymod, xobs, yobs))

        model_r, model_p, model_n = corr_and_p(xmod, ymod, minimum_years=config["analysis"].get("minimum_years", 20))
        obs_r, obs_p, obs_n = corr_and_p(xobs, yobs, minimum_years=config["analysis"].get("minimum_years", 20))
        model_r, obs_r = xr.align(model_r, obs_r, join="exact")

        metrics = weighted_spatial_metrics(
            model_r,
            obs_r,
            model_p,
            obs_p,
            latitude_bounds=config["analysis"].get("latitude_bounds", (-80.0, 80.0)),
            area_weighted=config["analysis"].get("area_weighted", True),
            sign_agreement_threshold=config["analysis"].get("sign_agreement_threshold", 0.0),
            alpha=config["analysis"].get("alpha", 0.10),
        )

        ds = xr.Dataset({
            "model_correlation": model_r,
            "model_pvalue": model_p,
            "observed_correlation": obs_r,
            "observed_pvalue": obs_p,
            "model_sample_count": model_n,
            "observed_sample_count": obs_n,
            **{name: da for name, da in metrics.data_vars.items()},
            "downstream_lead": xr.DataArray(int(field_lead)),
        }).expand_dims(L=[int(index_lead)])
        outputs.append(ds)

    result = xr.concat(outputs, dim="L").expand_dims(
        init_month=[init_month], system=[system], variable=[variable]
    )
    result.attrs["source_files"] = json.dumps([str(p) for p in sources])
    return result


def file_signature(paths: Sequence[Path | str]) -> list[dict[str, Any]]:
    """Compute file metadata signatures for provenance tracking."""
    records: list[dict[str, Any]] = []
    for path in sorted(set(map(Path, paths))):
        if path.is_file():
            stat = path.stat()
            records.append({
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
    return records


def compute_provenance_fingerprint(
    config: Mapping[str, Any], source_paths: Sequence[Path | str]
) -> str:
    """Generate a SHA-256 fingerprint hash for configuration and input file provenance."""
    provenance = {
        "schema": "teleconnection_metrics_v1",
        "selection": config["selection"],
        "analysis": config["analysis"],
        "sources": file_signature(source_paths),
    }
    return hashlib.sha256(
        json.dumps(provenance, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def standardize_spatial_grid(
    ds: xr.Dataset,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> xr.Dataset:
    """Align 2D spatial fields in a teleconnection dataset to a common target grid."""
    if "lat" not in ds.dims or "lon" not in ds.dims:
        return ds
    if np.array_equal(ds.lat.values, target_lat) and np.array_equal(ds.lon.values, target_lon):
        return ds

    spatial_vars = [k for k, v in ds.data_vars.items() if "lat" in v.dims and "lon" in v.dims]
    non_spatial_vars = [k for k, v in ds.data_vars.items() if k not in spatial_vars]
    new_vars = {k: ds[k] for k in non_spatial_vars}

    cur_lon = ds.lon.values
    cur_lat = ds.lat.values
    padded_ds = ds[spatial_vars]

    if cur_lon.min() > target_lon.min() or cur_lon.max() < target_lon.max():
        left_pad = padded_ds.isel(lon=[-1]).assign_coords(lon=[cur_lon[-1] - 360.0])
        right_pad = padded_ds.isel(lon=[0]).assign_coords(lon=[cur_lon[0] + 360.0])
        padded_ds = xr.concat([left_pad, padded_ds, right_pad], dim="lon")
    if cur_lat.min() > target_lat.min() or cur_lat.max() < target_lat.max():
        s_pad = padded_ds.isel(lat=[0]).assign_coords(lat=[target_lat.min()])
        n_pad = padded_ds.isel(lat=[-1]).assign_coords(lat=[target_lat.max()])
        padded_ds = xr.concat([s_pad, padded_ds, n_pad], dim="lat")

    interp_spatial = padded_ds.interp(lat=target_lat, lon=target_lon, method="linear")
    for k in spatial_vars:
        new_vars[k] = interp_spatial[k]

    return xr.Dataset(new_vars, attrs=ds.attrs)


def assemble_teleconnection_dataset(
    parts: Sequence[xr.Dataset],
    target_lat: np.ndarray | None = None,
    target_lon: np.ndarray | None = None,
) -> xr.Dataset:
    """Assemble individual teleconnection metric parts across systems, months, and variables.
    
    Standardizes spatial grids across atmospheric and land variables, reindexes each part
    to the global union of leads, and combines across systems, init_months, and variables.
    """
    if not parts:
        raise RuntimeError("No teleconnection datasets were computed from inventory.")

    if target_lat is None or target_lon is None:
        for p in parts:
            if "lat" in p.dims and len(p.lat) == 181:
                target_lat = p.lat.values
                target_lon = p.lon.values
                break
    if target_lat is None or target_lon is None:
        target_lat = np.linspace(-90.0, 90.0, 181)
        target_lon = np.linspace(0.0, 359.0, 360)

    all_leads = sorted(list(set(int(l) for p in parts for l in p.L.values)))

    aligned_parts = [
        standardize_spatial_grid(p, target_lat, target_lon).reindex(L=all_leads)
        for p in parts
    ]
    return xr.combine_by_coords(aligned_parts, combine_attrs="drop_conflicts")


def ensure_teleconnection_dataset(
    config: Mapping[str, Any],
    inventory: pd.DataFrame | None = None,
) -> tuple[xr.Dataset, Path, str]:
    """Load or compute cached teleconnection metrics across all configured products."""
    if inventory is None:
        inventory = build_teleconnection_inventory(config)

    missing = inventory.query("status == 'missing'")
    if not missing.empty:
        raise FileNotFoundError(
            "Required upstream products are missing. Please verify preparation workflows:\n"
            + missing.to_string(index=False)
        )

    all_source_paths: list[str] = []
    for row in inventory.itertuples():
        if getattr(row, "status", "") == "ready":
            all_source_paths.extend([
                row.index_forecast,
                row.index_observed,
                row.field_forecast,
                row.field_observed,
            ])

    fingerprint = compute_provenance_fingerprint(config, all_source_paths)
    index_name = config["selection"]["upstream_index"]
    output_dir = Path(config["paths"].get("output_dir", DEFAULT_OUTPUT_DIR))
    out_file = output_dir / f"teleconnection_{index_name.replace('.', '')}_{fingerprint}.nc"
    cache_mode = config["cache"].get("mode", "auto")

    if out_file.is_file() and cache_mode != "rebuild":
        metrics_ds = xr.open_dataset(out_file)
        return metrics_ds, out_file, "loaded"

    if cache_mode == "require":
        raise FileNotFoundError(f"Required teleconnection cache is unavailable: {out_file}")

    parts: list[xr.Dataset] = []
    for row in inventory.itertuples():
        if getattr(row, "status", "") != "ready":
            continue
        print(f"Computing {row.system} init={row.init_month:02d} {index_name} -> {row.variable}")
        part = compute_system_teleconnection(row.system, row.init_month, row.variable, config)
        parts.append(part)

    metrics_ds = assemble_teleconnection_dataset(parts)
    metrics_ds.attrs.update({
        "schema": "teleconnection_metrics_v1",
        "fingerprint": fingerprint,
        "upstream_index": index_name,
        "configuration_json": json.dumps(config, sort_keys=True, default=str),
        "source_inventory_json": json.dumps(file_signature(all_source_paths), sort_keys=True),
        "forecast_definition": "correlation across initialization years using ensemble-mean SST index and ensemble-mean downstream field",
        "observed_definition": "correlation across matching target dates using observed SST index and observed downstream anomaly",
        "season_definition": "centered three-month means inherited from upstream caches",
        "pvalue_note": "classical Pearson t test; no field-significance or autocorrelation correction",
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp.nc")
    metrics_ds.to_netcdf(tmp)
    tmp.replace(out_file)
    return metrics_ds, out_file, "computed"
