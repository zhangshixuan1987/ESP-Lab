"""
physical_core.py
================
S2D Physical-Consistency Diagnostics — pure core module.

Design principles
-----------------
* Contains NO I/O and NO ESP-Lab filesystem dependencies.
  Fully unit-testable without NERSC access or real files.
* Neutral S2D terminology: JRA55_FOSIRL, Reanalysis, JRA55_FOSIRL − Reanalysis.
* Processes non-linear physical metrics by evaluating the metric in each
  window first, then taking adjustments/differences (avoiding ratios of
  drift fields).

Diagnostics
-----------
1. Flux partitioning            (compute_evaporative_fraction, compute_bowen_ratio)
2. Ocean-atmosphere coupling    (compute_sst_trefht_contrast)
3. Soil moisture integration    (integrate_soil_moisture)
4. Soil moisture coupling       (compute_coupling_slope)
5. Precip-SM lag response       (compute_precip_sm_lag_response)
6. Apparent energy residual     (compute_apparent_energy_residual)
7. Paired bootstrap CI          (bootstrap_physical_metric_ci)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


# ===========================================================================
# Section 1 — Flux Partitioning
# ===========================================================================


def compute_evaporative_fraction(
    lhflx: xr.DataArray,
    shflx: xr.DataArray,
    min_flux_threshold: float = 1.0,
) -> Tuple[xr.DataArray, float]:
    """Compute Evaporative Fraction EF = sum(LHFLX) / sum(LHFLX + SHFLX).

    Parameters
    ----------
    lhflx:
        Latent heat flux DataArray (W/m²). Summed or averaged over the window.
    shflx:
        Sensible heat flux DataArray (W/m²).
    min_flux_threshold:
        Threshold below which net available energy (LHFLX + SHFLX) is masked
        (W/m²). Default 1.0.

    Returns
    -------
    (EF, masked_fraction) : (xr.DataArray, float)
        EF DataArray with invalid/negative-net-flux cells set to NaN.
        masked_fraction = fraction of spatial cells masked out.
    """
    net_flux = lhflx + shflx
    invalid_mask = (net_flux <= min_flux_threshold) | (lhflx < 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ef = lhflx / net_flux

    ef = ef.where(~invalid_mask)
    ef.name = "evaporative_fraction"
    ef.attrs["units"] = "dimensionless"
    ef.attrs["long_name"] = "Evaporative Fraction EF = LH / (LH + SH)"

    # QC masked fraction
    total_cells = float(ef.size)
    valid_cells = float(np.sum(np.isfinite(ef.values)))
    masked_fraction = 1.0 - (valid_cells / max(total_cells, 1.0))

    return ef, masked_fraction


def compute_bowen_ratio(
    lhflx: xr.DataArray,
    shflx: xr.DataArray,
    min_lhflx_threshold: float = 1.0,
    max_bowen_threshold: float = 100.0,
) -> Tuple[xr.DataArray, float]:
    """Compute Bowen Ratio BR = sum(SHFLX) / sum(LHFLX).

    Parameters
    ----------
    lhflx:
        Latent heat flux DataArray (W/m²).
    shflx:
        Sensible heat flux DataArray (W/m²).
    min_lhflx_threshold:
        Threshold below which LHFLX is considered near-zero and masked.
    max_bowen_threshold:
        Upper cap for extreme Bowen ratios to prevent numerical blowup.

    Returns
    -------
    (BR, masked_fraction) : (xr.DataArray, float)
    """
    invalid_mask = (lhflx <= min_lhflx_threshold)

    with np.errstate(divide="ignore", invalid="ignore"):
        br = shflx / lhflx

    br = br.where(~invalid_mask & (np.abs(br) <= max_bowen_threshold))
    br.name = "bowen_ratio"
    br.attrs["units"] = "dimensionless"
    br.attrs["long_name"] = "Bowen Ratio BR = SH / LH"

    total_cells = float(br.size)
    valid_cells = float(np.sum(np.isfinite(br.values)))
    masked_fraction = 1.0 - (valid_cells / max(total_cells, 1.0))

    return br, masked_fraction


# ===========================================================================
# Section 2 — Ocean-Atmosphere Coupling
# ===========================================================================


def compute_sst_trefht_contrast(
    trefht: xr.DataArray,
    sst: xr.DataArray,
) -> xr.DataArray:
    """Compute air-sea temperature contrast ΔT_AO = TREFHT - SST (°C or K).

    Parameters
    ----------
    trefht:
        Near-surface air temperature DataArray.
    sst:
        Sea surface temperature DataArray.

    Returns
    -------
    xr.DataArray named ``"sst_trefht_contrast"``.
    """
    diff = trefht - sst
    diff.name = "sst_trefht_contrast"
    diff.attrs["units"] = trefht.attrs.get("units", "degC")
    diff.attrs["long_name"] = "Air-sea temperature contrast TREFHT - SST"
    return diff


# ===========================================================================
# Section 3 — Soil Moisture Integration
# ===========================================================================


def integrate_soil_moisture(
    h2osoi: xr.DataArray,
    layer_depths: Optional[Sequence[float]] = None,
    lev_dim: str = "levgrnd",
) -> xr.DataArray:
    """Integrate soil moisture over layer depths (e.g. 0–1.6 m).

    Parameters
    ----------
    h2osoi:
        Soil moisture DataArray with a level dimension.
    layer_depths:
        Optional depth weights for each layer in meters. If None, simple mean across levels.
    lev_dim:
        Name of level dimension.

    Returns
    -------
    xr.DataArray integrated soil moisture.
    """
    if lev_dim not in h2osoi.dims:
        return h2osoi

    if layer_depths is not None:
        w = np.asarray(layer_depths, dtype=float)
        w_da = xr.DataArray(w, dims=[lev_dim], coords={lev_dim: h2osoi.coords[lev_dim]})
        sm = (h2osoi * w_da).sum(lev_dim, skipna=True)
    else:
        sm = h2osoi.mean(lev_dim, skipna=True)

    sm.name = "integrated_soil_moisture"
    sm.attrs["long_name"] = "Integrated soil moisture"
    return sm


# ===========================================================================
# Section 4 — Soil Moisture Coupling Slopes
# ===========================================================================


def compute_coupling_slope(
    x_anom: xr.DataArray,
    y_anom: xr.DataArray,
    sample_dim: str = "sample",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute linear regression slope β and correlation r between x_anom and y_anom.

    β = Cov(X', Y') / Var(X')
    r = Corr(X', Y')

    Parameters
    ----------
    x_anom:
        Predictor anomaly DataArray (e.g., Soil Moisture anomaly SM').
    y_anom:
        Target anomaly DataArray (e.g., LHFLX' or TREFHT').
    sample_dim:
        Dimension over which covariance/variance are computed (e.g., "Y" or "sample").

    Returns
    -------
    (slope, corr) : (xr.DataArray, xr.DataArray)
    """
    cov = (x_anom * y_anom).mean(sample_dim, skipna=True)
    var_x = (x_anom ** 2).mean(sample_dim, skipna=True)
    var_y = (y_anom ** 2).mean(sample_dim, skipna=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = cov / var_x
        corr  = cov / np.sqrt(var_x * var_y)

    slope.name = "coupling_slope"
    corr.name  = "coupling_corr"
    return slope, corr


# ===========================================================================
# Section 5 — Precipitation to Soil-Moisture Lag Response
# ===========================================================================


def compute_precip_sm_lag_response(
    prect: xr.DataArray,
    sm: xr.DataArray,
    max_lag: int = 7,
    day_dim: str = "d",
) -> xr.DataArray:
    """Compute lagged response slope β_{P -> ΔSM}(ℓ) for lags ℓ = 0..max_lag days.

    ΔSM_d = SM_{d+1} - SM_d
    β(ℓ) = Cov(PRECT_d', ΔSM_{d+ℓ}') / Var(PRECT_d')

    Parameters
    ----------
    prect:
        Daily precipitation rate DataArray with dimension ``d`` (1..84).
    sm:
        Daily integrated soil moisture DataArray with dimension ``d`` (1..84).
    max_lag:
        Maximum lag in days (default 7).
    day_dim:
        Name of daily forecast lead dimension.

    Returns
    -------
    xr.DataArray with dimension ``lag`` (0..max_lag).
    """
    if day_dim not in prect.dims or day_dim not in sm.dims:
        raise ValueError(f"Dimension '{day_dim}' missing from input fields.")

    days = [int(x) for x in prect.coords[day_dim].values]
    if len(days) < max_lag + 2:
        raise ValueError(f"Not enough days ({len(days)}) for max_lag={max_lag}.")

    # Soil moisture tendency ΔSM_d = SM_{d+1} - SM_d
    sm_vals = sm.sel({day_dim: days[:-1]})
    sm_next = sm.sel({day_dim: days[1:]}).assign_coords({day_dim: days[:-1]})
    delta_sm = sm_next - sm_vals

    # Anomalies relative to lead mean
    prect_anom = prect - prect.mean("Y" if "Y" in prect.dims else day_dim, skipna=True)
    dsm_anom   = delta_sm - delta_sm.mean("Y" if "Y" in delta_sm.dims else day_dim, skipna=True)

    lag_slopes = []
    for lag in range(max_lag + 1):
        if lag == 0:
            p_slice   = prect_anom.sel({day_dim: days[:-1]})
            dsm_slice = dsm_anom
        else:
            p_slice   = prect_anom.sel({day_dim: days[:-1-lag]})
            dsm_slice = dsm_anom.sel({day_dim: days[lag:-1]}).assign_coords({day_dim: days[:-1-lag]})

        sample_dims = [dim for dim in p_slice.dims if dim in ("Y", "M", day_dim)]
        cov   = (p_slice * dsm_slice).mean(sample_dims, skipna=True)
        var_p = (p_slice ** 2).mean(sample_dims, skipna=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            slope_lag = cov / var_p
        slope_lag = slope_lag.assign_coords(lag=lag)
        lag_slopes.append(slope_lag)

    result = xr.concat(lag_slopes, dim=pd.Index(range(max_lag + 1), name="lag"))
    result.name = "precip_sm_lag_response"
    result.attrs["long_name"] = "Precipitation to soil-moisture tendency lag response slope β(ℓ)"
    return result


# ===========================================================================
# Section 6 — Apparent Surface Energy Residual
# ===========================================================================


def compute_apparent_energy_residual(
    fsns: xr.DataArray,
    flns: xr.DataArray,
    lhflx: xr.DataArray,
    shflx: xr.DataArray,
) -> xr.DataArray:
    """Compute apparent surface energy residual R_apparent = FSNS - FLNS - LHFLX - SHFLX.

    Parameters
    ----------
    fsns:
        Net downward shortwave radiation at surface (W/m²).
    flns:
        Net upward longwave radiation at surface (W/m²).
    lhflx:
        Latent heat flux (upward positive, W/m²).
    shflx:
        Sensible heat flux (upward positive, W/m²).

    Returns
    -------
    xr.DataArray named ``"apparent_energy_residual"``.
    """
    r_app = fsns - flns - lhflx - shflx
    r_app.name = "apparent_energy_residual"
    r_app.attrs["units"] = "W/m²"
    r_app.attrs["long_name"] = "Apparent surface energy residual R = FSNS - FLNS - LH - SH"
    return r_app


# ===========================================================================
# Section 7 — Paired Bootstrap for Physical Metrics
# ===========================================================================


def bootstrap_physical_metric_ci(
    paired_metric_by_year: xr.DataArray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Bootstrap 95% CI for paired physical metric differences ΔM across init years.

    Parameters
    ----------
    paired_metric_by_year:
        DataArray with dim ``Y`` (init years), spatial or regional dims.

    Returns
    -------
    (lower, upper) DataArrays with same non-Y dims.
    """
    rng = np.random.default_rng(seed)
    data = paired_metric_by_year.load()
    n_y = data.sizes["Y"]

    if n_y < 2:
        raise ValueError(f"bootstrap_physical_metric_ci requires ≥ 2 init years; got {n_y}.")

    vals = data.values
    boot_means = np.empty((n_boot, *vals.shape[1:]), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n_y, size=n_y)
        boot_means[b] = np.nanmean(vals[idx], axis=0)

    lo_vals = np.nanquantile(boot_means, alpha / 2,     axis=0)
    hi_vals = np.nanquantile(boot_means, 1 - alpha / 2, axis=0)

    spatial_dims   = [dim for dim in data.dims if dim != "Y"]
    spatial_coords = {dim: data.coords[dim] for dim in spatial_dims if dim in data.coords}

    lower = xr.DataArray(lo_vals, dims=spatial_dims, coords=spatial_coords, name="ci_lower")
    upper = xr.DataArray(hi_vals, dims=spatial_dims, coords=spatial_coords, name="ci_upper")
    return lower, upper


__all__ = [
    "compute_evaporative_fraction",
    "compute_bowen_ratio",
    "compute_sst_trefht_contrast",
    "integrate_soil_moisture",
    "compute_coupling_slope",
    "compute_precip_sm_lag_response",
    "compute_apparent_energy_residual",
    "bootstrap_physical_metric_ci",
]
