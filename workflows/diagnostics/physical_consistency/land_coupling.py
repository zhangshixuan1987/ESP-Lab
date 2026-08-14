"""
land_coupling.py
================
Calculates soil-moisture coupling slopes β and correlations r for:
  1. SM -> LHFLX  (β_{SM->LH})
  2. SM -> TREFHT (β_{SM->T})
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import xarray as xr

from esp_lab.diagnostics.physical_core import (
    bootstrap_physical_metric_ci,
    compute_coupling_slope,
    integrate_soil_moisture,
)


def run_land_coupling(
    sm_ref: xr.DataArray,
    lhflx_ref: xr.DataArray,
    trefht_ref: xr.DataArray,
    sm_test: xr.DataArray,
    lhflx_test: xr.DataArray,
    trefht_test: xr.DataArray,
    window_defs: Dict[str, Tuple[int, int]],
    day_dim: str = "d",
    layer_thickness_ref: list[float] | None = None,
    layer_thickness_test: list[float] | None = None,
) -> Dict[str, dict]:
    sm_ref = integrate_soil_moisture(
        sm_ref, layer_thickness_ref, require_thickness=True
    )
    sm_test = integrate_soil_moisture(
        sm_test, layer_thickness_test, require_thickness=True
    )
    results = {}

    for win_name, (w_first, w_last) in window_defs.items():
        sel_days = [d for d in sm_ref.coords[day_dim].values if w_first <= d <= w_last]
        if not sel_days:
            continue

        # Slice window & average across window days to build anomalies across (Y, M)
        sm_r_w = sm_ref.sel({day_dim: sel_days}).mean(day_dim, skipna=True)
        lh_r_w = lhflx_ref.sel({day_dim: sel_days}).mean(day_dim, skipna=True)
        t_r_w  = trefht_ref.sel({day_dim: sel_days}).mean(day_dim, skipna=True)

        sm_t_w = sm_test.sel({day_dim: sel_days}).mean(day_dim, skipna=True)
        lh_t_w = lhflx_test.sel({day_dim: sel_days}).mean(day_dim, skipna=True)
        t_t_w  = trefht_test.sel({day_dim: sel_days}).mean(day_dim, skipna=True)

        # Compute anomalies relative to ensemble/year mean
        sm_r_anom = sm_r_w - sm_r_w.mean(["Y", "M"], skipna=True)
        lh_r_anom = lh_r_w - lh_r_w.mean(["Y", "M"], skipna=True)
        t_r_anom  = t_r_w  - t_r_w.mean(["Y", "M"],  skipna=True)

        sm_t_anom = sm_t_w - sm_t_w.mean(["Y", "M"], skipna=True)
        lh_t_anom = lh_t_w - lh_t_w.mean(["Y", "M"], skipna=True)
        t_t_anom  = t_t_w  - t_t_w.mean(["Y", "M"],  skipna=True)

        # Slopes
        slope_lh_ref,  corr_lh_ref  = compute_coupling_slope(sm_r_anom, lh_r_anom, sample_dim="Y")
        slope_lh_test, corr_lh_test = compute_coupling_slope(sm_t_anom, lh_t_anom, sample_dim="Y")

        slope_t_ref,   corr_t_ref   = compute_coupling_slope(sm_r_anom, t_r_anom,  sample_dim="Y")
        slope_t_test,  corr_t_test  = compute_coupling_slope(sm_t_anom, t_t_anom,  sample_dim="Y")

        # Start-level relationships use ensemble members as within-start samples;
        # uncertainty then resamples the paired initialization years.
        sm_r_start = sm_r_w - sm_r_w.mean("M", skipna=True)
        sm_t_start = sm_t_w - sm_t_w.mean("M", skipna=True)
        lh_r_start = lh_r_w - lh_r_w.mean("M", skipna=True)
        lh_t_start = lh_t_w - lh_t_w.mean("M", skipna=True)
        t_r_start = t_r_w - t_r_w.mean("M", skipna=True)
        t_t_start = t_t_w - t_t_w.mean("M", skipna=True)
        slope_lh_ref_by_start, _ = compute_coupling_slope(
            sm_r_start, lh_r_start, sample_dim="M"
        )
        slope_lh_test_by_start, _ = compute_coupling_slope(
            sm_t_start, lh_t_start, sample_dim="M"
        )
        slope_t_ref_by_start, _ = compute_coupling_slope(
            sm_r_start, t_r_start, sample_dim="M"
        )
        slope_t_test_by_start, _ = compute_coupling_slope(
            sm_t_start, t_t_start, sample_dim="M"
        )
        paired_lh_by_start = slope_lh_test_by_start - slope_lh_ref_by_start
        paired_t_by_start = slope_t_test_by_start - slope_t_ref_by_start
        lh_ci_lower, lh_ci_upper = bootstrap_physical_metric_ci(paired_lh_by_start)
        t_ci_lower, t_ci_upper = bootstrap_physical_metric_ci(paired_t_by_start)

        results[win_name] = {
            "slope_lh_ref":      slope_lh_ref,
            "slope_lh_test":     slope_lh_test,
            "paired_diff_sm_lh": slope_lh_test - slope_lh_ref,
            "corr_lh_ref":       corr_lh_ref,
            "corr_lh_test":      corr_lh_test,
            "slope_t_ref":       slope_t_ref,
            "slope_t_test":      slope_t_test,
            "paired_diff_sm_t":  slope_t_test - slope_t_ref,
            "corr_t_ref":        corr_t_ref,
            "corr_t_test":       corr_t_test,
            "paired_diff_sm_lh_by_start": paired_lh_by_start,
            "paired_diff_sm_t_by_start": paired_t_by_start,
            "paired_diff_sm_lh_ci_lower": lh_ci_lower,
            "paired_diff_sm_lh_ci_upper": lh_ci_upper,
            "paired_diff_sm_t_ci_lower": t_ci_lower,
            "paired_diff_sm_t_ci_upper": t_ci_upper,
        }

    return results
