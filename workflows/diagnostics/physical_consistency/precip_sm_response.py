"""
precip_sm_response.py
=====================
Calculates daily lagged response slope β_{P -> ΔSM}(ℓ) for lags ℓ = 0..7 days:
    ΔSM_d = SM_{d+1} - SM_d
    β(ℓ) = Cov(PRECT_d', ΔSM_{d+ℓ}') / Var(PRECT_d')
"""

from __future__ import annotations

import warnings
import xarray as xr

from esp_lab.diagnostics.physical_core import compute_precip_sm_lag_response


def run_precip_sm_response(
    prect_ref: xr.DataArray,
    sm_ref: xr.DataArray,
    prect_test: xr.DataArray,
    sm_test: xr.DataArray,
    max_lag: int = 7,
    day_dim: str = "d",
) -> dict:
    beta_ref  = compute_precip_sm_lag_response(prect_ref, sm_ref, max_lag=max_lag, day_dim=day_dim)
    beta_test = compute_precip_sm_lag_response(prect_test, sm_test, max_lag=max_lag, day_dim=day_dim)

    return {
        "beta_ref":        beta_ref,
        "beta_test":       beta_test,
        "paired_diff_lag": beta_test - beta_ref,
    }
