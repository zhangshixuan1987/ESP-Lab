"""
branch_a_drift.py
=================
Branch A: Field Drift

Computes:
  - Observation-relative bias: B^obs_m,s(τ) = X_m,s(τ) - O_X(t_valid)
  - Bias evolution: D^obs_m,s(τ) = B^obs_m,s(τ) - B^obs_m,s(τ0)
  - Direct state adjustment: J_m,s(τ) = X_m,s(τ) - X_m,s(τ0)
"""

from __future__ import annotations

import xarray as xr


def run_branch_a(
    model_field: xr.DataArray,
    obs_ref: xr.DataArray | None,
    baseline_lead: int = 1,
    lead_dim: str = "L",
) -> dict:
    if "M" in model_field.dims:
        model_em = model_field.mean("M", skipna=True)
    else:
        model_em = model_field

    # Direct state adjustment J = X(τ) - X(τ0)
    anchor_x = model_em.sel({lead_dim: baseline_lead}, drop=True)
    state_adjustment_J = model_em - anchor_x
    state_adjustment_J.name = "state_adjustment_J"

    # Bias B^obs and bias evolution D^obs
    b_obs = None
    d_obs = None
    if obs_ref is not None:
        b_obs = model_em - obs_ref
        b_obs.name = "bias_obs"
        anchor_b = b_obs.sel({lead_dim: baseline_lead}, drop=True)
        d_obs = b_obs - anchor_b
        d_obs.name = "bias_evolution_D_obs"

    return {
        "model_em":           model_em,
        "state_adjustment_J": state_adjustment_J,
        "bias_obs":           b_obs,
        "bias_evolution_D_obs": d_obs,
    }
