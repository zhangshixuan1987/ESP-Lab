"""
branch_c_attractor.py
=====================
Branch C: Model Attractor (Movement Toward Model State)

Computes:
  - Distances: d^obs(τ), d^model(τ)
  - Movement relative to W0: δd^obs(W), δd^model(W)
  - Initial displacements: d^obs(W0), d^model(W0)
  - 4-Category Movement Classification
  - 2-Axis Trajectory Data Points (x=δd^model, y=δd^obs)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr

from esp_lab.diagnostics.attractor_core import (
    MovementCategory,
    build_2axis_trajectory,
    classify_movement_category,
    classify_spatial_movement,
    compute_attractor_distances,
    compute_relative_movement,
)


def run_branch_c(
    model_field: xr.DataArray,
    obs_ref: Optional[xr.DataArray],
    e3sm_clim: Optional[xr.DataArray],
    window_defs: Dict[str, Tuple[int, int]],
    lead_dim: str = "L",
    baseline_lead: int = 1,
) -> dict:
    if "M" in model_field.dims:
        model_em = model_field.mean("M", skipna=True)
    else:
        model_em = model_field

    d_obs, d_model = compute_attractor_distances(model_em, obs_ref, e3sm_clim)

    delta_d_obs, d_init_obs = None, None
    if d_obs is not None:
        delta_d_obs, d_init_obs = compute_relative_movement(d_obs, baseline_coord_val=baseline_lead, lead_dim=lead_dim)

    delta_d_model, d_init_model = None, None
    if d_model is not None:
        delta_d_model, d_init_model = compute_relative_movement(d_model, baseline_coord_val=baseline_lead, lead_dim=lead_dim)

    cat_map = None
    if delta_d_obs is not None and delta_d_model is not None:
        cat_map = classify_spatial_movement(delta_d_obs, delta_d_model)

    window_results = {}
    for name, (first, last) in window_defs.items():
        leads = [value for value in model_em[lead_dim].values if first <= value <= last]
        if not leads:
            continue
        entry = {}
        for key, value in (
            ("d_obs", d_obs), ("d_model", d_model),
            ("delta_d_obs", delta_d_obs), ("delta_d_model", delta_d_model),
        ):
            if value is not None:
                entry[key] = value.sel({lead_dim: leads}).mean(lead_dim, skipna=True)
        if "delta_d_obs" in entry and "delta_d_model" in entry:
            entry["movement_category_map"] = classify_spatial_movement(
                entry["delta_d_obs"], entry["delta_d_model"]
            )
        window_results[name] = entry

    return {
        "d_obs":          d_obs,
        "d_model":        d_model,
        "delta_d_obs":    delta_d_obs,
        "delta_d_model":  delta_d_model,
        "d_init_obs":     d_init_obs,
        "d_init_model":   d_init_model,
        "movement_category_map": cat_map,
        "windows": window_results,
    }
