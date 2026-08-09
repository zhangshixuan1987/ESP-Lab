"""
attractor_core.py
=================
Branch C: Model-Attractor Diagnostic Core (Movement Toward Model State) — pure core module.

Design principles
-----------------
* Contains NO I/O and NO ESP-Lab filesystem dependencies.
  Fully unit-testable without NERSC access or real files.
* Measures movement relative to TWO reference states:
    1. Observation / Reanalysis reference  O_X(t_valid)
    2. Free-running E3SM historical climatology  M_X(c_valid)

Quantities
----------
1. Distances at lead τ:
     d^{obs}(τ)   = || X(τ) - O_X(t_valid) ||
     d^{model}(τ) = || X(τ) - M_X(c_valid) ||
2. Movement from baseline window W0:
     δd^{obs}(W)   = d^{obs}(W) - d^{obs}(W0)
     δd^{model}(W) = d^{model}(W) - d^{model}(W0)
3. Initial displacements:
     d^{obs}(W0), d^{model}(W0)
4. 4-Category Movement Classification:
     1. "Toward Both"               (δd^{obs} < 0, δd^{model} < 0)
     2. "Toward Obs, Away from E3SM" (δd^{obs} < 0, δd^{model} > 0)
     3. "Toward E3SM, Away from Obs" (δd^{obs} > 0, δd^{model} < 0)
     4. "Away from Both"            (δd^{obs} > 0, δd^{model} > 0)
     5. "Near Zero / Uncertain"     (within bootstrap CI)
5. 2-Axis Regional Trajectory Data Structure:
     x = δd^{model}, y = δd^{obs}
"""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


class MovementCategory(enum.Enum):
    TOWARD_BOTH            = "TOWARD_BOTH"
    TOWARD_OBS_AWAY_E3SM   = "TOWARD_OBS_AWAY_E3SM"
    TOWARD_E3SM_AWAY_OBS   = "TOWARD_E3SM_AWAY_OBS"
    AWAY_FROM_BOTH         = "AWAY_FROM_BOTH"
    NEAR_ZERO              = "NEAR_ZERO"


def spatial_rmsd(
    da1: xr.DataArray,
    da2: xr.DataArray,
    weights: Optional[xr.DataArray] = None,
    spatial_dims: Tuple[str, ...] = ("lat", "lon"),
) -> xr.DataArray:
    """Compute spatial area-weighted Root-Mean-Square Distance || da1 - da2 ||.

    Parameters
    ----------
    da1, da2:
        Input DataArrays.
    weights:
        Optional spatial weights (e.g. cos(lat)).
    spatial_dims:
        Spatial dimensions to average over.

    Returns
    -------
    xr.DataArray with spatial_dims dropped.
    """
    diff_sq = (da1 - da2) ** 2
    dims_to_avg = [d for d in spatial_dims if d in diff_sq.dims]

    if weights is not None:
        weighted = diff_sq * weights
        rmsd = np.sqrt(weighted.sum(dims_to_avg, skipna=True) / weights.sum(dims_to_avg, skipna=True))
    else:
        rmsd = np.sqrt(diff_sq.mean(dims_to_avg, skipna=True))

    rmsd.name = "rmsd"
    return rmsd


def compute_attractor_distances(
    model_field: xr.DataArray,
    obs_field: Optional[xr.DataArray],
    e3sm_clim: Optional[xr.DataArray],
    weights: Optional[xr.DataArray] = None,
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Compute d^{obs}(τ) and d^{model}(τ).

    Returns
    -------
    (d_obs, d_model) : tuple of DataArrays
    """
    d_obs = None
    if obs_field is not None:
        d_obs = spatial_rmsd(model_field, obs_field, weights=weights)
        d_obs.name = "d_obs"

    d_model = None
    if e3sm_clim is not None:
        d_model = spatial_rmsd(model_field, e3sm_clim, weights=weights)
        d_model.name = "d_model"

    return d_obs, d_model


def compute_relative_movement(
    distance: xr.DataArray,
    baseline_coord_val: int = 1,
    lead_dim: str = "L",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute relative drift δd(W) = d(W) - d(W0) and record initial displacement d(W0).

    Parameters
    ----------
    distance:
        DataArray of distances with lead_dim.
    baseline_coord_val:
        Value of lead_dim at W0 (e.g. 1).
    lead_dim:
        Dimension name ("L" or "d").

    Returns
    -------
    (delta_d, d_initial) : tuple of DataArrays
    """
    d_w0 = distance.sel({lead_dim: baseline_coord_val}, drop=True)
    delta_d = distance - d_w0
    delta_d.name = f"delta_{distance.name}"
    return delta_d, d_w0


def classify_movement_category(
    delta_d_obs: float,
    delta_d_model: float,
    tol_obs: float = 0.0,
    tol_model: float = 0.0,
) -> MovementCategory:
    """Classify a single (δd^{obs}, δd^{model}) pair into 4 categories or Near Zero.

    Parameters
    ----------
    delta_d_obs:
        Change in distance to observations.
    delta_d_model:
        Change in distance to free-running E3SM climatology.
    tol_obs, tol_model:
        Near-zero tolerance thresholds (e.g., half bootstrap CI width).

    Returns
    -------
    MovementCategory enum.
    """
    if abs(delta_d_obs) <= tol_obs and abs(delta_d_model) <= tol_model:
        return MovementCategory.NEAR_ZERO

    if delta_d_obs < -tol_obs and delta_d_model < -tol_model:
        return MovementCategory.TOWARD_BOTH
    elif delta_d_obs < -tol_obs and delta_d_model > tol_model:
        return MovementCategory.TOWARD_OBS_AWAY_E3SM
    elif delta_d_obs > tol_obs and delta_d_model < -tol_model:
        return MovementCategory.TOWARD_E3SM_AWAY_OBS
    elif delta_d_obs > tol_obs and delta_d_model > tol_model:
        return MovementCategory.AWAY_FROM_BOTH
    else:
        return MovementCategory.NEAR_ZERO


def classify_spatial_movement(
    delta_d_obs_map: xr.DataArray,
    delta_d_model_map: xr.DataArray,
    ci_obs: Optional[xr.DataArray] = None,
    ci_model: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Classify movement at every spatial cell into integer category codes.

    Codes:
      1: Toward Both
      2: Toward Obs, Away from E3SM (Improvement)
      3: Toward E3SM, Away from Obs (Model Attractor Drift)
      4: Away from Both
      0: Near Zero / Uncertain

    Returns
    -------
    Integer DataArray with spatial dimensions.
    """
    d_obs_vals   = delta_d_obs_map.values
    d_model_vals = delta_d_model_map.values

    tol_obs   = ci_obs.values / 2.0 if ci_obs is not None else np.zeros_like(d_obs_vals)
    tol_model = ci_model.values / 2.0 if ci_model is not None else np.zeros_like(d_model_vals)

    cat_grid = np.zeros_like(d_obs_vals, dtype=int)

    for idx in np.ndindex(d_obs_vals.shape):
        do = d_obs_vals[idx]
        dm = d_model_vals[idx]
        to = tol_obs[idx] if tol_obs.ndim > 0 else tol_obs
        tm = tol_model[idx] if tol_model.ndim > 0 else tol_model

        if np.isnan(do) or np.isnan(dm):
            cat_grid[idx] = 0
            continue

        cat = classify_movement_category(do, dm, tol_obs=to, tol_model=tm)
        cat_code_map = {
            MovementCategory.TOWARD_BOTH: 1,
            MovementCategory.TOWARD_OBS_AWAY_E3SM: 2,
            MovementCategory.TOWARD_E3SM_AWAY_OBS: 3,
            MovementCategory.AWAY_FROM_BOTH: 4,
            MovementCategory.NEAR_ZERO: 0,
        }
        cat_grid[idx] = cat_code_map[cat]

    result = xr.DataArray(
        cat_grid,
        dims=delta_d_obs_map.dims,
        coords=delta_d_obs_map.coords,
        name="movement_category_code",
    )
    result.attrs["categories"] = {
        1: "Toward Both",
        2: "Toward Obs, Away E3SM (Improvement)",
        3: "Toward E3SM, Away Obs (Model Attractor)",
        4: "Away from Both",
        0: "Near Zero / Uncertain",
    }
    return result


@dataclass
class TrajectoryPoint2Axis:
    """One point in the 2-axis regional trajectory plot (x=δd^{model}, y=δd^{obs})."""

    lead_name: str
    delta_d_model: float
    delta_d_obs: float
    d_initial_model: float
    d_initial_obs: float
    category: MovementCategory


def build_2axis_trajectory(
    delta_d_model_series: xr.DataArray,
    delta_d_obs_series: xr.DataArray,
    d_initial_model: float,
    d_initial_obs: float,
    lead_dim: str = "L",
) -> List[TrajectoryPoint2Axis]:
    """Build list of TrajectoryPoint2Axis for plotting 2-axis trajectory curve."""
    points = []
    leads = delta_d_model_series.coords[lead_dim].values

    for L in leads:
        dm = float(delta_d_model_series.sel({lead_dim: L}))
        do = float(delta_d_obs_series.sel({lead_dim: L}))
        cat = classify_movement_category(do, dm)
        points.append(
            TrajectoryPoint2Axis(
                lead_name=str(L),
                delta_d_model=dm,
                delta_d_obs=do,
                d_initial_model=d_initial_model,
                d_initial_obs=d_initial_obs,
                category=cat,
            )
        )
    return points


__all__ = [
    "MovementCategory",
    "spatial_rmsd",
    "compute_attractor_distances",
    "compute_relative_movement",
    "classify_movement_category",
    "classify_spatial_movement",
    "TrajectoryPoint2Axis",
    "build_2axis_trajectory",
]
