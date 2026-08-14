"""IC-to-drift attribution metrics used by the unified 5f consumer."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr


def _broadcast_weights(field: xr.DataArray, weights: xr.DataArray | None) -> xr.DataArray:
    if weights is None:
        return xr.ones_like(field, dtype=float)
    return xr.broadcast(weights, field)[0]


def weighted_ic_alignment(
    ic_difference: xr.DataArray,
    drift_difference: xr.DataArray,
    *,
    spatial_dims: Iterable[str],
    weights: xr.DataArray | None = None,
    variance_epsilon: float = 1.0e-12,
) -> xr.Dataset:
    """Compute weighted pattern correlation and projection onto the IC pattern.

    ``beta_ic = <drift, ic>_w / <ic, ic>_w`` and ``r_ic`` is the weighted,
    centered spatial correlation. Non-spatial dimensions (normally lead and
    initialization year) are retained.
    """
    dims = list(spatial_dims)
    missing = [dim for dim in dims if dim not in ic_difference.dims or dim not in drift_difference.dims]
    if missing:
        raise ValueError(f"Spatial dimensions missing from IC or drift fields: {missing}")
    ic, drift = xr.align(ic_difference, drift_difference, join="inner")
    w = _broadcast_weights(ic, weights).where(np.isfinite(ic) & np.isfinite(drift))
    wsum = w.sum(dims, skipna=True)
    ic_mean = (w * ic).sum(dims, skipna=True) / wsum
    drift_mean = (w * drift).sum(dims, skipna=True) / wsum
    ic_anom = ic - ic_mean
    drift_anom = drift - drift_mean
    covariance = (w * ic_anom * drift_anom).sum(dims, skipna=True)
    ic_variance = (w * ic_anom**2).sum(dims, skipna=True)
    drift_variance = (w * drift_anom**2).sum(dims, skipna=True)
    denominator = np.sqrt(ic_variance * drift_variance)
    r_ic = (covariance / denominator).where(denominator > variance_epsilon).rename("r_ic")

    projection_denominator = (w * ic**2).sum(dims, skipna=True)
    beta_ic = (
        (w * drift * ic).sum(dims, skipna=True) / projection_denominator
    ).where(projection_denominator > variance_epsilon).rename("beta_ic")
    sample_count = w.notnull().sum(dims).rename("sample_count")
    return xr.Dataset({"r_ic": r_ic, "beta_ic": beta_ic, "sample_count": sample_count})


def across_start_attribution(
    indicator: xr.DataArray,
    drift_response: xr.DataArray,
    *,
    start_dim: str = "Y",
    variance_epsilon: float = 1.0e-12,
) -> xr.Dataset:
    """Fit ``Delta D_s = alpha + beta I_s + epsilon_s`` across paired starts."""
    if start_dim not in indicator.dims or start_dim not in drift_response.dims:
        raise ValueError(f"Both inputs must contain start dimension {start_dim!r}.")
    x, y = xr.align(indicator, drift_response, join="inner")
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(start_dim).rename("sample_count")
    x_valid = x.where(valid)
    y_valid = y.where(valid)
    x_mean = x_valid.mean(start_dim, skipna=True)
    y_mean = y_valid.mean(start_dim, skipna=True)
    xa = x_valid - x_mean
    ya = y_valid - y_mean
    ssx = (xa**2).sum(start_dim, skipna=True)
    ssy = (ya**2).sum(start_dim, skipna=True)
    cov = (xa * ya).sum(start_dim, skipna=True)
    beta = (cov / ssx).where(ssx > variance_epsilon).rename("beta")
    alpha = (y_mean - beta * x_mean).rename("alpha")
    denom = np.sqrt(ssx * ssy)
    correlation = (cov / denom).where(denom > variance_epsilon).rename("correlation")
    return xr.Dataset({"alpha": alpha, "beta": beta, "correlation": correlation, "sample_count": count})


__all__ = ["across_start_attribution", "weighted_ic_alignment"]
