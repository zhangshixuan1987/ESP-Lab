"""Branch D: spatial and across-start IC-to-drift attribution."""

from __future__ import annotations

from collections.abc import Iterable

import xarray as xr

from esp_lab.diagnostics.attribution_core import (
    across_start_attribution,
    weighted_ic_alignment,
)


def run_branch_d(
    ic_difference: xr.DataArray,
    paired_drift: xr.DataArray,
    *,
    spatial_dims: Iterable[str] = ("lat", "lon"),
    weights: xr.DataArray | None = None,
    start_indicator: xr.DataArray | None = None,
    regional_drift_response: xr.DataArray | None = None,
    start_dim: str = "Y",
) -> dict[str, xr.Dataset]:
    """Compute spatial IC alignment and, when supplied, across-start regression."""
    results = {
        "spatial_alignment": weighted_ic_alignment(
            ic_difference,
            paired_drift,
            spatial_dims=spatial_dims,
            weights=weights,
        )
    }
    if (start_indicator is None) != (regional_drift_response is None):
        raise ValueError(
            "start_indicator and regional_drift_response must be supplied together."
        )
    if start_indicator is not None:
        results["across_start_regression"] = across_start_attribution(
            start_indicator, regional_drift_response, start_dim=start_dim
        )
    return results


__all__ = ["run_branch_d"]
