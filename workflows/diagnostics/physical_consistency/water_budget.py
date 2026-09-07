"""Land-water budget consistency for fields already normalized to common units."""

from __future__ import annotations

from typing import Dict, Tuple

import xarray as xr

from esp_lab.diagnostics.physical_core import compute_land_water_residual


def run_water_budget(
    storage_tendency_ref: xr.DataArray,
    precipitation_ref: xr.DataArray,
    evapotranspiration_ref: xr.DataArray,
    runoff_ref: xr.DataArray,
    storage_tendency_test: xr.DataArray,
    precipitation_test: xr.DataArray,
    evapotranspiration_test: xr.DataArray,
    runoff_test: xr.DataArray,
    window_defs: Dict[str, Tuple[int, int]],
    lead_dim: str = "d",
) -> Dict[str, dict]:
    """Compute ``dS/dt - (P - ET - R)`` and its paired difference by window."""
    residual_ref = compute_land_water_residual(
        storage_tendency_ref, precipitation_ref, evapotranspiration_ref, runoff_ref
    )
    residual_test = compute_land_water_residual(
        storage_tendency_test, precipitation_test, evapotranspiration_test, runoff_test
    )
    results: Dict[str, dict] = {}
    for name, (first, last) in window_defs.items():
        leads = [
            lead for lead in residual_ref[lead_dim].values
            if first <= lead <= last
        ]
        if not leads:
            continue
        reduce_dims = [dim for dim in (lead_dim, "Y", "M") if dim in residual_ref.dims]
        ref = residual_ref.sel({lead_dim: leads}).mean(reduce_dims, skipna=True)
        test = residual_test.sel({lead_dim: leads}).mean(reduce_dims, skipna=True)
        results[name] = {
            "land_water_residual_ref": ref,
            "land_water_residual_test": test,
            "paired_diff_land_water_residual": test - ref,
        }
    return results


__all__ = ["run_water_budget"]
