"""
branch_b_physics.py
===================
Branch B: Physical Consistency

Calculates EF and Bowen ratio from the flux fields accepted by this branch.
The full multi-field physical suite is orchestrated by
``workflows.diagnostics.physical_consistency.run_physical``.
"""

from __future__ import annotations

from typing import Dict, Tuple
import xarray as xr

from esp_lab.diagnostics.physical_core import (
    compute_apparent_energy_residual,
    compute_bowen_ratio,
    compute_coupling_slope,
    compute_evaporative_fraction,
    compute_sst_trefht_contrast,
)


def run_branch_b(
    lhflx_ref: xr.DataArray,
    shflx_ref: xr.DataArray,
    lhflx_test: xr.DataArray,
    shflx_test: xr.DataArray,
    window_defs: Dict[str, Tuple[int, int]],
    lead_dim: str = "L",
) -> dict:
    results = {}
    for win_name, (w_first, w_last) in window_defs.items():
        sel_leads = [L for L in lhflx_ref.coords[lead_dim].values if w_first <= L <= w_last]
        if not sel_leads:
            continue

        lh_r = lhflx_ref.sel({lead_dim: sel_leads}).sum(lead_dim, skipna=True).mean(["Y", "M"], skipna=True)
        sh_r = shflx_ref.sel({lead_dim: sel_leads}).sum(lead_dim, skipna=True).mean(["Y", "M"], skipna=True)

        lh_t = lhflx_test.sel({lead_dim: sel_leads}).sum(lead_dim, skipna=True).mean(["Y", "M"], skipna=True)
        sh_t = shflx_test.sel({lead_dim: sel_leads}).sum(lead_dim, skipna=True).mean(["Y", "M"], skipna=True)

        ef_ref, _  = compute_evaporative_fraction(lh_r, sh_r)
        ef_test, _ = compute_evaporative_fraction(lh_t, sh_t)
        br_ref, _ = compute_bowen_ratio(lh_r, sh_r)
        br_test, _ = compute_bowen_ratio(lh_t, sh_t)

        results[win_name] = {
            "ef_ref":         ef_ref,
            "ef_test":        ef_test,
            "paired_diff_ef": ef_test - ef_ref,
            "br_ref":         br_ref,
            "br_test":        br_test,
            "paired_diff_br": br_test - br_ref,
        }

    return results
