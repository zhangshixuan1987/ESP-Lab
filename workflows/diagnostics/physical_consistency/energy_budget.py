"""
energy_budget.py
================
Calculates apparent surface energy residual:
    R_apparent = FSNS - FLNS - LHFLX - SHFLX
"""

from __future__ import annotations

from typing import Dict, Tuple

import xarray as xr

from esp_lab.diagnostics.physical_core import compute_apparent_energy_residual


def run_energy_budget(
    fsns_ref: xr.DataArray,
    flns_ref: xr.DataArray,
    lhflx_ref: xr.DataArray,
    shflx_ref: xr.DataArray,
    fsns_test: xr.DataArray,
    flns_test: xr.DataArray,
    lhflx_test: xr.DataArray,
    shflx_test: xr.DataArray,
    window_defs: Dict[str, Tuple[int, int]],
    day_dim: str = "d",
) -> Dict[str, dict]:
    results = {}

    for win_name, (w_first, w_last) in window_defs.items():
        sel_days = [d for d in fsns_ref.coords[day_dim].values if w_first <= d <= w_last]
        if not sel_days:
            continue

        fs_r = fsns_ref.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)
        fl_r = flns_ref.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)
        lh_r = lhflx_ref.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)
        sh_r = shflx_ref.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)

        fs_t = fsns_test.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)
        fl_t = flns_test.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)
        lh_t = lhflx_test.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)
        sh_t = shflx_test.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)

        res_ref  = compute_apparent_energy_residual(fs_r, fl_r, lh_r, sh_r)
        res_test = compute_apparent_energy_residual(fs_t, fl_t, lh_t, sh_t)

        results[win_name] = {
            "res_ref":         res_ref,
            "res_test":        res_test,
            "paired_diff_res": res_test - res_ref,
        }

    return results
