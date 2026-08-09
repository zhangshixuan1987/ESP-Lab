"""
ocean_coupling.py
=================
Calculates air-sea contrast ΔT_AO = TREFHT - SST over ocean.
"""

from __future__ import annotations

from typing import Dict, Tuple

import xarray as xr

from esp_lab.diagnostics.physical_core import compute_sst_trefht_contrast


def run_ocean_coupling(
    trefht_ref: xr.DataArray,
    sst_ref: xr.DataArray,
    trefht_test: xr.DataArray,
    sst_test: xr.DataArray,
    window_defs: Dict[str, Tuple[int, int]],
    day_dim: str = "d",
) -> Dict[str, dict]:
    results = {}

    for win_name, (w_first, w_last) in window_defs.items():
        sel_days = [d for d in trefht_ref.coords[day_dim].values if w_first <= d <= w_last]
        if not sel_days:
            continue

        t_r = trefht_ref.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)
        s_r = sst_ref.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)

        t_t = trefht_test.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)
        s_t = sst_test.sel({day_dim: sel_days}).mean([day_dim, "Y", "M"], skipna=True)

        dt_ref  = compute_sst_trefht_contrast(t_r, s_r)
        dt_test = compute_sst_trefht_contrast(t_t, s_t)

        results[win_name] = {
            "dt_ao_ref":       dt_ref,
            "dt_ao_test":      dt_test,
            "paired_diff_dt":  dt_test - dt_ref,
        }

    return results
