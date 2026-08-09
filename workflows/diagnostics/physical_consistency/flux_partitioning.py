"""
flux_partitioning.py
====================
Calculates Evaporative Fraction (EF) and Bowen Ratio (BR) for each window W:

    EF(W) = sum_{d in W} LHFLX / sum_{d in W} (LHFLX + SHFLX)
    BR(W) = sum_{d in W} SHFLX / sum_{d in W} LHFLX

Evaluated separately per window W first, then differenced:
    A_EF(W) = EF(W) - EF(W0)
    ΔEF(W)  = EF_JRA55_FOSIRL(W) - EF_Reanalysis(W)
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import xarray as xr

from esp_lab.diagnostics.physical_core import compute_bowen_ratio, compute_evaporative_fraction


def run_flux_partitioning(
    lhflx_ref: xr.DataArray,
    shflx_ref: xr.DataArray,
    lhflx_test: xr.DataArray,
    shflx_test: xr.DataArray,
    window_defs: Dict[str, Tuple[int, int]],
    day_dim: str = "d",
) -> Dict[str, dict]:
    """Run EF and Bowen ratio diagnostics across lead windows.

    Returns dict of window_name -> {ef_ref, ef_test, paired_diff_ef, br_ref, br_test, paired_diff_br, qc_masked_frac}.
    """
    results = {}

    for win_name, (w_first, w_last) in window_defs.items():
        sel_days = [d for d in lhflx_ref.coords[day_dim].values if w_first <= d <= w_last]
        if not sel_days:
            continue

        # Sum fluxes over window first before taking ratios
        lh_ref_w = lhflx_ref.sel({day_dim: sel_days}).sum(day_dim, skipna=True).mean(["Y", "M"], skipna=True)
        sh_ref_w = shflx_ref.sel({day_dim: sel_days}).sum(day_dim, skipna=True).mean(["Y", "M"], skipna=True)

        lh_test_w = lhflx_test.sel({day_dim: sel_days}).sum(day_dim, skipna=True).mean(["Y", "M"], skipna=True)
        sh_test_w = shflx_test.sel({day_dim: sel_days}).sum(day_dim, skipna=True).mean(["Y", "M"], skipna=True)

        ef_ref, frac_ref   = compute_evaporative_fraction(lh_ref_w, sh_ref_w)
        ef_test, frac_test = compute_evaporative_fraction(lh_test_w, sh_test_w)

        br_ref, _  = compute_bowen_ratio(lh_ref_w, sh_ref_w)
        br_test, _ = compute_bowen_ratio(lh_test_w, sh_test_w)

        results[win_name] = {
            "ef_ref":          ef_ref,
            "ef_test":         ef_test,
            "paired_diff_ef":  ef_test - ef_ref,
            "br_ref":          br_ref,
            "br_test":         br_test,
            "paired_diff_br":  br_test - br_ref,
            "qc_masked_frac": (frac_ref + frac_test) / 2.0,
        }

    return results
