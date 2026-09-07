"""
bootstrap.py
============
Step 8: Bootstrap across paired initialization years to quantify uncertainty.

Algorithm
---------
1. For each window and init_month, take the per-year ΔD array (Y, lat, lon)
   from diagnostics.run_diagnostics().
2. Resample paired years with replacement (N_BOOT replicates).
3. Compute alpha/2 and 1-alpha/2 quantiles → CI arrays.
4. Build significance mask where CI excludes zero.

May and November bootstraps are run separately.

Note: ensemble members are averaged BEFORE this step (inside diagnostics.py).
      Members contribute to spread but are NOT treated as independent years.

Usage
-----
    from bootstrap import run_bootstrap
    boot_results = run_bootstrap(results, config)

    # boot_results[5]["months_4_6"]["lower"]     → xr.DataArray (lat, lon)
    # boot_results[5]["months_4_6"]["upper"]     → xr.DataArray (lat, lon)
    # boot_results[5]["months_4_6"]["significant"] → bool DataArray (lat, lon)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

import xarray as xr

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.monthly_core import (
    MonthlyConfig,
    bootstrap_spatial_ci,
    significance_mask,
)


def run_bootstrap(
    results: Dict[int, Dict[str, Any]],
    config: MonthlyConfig,
    verbose: bool = True,
) -> Dict[int, Dict[str, Dict[str, xr.DataArray]]]:
    """Run bootstrap CI for every (init_month, window) pair.

    Parameters
    ----------
    results:
        Output of ``diagnostics.run_diagnostics()``.
    config:
        MonthlyConfig (provides n_bootstrap, bootstrap_seed, bootstrap_alpha).

    Returns
    -------
    Nested dict: ``boot[init_month][window_name]`` →
        {"lower": DataArray, "upper": DataArray, "significant": DataArray}
    """
    boot: Dict[int, Dict[str, Dict[str, xr.DataArray]]] = {}

    for init_month, month_results in results.items():
        season = {5: "May", 11: "November"}.get(init_month, f"Month{init_month:02d}")
        if verbose:
            print("=" * 70)
            print(f"Step 8: Bootstrap — {season}")
            print("=" * 70)

        boot[init_month] = {}

        for win_name, win_dict in month_results.items():
            if not isinstance(win_dict, dict):
                continue   # skip scalar entries like bias_ref_lead1

            paired_by_year = win_dict.get("paired_diff_by_year")
            if paired_by_year is None:
                if verbose:
                    print(f"  {win_name}: no per-year data — skipping bootstrap")
                continue

            if paired_by_year.sizes.get("Y", 0) < 2:
                if verbose:
                    print(f"  {win_name}: fewer than 2 years — skipping bootstrap")
                continue

            try:
                lower, upper = bootstrap_spatial_ci(
                    paired_diff_by_year=paired_by_year,
                    n_boot=config.n_bootstrap,
                    seed=config.bootstrap_seed,
                    alpha=config.bootstrap_alpha,
                )
                sig = significance_mask(lower, upper)

                boot[init_month][win_name] = {
                    "lower": lower,
                    "upper": upper,
                    "significant": sig,
                }

                if verbose:
                    frac_sig = float(sig.mean(skipna=True))
                    print(
                        f"  {win_name:<18s}  "
                        f"n_years={paired_by_year.sizes['Y']}  "
                        f"frac_significant={frac_sig:.2%}"
                    )

            except Exception as exc:
                warnings.warn(
                    f"bootstrap: {season}/{win_name}: {exc}",
                    stacklevel=2,
                )

    return boot
