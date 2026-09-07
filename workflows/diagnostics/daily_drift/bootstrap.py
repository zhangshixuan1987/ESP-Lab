"""
bootstrap.py
============
Step 10: Bootstrap across paired initialization years for daily diagnostics.

Resamples paired init years with replacement (default N=1000).
May and November starts are bootstrapped separately.
Generates 95% CI bounds and significance masks (where CI excludes zero).
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

from esp_lab.diagnostics.daily_core import (
    DailyDriftConfig,
    bootstrap_daily_spatial_ci,
    daily_significance_mask,
)


def run_bootstrap(
    results: Dict[int, Dict[str, Any]],
    config: DailyDriftConfig,
    verbose: bool = True,
) -> Dict[int, Dict[str, Dict[str, xr.DataArray]]]:
    boot: Dict[int, Dict[str, Dict[str, xr.DataArray]]] = {}

    for init_month, month_results in results.items():
        season = {5: "May", 11: "November"}.get(init_month, f"Month{init_month:02d}")
        if verbose:
            print("=" * 70)
            print(f"Daily Step 10: Bootstrap — {season}")
            print("=" * 70)

        boot[init_month] = {}

        for win_name, win_dict in month_results.items():
            if not isinstance(win_dict, dict):
                continue

            paired_by_year = win_dict.get("paired_diff_by_year")
            if paired_by_year is None:
                continue

            if paired_by_year.sizes.get("Y", 0) < 2:
                if verbose:
                    print(f"  {win_name}: fewer than 2 years — skipping bootstrap")
                continue

            try:
                lower, upper = bootstrap_daily_spatial_ci(
                    paired_diff_by_year=paired_by_year,
                    n_boot=config.n_bootstrap,
                    seed=config.bootstrap_seed,
                    alpha=config.bootstrap_alpha,
                )
                sig = daily_significance_mask(lower, upper)

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
                warnings.warn(f"bootstrap: {season}/{win_name}: {exc}", stacklevel=2)

    return boot
