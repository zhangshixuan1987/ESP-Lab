"""
preprocess.py
=============
Steps 6–7: Open approved daily data, assign forecast lead days 1–84,
           retain cftime noleap calendar, apply unit conversions,
           apply masks, and return campaign-ready (Y, M, d, lat, lon) DataArrays.

Only called after the daily inventory gate passes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.daily_core import (
    DailyDriftConfig,
    DailyPairedReadinessReport,
    apply_mask,
)
from esp_lab.diagnostics.daily_io import (
    load_daily_campaign_field,
    load_daily_obs_spatial,
)


def run_preprocess(
    config: DailyDriftConfig,
    inventory_df: pd.DataFrame,
    report: DailyPairedReadinessReport,
    field: str = "TREFHT",
    land_mask: Optional[xr.DataArray] = None,
    obs_path: Optional[str] = None,
    chunks: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[int, xr.DataArray]], Optional[xr.DataArray]]:
    """Load and preprocess daily campaign data for one variable.

    Returns
    -------
    data:
        Nested dict: ``data[experiment][init_month]`` → DataArray (Y, M, d, lat, lon).
    obs_da:
        Daily observation DataArray or None.
    """
    var_spec = config.get_variable(field)

    approved_pairs = report.common_paired_inits
    if not approved_pairs:
        warnings.warn("No approved paired combinations — preprocess will be empty.", stacklevel=2)

    from esp_lab.data_access_e3sm import build_init_tags
    approved_tags = {tag for (tag, _) in approved_pairs}

    approved_years_per_month: Dict[int, list] = {}
    for init_month in config.active_months:
        tags = build_init_tags(config.active_years, init_month)
        approved_years_per_month[init_month] = [
            yr for yr, tag in zip(config.active_years, tags)
            if tag in approved_tags
        ]

    if verbose:
        print("=" * 70)
        print(f"Daily Step 6–7: Preprocess — {field}")
        print("=" * 70)

    data: Dict[str, Dict[int, xr.DataArray]] = {}

    for exp_label in config.experiments:
        data[exp_label] = {}
        for init_month in config.active_months:
            approved_years = approved_years_per_month.get(init_month, [])
            if not approved_years:
                continue

            if verbose:
                print(f"\n  [{exp_label}] init_month={init_month}  years={approved_years}")

            try:
                da = load_daily_campaign_field(
                    config=config,
                    experiment=exp_label,
                    field=field,
                    init_month=init_month,
                    approved_years=approved_years,
                    chunks=chunks,
                    verbose=verbose,
                )

                # Model unit conversion
                if var_spec.model_convert is not None:
                    da = var_spec.model_convert(da)

                # Masking
                if land_mask is not None:
                    da = apply_mask(da, var_spec.mask, land_mask)

                data[exp_label][init_month] = da

                if verbose:
                    print(f"    shape={da.shape}  dims={da.dims}")

            except Exception as exc:
                warnings.warn(f"preprocess: [{exp_label}] init_month={init_month}: {exc}", stacklevel=2)

    # Observation loading
    obs_da = None
    if obs_path and var_spec.obs_product:
        all_years = sorted({y for yrs in approved_years_per_month.values() for y in yrs})
        if all_years:
            obs_da = load_daily_obs_spatial(
                obs_path=obs_path,
                field=var_spec.obs_var or field,
                years_range=(min(all_years), max(all_years) + 1),
                chunks={"time": 60},
            )
            if obs_da is not None and var_spec.obs_convert is not None:
                obs_da = var_spec.obs_convert(obs_da)

    return data, obs_da
