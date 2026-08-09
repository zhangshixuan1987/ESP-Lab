"""
preprocess.py
=============
Step 5: Open approved monthly data, assign leads, apply unit conversions,
        apply masks, and return campaign-ready (Y, M, L, lat, lon) arrays.

Only called after the inventory gate passes.

Usage (interactive)
-------------------
    from preprocess import run_preprocess
    data, obs = run_preprocess(config, inventory_df, report)

    # data["JRA55_FOSIRL"][5] → xr.DataArray (Y, M, L, lat, lon)  May starts
    # data["Reanalysis"][11]  → xr.DataArray (Y, M, L, lat, lon)  November starts
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

from esp_lab.diagnostics.monthly_core import (
    MonthlyConfig,
    PairedReadinessReport,
    apply_mask,
)
from esp_lab.diagnostics.monthly_io import load_campaign_field, load_obs_spatial


def run_preprocess(
    config: MonthlyConfig,
    inventory_df: pd.DataFrame,
    report: PairedReadinessReport,
    field: str = "PRECT",
    land_mask: Optional[xr.DataArray] = None,
    obs_path: Optional[str] = None,
    chunks: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[int, xr.DataArray]], Optional[xr.DataArray]]:
    """Load and preprocess campaign data for one variable.

    Parameters
    ----------
    config:
        MonthlyConfig.
    inventory_df:
        Inventory DataFrame from inventory.run().
    report:
        PairedReadinessReport from inventory.run().
    field:
        Variable native name (e.g. ``"PRECT"``).
    land_mask:
        Boolean DataArray (True = land) with (lat, lon) dims.
        Required for land/ocean masking.
    obs_path:
        Path to observation NetCDF file.  If None, obs is skipped.
    chunks:
        Dask chunks for model data loading.
    verbose:
        Print progress.

    Returns
    -------
    data:
        Nested dict: ``data[experiment][init_month]`` → DataArray (Y, M, L, lat, lon).
    obs_da:
        Observation DataArray with time coordinate, or None.
    """
    var_spec = config.get_variable(field)

    # Approved members from paired readiness
    approved_pairs = report.common_paired_inits   # {(init_tag, member)}
    if not approved_pairs:
        warnings.warn("No approved paired combinations — preprocess will be empty.", stacklevel=2)

    # Determine approved years per init_month
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
        print(f"Step 5: Preprocess — {field}")
        print("=" * 70)
        for m, yrs in approved_years_per_month.items():
            print(f"  init_month={m}  approved_years={yrs}")

    data: Dict[str, Dict[int, xr.DataArray]] = {}

    for exp_label in config.experiments:
        data[exp_label] = {}
        for init_month in config.active_months:
            approved_years = approved_years_per_month.get(init_month, [])
            if not approved_years:
                if verbose:
                    print(f"  [{exp_label}] init_month={init_month}: no approved years")
                continue

            if verbose:
                print(f"\n  [{exp_label}] init_month={init_month}  years={approved_years}")

            try:
                da = load_campaign_field(
                    config=config,
                    experiment=exp_label,
                    field=field,
                    init_month=init_month,
                    approved_years=approved_years,
                    chunks=chunks,
                    verbose=verbose,
                )

                # Unit conversion
                if var_spec.model_convert is not None:
                    da = var_spec.model_convert(da)

                # Mask
                if land_mask is not None:
                    da = apply_mask(da, var_spec.mask_type, land_mask)

                data[exp_label][init_month] = da

                if verbose:
                    print(f"    shape={da.shape}  dims={da.dims}")

            except Exception as exc:
                warnings.warn(
                    f"preprocess: [{exp_label}] init_month={init_month}: {exc}",
                    stacklevel=2,
                )

    # Observation loading
    obs_da = None
    if obs_path and var_spec.obs_product:
        all_years = sorted({y for yrs in approved_years_per_month.values() for y in yrs})
        if all_years:
            obs_da = load_obs_spatial(
                obs_path=obs_path,
                field=var_spec.obs_var or field,
                years_range=(min(all_years), max(all_years) + 3),
                chunks={"time": 60},
            )
            if obs_da is not None and var_spec.obs_convert is not None:
                obs_da = var_spec.obs_convert(obs_da)
            if verbose:
                print(f"\n  Obs loaded: {obs_path}" if obs_da is not None
                      else "\n  Obs: NOT loaded (file missing)")

    return data, obs_da
