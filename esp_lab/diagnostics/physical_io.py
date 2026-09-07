"""
physical_io.py
==============
S2D Physical-Consistency Diagnostics — I/O bridge.

Handles multi-variable assembly for process diagnostics (e.g. reading LHFLX + SHFLX
together for EF, or FSNS + FLNS + LHFLX + SHFLX together for energy budget).

Wraps daily_io and monthly_io dataset loaders.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr

from esp_lab.diagnostics.daily_io import load_daily_campaign_field
from esp_lab.diagnostics.monthly_io import load_campaign_field as load_monthly_campaign_field


def load_physical_variables_daily(
    config,
    experiment: str,
    fields: List[str],
    init_month: int,
    approved_years: Optional[List[int]] = None,
    approved_members: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    verbose: bool = True,
) -> Dict[str, xr.DataArray]:
    """Load multiple daily fields into a dictionary for physical process calculations.

    Returns dict of field_name -> DataArray(Y, M, d, lat, lon).
    """
    field_das: Dict[str, xr.DataArray] = {}
    for f in fields:
        try:
            da = load_daily_campaign_field(
                config=config,
                experiment=experiment,
                field=f,
                init_month=init_month,
                approved_years=approved_years,
                approved_members=approved_members,
                chunks=chunks,
                verbose=verbose,
            )
            field_das[f] = da
        except Exception as exc:
            warnings.warn(f"load_physical_variables_daily: cannot load {experiment}/{f}: {exc}", stacklevel=2)

    return field_das


def load_physical_variables_monthly(
    config,
    experiment: str,
    fields: List[str],
    init_month: int,
    approved_years: Optional[List[int]] = None,
    approved_members: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    verbose: bool = True,
) -> Dict[str, xr.DataArray]:
    """Load multiple monthly fields into a dictionary for process persistence.

    Returns dict of field_name -> DataArray(Y, M, L, lat, lon).
    """
    field_das: Dict[str, xr.DataArray] = {}
    for f in fields:
        try:
            da = load_monthly_campaign_field(
                config=config,
                experiment=experiment,
                field=f,
                init_month=init_month,
                approved_years=approved_years,
                approved_members=approved_members,
                chunks=chunks,
                verbose=verbose,
            )
            field_das[f] = da
        except Exception as exc:
            warnings.warn(f"load_physical_variables_monthly: cannot load {experiment}/{f}: {exc}", stacklevel=2)

    return field_das


__all__ = [
    "load_physical_variables_daily",
    "load_physical_variables_monthly",
]
