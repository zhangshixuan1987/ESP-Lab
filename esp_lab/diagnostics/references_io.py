"""
references_io.py
================
Constructs observation references O_X(t_valid) and free-running E3SM
historical climatology M_X(c_valid).

Functions
---------
build_observation_reference   — align observation dataset to hindcast valid times
build_e3sm_historical_climatology — build calendar-conditioned climatology M_X(c)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from esp_lab.data_access_e3sm import drop_feb29, time_set_midmonth


def build_observation_reference(
    obs_da: xr.DataArray,
    valid_times: Sequence[object],
) -> Optional[xr.DataArray]:
    """Select observation slices corresponding to valid_times.

    Parameters
    ----------
    obs_da:
        Observation DataArray with time coordinate.
    valid_times:
        List of cftime or datetime objects for forecast verification times.

    Returns
    -------
    xr.DataArray aligned to valid_times, or None if unavailable.
    """
    if obs_da is None:
        return None

    try:
        # Align time coordinates
        obs_da = time_set_midmonth(obs_da, "time")
        matched = []
        matched_times = []
        for vt in valid_times:
            try:
                vy, vm = int(vt.year), int(vt.month)
            except AttributeError:
                ts = pd.Timestamp(vt)
                vy, vm = ts.year, ts.month

            # Select by year+month
            sl = None
            for tv in obs_da.time.values:
                try:
                    ty, tm = int(tv.year), int(tv.month)
                except AttributeError:
                    ts = pd.Timestamp(tv)
                    ty, tm = ts.year, ts.month
                if ty == vy and tm == vm:
                    sl = obs_da.sel(time=tv, drop=True)
                    break
            if sl is not None:
                matched.append(sl)
                matched_times.append(vt)

        if not matched:
            return None
        return xr.concat(matched, dim=pd.Index(matched_times, name="valid_time"))
    except Exception as exc:
        warnings.warn(f"build_observation_reference error: {exc}", stacklevel=2)
        return None


def build_e3sm_historical_climatology(
    historical_ds: Optional[xr.Dataset],
    field: str,
    frequency: str = "monthly",
) -> Optional[xr.DataArray]:
    """Construct calendar-conditioned climatology M_X(c) from free-running E3SM historical dataset.

    - Monthly frequency: calendar-month climatology (1..12).
    - Daily frequency: smoothed day-of-year climatology (1..365).

    Parameters
    ----------
    historical_ds:
        Free-running E3SM historical Dataset with time coordinate.
    field:
        Variable name.
    frequency:
        ``"monthly"`` or ``"daily"``.

    Returns
    -------
    DataArray with coordinate ``month`` (1..12) or ``dayofyear`` (1..365).
    """
    if historical_ds is None or field not in historical_ds.data_vars:
        warnings.warn(
            f"build_e3sm_historical_climatology: historical dataset missing or '{field}' not found. "
            "Attractor branch will run in paired observation distance mode.",
            stacklevel=2,
        )
        return None

    try:
        da = historical_ds[field]
        da = drop_feb29(da, "time")
        da = time_set_midmonth(da, "time")

        if frequency == "monthly":
            clim = da.groupby("time.month").mean("time", skipna=True)
            clim.name = "e3sm_monthly_climatology"
        else:
            clim = da.groupby("time.dayofyear").mean("time", skipna=True)
            # Smooth 5-day rolling window for daily noise reduction
            clim = clim.rolling(dayofyear=5, center=True, min_periods=1).mean()
            clim.name = "e3sm_daily_climatology"

        clim.attrs["description"] = f"Free-running E3SM historical {frequency} climatology for {field}"
        return clim
    except Exception as exc:
        warnings.warn(f"build_e3sm_historical_climatology failed: {exc}", stacklevel=2)
        return None


__all__ = [
    "build_observation_reference",
    "build_e3sm_historical_climatology",
]
