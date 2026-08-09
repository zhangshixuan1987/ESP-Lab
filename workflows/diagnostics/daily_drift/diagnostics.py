"""
diagnostics.py
==============
Steps 8–9: Calculate daily spatial bias, adjustment D(d) (D(1)=0),
           paired difference ΔD, weekly window averages, and per-year
           ΔD arrays for bootstrap.

Delegates core calculation to daily_core.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.daily_core import (
    DailyDriftConfig,
    DailyWindowDef,
    daily_spatial_adjustment,
    daily_spatial_bias,
    daily_spatial_paired_diff,
    daily_window_average,
    valid_date_from_init_and_lead,
)


def _select_obs_day(
    obs_da: xr.DataArray, year: int, month: int, day: int
) -> Optional[xr.DataArray]:
    for tv in obs_da.time.values:
        try:
            ty, tm, td = int(tv.year), int(tv.month), int(tv.day)
        except AttributeError:
            import pandas as pd
            ts = pd.Timestamp(tv)
            ty, tm, td = ts.year, ts.month, ts.day
        if (ty, tm, td) == (year, month, day):
            return obs_da.sel(time=tv, drop=True)
    return None


def _extract_obs_year(
    obs_da: xr.DataArray,
    init_year: int,
    init_month: int,
    lead_days: list[int],
) -> Optional[xr.DataArray]:
    slices = []
    for day in lead_days:
        vy, vm, vd = valid_date_from_init_and_lead(init_year, init_month, 1, day)
        matched = _select_obs_day(obs_da, vy, vm, vd)
        if matched is None:
            return None
        slices.append(matched.assign_coords(d=int(day)))
    return xr.concat(slices, dim="d")


def _build_obs_matrix(
    obs_da: xr.DataArray,
    init_years: list[int],
    init_month: int,
    lead_days: list[int],
) -> Optional[xr.DataArray]:
    import pandas as pd
    arrays = []
    valid_years = []
    for year in init_years:
        obs_year = _extract_obs_year(obs_da, year, init_month, lead_days)
        if obs_year is not None:
            arrays.append(obs_year)
            valid_years.append(year)
    if not arrays:
        return None
    return xr.concat(arrays, dim=pd.Index(valid_years, name="Y"))


def _align_obs_grid(obs: xr.DataArray, model: xr.DataArray) -> xr.DataArray:
    spatial = [name for name in ("lat", "lon") if name in model.coords and name in obs.coords]
    targets = {
        name: model.coords[name]
        for name in spatial
        if not np.array_equal(obs.coords[name].values, model.coords[name].values)
    }
    return obs.interp(targets, method="linear") if targets else obs


def _area_mean(field: xr.DataArray) -> xr.DataArray:
    spatial = [name for name in ("lat", "lon") if name in field.dims]
    if "lat" in field.dims:
        weights = np.cos(np.deg2rad(field["lat"])).clip(min=0)
        return field.weighted(weights).mean(spatial, skipna=True)
    return field.mean(spatial, skipna=True) if spatial else field


def _compute_per_year_daily_paired_diff(
    da_test: xr.DataArray,
    da_ref: xr.DataArray,
    obs_da: Optional[xr.DataArray],
    init_month: int,
    config: DailyDriftConfig,
    window_def: DailyWindowDef,
) -> xr.DataArray:
    common_years = sorted(
        set(int(y) for y in da_test.coords["Y"].values) &
        set(int(y) for y in da_ref.coords["Y"].values)
    )
    if not common_years:
        raise ValueError("No common init years between test and ref.")

    sel_days = [d for d in config.lead_days
                if window_def.day_first <= d <= window_def.day_last]
    if not sel_days:
        raise ValueError(f"No days in window {window_def.name}.")

    year_diffs = []
    valid_years = []
    for yr in common_years:
        try:
            test_yr = da_test.sel(Y=yr).mean("M", skipna=True)
            ref_yr  = da_ref.sel(Y=yr).mean("M", skipna=True)

            if obs_da is not None:
                obs_yr = _extract_obs_year(
                    obs_da, yr, init_month, config.lead_days
                )
                if obs_yr is None:
                    raise ValueError("observation coverage is incomplete")
                obs_yr = _align_obs_grid(obs_yr, ref_yr)
                b_test = test_yr - obs_yr
                b_ref = ref_yr - obs_yr
            else:
                b_test = test_yr
                b_ref = ref_yr

            anchor_test = b_test.sel(d=config.baseline_day, drop=True)
            anchor_ref  = b_ref.sel(d=config.baseline_day,  drop=True)

            d_test = b_test - anchor_test
            d_ref  = b_ref  - anchor_ref

            win_da = daily_window_average(d_test - d_ref, window_def, verify_complete=False)
            year_diffs.append(win_da)
            valid_years.append(yr)
        except Exception as exc:
            warnings.warn(f"Per-year ΔD failed for yr={yr}: {exc}", stacklevel=2)

    if not year_diffs:
        raise ValueError(f"Could not compute per-year ΔD for window {window_def.name}.")

    import pandas as pd
    return xr.concat(year_diffs, dim=pd.Index(valid_years, name="Y"))


def run_diagnostics(
    data: Dict[str, Dict[int, xr.DataArray]],
    config: DailyDriftConfig,
    obs_da: Optional[xr.DataArray] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, Any]]:
    experiments = list(config.experiments.keys())
    if len(experiments) < 2:
        raise ValueError("Need at least 2 experiments for daily diagnostics.")
    ref_label, test_label = experiments[0], experiments[1]

    results: Dict[int, Dict[str, Any]] = {}

    for init_month in config.active_months:
        season_name = {5: "May", 11: "November"}.get(init_month, f"Month{init_month:02d}")
        if verbose:
            print("=" * 70)
            print(f"Daily Steps 8–9: Diagnostics — {season_name}")
            print("=" * 70)

        da_ref  = data.get(ref_label,  {}).get(init_month)
        da_test = data.get(test_label, {}).get(init_month)

        if da_ref is None or da_test is None:
            warnings.warn(f"Missing daily data for init_month={init_month}", stacklevel=2)
            continue

        init_years = sorted(
            set(int(y) for y in da_ref.coords["Y"].values) &
            set(int(y) for y in da_test.coords["Y"].values)
        )

        results[init_month] = {}

        ref_em  = da_ref.sel(Y=init_years).mean("M",  skipna=True)
        test_em = da_test.sel(Y=init_years).mean("M", skipna=True)

        obs_matrix = None
        if obs_da is not None:
            obs_matrix = _build_obs_matrix(
                obs_da, init_years, init_month, config.lead_days
            )
            if obs_matrix is not None:
                obs_matrix = _align_obs_grid(obs_matrix, ref_em)
            else:
                warnings.warn(
                    f"No initialization year has complete daily observation coverage "
                    f"for init_month={init_month}; using model-state adjustment.",
                    stacklevel=2,
                )

        # Lead day 1 bias & adjustment
        ref_day1  = ref_em.sel(d=config.baseline_day).mean("Y", skipna=True)
        test_day1 = test_em.sel(d=config.baseline_day).mean("Y", skipna=True)

        results[init_month]["model_ref_day1"]  = ref_day1
        results[init_month]["model_test_day1"] = test_day1
        if obs_matrix is not None:
            obs_day1 = obs_matrix.sel(d=config.baseline_day).mean("Y", skipna=True)
            results[init_month]["bias_ref_day1"] = ref_day1 - obs_day1
            results[init_month]["bias_test_day1"] = test_day1 - obs_day1

        if obs_matrix is not None:
            obs_years = [int(y) for y in obs_matrix.Y.values]
            ref_bias = ref_em.sel(Y=obs_years) - obs_matrix
            test_bias = test_em.sel(Y=obs_years) - obs_matrix
        else:
            ref_bias = ref_em
            test_bias = test_em

        ref_anchor = ref_bias.sel(d=config.baseline_day)
        test_anchor = test_bias.sel(d=config.baseline_day)
        ref_ts_d = _area_mean(ref_bias - ref_anchor).mean("Y", skipna=True)
        test_ts_d = _area_mean(test_bias - test_anchor).mean("Y", skipna=True)
        results[init_month]["ref_ts_d"] = ref_ts_d
        results[init_month]["test_ts_d"] = test_ts_d

        # Per-window diagnostics
        for win_name, (df_day, dl_day) in config.window_defs.items():
            win_def  = DailyWindowDef(name=win_name, day_first=df_day, day_last=dl_day)
            sel_days = [d for d in config.lead_days if df_day <= d <= dl_day]

            if not sel_days:
                continue

            if verbose:
                print(f"\n  Window: {win_name}  days={sel_days[0]}..{sel_days[-1]}")

            try:
                # D(d) = B(d) - B(1)
                ref_adj_d  = ref_bias.sel(d=sel_days)  - ref_anchor
                test_adj_d = test_bias.sel(d=sel_days) - test_anchor

                ref_adj_win  = ref_adj_d.mean(["Y", "d"],  skipna=True)
                test_adj_win = test_adj_d.mean(["Y", "d"], skipna=True)
                paired_win   = test_adj_win - ref_adj_win

                paired_by_year = _compute_per_year_daily_paired_diff(
                    da_test=da_test, da_ref=da_ref,
                    obs_da=obs_da, init_month=init_month,
                    config=config, window_def=win_def,
                )

                results[init_month][win_name] = {
                    "adj_ref":             ref_adj_win,
                    "adj_test":            test_adj_win,
                    "paired_diff":         paired_win,
                    "paired_diff_by_year": paired_by_year,
                    "sel_days":            sel_days,
                    "init_years":          init_years,
                }

                if verbose:
                    print(
                        f"    global mean paired ΔD = "
                        f"{float(paired_win.mean(skipna=True)):.4f} {config.variables[0].plot_units}"
                    )

            except Exception as exc:
                warnings.warn(f"Window {win_name} init_month={init_month}: {exc}", stacklevel=2)

    return results
