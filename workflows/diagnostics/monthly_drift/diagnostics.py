"""
diagnostics.py
==============
Steps 6–7: Compute spatial bias, drift adjustment, paired differences,
           and window averages.

All computation is delegated to monthly_core.py (pure xarray, no I/O).

Returns
-------
A nested dict ``results`` with the following structure::

    results[init_month][window_name] = {
        "bias_ref":       xr.DataArray (lat, lon),  # B_ref averaged over window
        "bias_test":      xr.DataArray (lat, lon),
        "adj_ref":        xr.DataArray (lat, lon),  # D_ref averaged over window
        "adj_test":       xr.DataArray (lat, lon),
        "paired_diff":    xr.DataArray (lat, lon),  # ΔD = D_test − D_ref
        "paired_diff_by_year": xr.DataArray (Y, lat, lon),  # for bootstrap
        # lead-1 only (presented as bias map, not adjustment)
        "bias_ref_lead1":    xr.DataArray (lat, lon),
        "bias_test_lead1":   xr.DataArray (lat, lon),
    }
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
from esp_lab.diagnostics.products import resolve_experiment_roles

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.monthly_core import (
    MonthlyConfig,
    WindowDef,
    spatial_adjustment,
    spatial_bias,
    spatial_paired_diff,
    window_average,
    paired_effect_size,
    paired_fdr_significance,
    valid_year_month_spatial,
)


def _obs_da_for_month(
    obs_da: Optional[xr.DataArray],
    init_years: list,
    init_month: int,
    leads: list,
) -> Optional[xr.DataArray]:
    """Return obs time-slices aligned with the hindcast valid months.

    Returns None if obs_da is None (paired-difference-only mode).
    """
    if obs_da is None:
        return None
    return obs_da


def _compute_per_year_paired_diff(
    da_test: xr.DataArray,
    da_ref: xr.DataArray,
    obs_da: Optional[xr.DataArray],
    init_month: int,
    config: MonthlyConfig,
    window_def: WindowDef,
) -> xr.DataArray:
    """Compute paired ΔD for each init year separately (needed for bootstrap).

    Returns DataArray with dims (Y, lat, lon), one value per init year.
    """
    common_years = sorted(
        set(int(y) for y in da_test.coords["Y"].values) &
        set(int(y) for y in da_ref.coords["Y"].values)
    )
    if not common_years:
        raise ValueError("No common init years between test and ref.")

    sel_leads = [L for L in config.leads
                 if window_def.lead_first <= L <= window_def.lead_last]
    if not sel_leads:
        raise ValueError(f"No leads in window {window_def.name}.")

    year_diffs = []
    valid_years = []
    for yr in common_years:
        try:
            test_yr = da_test.sel(Y=yr).mean("M", skipna=True)
            ref_yr  = da_ref.sel(Y=yr).mean("M", skipna=True)

            # obs for this year (needed for bias; skip if no obs)
            obs_yr_ref = obs_yr_test = None
            if obs_da is not None:
                obs_yr_ref = obs_yr_test = _extract_obs_year(
                    obs_da, yr, init_month, config.leads
                )
                if obs_yr_ref is None:
                    raise ValueError("observation coverage is incomplete")

            if obs_yr_test is not None:
                b_test = test_yr - obs_yr_test
                b_ref  = ref_yr  - obs_yr_ref
            else:
                # Paired difference without absolute obs — use ref as surrogate obs
                b_test = test_yr - ref_yr  # ΔF directly
                b_ref  = xr.zeros_like(b_test)

            anchor_test = b_test.sel(L=config.baseline_lead, drop=True)
            anchor_ref  = b_ref.sel(L=config.baseline_lead,  drop=True)
            d_test = b_test - anchor_test
            d_ref  = b_ref  - anchor_ref

            # Window average
            win_da = window_average(d_test - d_ref, window_def, verify_complete=False)
            year_diffs.append(win_da)
            valid_years.append(yr)
        except Exception as exc:
            warnings.warn(f"  Per-year ΔD failed for year={yr}: {exc}", stacklevel=2)

    if not year_diffs:
        raise ValueError(f"Could not compute per-year ΔD for window {window_def.name}.")

    import pandas as pd
    return xr.concat(year_diffs, dim=pd.Index(valid_years, name="Y"))


def _extract_obs_year(
    obs_da: xr.DataArray,
    init_year: int,
    init_month: int,
    leads: list,
) -> Optional[xr.DataArray]:
    """Extract obs slices for all leads of one init year."""
    slices = []
    for L in leads:
        vy, vm = valid_year_month_spatial(init_year, init_month, L)
        try:
            matched = _select_obs_month(obs_da, vy, vm)
            if matched is not None:
                slices.append(matched.assign_coords(L=int(L)))
        except Exception:
            pass
    if len(slices) != len(leads):
        return None
    return xr.concat(slices, dim="L")


def _select_obs_month(obs_da: xr.DataArray, year: int, month: int) -> Optional[xr.DataArray]:
    """Select obs at a specific (year, month) from the time coordinate."""
    for tv in obs_da.time.values:
        try:
            ty, tm = int(tv.year), int(tv.month)
        except AttributeError:
            import pandas as _pd
            ts = _pd.Timestamp(tv)
            ty, tm = ts.year, ts.month
        if ty == year and tm == month:
            return obs_da.sel(time=tv, drop=True)
    return None


def _align_obs_grid(obs: xr.DataArray, model: xr.DataArray) -> xr.DataArray:
    """Interpolate rectilinear observations to the model grid when required."""
    spatial = [name for name in ("lat", "lon") if name in model.coords and name in obs.coords]
    if not spatial:
        return obs
    targets = {
        name: model.coords[name]
        for name in spatial
        if not np.array_equal(obs.coords[name].values, model.coords[name].values)
    }
    return obs.interp(targets, method="linear") if targets else obs


def run_diagnostics(
    data: Dict[str, Dict[int, xr.DataArray]],
    config: MonthlyConfig,
    obs_da: Optional[xr.DataArray] = None,
    verbose: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Run Steps 6–7: bias, adjustment, paired diff, and window averages.

    Parameters
    ----------
    data:
        Preprocessed campaign data: ``data[exp_label][init_month]``.
    config:
        MonthlyConfig.
    obs_da:
        Optional spatial obs DataArray with time coordinate.
    verbose:
        Print progress.

    Returns
    -------
    Nested results dict per init_month per window.
    """
    experiments = list(config.experiments.keys())
    if len(experiments) < 2:
        raise ValueError("Need at least 2 experiments.")
    # Preserve established ref/test display products while resolving the paired
    # subtraction independently of insertion order.
    ref_label, test_label = experiments[0], experiments[1]
    sign_ref_label, sign_test_label = resolve_experiment_roles(experiments)

    results: Dict[int, Dict[str, Any]] = {}

    for init_month in config.active_months:
        season_name = {5: "May", 11: "November"}.get(init_month, f"Month{init_month:02d}")
        if verbose:
            print("=" * 70)
            print(f"Steps 6–7: Diagnostics — {season_name}")
            print("=" * 70)

        da_ref  = data.get(ref_label,  {}).get(init_month)
        da_test = data.get(test_label, {}).get(init_month)

        if da_ref is None or da_test is None:
            warnings.warn(
                f"Missing data for init_month={init_month}: "
                f"ref={da_ref is None}, test={da_test is None}",
                stacklevel=2,
            )
            continue

        init_years = sorted(
            set(int(y) for y in da_ref.coords["Y"].values) &
            set(int(y) for y in da_test.coords["Y"].values)
        )

        results[init_month] = {}

        # Lead-1 bias maps (presented as bias, not adjustment)
        ref_em  = da_ref.sel(Y=init_years).mean("M",  skipna=True)
        test_em = da_test.sel(Y=init_years).mean("M", skipna=True)

        obs_matrix = None
        if obs_da is not None:
            obs_matrix = _build_obs_matrix(
                obs_da, init_years, init_month, config.leads
            )
            if obs_matrix is not None:
                obs_matrix = _align_obs_grid(obs_matrix, ref_em)
            if obs_matrix is None:
                warnings.warn(
                    f"No initialization year has complete observation coverage "
                    f"for init_month={init_month}; observation-relative products skipped.",
                    stacklevel=2,
                )

        ref_lead1  = ref_em.sel(L=config.baseline_lead).mean("Y", skipna=True)
        test_lead1 = test_em.sel(L=config.baseline_lead).mean("Y", skipna=True)

        if obs_matrix is not None:
            obs_lead1 = obs_matrix.sel(L=config.baseline_lead).mean("Y", skipna=True)
            if obs_lead1 is not None:
                results[init_month]["bias_ref_lead1"]  = ref_lead1  - obs_lead1
                results[init_month]["bias_test_lead1"] = test_lead1 - obs_lead1
        results[init_month]["model_ref_lead1"]  = ref_lead1
        results[init_month]["model_test_lead1"] = test_lead1

        # Per-window diagnostics
        for win_name, (lf, ll) in config.window_defs.items():
            win_def = WindowDef(name=win_name, lead_first=lf, lead_last=ll)
            sel_leads = [L for L in config.leads if lf <= L <= ll]

            if not sel_leads:
                continue

            if verbose:
                print(f"\n  Window: {win_name}  leads={sel_leads}")

            try:
                # Ensemble mean → year average bias
                if obs_matrix is not None:
                    valid_years = [int(y) for y in obs_matrix.Y.values]
                    obs_win = obs_matrix.sel(L=sel_leads)
                    ref_bias_L = ref_em.sel(Y=valid_years, L=sel_leads) - obs_win
                    test_bias_L = test_em.sel(Y=valid_years, L=sel_leads) - obs_win
                    ref_anchor = (
                        ref_em.sel(Y=valid_years, L=config.baseline_lead)
                        - obs_matrix.sel(L=config.baseline_lead)
                    )
                    test_anchor = (
                        test_em.sel(Y=valid_years, L=config.baseline_lead)
                        - obs_matrix.sel(L=config.baseline_lead)
                    )
                else:
                    ref_bias_L = ref_em.sel(L=sel_leads)
                    test_bias_L = test_em.sel(L=sel_leads)
                    ref_anchor = ref_em.sel(L=config.baseline_lead)
                    test_anchor = test_em.sel(L=config.baseline_lead)

                # Without obs: bias = model mean
                ref_bias_mean  = ref_bias_L.mean(["Y", "L"], skipna=True)
                test_bias_mean = test_bias_L.mean(["Y", "L"], skipna=True)

                # Adjustment: D(τ) = B(τ) − B(1)
                ref_adj_L  = ref_bias_L  - ref_anchor
                test_adj_L = test_bias_L - test_anchor

                ref_adj_win  = ref_adj_L.mean(["Y", "L"],  skipna=True)
                test_adj_win = test_adj_L.mean(["Y", "L"], skipna=True)
                sign_factor = 1.0 if test_label == sign_test_label else -1.0
                paired_win = sign_factor * (test_adj_win - ref_adj_win)

                # Per-year paired diff for bootstrap
                paired_by_year = _compute_per_year_paired_diff(
                    da_test=data[sign_test_label][init_month],
                    da_ref=data[sign_ref_label][init_month],
                    obs_da=obs_da, init_month=init_month,
                    config=config, window_def=win_def,
                )

                results[init_month][win_name] = {
                    "bias_ref":            ref_bias_mean,
                    "bias_test":           test_bias_mean,
                    "adj_ref":             ref_adj_win,
                    "adj_test":            test_adj_win,
                    "paired_diff":         paired_win,
                    "paired_diff_by_year": paired_by_year,
                    "paired_effect_size":  paired_effect_size(paired_by_year),
                    "fdr_significant":     paired_fdr_significance(
                        paired_by_year, alpha=config.bootstrap_alpha
                    ),
                    "sel_leads":           sel_leads,
                    "init_years":          init_years,
                }

                if verbose:
                    print(
                        f"    global mean paired ΔD = "
                        f"{float(paired_win.mean(skipna=True)):.4f} {config.variables[0].plot_units}"
                    )

            except Exception as exc:
                warnings.warn(
                    f"  Window {win_name} init_month={init_month}: {exc}",
                    stacklevel=2,
                )

    return results


def _extract_obs_year_mean(obs_da, init_years, init_month, leads):
    """Mean obs over init years at given leads."""
    obs_slices = []
    valid_years = []
    for yr in init_years:
        obs_yr = _extract_obs_year(obs_da, yr, init_month, leads)
        if obs_yr is not None:
            obs_slices.append(obs_yr.mean("L", skipna=True))
            valid_years.append(yr)
    if not obs_slices:
        return None
    import pandas as pd
    return xr.concat(obs_slices, dim=pd.Index(valid_years, name="Y")).mean("Y", skipna=True)


def _build_obs_matrix(obs_da, init_years, init_month, leads):
    """Build a (Y, L, lat, lon) obs DataArray aligned with the model."""
    if obs_da is None:
        return 0.0
    slices_yr = []
    valid_years = []
    for yr in init_years:
        yr_slices = []
        for L in leads:
            vy, vm = valid_year_month_spatial(yr, init_month, L)
            sl = _select_obs_month(obs_da, vy, vm)
            if sl is not None:
                yr_slices.append(sl.assign_coords(L=int(L)))
        if len(yr_slices) == len(leads):
            slices_yr.append(xr.concat(yr_slices, dim="L"))
            valid_years.append(yr)
    if not slices_yr:
        return None
    import pandas as pd
    return xr.concat(slices_yr, dim=pd.Index(valid_years, name="Y"))
