"""
s2s_core.py
===========
Subseasonal-to-Seasonal (S2S) Weekly Skill and Diagnostic Core Module.

Pure calculation module for subseasonal weekly lead diagnostics:
- Lead Weeks 1 to 8: Days 1–7, 8–14, 15–21, 22–28, 29–35, 36–42, 43–49, 50–56
- 7-day non-overlapping weekly block averaging for hindcasts and observations
- Lead-dependent weekly climatology and drift-corrected weekly anomalies
- Weekly Anomaly Correlation Coefficient (ACC), RMSE, and significance testing
- Multi-model and multi-experiment weekly comparative metrics

Design Principles:
------------------
- Pure computation: No file system I/O dependencies in core metric functions.
- Can be imported, unit tested, and run without NERSC-specific filesystem access.
- Works seamlessly with both xarray DataArray and Dataset structures.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr


# ===========================================================================
# Section 1 — Weekly Lead Window Definitions
# ===========================================================================


@dataclass(frozen=True)
class WeeklyWindowDef:
    """Definition of a 7-day non-overlapping weekly lead window."""

    week: int  # 1-based week index: 1..8
    day_first: int  # 1-based lead day start (e.g. 1 for Week 1)
    day_last: int  # 1-based lead day end (e.g. 7 for Week 1)
    name: str  # e.g. "Week 1"
    label: str  # e.g. "Week 1 (Days 1–7)"

    def __post_init__(self) -> None:
        if self.week < 1:
            raise ValueError(f"WeeklyWindowDef.week must be >= 1, got {self.week}")
        if self.day_first < 1:
            raise ValueError(f"WeeklyWindowDef.day_first must be >= 1, got {self.day_first}")
        if self.day_last < self.day_first:
            raise ValueError(
                f"day_last ({self.day_last}) cannot be less than day_first ({self.day_first})"
            )

    @property
    def lead_days(self) -> List[int]:
        """List of 1-based lead days in this window."""
        return list(range(self.day_first, self.day_last + 1))

    @property
    def num_days(self) -> int:
        return self.day_last - self.day_first + 1


def build_weekly_window(week: int) -> WeeklyWindowDef:
    """Build a standard 7-day non-overlapping window definition for week 1..8."""
    if week < 1:
        raise ValueError(f"Week must be >= 1, got {week}")
    d_first = 7 * (week - 1) + 1
    d_last = 7 * week
    return WeeklyWindowDef(
        week=week,
        day_first=d_first,
        day_last=d_last,
        name=f"Week {week}",
        label=f"Week {week} (Days {d_first}–{d_last})",
    )


# Authoritative S2S 8-week definitions (Days 1 to 56)
S2S_WEEKLY_WINDOWS: Dict[int, WeeklyWindowDef] = {
    w: build_weekly_window(w) for w in range(1, 9)
}

WEEK_NAMES: List[str] = [S2S_WEEKLY_WINDOWS[w].name for w in range(1, 9)]
WEEK_LABELS: List[str] = [S2S_WEEKLY_WINDOWS[w].label for w in range(1, 9)]


def get_weekly_window(week: int) -> WeeklyWindowDef:
    """Retrieve WeeklyWindowDef for week 1..8."""
    if week in S2S_WEEKLY_WINDOWS:
        return S2S_WEEKLY_WINDOWS[week]
    return build_weekly_window(week)


# ===========================================================================
# Section 2 — Calendar and Valid Date Helpers
# ===========================================================================


def valid_date_from_init_and_lead(
    init_year: int,
    init_month: int,
    init_day: int,
    lead_day: int,
    calendar: str = "noleap",
) -> Tuple[int, int, int]:
    """Compute verification calendar date (year, month, day) from initialization and lead day.

    Parameters
    ----------
    init_year, init_month, init_day:
        Initialization date (lead day 0).
    lead_day:
        1-based forecast lead day (lead day 1 = day after initialization).
    calendar:
        Calendar type ('noleap' or 'standard').

    Returns
    -------
    (year, month, day) tuple of verification date.
    """
    if calendar == "noleap":
        import cftime

        init_dt = cftime.DatetimeNoLeap(init_year, init_month, init_day)
        valid_dt = init_dt + pd.Timedelta(days=lead_day)
        return int(valid_dt.year), int(valid_dt.month), int(valid_dt.day)
    else:
        init_ts = pd.Timestamp(year=init_year, month=init_month, day=init_day)
        valid_ts = init_ts + pd.Timedelta(days=lead_day)
        return int(valid_ts.year), int(valid_ts.month), int(valid_ts.day)


def valid_dates_for_week(
    init_year: int,
    init_month: int,
    week: int,
    init_day: int = 1,
    calendar: str = "noleap",
) -> List[Tuple[int, int, int]]:
    """Return all verification calendar dates (year, month, day) for a given lead week."""
    w_def = get_weekly_window(week)
    return [
        valid_date_from_init_and_lead(init_year, init_month, init_day, d, calendar=calendar)
        for d in w_def.lead_days
    ]


# ===========================================================================
# Section 3 — Weekly Aggregation (Daily / 6-hourly -> Weekly Averages)
# ===========================================================================


def aggregate_daily_to_weekly(
    field: xr.DataArray,
    day_dim: str = "d",
    weeks: Sequence[int] = range(1, 9),
    output_dim: str = "L",
) -> xr.DataArray:
    """Average a daily spatial array along lead days into 7-day weekly blocks.

    Parameters
    ----------
    field:
        DataArray containing lead day dimension `day_dim` (e.g. d=1..84).
    day_dim:
        Name of the lead day dimension.
    weeks:
        Sequence of 1-based weeks to compute (default: 1..8).
    output_dim:
        Name of output week lead dimension (default: "L").

    Returns
    -------
    xr.DataArray with `output_dim` coordinate matching `weeks`.
    """
    if day_dim not in field.dims:
        raise ValueError(f"Dimension {day_dim!r} not found in field dims {field.dims}")

    available_days = set(int(d) for d in field[day_dim].values)
    week_slices = []
    valid_weeks = []

    for w in weeks:
        w_def = get_weekly_window(w)
        req_days = w_def.lead_days
        present_days = [d for d in req_days if d in available_days]
        if not present_days:
            continue
        # Average over the days in this weekly block
        w_mean = field.sel({day_dim: present_days}).mean(dim=day_dim, skipna=True)
        week_slices.append(w_mean)
        valid_weeks.append(w)

    if not week_slices:
        raise ValueError("No matching weekly lead days found in field.")

    result = xr.concat(week_slices, dim=pd.Index(valid_weeks, name=output_dim))
    result.name = field.name
    return result


def aggregate_6hourly_to_daily(
    field: xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray:
    """Aggregate 6-hourly DataArray to daily means using calendar day grouping."""
    if time_dim not in field.dims:
        raise ValueError(f"Dimension {time_dim!r} not found in field dims {field.dims}")

    # Resample or group by day
    try:
        daily = field.resample({time_dim: "1D"}).mean(skipna=True)
    except Exception:
        # Fallback for cftime index
        time_vals = field[time_dim].values
        dates = pd.Index([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in time_vals], name=time_dim)
        daily = field.assign_coords({time_dim: dates}).groupby(time_dim).mean(skipna=True)
    return daily


# ===========================================================================
# Section 4 — Weekly Climatology & Anomalies
# ===========================================================================


def compute_weekly_climatology(
    field: xr.DataArray,
    year_dim: str = "Y",
    clim_years: Optional[Sequence[int]] = None,
) -> xr.DataArray:
    """Compute lead-dependent weekly climatology across initialization years.

    In S2S, drift correction is achieved by calculating the lead-dependent
    mean state for each lead week separately:
        Clim(L, lat, lon) = mean_Y( Field(Y, L, lat, lon) )
    """
    target = field
    if clim_years is not None and year_dim in field.dims:
        available_years = [y for y in clim_years if y in field[year_dim].values]
        if available_years:
            target = field.sel({year_dim: available_years})

    if "M" in target.dims:
        # Average across ensemble members and years
        clim = target.mean(dim=["M", year_dim], skipna=True)
    elif year_dim in target.dims:
        clim = target.mean(dim=year_dim, skipna=True)
    else:
        clim = target
    return clim


def compute_weekly_anomalies(
    field: xr.DataArray,
    year_dim: str = "Y",
    clim_years: Optional[Sequence[int]] = None,
    climatology: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Compute lead-dependent weekly anomalies relative to the hindcast climatology.

    Anom(Y, [M], L, lat, lon) = Field(Y, [M], L, lat, lon) - Clim(L, lat, lon)
    """
    if climatology is None:
        climatology = compute_weekly_climatology(field, year_dim=year_dim, clim_years=clim_years)
    anom = field - climatology
    anom.name = f"{field.name}_anom" if field.name else "anom"
    return anom


# ===========================================================================
# Section 5 — Observation Alignment to Weekly Leads
# ===========================================================================


def align_obs_to_weekly_leads(
    obs_da: xr.DataArray,
    init_years: Sequence[int],
    init_month: int,
    weeks: Sequence[int] = range(1, 9),
    init_day: int = 1,
    time_dim: str = "time",
    output_lead_dim: str = "L",
    output_year_dim: str = "Y",
    calendar: str = "noleap",
) -> xr.DataArray:
    """Extract and aggregate daily observations into matching weekly lead arrays.

    For each initialization year and weekly lead window W, matches verification
    calendar dates and computes the weekly average observation.

    Returns
    -------
    xr.DataArray with dimensions (Y, L, lat, lon).
    """
    if time_dim not in obs_da.dims:
        raise ValueError(f"Time dimension {time_dim!r} not found in observation dims {obs_da.dims}")

    # Build date lookup table for observation time index
    obs_time_vals = obs_da[time_dim].values
    obs_dates = {}
    for i, tv in enumerate(obs_time_vals):
        try:
            k = (int(tv.year), int(tv.month), int(tv.day))
        except AttributeError:
            ts = pd.Timestamp(tv)
            k = (ts.year, ts.month, ts.day)
        obs_dates[k] = i

    year_arrays = []
    valid_years = []

    for yr in init_years:
        week_arrays = []
        valid_weeks = []
        for w in weeks:
            dates = valid_dates_for_week(yr, init_month, w, init_day=init_day, calendar=calendar)
            matched_indices = [obs_dates[d] for d in dates if d in obs_dates]
            if not matched_indices:
                continue
            # Extract slices and take 7-day average
            obs_w = obs_da.isel({time_dim: matched_indices}).mean(dim=time_dim, skipna=True)
            week_arrays.append(obs_w)
            valid_weeks.append(w)

        if week_arrays:
            da_y = xr.concat(week_arrays, dim=pd.Index(valid_weeks, name=output_lead_dim))
            year_arrays.append(da_y)
            valid_years.append(yr)

    if not year_arrays:
        raise ValueError("No observation dates matched the requested initialization years and weeks.")

    result = xr.concat(year_arrays, dim=pd.Index(valid_years, name=output_year_dim))
    result.name = obs_da.name or "obs"
    return result


# ===========================================================================
# Section 6 — Skill Metrics: ACC, RMSE, Bias & Significance
# ===========================================================================


def compute_weekly_acc(
    anom_model: xr.DataArray,
    anom_obs: xr.DataArray,
    year_dim: str = "Y",
    lead_dim: str = "L",
    ensemble_dim: Optional[str] = "M",
) -> xr.DataArray:
    """Compute Anomaly Correlation Coefficient (ACC) for each weekly lead.

    Parameters
    ----------
    anom_model:
        Model weekly anomaly DataArray with dimensions (Y, [M], L, lat, lon).
    anom_obs:
        Observation weekly anomaly DataArray with dimensions (Y, L, lat, lon).
    year_dim:
        Dimension representing initialization years (sample axis).
    lead_dim:
        Dimension representing weekly leads (1..8).
    ensemble_dim:
        Optional ensemble member dimension. If present in model, ensemble mean is used.

    Returns
    -------
    xr.DataArray with dimensions (L, lat, lon) containing Pearson correlation [-1, 1].
    """
    m = anom_model
    if ensemble_dim and ensemble_dim in m.dims:
        m = m.mean(dim=ensemble_dim, skipna=True)

    # Find common initialization years and leads
    common_years = np.intersect1d(m[year_dim].values, anom_obs[year_dim].values)
    if len(common_years) < 3:
        raise ValueError(f"At least 3 initialization years required for ACC, found {len(common_years)}.")

    common_leads = np.intersect1d(m[lead_dim].values, anom_obs[lead_dim].values)
    if len(common_leads) == 0:
        raise ValueError("No overlapping weekly leads between model and observations.")

    m = m.sel({year_dim: common_years, lead_dim: common_leads})
    o = anom_obs.sel({year_dim: common_years, lead_dim: common_leads})

    # Center anomalies across the verification sample
    m_prime = m - m.mean(dim=year_dim, skipna=True)
    o_prime = o - o.mean(dim=year_dim, skipna=True)

    cov = (m_prime * o_prime).mean(dim=year_dim, skipna=True)
    var_m = (m_prime**2).mean(dim=year_dim, skipna=True)
    var_o = (o_prime**2).mean(dim=year_dim, skipna=True)

    denom = np.sqrt(var_m * var_o)
    acc = xr.where(denom > 1e-12, cov / denom, np.nan)
    acc.name = "acc"
    acc.attrs["long_name"] = "Anomaly Correlation Coefficient"
    acc.attrs["units"] = "1"
    return acc


def compute_weekly_rmse(
    anom_model: xr.DataArray,
    anom_obs: xr.DataArray,
    year_dim: str = "Y",
    lead_dim: str = "L",
    ensemble_dim: Optional[str] = "M",
) -> xr.DataArray:
    """Compute Root Mean Square Error (RMSE) of anomalies for each weekly lead."""
    m = anom_model
    if ensemble_dim and ensemble_dim in m.dims:
        m = m.mean(dim=ensemble_dim, skipna=True)

    common_years = np.intersect1d(m[year_dim].values, anom_obs[year_dim].values)
    common_leads = np.intersect1d(m[lead_dim].values, anom_obs[lead_dim].values)

    m = m.sel({year_dim: common_years, lead_dim: common_leads})
    o = anom_obs.sel({year_dim: common_years, lead_dim: common_leads})

    rmse = np.sqrt(((m - o) ** 2).mean(dim=year_dim, skipna=True))
    rmse.name = "rmse"
    rmse.attrs["long_name"] = "Root Mean Square Error"
    return rmse


def compute_weekly_acc_significance(
    acc: xr.DataArray,
    n_samples: int,
    alpha: float = 0.05,
    two_sided: bool = False,
) -> xr.DataArray:
    """Determine statistical significance of ACC values via Student's t-distribution.

    Parameters
    ----------
    acc:
        DataArray of ACC values.
    n_samples:
        Effective degrees of freedom / number of independent initialization years.
    alpha:
        Significance level (e.g. 0.05 for 95% confidence).
    two_sided:
        Whether to perform a two-sided test (default False: one-sided test testing ACC > 0).

    Returns
    -------
    xr.DataArray of boolean masks where True indicates significant skill.
    """
    df = max(1, n_samples - 2)
    # Critical t value
    if two_sided:
        t_crit = stats.t.ppf(1 - alpha / 2, df)
    else:
        t_crit = stats.t.ppf(1 - alpha, df)

    # Convert t_crit to critical r threshold: r = t / sqrt(df + t^2)
    r_crit = t_crit / np.sqrt(df + t_crit**2)

    sig_mask = acc >= r_crit
    sig_mask.name = "sig_mask"
    sig_mask.attrs["critical_acc"] = float(r_crit)
    sig_mask.attrs["alpha"] = alpha
    sig_mask.attrs["degrees_of_freedom"] = df
    return sig_mask


def paired_acc_difference(
    acc_test: xr.DataArray,
    acc_ref: xr.DataArray,
    lead_dim: str = "L",
) -> xr.DataArray:
    """Compute difference in ACC skill: Delta_ACC = ACC_test - ACC_ref."""
    common_leads = np.intersect1d(acc_test[lead_dim].values, acc_ref[lead_dim].values)
    diff = acc_test.sel({lead_dim: common_leads}) - acc_ref.sel({lead_dim: common_leads})
    diff.name = "acc_diff"
    diff.attrs["long_name"] = "ACC Skill Difference"
    return diff


__all__ = [
    "WeeklyWindowDef",
    "build_weekly_window",
    "S2S_WEEKLY_WINDOWS",
    "WEEK_NAMES",
    "WEEK_LABELS",
    "get_weekly_window",
    "valid_date_from_init_and_lead",
    "valid_dates_for_week",
    "aggregate_daily_to_weekly",
    "aggregate_6hourly_to_daily",
    "compute_weekly_climatology",
    "compute_weekly_anomalies",
    "align_obs_to_weekly_leads",
    "compute_weekly_acc",
    "compute_weekly_rmse",
    "compute_weekly_acc_significance",
    "paired_acc_difference",
]

