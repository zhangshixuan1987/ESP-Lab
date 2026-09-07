"""
daily_core.py
=============
Daily S2D spatial bias and drift analysis — pure core module.

Design principles
-----------------
* Sections 1–7 contain NO I/O and NO ESP-Lab dependency.
  They can be imported and unit-tested without NERSC access or real files.
* Section 8 (unit conversions) imports only numpy/xarray.
* Section 9 (inventory output helpers) uses only stdlib + pandas.
* All spatial diagnostics return xr.DataArray with dimensions
  (lat, lon, d), (Y, lat, lon, d), or (lat, lon) as documented per function,
  where `d` represents 1-based lead days (1..84).

Sections
--------
1. Configuration dataclasses   (DailyDriftConfig, DailyVariableSpec,
                                DailyWindowDef, ExperimentSpec)
2. Inventory types             (DailyInventoryStatus, DailyGateStatus,
                                DailyLeadCoverageResult, DailyInventoryRecord,
                                DailyPairedReadinessReport)
3. Inventory validation        (check_daily_lead_coverage, classify_daily_window_status,
                                classify_daily_archive_status, build_daily_paired_readiness)
4. Spatial bias and drift      (daily_spatial_bias, daily_spatial_adjustment,
                                daily_spatial_paired_diff, daily_window_average)
5. Bootstrap                   (bootstrap_daily_spatial_ci, daily_significance_mask)
6. Regional time series        (daily_regional_timeseries)
7. Utility helpers             (valid_date_from_init_and_lead, apply_mask)
8. Unit conversions            (convert_prect_to_mmday, convert_kelvin_to_celsius_if_needed,
                                DAILY_VARIABLE_CONVERSIONS)
9. Inventory output            (DailyInventoryRecord, daily_inventory_records_to_df,
                                write_daily_inventory_csv, write_daily_missing_report,
                                write_daily_inventory_json)
"""

from __future__ import annotations

import enum
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# ===========================================================================
# Section 1 — Configuration dataclasses
# ===========================================================================


@dataclass
class ExperimentSpec:
    """Metadata for a single hindcast experiment."""

    label: str
    case_prefix: str
    init_description: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ExperimentSpec.label must not be empty.")
        if not self.case_prefix:
            raise ValueError("ExperimentSpec.case_prefix must not be empty.")


@dataclass
class DailyWindowDef:
    """Named lead day window [day_first, day_last], both 1-based inclusive."""

    name: str
    day_first: int
    day_last: int

    def __post_init__(self) -> None:
        if self.day_first < 1:
            raise ValueError(f"DailyWindowDef '{self.name}': day_first must be >= 1.")
        if self.day_last < self.day_first:
            raise ValueError(
                f"DailyWindowDef '{self.name}': day_last ({self.day_last}) "
                f"< day_first ({self.day_first})."
            )

    @property
    def lead_days(self) -> List[int]:
        return list(range(self.day_first, self.day_last + 1))


@dataclass
class DailyVariableSpec:
    """Variable-specific processing options for daily S2D output.

    Parameters
    ----------
    native_field:
        Variable name in model output (e.g. ``"TREFHT"`` or ``"PRECT"``).
    plot_name:
        Human-readable label for figure titles.
    plot_units:
        Target units string (e.g. ``"degC"``, ``"mm/day"``).
    obs_product:
        Observation product key (e.g. ``"ERA5_daily"``) or ``None``.
    obs_var:
        Variable name in observation dataset (e.g. ``"tas"``).
    mask:
        One of ``"land"``, ``"ocean"``, ``"none"``.
    model_convert:
        Callable DataArray → DataArray for model units.
    obs_convert:
        Callable DataArray → DataArray for obs units.
    daily_aggregation:
        ``"mean"``, ``"sum"``, etc.
    """

    native_field: str
    plot_name: str
    plot_units: str
    obs_product: Optional[str] = None
    obs_var: Optional[str] = None
    mask: str = "none"
    model_convert: Optional[Callable] = field(default=None, repr=False)
    obs_convert: Optional[Callable] = field(default=None, repr=False)
    daily_aggregation: str = "mean"

    VALID_MASKS = {"land", "ocean", "none"}

    def __post_init__(self) -> None:
        if self.mask not in self.VALID_MASKS:
            raise ValueError(
                f"DailyVariableSpec.mask={self.mask!r} not in {self.VALID_MASKS}."
            )


@dataclass
class DailyDriftConfig:
    """Global configuration for the daily S2D spatial analysis.

    Parameters
    ----------
    experiments:
        Dict of label → ExperimentSpec.
    init_years:
        List of initialization years [1980..1986].
    init_months:
        List of initialization calendar months [5, 11].
    members:
        Ensemble member tags (e.g. ``["EN00", ..., "EN09"]``).
    lead_days:
        1-based lead days (e.g. ``list(range(1, 85))``).
    window_defs:
        Named lead day windows (e.g. week_1: 1..7, weeks_2_3: 8..21, etc.).
    variables:
        List of DailyVariableSpec objects.
    data_dir:
        Root directory containing case directories.
    baseline_day:
        Lead day at which adjustment D(d) = 0 (default 1).
    pilot_only:
        If True, restrict to ``pilot_years`` and ``pilot_months``.
    pilot_years:
        Subset of init_years for pilot run.
    pilot_months:
        Subset of init_months for pilot run.
    gate_strict:
        If True, raise SystemExit on inventory gate failure.
    n_bootstrap:
        Number of bootstrap resamples.
    bootstrap_seed:
        Random seed.
    freq_tag:
        ``"day"``.
    output_root:
        Output directory root.
    """

    experiments: Dict[str, ExperimentSpec]
    init_years: List[int]
    init_months: List[int]
    members: List[str]
    lead_days: List[int]
    window_defs: Dict[str, Tuple[int, int]]
    variables: List[DailyVariableSpec]
    data_dir: str

    baseline_day: int = 1
    baseline_window: Optional[Tuple[int, int]] = (1, 3)

    pilot_only: bool = True
    pilot_years: List[int] = field(default_factory=lambda: [1980, 1981, 1982])
    pilot_months: List[int] = field(default_factory=lambda: [5])

    gate_strict: bool = False

    n_bootstrap: int = 1000
    bootstrap_seed: int = 42
    bootstrap_alpha: float = 0.05
    freq_tag: str = "day"

    output_root: str = "outputs_daily"

    def __post_init__(self) -> None:
        if not self.experiments:
            raise ValueError("DailyDriftConfig.experiments must not be empty.")
        if not self.init_years:
            raise ValueError("DailyDriftConfig.init_years must not be empty.")
        if not self.members:
            raise ValueError("DailyDriftConfig.members must not be empty.")
        if not self.lead_days:
            raise ValueError("DailyDriftConfig.lead_days must not be empty.")
        if self.baseline_day not in self.lead_days:
            raise ValueError(
                f"DailyDriftConfig.baseline_day={self.baseline_day} not in lead_days."
            )
        if self.baseline_window is not None:
            first, last = self.baseline_window
            if first > last:
                raise ValueError("DailyDriftConfig.baseline_window must be ordered.")
        if not self.variables:
            raise ValueError("DailyDriftConfig.variables must not be empty.")
        for name, (df, dl) in self.window_defs.items():
            DailyWindowDef(name=name, day_first=df, day_last=dl)

    @property
    def active_years(self) -> List[int]:
        return self.pilot_years if self.pilot_only else list(self.init_years)

    @property
    def active_months(self) -> List[int]:
        return self.pilot_months if self.pilot_only else list(self.init_months)

    @property
    def n_experiments(self) -> int:
        return len(self.experiments)

    def get_variable(self, native_field: str) -> DailyVariableSpec:
        for v in self.variables:
            if v.native_field == native_field:
                return v
        raise KeyError(f"Variable {native_field!r} not found in DailyDriftConfig.")


# Default specs for JRA55_FOSIRL vs Reanalysis
DEFAULT_DAILY_EXPERIMENT_SPECS = {
    "JRA55_FOSIRL": ExperimentSpec(
        label="JRA55_FOSIRL",
        case_prefix="WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL",
        init_description="JRA55-forced FOSIRL ocean initialization",
    ),
    "Reanalysis": ExperimentSpec(
        label="Reanalysis",
        case_prefix="WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_BruteForce",
        init_description="Brute-force reanalysis initialization",
    ),
}

DEFAULT_DAILY_WINDOW_DEFS: Dict[str, Tuple[int, int]] = {
    "week_1":     (1, 7),
    "weeks_2_3":  (8, 21),
    "weeks_4_6":  (22, 42),
    "weeks_7_12": (43, 84),
}

RECOMMENDED_DAILY_WINDOW_DEFS: Dict[str, Tuple[int, int]] = {
    "days_1_3":   (1, 3),
    "days_4_7":   (4, 7),
    "days_8_21":  (8, 21),
    "days_22_42": (22, 42),
    "days_43_84": (43, 84),
}


# ===========================================================================
# Section 2 — Inventory types
# ===========================================================================


class DailyInventoryStatus(enum.Enum):
    COMPLETE  = "COMPLETE"
    PARTIAL   = "PARTIAL"
    MISSING   = "MISSING"
    DUPLICATE = "DUPLICATE"
    ERROR     = "ERROR"


class DailyGateStatus(enum.Enum):
    ANALYSIS_READY   = "ANALYSIS_READY"
    ARCHIVE_COMPLETE = "ARCHIVE_COMPLETE"
    BLOCKED          = "BLOCKED"


@dataclass
class DailyLeadCoverageResult:
    available_days: List[int]
    expected_days: List[int]
    missing_days: List[int]
    duplicate_days: List[int]
    extra_days: List[int]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_days) == 0 and len(self.duplicate_days) == 0

    @property
    def has_duplicates(self) -> bool:
        return len(self.duplicate_days) > 0


@dataclass
class DailyPairedReadinessReport:
    common_paired_inits: Set[Tuple[str, str]]
    ref_only_inits: Set[Tuple[str, str]]
    test_only_inits: Set[Tuple[str, str]]
    window_blocked_inits: Dict[str, Set[Tuple[str, str]]]
    requested_inits: Set[Tuple[str, str]]
    gate_pass: bool

    def format_report(self) -> str:
        lines = [
            "=== Daily Paired Readiness Report ===",
            f"  Requested (init_tag, member): {len(self.requested_inits)}",
            f"  Common paired complete:        {len(self.common_paired_inits)}",
            f"  Ref-only:                      {len(self.ref_only_inits)}",
            f"  Test-only:                     {len(self.test_only_inits)}",
            f"  Gate pass: {self.gate_pass}",
            "",
            "  Window blocking:",
        ]
        for win, blocked in self.window_blocked_inits.items():
            lines.append(f"    {win}: {len(blocked)} blocked")
        return "\n".join(lines)


# ===========================================================================
# Section 3 — Inventory validation (pure, no I/O)
# ===========================================================================


def check_daily_lead_coverage(
    available_days: Sequence[int],
    expected_days: Sequence[int],
) -> DailyLeadCoverageResult:
    available = list(available_days)
    expected  = sorted(set(expected_days))
    available_set = set(available)

    missing = sorted(set(expected) - available_set)
    extra   = sorted(available_set - set(expected))

    from collections import Counter
    counts = Counter(available)
    duplicates = sorted(k for k, v in counts.items() if v > 1)

    return DailyLeadCoverageResult(
        available_days=sorted(available_set),
        expected_days=expected,
        missing_days=missing,
        duplicate_days=duplicates,
        extra_days=extra,
    )


def classify_daily_window_status(
    coverage: DailyLeadCoverageResult,
    window_defs: Dict[str, Tuple[int, int]],
) -> Dict[str, DailyGateStatus]:
    """Classify readiness window-by-window for a single file.

    Returns dict of window_name → ANALYSIS_READY | BLOCKED.
    """
    statuses: Dict[str, DailyGateStatus] = {}
    if coverage.has_duplicates:
        for win_name in window_defs:
            statuses[win_name] = DailyGateStatus.BLOCKED
        return statuses

    missing_set = set(coverage.missing_days)
    for win_name, (df, dl) in window_defs.items():
        window_days = set(range(df, dl + 1))
        expected_in_window = window_days & set(coverage.expected_days)
        if missing_set & expected_in_window:
            statuses[win_name] = DailyGateStatus.BLOCKED
        else:
            statuses[win_name] = DailyGateStatus.ANALYSIS_READY
    return statuses


def classify_daily_analysis_status(
    coverage: DailyLeadCoverageResult,
    window_defs: Dict[str, Tuple[int, int]],
) -> DailyGateStatus:
    """Overall analysis status across all requested windows."""
    win_statuses = classify_daily_window_status(coverage, window_defs)
    if any(s == DailyGateStatus.BLOCKED for s in win_statuses.values()):
        return DailyGateStatus.BLOCKED
    return DailyGateStatus.ANALYSIS_READY


def classify_daily_archive_status(
    coverage: DailyLeadCoverageResult,
    ndays: int = 84,
) -> DailyGateStatus:
    if coverage.has_duplicates:
        return DailyGateStatus.BLOCKED
    all_days = list(range(1, ndays + 1))
    missing = set(all_days) - set(coverage.available_days)
    if missing:
        return DailyGateStatus.BLOCKED
    return DailyGateStatus.ARCHIVE_COMPLETE


def build_daily_paired_readiness(
    ref_inventory: pd.DataFrame,
    test_inventory: pd.DataFrame,
    window_defs: Dict[str, Tuple[int, int]],
    init_tags: Sequence[str],
    members: Sequence[str],
) -> DailyPairedReadinessReport:
    requested = {(tag, m) for tag in init_tags for m in members}

    def _ready_set(df: pd.DataFrame) -> Set[Tuple[str, str]]:
        if df.empty or "daily_status" not in df.columns:
            return set()
        ready = df[df["daily_status"] == DailyGateStatus.ANALYSIS_READY.value]
        return {(row["init_tag"], row["member"]) for _, row in ready.iterrows()}

    ref_ready  = _ready_set(ref_inventory)
    test_ready = _ready_set(test_inventory)

    common   = ref_ready & test_ready
    ref_only  = ref_ready - test_ready
    test_only = test_ready - ref_ready

    window_blocked: Dict[str, Set[Tuple[str, str]]] = {}
    for win_name, (df_day, dl_day) in window_defs.items():
        win_days = set(range(df_day, dl_day + 1))

        def _blocked_for_win(df):
            blocked = set()
            if df.empty:
                return {(t, m) for t in init_tags for m in members}
            for _, row in df.iterrows():
                avail = _parse_day_list(row.get("available_days", ""))
                if win_days - set(avail):
                    blocked.add((row["init_tag"], row["member"]))
            return blocked

        ref_blocked  = _blocked_for_win(ref_inventory)
        test_blocked = _blocked_for_win(test_inventory)
        window_blocked[win_name] = ref_blocked | test_blocked

    gate_pass = len(common) > 0

    return DailyPairedReadinessReport(
        common_paired_inits=common,
        ref_only_inits=ref_only,
        test_only_inits=test_only,
        window_blocked_inits=window_blocked,
        requested_inits=requested,
        gate_pass=gate_pass,
    )


def _parse_day_list(value) -> List[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    try:
        return [int(x.strip()) for x in str(value).split(",") if x.strip()]
    except Exception:
        return []


# ===========================================================================
# Section 4 — Spatial bias and drift diagnostics
# ===========================================================================


def valid_date_from_init_and_lead(
    init_year: int, init_month: int, init_day: int, lead_day: int
) -> Tuple[int, int, int]:
    """Calculate valid (year, month, day) from init date and 1-based lead day.

    Lead day 1 = init_date + 1 day.
    """
    import cftime
    init_dt = cftime.DatetimeNoLeap(init_year, init_month, init_day)
    # Add lead_day days
    val_dt = init_dt + pd.Timedelta(days=int(lead_day))
    return int(val_dt.year), int(val_dt.month), int(val_dt.day)


def daily_spatial_bias(
    model_field: xr.DataArray,
    obs_field: Optional[xr.DataArray],
    init_years: Sequence[int],
    init_month: int,
    lead_days: Sequence[int],
) -> xr.DataArray:
    """Compute spatial daily bias B(x,y,d) = F̄(x,y,d) − O(x,y,d).

    Parameters
    ----------
    model_field:
        DataArray with dims (Y, M, d, lat, lon) or (Y, d, lat, lon).
        d coordinate = 1-based lead days.
    obs_field:
        DataArray with time coordinate, or None.
    init_years, init_month, lead_days:
        Hindcast settings.

    Returns
    -------
    xr.DataArray with dims (d, lat, lon), named ``"bias"``.
    """
    if "M" in model_field.dims:
        model_em = model_field.mean("M", skipna=True)
    else:
        model_em = model_field

    if obs_field is None:
        # Without obs: bias = model mean directly
        bias = model_em.mean("Y", skipna=True)
        bias.name = "bias"
        return bias

    # With obs: compute model_mean - obs_mean per lead day
    bias_slices = []
    for d in lead_days:
        model_d = model_em.sel(d=d, drop=False).mean("Y", skipna=True)
        # Match obs valid dates
        obs_slices = []
        for yr in init_years:
            vy, vm, vd = valid_date_from_init_and_lead(yr, init_month, 1, d)
            sl = _select_obs_day(obs_field, vy, vm, vd)
            if sl is not None:
                obs_slices.append(sl)
        if obs_slices:
            obs_d = xr.concat(obs_slices, dim="sample").mean("sample", skipna=True)
            diff = (model_d - obs_d).assign_coords(d=int(d))
        else:
            diff = model_d.assign_coords(d=int(d))
        bias_slices.append(diff)

    result = xr.concat(bias_slices, dim="d")
    result.name = "bias"
    return result


def _select_obs_day(obs_field: xr.DataArray, year: int, month: int, day: int) -> Optional[xr.DataArray]:
    for tv in obs_field.time.values:
        try:
            ty, tm, td = int(tv.year), int(tv.month), int(tv.day)
        except AttributeError:
            ts = pd.Timestamp(tv)
            ty, tm, td = ts.year, ts.month, ts.day
        if ty == year and tm == month and td == day:
            return obs_field.sel(time=tv, drop=True)
    return None


def daily_spatial_adjustment(
    bias: xr.DataArray,
    baseline_day: int = 1,
) -> xr.DataArray:
    """Adjustment relative to baseline day: D(x,y,d) = B(x,y,d) − B(x,y,d₀).

    By construction D(x,y,d₀) = 0 at every grid point.

    Parameters
    ----------
    bias:
        Output of ``daily_spatial_bias``: DataArray with dim ``d``.
    baseline_day:
        Lead day at which D = 0 (default 1).

    Returns
    -------
    DataArray with dim ``d``, named ``"adjustment"``.
    """
    if baseline_day not in bias.d.values:
        raise ValueError(
            f"daily_spatial_adjustment: baseline_day={baseline_day} "
            f"not in bias.d={list(bias.d.values)}."
        )
    anchor = bias.sel(d=baseline_day, drop=False)
    return (bias - anchor).rename("adjustment")


def daily_spatial_paired_diff(
    adj_test: xr.DataArray,
    adj_ref: xr.DataArray,
) -> xr.DataArray:
    """Paired method difference: ΔD(x,y,d) = D_test(x,y,d) − D_ref(x,y,d)."""
    common_days = np.intersect1d(adj_test.d.values, adj_ref.d.values)
    return (
        adj_test.sel(d=common_days) - adj_ref.sel(d=common_days)
    ).rename("paired_diff")


def rapid_adjustment_metrics(
    adjustment: xr.DataArray,
    *,
    day_dim: str = "d",
    early_days: Tuple[int, int] = (1, 7),
    late_days: Tuple[int, int] = (22, 84),
) -> xr.Dataset:
    """Summarize rapid adjustment while retaining start and spatial axes."""
    if day_dim not in adjustment.dims or adjustment.sizes[day_dim] < 2:
        raise ValueError(f"adjustment must contain at least two {day_dim!r} values.")
    rate = adjustment.diff(day_dim)
    abs_rate = np.abs(rate)
    maximum_rate = abs_rate.max(day_dim, skipna=True).rename("maximum_adjustment_rate")
    day_of_maximum = abs_rate.idxmax(day_dim, skipna=True).rename("day_of_maximum_adjustment")
    cumulative = adjustment.sum(day_dim, skipna=True).rename("cumulative_adjustment")
    sign_reversal = (
        (adjustment.max(day_dim, skipna=True) > 0)
        & (adjustment.min(day_dim, skipna=True) < 0)
    ).rename("sign_reversal_or_overshoot")

    early = rate.sel({day_dim: slice(*early_days)}).pipe(np.abs).mean(day_dim, skipna=True)
    late = rate.sel({day_dim: slice(*late_days)}).pipe(np.abs).mean(day_dim, skipna=True)
    early_late_ratio = (early / late.where(late > 0)).rename("early_late_adjustment_ratio")
    return xr.Dataset({
        maximum_rate.name: maximum_rate,
        day_of_maximum.name: day_of_maximum,
        sign_reversal.name: sign_reversal,
        cumulative.name: cumulative,
        early_late_ratio.name: early_late_ratio,
    })


def daily_window_average(
    field: xr.DataArray,
    window_def: DailyWindowDef,
    verify_complete: bool = True,
) -> xr.DataArray:
    """Average a daily spatial field over a named lead day window."""
    available = set(int(x) for x in field.d.values)
    expected  = set(window_def.lead_days)

    if verify_complete:
        missing = expected - available
        if missing:
            raise ValueError(
                f"daily_window_average: window '{window_def.name}' requires days "
                f"{sorted(missing)} which are missing from field."
            )

    sel_days = sorted(expected & available)
    if not sel_days:
        raise ValueError(f"No days from window '{window_def.name}' available.")

    result = field.sel(d=sel_days).mean("d", skipna=True)
    result.attrs["window"] = window_def.name
    result.attrs["day_first"] = window_def.day_first
    result.attrs["day_last"] = window_def.day_last
    return result


# ===========================================================================
# Section 5 — Bootstrap
# ===========================================================================


def bootstrap_daily_spatial_ci(
    paired_diff_by_year: xr.DataArray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Bootstrap 95% CI for mean ΔD across paired initialization years.

    Parameters
    ----------
    paired_diff_by_year:
        DataArray with dim ``Y`` (init years), spatial dims (lat, lon) or (x,).

    Returns
    -------
    (lower, upper) DataArrays with same spatial dims.
    """
    rng = np.random.default_rng(seed)
    data = paired_diff_by_year
    n_y = data.sizes["Y"]

    if n_y < 2:
        raise ValueError(f"bootstrap_daily_spatial_ci requires ≥ 2 init years; got {n_y}.")

    if getattr(data.data, "chunks", None) is not None:
        indexer = xr.DataArray(
            rng.integers(0, n_y, size=(n_boot, n_y)),
            dims=("bootstrap", "sample"),
        )
        boot_means = data.isel(Y=indexer).mean("sample", skipna=True)
        quantiles = boot_means.quantile(
            [alpha / 2, 1 - alpha / 2], dim="bootstrap", skipna=True
        )
        lower = quantiles.isel(quantile=0, drop=True).rename("ci_lower")
        upper = quantiles.isel(quantile=1, drop=True).rename("ci_upper")
        return lower, upper

    vals = data.values
    boot_means = np.empty((n_boot, *vals.shape[1:]), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n_y, size=n_y)
        boot_means[b] = np.nanmean(vals[idx], axis=0)

    lo_vals = np.nanquantile(boot_means, alpha / 2,     axis=0)
    hi_vals = np.nanquantile(boot_means, 1 - alpha / 2, axis=0)

    spatial_dims  = [dim for dim in data.dims if dim != "Y"]
    spatial_coords = {dim: data.coords[dim] for dim in spatial_dims if dim in data.coords}

    lower = xr.DataArray(lo_vals, dims=spatial_dims, coords=spatial_coords, name="ci_lower")
    upper = xr.DataArray(hi_vals, dims=spatial_dims, coords=spatial_coords, name="ci_upper")
    return lower, upper


def daily_significance_mask(
    ci_lower: xr.DataArray,
    ci_upper: xr.DataArray,
) -> xr.DataArray:
    return ((ci_lower > 0) | (ci_upper < 0)).rename("significant")


# ===========================================================================
# Section 6 — Regional time series (days 1–84)
# ===========================================================================


def daily_regional_timeseries(
    field: xr.DataArray,
    region: Optional[Dict[str, float]] = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Compute regional average time series for days 1–84.

    Parameters
    ----------
    field:
        DataArray with dims (d, lat, lon) or (Y, M, d, lat, lon).
    region:
        Dict with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.
        If None, computes global area-unweighted mean.

    Returns
    -------
    DataArray with dim ``d`` (1..84), containing the 1-84 day regional curve.
    """
    da = field
    if region is not None and lat_name in da.coords and lon_name in da.coords:
        da = da.sel(
            **{
                lat_name: slice(region["lat_min"], region["lat_max"]),
                lon_name: slice(region["lon_min"], region["lon_max"]),
            }
        )

    spatial_dims = [dim for dim in (lat_name, lon_name, "x", "ncol", "nCells") if dim in da.dims]
    other_dims   = [dim for dim in da.dims if dim != "d" and dim not in spatial_dims]

    # Average over spatial dims and any extra non-d dims
    avg = da.mean(spatial_dims + other_dims, skipna=True)
    avg.name = "regional_timeseries"
    return avg


# ===========================================================================
# Section 7 — Utility helpers
# ===========================================================================


def apply_mask(
    field: xr.DataArray,
    mask_type: str,
    land_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    if mask_type == "none" or land_mask is None:
        return field
    if mask_type == "land":
        return field.where(land_mask)
    if mask_type == "ocean":
        return field.where(~land_mask)
    raise ValueError(f"apply_mask: unknown mask_type={mask_type!r}")


# ===========================================================================
# Section 8 — Unit conversions
# ===========================================================================


def convert_prect_to_mmday(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower().strip()
    mm_day_aliases = {"mm/day", "mm day-1", "mm/d", "mmday-1", "mmd-1"}
    if units in mm_day_aliases:
        return da
    out = da * 1000.0 * 86400.0
    out.attrs.update(da.attrs)
    out.attrs["units"] = "mm/day"
    return out


def convert_kelvin_to_celsius_if_needed(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower().strip()
    celsius_aliases = {"degc", "celsius", "°c", "degree_c"}
    if any(c in units for c in celsius_aliases):
        return da
    try:
        sample = float(da.mean(skipna=True))
        if sample < 100.0:   # already in degC
            return da
    except Exception:
        pass
    out = da - 273.15
    out.attrs.update(da.attrs)
    out.attrs["units"] = "degC"
    return out


def no_conversion(da: xr.DataArray) -> xr.DataArray:
    return da


DAILY_VARIABLE_CONVERSIONS: Dict[str, Callable] = {
    "PRECT":  convert_prect_to_mmday,
    "PRECC":  convert_prect_to_mmday,
    "PRECL":  convert_prect_to_mmday,
    "TREFHT": convert_kelvin_to_celsius_if_needed,
    "TS":     convert_kelvin_to_celsius_if_needed,
    "LHFLX":  no_conversion,
    "SHFLX":  no_conversion,
}


# ===========================================================================
# Section 9 — Inventory output helpers
# ===========================================================================


@dataclass
class DailyInventoryRecord:
    """One row in the daily inventory CSV (27 columns)."""

    experiment:         str
    case_prefix:        str
    init_date:          str
    init_year:          int
    init_month:         int
    member:             str
    variable:           str
    stream:             str = "daily/0yr"
    file_count:         int = 0
    first_timestamp:    str = ""
    last_timestamp:     str = ""
    expected_days:      str = "1..84"
    available_days:     str = ""
    missing_days:       str = ""
    duplicate_days:     str = ""
    variable_found:     bool = False
    units:              str = "NOT_CHECKED"
    calendar:           str = "NOT_CHECKED"
    grid_signature:     str = "NOT_CHECKED"
    member_valid:       bool = True
    week_1_status:      str = DailyGateStatus.BLOCKED.value
    weeks_2_3_status:   str = DailyGateStatus.BLOCKED.value
    weeks_4_6_status:   str = DailyGateStatus.BLOCKED.value
    weeks_7_12_status:  str = DailyGateStatus.BLOCKED.value
    daily_status:       str = DailyGateStatus.BLOCKED.value
    paired_status:      str = "NOT_CHECKED"
    notes:              str = ""

    def to_dict(self) -> dict:
        return {
            "experiment":         self.experiment,
            "case_prefix":        self.case_prefix,
            "init_date":          self.init_date,
            "init_year":          self.init_year,
            "init_month":         self.init_month,
            "member":             self.member,
            "variable":           self.variable,
            "stream":             self.stream,
            "file_count":         self.file_count,
            "first_timestamp":    self.first_timestamp,
            "last_timestamp":     self.last_timestamp,
            "expected_days":      self.expected_days,
            "available_days":     self.available_days,
            "missing_days":       self.missing_days,
            "duplicate_days":     self.duplicate_days,
            "variable_found":     self.variable_found,
            "units":              self.units,
            "calendar":           self.calendar,
            "grid_signature":     self.grid_signature,
            "member_valid":       self.member_valid,
            "week_1_status":      self.week_1_status,
            "weeks_2_3_status":   self.weeks_2_3_status,
            "weeks_4_6_status":   self.weeks_4_6_status,
            "weeks_7_12_status":  self.weeks_7_12_status,
            "daily_status":       self.daily_status,
            "paired_status":      self.paired_status,
            "notes":              self.notes,
        }


def daily_inventory_records_to_df(records: List[DailyInventoryRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in records])


def write_daily_inventory_csv(df: pd.DataFrame, outpath: Path) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    return outpath


def write_daily_missing_report(
    df: pd.DataFrame,
    outpath: Path,
    config_summary: Optional[str] = None,
) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    blocked = df[df["daily_status"] == DailyGateStatus.BLOCKED.value]

    lines = [
        "=" * 70,
        "Daily S2D Inventory — Missing Data Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 70,
    ]
    if config_summary:
        lines += ["", "Configuration:", config_summary, ""]

    lines += [
        f"Total inventory rows:     {len(df)}",
        f"Blocked (daily gate):     {len(blocked)}",
        "",
    ]

    if not blocked.empty:
        lines.append("--- Blocked rows ---")
        for _, row in blocked.iterrows():
            lines.append(
                f"  {row['experiment']} | {row['init_date']} | "
                f"{row['member']} | {row['variable']} | "
                f"missing_days={row['missing_days']}"
            )
        lines.append("")

    outpath.write_text("\n".join(lines))
    return outpath


def write_daily_inventory_json(
    df: pd.DataFrame,
    outpath: Path,
    extra_meta: Optional[dict] = None,
) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": len(df),
        "daily_status_counts": df["daily_status"].value_counts().to_dict(),
        "experiments": sorted(df["experiment"].unique().tolist()),
        "variables": sorted(df["variable"].unique().tolist()),
    }
    if extra_meta:
        summary.update(extra_meta)

    outpath.write_text(json.dumps(summary, indent=2, default=str))
    return outpath


__all__ = [
    "ExperimentSpec",
    "DailyWindowDef",
    "DailyVariableSpec",
    "DailyDriftConfig",
    "DEFAULT_DAILY_EXPERIMENT_SPECS",
    "DEFAULT_DAILY_WINDOW_DEFS",
    "RECOMMENDED_DAILY_WINDOW_DEFS",
    "DailyInventoryStatus",
    "DailyGateStatus",
    "DailyLeadCoverageResult",
    "DailyPairedReadinessReport",
    "check_daily_lead_coverage",
    "classify_daily_window_status",
    "classify_daily_analysis_status",
    "classify_daily_archive_status",
    "build_daily_paired_readiness",
    "valid_date_from_init_and_lead",
    "daily_spatial_bias",
    "daily_spatial_adjustment",
    "daily_spatial_paired_diff",
    "rapid_adjustment_metrics",
    "daily_window_average",
    "bootstrap_daily_spatial_ci",
    "daily_significance_mask",
    "daily_regional_timeseries",
    "apply_mask",
    "convert_prect_to_mmday",
    "convert_kelvin_to_celsius_if_needed",
    "no_conversion",
    "DAILY_VARIABLE_CONVERSIONS",
    "DailyInventoryRecord",
    "daily_inventory_records_to_df",
    "write_daily_inventory_csv",
    "write_daily_missing_report",
    "write_daily_inventory_json",
]
