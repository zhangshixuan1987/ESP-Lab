"""
monthly_core.py
===============
Monthly S2D spatial bias and drift analysis — pure core module.

Design principles
-----------------
* Sections 1–6 contain NO I/O and NO ESP-Lab dependency.
  They can be imported and unit-tested without NERSC access or real files.
* Section 7 (unit conversions) imports only numpy/xarray.
* Section 8 (inventory output helpers) uses only stdlib + pandas.
* All spatial diagnostics return xr.DataArray with dimensions
  (lat, lon, L), (Y, lat, lon, L), or (lat, lon) as documented per function.

Sections
--------
1. Configuration dataclasses   (MonthlyConfig, VariableConversionSpec,
                                WindowDef, ExperimentSpec)
2. Inventory types             (InventoryStatus, GateStatus, LeadCoverageResult,
                                InventoryRecord, PairedReadinessReport)
3. Inventory validation        (check_lead_coverage, classify_analysis_status,
                                classify_archive_status, build_paired_readiness)
4. Spatial bias and drift      (spatial_bias, spatial_adjustment,
                                spatial_paired_diff, window_average)
5. Bootstrap                   (bootstrap_spatial_ci)
6. Utility helpers             (valid_year_month_spatial, apply_mask)
7. Unit conversions            (convert_prect_to_mmday, VARIABLE_CONVERSIONS)
8. Inventory output            (InventoryRecord, inventory_records_to_df,
                                write_inventory_csv, write_missing_report,
                                write_inventory_json)
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
    """Metadata for a single hindcast experiment (mirrors drift.py)."""

    label: str
    case_prefix: str
    init_description: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("ExperimentSpec.label must not be empty.")
        if not self.case_prefix:
            raise ValueError("ExperimentSpec.case_prefix must not be empty.")


@dataclass
class WindowDef:
    """Named lead window [lead_first, lead_last], both 1-based inclusive."""

    name: str
    lead_first: int
    lead_last: int

    def __post_init__(self) -> None:
        if self.lead_first < 1:
            raise ValueError(f"WindowDef '{self.name}': lead_first must be >= 1.")
        if self.lead_last < self.lead_first:
            raise ValueError(
                f"WindowDef '{self.name}': lead_last ({self.lead_last}) "
                f"< lead_first ({self.lead_first})."
            )

    @property
    def leads(self) -> List[int]:
        return list(range(self.lead_first, self.lead_last + 1))


@dataclass
class VariableConversionSpec:
    """Variable-specific processing options.

    Parameters
    ----------
    native_name:
        Variable name in the model output (e.g. ``"PRECT"``).
    plot_name:
        Human-readable label for figure titles.
    plot_units:
        Target units string (e.g. ``"mm/day"``).
    mask_type:
        One of ``"land"``, ``"ocean"``, ``"none"``.
    obs_product:
        Observation product key, or ``None`` to skip obs-based bias.
    obs_var:
        Variable name inside the observation dataset.
    NOT_CHECKED fields are recorded in the inventory but not validated.
    """

    native_name: str
    plot_name: str
    plot_units: str
    mask_type: str = "none"
    obs_product: Optional[str] = None
    obs_var: Optional[str] = None
    # conversion callable: DataArray → DataArray
    model_convert: Optional[Callable] = field(default=None, repr=False)
    # Observation products often use different native units than the model.
    obs_convert: Optional[Callable] = field(default=None, repr=False)

    VALID_MASKS = {"land", "ocean", "none"}

    def __post_init__(self) -> None:
        if self.mask_type not in self.VALID_MASKS:
            raise ValueError(
                f"VariableConversionSpec.mask_type={self.mask_type!r} "
                f"not in {self.VALID_MASKS}."
            )


@dataclass
class MonthlyConfig:
    """Global configuration for the monthly S2D spatial analysis.

    Parameters
    ----------
    experiments:
        Dict of label → ExperimentSpec.
    init_years:
        List of initialization years.
    init_months:
        List of initialization calendar months (1-based).
    members:
        Ensemble member tags (e.g. ``["EN00", ..., "EN09"]``).
    leads:
        1-based lead indices (e.g. ``list(range(1, 25))``).
    window_defs:
        Named lead windows; keys = name, values = (lead_first, lead_last).
    variables:
        List of VariableConversionSpec objects.
    data_dir:
        Root directory containing case directories.
    realm, grid, freq, ts_split:
        Data access options matching data_access_e3sm conventions.
    baseline_lead:
        Lead at which adjustment D(τ) = 0.
    pilot_only:
        If True, restrict to ``pilot_years`` and ``pilot_months``.
    pilot_years:
        Subset of init_years for the pilot run.
    pilot_months:
        Subset of init_months for the pilot run.
    gate_strict:
        If True, raise SystemExit on inventory gate failure (batch mode).
        If False, warn and continue (notebook mode).
    n_bootstrap:
        Number of bootstrap resamples.
    bootstrap_seed:
        Random seed.
    output_root:
        Root directory for all outputs.
    """

    experiments: Dict[str, ExperimentSpec]
    init_years: List[int]
    init_months: List[int]
    members: List[str]
    leads: List[int]
    window_defs: Dict[str, Tuple[int, int]]
    variables: List[VariableConversionSpec]
    data_dir: str

    realm: str = "atm"
    grid: str = "180x360_aave"
    freq: str = "monthly"
    ts_split: str = "2yr"

    baseline_lead: int = 1

    pilot_only: bool = True
    pilot_years: List[int] = field(default_factory=lambda: [1980, 1981, 1982])
    pilot_months: List[int] = field(default_factory=lambda: [5])

    gate_strict: bool = False

    n_bootstrap: int = 1000
    bootstrap_seed: int = 42
    bootstrap_alpha: float = 0.05

    output_root: str = "outputs"

    def __post_init__(self) -> None:
        if not self.experiments:
            raise ValueError("MonthlyConfig.experiments must not be empty.")
        if not self.init_years:
            raise ValueError("MonthlyConfig.init_years must not be empty.")
        if not self.members:
            raise ValueError("MonthlyConfig.members must not be empty.")
        if not self.leads:
            raise ValueError("MonthlyConfig.leads must not be empty.")
        if self.baseline_lead not in self.leads:
            raise ValueError(
                f"MonthlyConfig.baseline_lead={self.baseline_lead} "
                f"not in leads."
            )
        if not self.variables:
            raise ValueError("MonthlyConfig.variables must not be empty.")
        # Convert window_defs tuples to WindowDef objects for validation
        for name, (lf, ll) in self.window_defs.items():
            WindowDef(name=name, lead_first=lf, lead_last=ll)  # validates

    @property
    def active_years(self) -> List[int]:
        return self.pilot_years if self.pilot_only else list(self.init_years)

    @property
    def active_months(self) -> List[int]:
        return self.pilot_months if self.pilot_only else list(self.init_months)

    @property
    def n_experiments(self) -> int:
        return len(self.experiments)

    def window_lead_sets(self) -> Dict[str, List[int]]:
        """Return {window_name: [leads...]} for all window defs."""
        return {
            name: [L for L in self.leads if lf <= L <= ll]
            for name, (lf, ll) in self.window_defs.items()
        }

    def get_variable(self, native_name: str) -> VariableConversionSpec:
        for v in self.variables:
            if v.native_name == native_name:
                return v
        raise KeyError(f"Variable {native_name!r} not in MonthlyConfig.variables.")


# Default configuration for the JRA55_FOSIRL vs Reanalysis cohort
DEFAULT_EXPERIMENT_SPECS = {
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

DEFAULT_WINDOW_DEFS: Dict[str, Tuple[int, int]] = {
    "months_1_3":   (1, 3),
    "months_4_6":   (4, 6),
    "months_10_12": (10, 12),
    "second_year":  (13, 24),
}


# ===========================================================================
# Section 2 — Inventory types
# ===========================================================================


class InventoryStatus(enum.Enum):
    """File-level inventory status."""
    COMPLETE   = "COMPLETE"    # all expected leads present, no duplicates
    PARTIAL    = "PARTIAL"     # some leads present, some missing
    MISSING    = "MISSING"     # file not found or cannot be opened
    DUPLICATE  = "DUPLICATE"   # one or more leads appear more than once
    ERROR      = "ERROR"       # unexpected error during probing


class GateStatus(enum.Enum):
    """Window-aware gate status for a single (experiment, init_tag, member) tuple."""
    ANALYSIS_READY    = "ANALYSIS_READY"    # all window leads complete
    ARCHIVE_COMPLETE  = "ARCHIVE_COMPLETE"  # all 24 leads complete
    BLOCKED           = "BLOCKED"           # at least one required window lead missing


@dataclass
class LeadCoverageResult:
    """Result of checking lead coverage for a single file.

    Attributes
    ----------
    available_leads:
        Sorted list of forecast leads found in the file.
    expected_leads:
        Sorted list of forecast leads that were requested.
    missing_leads:
        Leads in expected_leads but not in available_leads.
    duplicate_leads:
        Leads that appear more than once.
    extra_leads:
        Leads in available_leads but not in expected_leads.
    """
    available_leads: List[int]
    expected_leads: List[int]
    missing_leads: List[int]
    duplicate_leads: List[int]
    extra_leads: List[int]

    @property
    def is_complete(self) -> bool:
        return len(self.missing_leads) == 0 and len(self.duplicate_leads) == 0

    @property
    def has_duplicates(self) -> bool:
        return len(self.duplicate_leads) > 0


@dataclass
class PairedReadinessReport:
    """Summary of paired analysis readiness across both experiments.

    Attributes
    ----------
    common_paired_inits:
        Set of (init_tag, member) tuples that are COMPLETE in both experiments
        for all requested analysis windows.
    ref_only_inits:
        (init_tag, member) pairs present only in the reference experiment.
    test_only_inits:
        (init_tag, member) pairs present only in the test experiment.
    window_blocked_inits:
        Dict of window_name → set of (init_tag, member) that block that window.
    requested_inits:
        Full set of (init_tag, member) tuples requested by the config.
    gate_pass:
        True if all requested windows have at least one complete paired year.
    """
    common_paired_inits: Set[Tuple[str, str]]
    ref_only_inits: Set[Tuple[str, str]]
    test_only_inits: Set[Tuple[str, str]]
    window_blocked_inits: Dict[str, Set[Tuple[str, str]]]
    requested_inits: Set[Tuple[str, str]]
    gate_pass: bool

    def format_report(self) -> str:
        lines = [
            "=== Paired Readiness Report ===",
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


def check_lead_coverage(
    available_leads: Sequence[int],
    expected_leads: Sequence[int],
) -> LeadCoverageResult:
    """Check lead coverage for a single (experiment, init_tag, member).

    Parameters
    ----------
    available_leads:
        Lead numbers actually found in the file (may contain duplicates).
    expected_leads:
        Lead numbers required by the analysis configuration.

    Returns
    -------
    LeadCoverageResult
    """
    available = list(available_leads)
    expected = sorted(set(expected_leads))
    available_set = set(available)

    missing = sorted(set(expected) - available_set)
    extra = sorted(available_set - set(expected))

    # Detect duplicates
    from collections import Counter
    counts = Counter(available)
    duplicates = sorted(k for k, v in counts.items() if v > 1)

    return LeadCoverageResult(
        available_leads=sorted(available_set),
        expected_leads=expected,
        missing_leads=missing,
        duplicate_leads=duplicates,
        extra_leads=extra,
    )


def classify_analysis_status(
    lead_result: LeadCoverageResult,
    window_defs: Dict[str, Tuple[int, int]],
) -> GateStatus:
    """Classify the analysis readiness of a single file, window-by-window.

    A missing lead blocks only windows that require that lead.  A missing
    lead outside all window ranges does not block the analysis.

    Parameters
    ----------
    lead_result:
        Output of ``check_lead_coverage``.
    window_defs:
        Dict of window_name → (lead_first, lead_last).

    Returns
    -------
    GateStatus.ANALYSIS_READY  — all required window leads are present
    GateStatus.BLOCKED         — at least one window is missing required leads
    """
    if lead_result.has_duplicates:
        return GateStatus.BLOCKED

    missing_set = set(lead_result.missing_leads)
    for name, (lf, ll) in window_defs.items():
        window_leads = set(range(lf, ll + 1))
        # Only consider expected leads within this window
        expected_in_window = window_leads & set(lead_result.expected_leads)
        if missing_set & expected_in_window:
            return GateStatus.BLOCKED

    return GateStatus.ANALYSIS_READY


def classify_archive_status(
    lead_result: LeadCoverageResult,
    nlead: int = 24,
) -> GateStatus:
    """Check whether all 24 monthly leads are complete.

    Parameters
    ----------
    lead_result:
        Output of ``check_lead_coverage``.
    nlead:
        Total number of leads expected for archive completeness.

    Returns
    -------
    GateStatus.ARCHIVE_COMPLETE or GateStatus.BLOCKED
    """
    if lead_result.has_duplicates:
        return GateStatus.BLOCKED
    all_leads = list(range(1, nlead + 1))
    missing_archive = set(all_leads) - set(lead_result.available_leads)
    if missing_archive:
        return GateStatus.BLOCKED
    return GateStatus.ARCHIVE_COMPLETE


def build_paired_readiness(
    ref_inventory: pd.DataFrame,
    test_inventory: pd.DataFrame,
    window_defs: Dict[str, Tuple[int, int]],
    init_tags: Sequence[str],
    members: Sequence[str],
) -> PairedReadinessReport:
    """Build a PairedReadinessReport from two inventory DataFrames.

    Parameters
    ----------
    ref_inventory, test_inventory:
        DataFrames with columns: init_tag, member, analysis_status.
    window_defs:
        Same window_defs used for classification.
    init_tags:
        All requested init tags.
    members:
        All requested members.

    Returns
    -------
    PairedReadinessReport
    """
    requested = {
        (tag, m) for tag in init_tags for m in members
    }

    def _ready_set(df: pd.DataFrame) -> Set[Tuple[str, str]]:
        if df.empty or "analysis_status" not in df.columns:
            return set()
        ready = df[df["analysis_status"] == GateStatus.ANALYSIS_READY.value]
        return {(row["init_tag"], row["member"]) for _, row in ready.iterrows()}

    ref_ready  = _ready_set(ref_inventory)
    test_ready = _ready_set(test_inventory)

    common   = ref_ready & test_ready
    ref_only  = ref_ready - test_ready
    test_only = test_ready - ref_ready

    # Per-window blocking
    window_blocked: Dict[str, Set[Tuple[str, str]]] = {}
    for win_name, (lf, ll) in window_defs.items():
        win_leads = set(range(lf, ll + 1))

        def _blocked_for_window(df, win_l):
            blocked = set()
            if df.empty:
                return {(t, m) for t in init_tags for m in members}
            for _, row in df.iterrows():
                avail = _parse_lead_list(row.get("available_leads", ""))
                missing_in_win = win_l - set(avail)
                if missing_in_win:
                    blocked.add((row["init_tag"], row["member"]))
            return blocked

        ref_blocked  = _blocked_for_window(ref_inventory, win_leads)
        test_blocked = _blocked_for_window(test_inventory, win_leads)
        window_blocked[win_name] = ref_blocked | test_blocked

    gate_pass = len(common) > 0

    return PairedReadinessReport(
        common_paired_inits=common,
        ref_only_inits=ref_only,
        test_only_inits=test_only,
        window_blocked_inits=window_blocked,
        requested_inits=requested,
        gate_pass=gate_pass,
    )


def _parse_lead_list(value) -> List[int]:
    """Parse a serialised lead list (e.g. '1,2,3' or [1,2,3])."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    try:
        return [int(x.strip()) for x in str(value).split(",") if x.strip()]
    except Exception:
        return []


# ===========================================================================
# Section 4 — Spatial bias and drift diagnostics
# ===========================================================================


def valid_year_month_spatial(
    init_year: int, init_month: int, lead: int
) -> Tuple[int, int]:
    """Return (year, month) of a forecast valid time.

    Mirrors drift.py valid_year_month — independent implementation so this
    module has no external dependency.
    """
    month_index = (init_month - 1) + (int(lead) - 1)
    return int(init_year + month_index // 12), int(month_index % 12 + 1)


def spatial_bias(
    model_field: xr.DataArray,
    obs_field: xr.DataArray,
    init_years: Sequence[int],
    init_month: int,
    leads: Sequence[int],
) -> xr.DataArray:
    """Compute spatial absolute forecast bias B(x,y,τ).

    B(x,y,τ) = X̄_model(x,y,τ) − X_obs(x,y,τ)

    where the overbar is the ensemble mean averaged across init years.

    Parameters
    ----------
    model_field:
        DataArray with dims (Y, M, L, lat, lon) or (Y, L, lat, lon) if
        ensemble mean already computed.  Y coordinate = init years (int).
        L coordinate = 1-based leads.
    obs_field:
        DataArray with a ``time`` coordinate (cftime or datetime64).
        Must cover all valid (year, month) pairs required by the hindcast.
    init_years:
        List of initialization years.
    init_month:
        Initialization calendar month (1–12).
    leads:
        1-based lead indices.

    Returns
    -------
    xr.DataArray with dims (L, lat, lon), named ``"bias"``.
    """
    # Ensemble mean if M dimension present
    if "M" in model_field.dims:
        model_em = model_field.mean("M", skipna=True)
    else:
        model_em = model_field

    # Build obs lookup: valid (year, month) → obs slice
    obs_lookup = _build_obs_spatial_lookup(obs_field, init_years, init_month, leads)

    bias_slices = []
    for L in leads:
        # Model: mean over init years at this lead
        model_L = model_em.sel(L=L, drop=False).mean("Y", skipna=True)

        # Obs: mean over the corresponding valid months
        obs_slices = [obs_lookup[(yr, L)] for yr in init_years
                      if (yr, L) in obs_lookup]
        if not obs_slices:
            bias_slices.append(xr.full_like(model_L, fill_value=np.nan))
            continue
        obs_L = xr.concat(obs_slices, dim="sample").mean("sample", skipna=True)

        # Align spatial coordinates
        try:
            model_L_a, obs_L_a = xr.align(model_L, obs_L, join="inner")
        except Exception:
            model_L_a, obs_L_a = model_L, obs_L

        diff = (model_L_a - obs_L_a).assign_coords(L=int(L))
        bias_slices.append(diff)

    if not bias_slices:
        raise ValueError("spatial_bias: no valid leads could be computed.")

    result = xr.concat(bias_slices, dim="L")
    result.name = "bias"
    return result


def _build_obs_spatial_lookup(
    obs_field: xr.DataArray,
    init_years: Sequence[int],
    init_month: int,
    leads: Sequence[int],
) -> Dict[Tuple[int, int], xr.DataArray]:
    """Build {(year, lead): obs_slice} from a spatial obs DataArray."""
    lookup: Dict[Tuple[int, int], xr.DataArray] = {}
    if obs_field is None:
        return lookup

    for yr in init_years:
        for L in leads:
            valid_yr, valid_mo = valid_year_month_spatial(yr, init_month, int(L))
            try:
                # Select by year+month from time coordinate
                time_vals = obs_field.time.values
                matches = []
                for tv in time_vals:
                    try:
                        ty = int(tv.year); tm = int(tv.month)
                    except AttributeError:
                        import pandas as _pd
                        ts = _pd.Timestamp(tv)
                        ty, tm = ts.year, ts.month
                    if ty == valid_yr and tm == valid_mo:
                        matches.append(tv)
                if matches:
                    lookup[(yr, L)] = obs_field.sel(time=matches[0], drop=True)
            except Exception:
                pass
    return lookup


def spatial_adjustment(
    bias: xr.DataArray,
    baseline_lead: int = 1,
) -> xr.DataArray:
    """Adjustment relative to baseline lead: D(x,y,τ) = B(x,y,τ) − B(x,y,τ₀).

    By construction D(x,y,τ₀) = 0 at every grid point.
    Lead 1 should be presented as a bias map, not an adjustment map.

    Parameters
    ----------
    bias:
        Output of ``spatial_bias``: DataArray with dim ``L``.
    baseline_lead:
        Lead at which D = 0.

    Returns
    -------
    DataArray with dim ``L``, named ``"adjustment"``.
    """
    if baseline_lead not in bias.L.values:
        raise ValueError(
            f"spatial_adjustment: baseline_lead={baseline_lead} "
            f"not in bias.L={list(bias.L.values)}."
        )
    anchor = bias.sel(L=baseline_lead, drop=False)
    return (bias - anchor).rename("adjustment")


def spatial_paired_diff(
    adj_test: xr.DataArray,
    adj_ref: xr.DataArray,
) -> xr.DataArray:
    """Paired method difference: ΔD(x,y,τ) = D_test(x,y,τ) − D_ref(x,y,τ).

    Parameters
    ----------
    adj_test, adj_ref:
        Output of ``spatial_adjustment`` for the two experiments.

    Returns
    -------
    DataArray with dim ``L``, named ``"paired_diff"``.
    """
    common_leads = np.intersect1d(adj_test.L.values, adj_ref.L.values)
    return (
        adj_test.sel(L=common_leads) - adj_ref.sel(L=common_leads)
    ).rename("paired_diff")


def window_average(
    field: xr.DataArray,
    window_def: WindowDef,
    verify_complete: bool = True,
) -> xr.DataArray:
    """Average a spatial field over a named lead window.

    Parameters
    ----------
    field:
        DataArray with a ``L`` dimension.
    window_def:
        WindowDef specifying the lead range.
    verify_complete:
        If True, raise ValueError when any window lead is missing.

    Returns
    -------
    DataArray averaged over the window leads, with ``L`` dimension dropped.
    """
    available = set(int(x) for x in field.L.values)
    expected  = set(window_def.leads)

    if verify_complete:
        missing = expected - available
        if missing:
            raise ValueError(
                f"window_average: window '{window_def.name}' requires leads "
                f"{sorted(missing)} which are missing from the field "
                f"(available: {sorted(available)})."
            )

    sel_leads = sorted(expected & available)
    if not sel_leads:
        raise ValueError(
            f"window_average: no leads from window '{window_def.name}' "
            f"are available in the field."
        )

    result = field.sel(L=sel_leads).mean("L", skipna=True)
    result.attrs["window"] = window_def.name
    result.attrs["lead_first"] = window_def.lead_first
    result.attrs["lead_last"] = window_def.lead_last
    result.attrs["n_leads_averaged"] = len(sel_leads)
    return result


# ===========================================================================
# Section 5 — Bootstrap
# ===========================================================================


def bootstrap_spatial_ci(
    paired_diff_by_year: xr.DataArray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Bootstrap 95% CI for the mean ΔD, resampling paired init years.

    Algorithm
    ---------
    1. Input is the per-year paired difference: dims (Y, lat, lon) or (Y, lat, lon).
    2. For each bootstrap replicate: resample Y with replacement, compute mean.
    3. Return alpha/2 and 1-alpha/2 quantiles across replicates.

    Ensemble members are averaged before this function is called — they do NOT
    contribute as independent samples.

    Parameters
    ----------
    paired_diff_by_year:
        DataArray with dim ``Y`` (init years), spatial dims (lat, lon) or (x,).
        Represents ΔD averaged over the lead window, per init year.
    n_boot:
        Number of bootstrap replicates.
    seed:
        Random seed for reproducibility.
    alpha:
        Significance level (0.05 → 95% CI).

    Returns
    -------
    (lower, upper) : tuple of DataArrays with same spatial dims as input,
    named ``"ci_lower"`` and ``"ci_upper"``.
    """
    rng = np.random.default_rng(seed)
    data = paired_diff_by_year
    n_y = data.sizes["Y"]

    if n_y < 2:
        raise ValueError(
            f"bootstrap_spatial_ci requires ≥ 2 init years; got {n_y}."
        )

    # Keep Dask-backed spatial chunks lazy so bootstrap memory scales with a
    # spatial chunk rather than n_boot * the entire global grid.
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

    # In-memory fallback for small arrays.
    vals = data.values  # (n_y, ...)
    boot_means = np.empty((n_boot, *vals.shape[1:]), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n_y, size=n_y)
        boot_means[b] = np.nanmean(vals[idx], axis=0)

    lo_vals = np.nanquantile(boot_means, alpha / 2,     axis=0)
    hi_vals = np.nanquantile(boot_means, 1 - alpha / 2, axis=0)

    # Rebuild DataArrays with same spatial coords
    spatial_dims  = [d for d in data.dims if d != "Y"]
    spatial_coords = {d: data.coords[d] for d in spatial_dims if d in data.coords}

    lower = xr.DataArray(lo_vals, dims=spatial_dims, coords=spatial_coords, name="ci_lower")
    upper = xr.DataArray(hi_vals, dims=spatial_dims, coords=spatial_coords, name="ci_upper")
    return lower, upper


def significance_mask(
    ci_lower: xr.DataArray,
    ci_upper: xr.DataArray,
) -> xr.DataArray:
    """Return a boolean mask where the CI excludes zero (significant).

    Parameters
    ----------
    ci_lower, ci_upper:
        Bootstrap CI bounds from ``bootstrap_spatial_ci``.

    Returns
    -------
    Boolean DataArray: True where (lower > 0) or (upper < 0).
    """
    return ((ci_lower > 0) | (ci_upper < 0)).rename("significant")


# ===========================================================================
# Section 6 — Utility helpers
# ===========================================================================


def apply_mask(
    field: xr.DataArray,
    mask_type: str,
    land_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Apply land, ocean, or no mask to a spatial field.

    Parameters
    ----------
    field:
        Input DataArray with lat/lon dimensions.
    mask_type:
        One of ``"land"``, ``"ocean"``, ``"none"``.
    land_mask:
        Boolean DataArray; True = land cell.  Required when mask_type ≠ ``"none"``.

    Returns
    -------
    Masked DataArray (NaN where excluded).
    """
    if mask_type == "none" or land_mask is None:
        return field

    if mask_type == "land":
        return field.where(land_mask)
    if mask_type == "ocean":
        return field.where(~land_mask)

    raise ValueError(f"apply_mask: unknown mask_type={mask_type!r}")


def init_tag_from_year_month(
    year: int,
    month: int,
    day: int = 1,
    hour: int = 0,
) -> str:
    """Build E3SM init tag string YYYYMMDDHH."""
    return f"{year:04d}{month:02d}{day:02d}{hour:02d}"


# ===========================================================================
# Section 7 — Unit conversions
# ===========================================================================


def convert_prect_to_mmday(da: xr.DataArray) -> xr.DataArray:
    """Convert PRECT from m/s to mm/day.

    No-op if units metadata already indicates mm/day.
    """
    units = str(da.attrs.get("units", "")).lower().strip()
    mm_day_aliases = {"mm/day", "mm day-1", "mm/d", "mmday-1", "mmd-1"}
    if units in mm_day_aliases:
        return da
    out = da * 1000.0 * 86400.0
    out.attrs.update(da.attrs)
    out.attrs["units"] = "mm/day"
    return out


def no_conversion(da: xr.DataArray) -> xr.DataArray:
    """Identity conversion (no-op)."""
    return da


def convert_kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert K → °C."""
    units = str(da.attrs.get("units", "")).lower()
    celsius_aliases = {"degc", "celsius", "°c", "degree_c"}
    if any(c in units for c in celsius_aliases):
        return da
    out = da - 273.15
    out.attrs.update(da.attrs)
    out.attrs["units"] = "degC"
    return out


# Variable conversion lookup table
VARIABLE_CONVERSIONS: Dict[str, Callable] = {
    "PRECT":  convert_prect_to_mmday,
    "PRECC":  convert_prect_to_mmday,
    "PRECL":  convert_prect_to_mmday,
    "TREFHT": convert_kelvin_to_celsius,
    "TS":     convert_kelvin_to_celsius,
    "LHFLX":  no_conversion,
    "SHFLX":  no_conversion,
}


# ===========================================================================
# Section 8 — Inventory output helpers
# ===========================================================================


@dataclass
class InventoryRecord:
    """One row in the monthly inventory CSV.

    All fields have defaults so records can be built incrementally.
    NOT_CHECKED / NOT_APPLICABLE are used for fields outside this analysis scope.
    """
    experiment:         str
    case_prefix:        str
    init_tag:           str
    member:             str
    variable:           str
    stream:             str = ""
    file_count:         int = 0
    available_leads:    str = ""    # comma-separated integers
    missing_leads:      str = ""
    duplicate_leads:    str = ""
    units:              str = "NOT_CHECKED"
    calendar:           str = "NOT_CHECKED"
    grid_signature:     str = "NOT_CHECKED"
    variable_found:     bool = False
    analysis_status:    str = GateStatus.BLOCKED.value
    archive_status:     str = GateStatus.BLOCKED.value
    paired_status:      str = "NOT_CHECKED"
    notes:              str = ""

    _REQUIRED_COLUMNS: Tuple[str, ...] = field(
        default=(
            "experiment", "case_prefix", "init_tag", "member", "variable",
            "stream", "file_count", "available_leads", "missing_leads",
            "duplicate_leads", "units", "calendar", "grid_signature",
            "variable_found", "analysis_status", "archive_status",
            "paired_status", "notes",
        ),
        init=False, repr=False,
    )

    def to_dict(self) -> dict:
        return {
            "experiment":      self.experiment,
            "case_prefix":     self.case_prefix,
            "init_tag":        self.init_tag,
            "member":          self.member,
            "variable":        self.variable,
            "stream":          self.stream,
            "file_count":      self.file_count,
            "available_leads": self.available_leads,
            "missing_leads":   self.missing_leads,
            "duplicate_leads": self.duplicate_leads,
            "units":           self.units,
            "calendar":        self.calendar,
            "grid_signature":  self.grid_signature,
            "variable_found":  self.variable_found,
            "analysis_status": self.analysis_status,
            "archive_status":  self.archive_status,
            "paired_status":   self.paired_status,
            "notes":           self.notes,
        }


def inventory_records_to_df(records: List[InventoryRecord]) -> pd.DataFrame:
    """Convert a list of InventoryRecords to a DataFrame."""
    return pd.DataFrame([r.to_dict() for r in records])


def write_inventory_csv(df: pd.DataFrame, outpath: Path) -> Path:
    """Write inventory DataFrame to CSV."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    return outpath


def write_missing_report(
    df: pd.DataFrame,
    outpath: Path,
    config_summary: Optional[str] = None,
) -> Path:
    """Write a human-readable missing-data report."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    blocked = df[df["analysis_status"] == GateStatus.BLOCKED.value]
    missing_files = df[df["analysis_status"] == InventoryStatus.MISSING.value]

    lines = [
        "=" * 70,
        "Monthly S2D Inventory — Missing Data Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 70,
    ]
    if config_summary:
        lines += ["", "Configuration:", config_summary, ""]

    lines += [
        f"Total inventory rows:     {len(df)}",
        f"Blocked (analysis gate):  {len(blocked)}",
        f"Missing files:            {len(missing_files)}",
        "",
    ]

    if not blocked.empty:
        lines.append("--- Blocked rows ---")
        for _, row in blocked.iterrows():
            lines.append(
                f"  {row['experiment']} | {row['init_tag']} | "
                f"{row['member']} | {row['variable']} | "
                f"missing_leads={row['missing_leads']}"
            )
        lines.append("")

    outpath.write_text("\n".join(lines))
    return outpath


def write_inventory_json(
    df: pd.DataFrame,
    outpath: Path,
    extra_meta: Optional[dict] = None,
) -> Path:
    """Write inventory summary JSON."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    status_counts = df["analysis_status"].value_counts().to_dict()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": len(df),
        "analysis_status_counts": status_counts,
        "experiments": sorted(df["experiment"].unique().tolist()),
        "variables": sorted(df["variable"].unique().tolist()),
        "init_tags": sorted(df["init_tag"].unique().tolist()),
        "members": sorted(df["member"].unique().tolist()),
    }
    if extra_meta:
        summary.update(extra_meta)

    outpath.write_text(json.dumps(summary, indent=2, default=str))
    return outpath


__all__ = [
    # Section 1 — Config
    "ExperimentSpec",
    "WindowDef",
    "VariableConversionSpec",
    "MonthlyConfig",
    "DEFAULT_EXPERIMENT_SPECS",
    "DEFAULT_WINDOW_DEFS",
    # Section 2 — Inventory types
    "InventoryStatus",
    "GateStatus",
    "LeadCoverageResult",
    "PairedReadinessReport",
    # Section 3 — Inventory validation
    "check_lead_coverage",
    "classify_analysis_status",
    "classify_archive_status",
    "build_paired_readiness",
    # Section 4 — Spatial diagnostics
    "valid_year_month_spatial",
    "spatial_bias",
    "spatial_adjustment",
    "spatial_paired_diff",
    "window_average",
    # Section 5 — Bootstrap
    "bootstrap_spatial_ci",
    "significance_mask",
    # Section 6 — Helpers
    "apply_mask",
    "init_tag_from_year_month",
    # Section 7 — Unit conversions
    "convert_prect_to_mmday",
    "convert_kelvin_to_celsius",
    "no_conversion",
    "VARIABLE_CONVERSIONS",
    # Section 8 — Inventory output
    "InventoryRecord",
    "inventory_records_to_df",
    "write_inventory_csv",
    "write_missing_report",
    "write_inventory_json",
]
