"""
drift.py
========
S2D initialization-drift diagnostics — unified module.

Sections
--------
1. Configuration dataclasses  (ExperimentSpec, VariableSpec, DriftConfig)
2. Data contract               (CaseArray)
3. Calendar utilities          (valid_year_month, lead_to_cal_month)
4. Observation lookup          (build_obs_lookup)
5. Core drift diagnostics      (absolute_bias, adjustment, paired_difference)
6. Skill diagnostics           (compute_skill)
7. Paired bootstrap            (bootstrap_paired_ci)
8. Window-level summary        (window_summary)
9. Provenance manifest         (write_manifest)
10. Unit conversions           (convert_*, no_conversion)
11. ESP-Lab I/O                (open_regional_cache, cache_to_case_array,
                                build_obs_regional)
12. Pipeline runner            (run_pipeline)

Design principles
-----------------
* Sections 1–9 contain no I/O and no ESP-Lab dependency; they can be
  imported and unit-tested without access to NERSC or real data files.
* Sections 10–12 are the only code that touches file paths, xarray I/O,
  and ESP-Lab modules.  They convert environment-specific structures into
  the strict CaseArray data contract before passing them to the core.
* No default EXPERIMENTS dict is provided.  Always define EXPERIMENTS
  explicitly at the top of your workflow notebook.

Environment-specific items that need verification on NERSC
----------------------------------------------------------
1. The exact Reanalysis and DART case prefixes in ExperimentSpec.
2. The regional-index cache paths (depend on _safe_token(exp_name)).
3. The obs_access observation loader and its field-map conventions.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from esp_lab.diagnostics.products import (
    PAIRED_SIGN_CONVENTION,
    resolve_experiment_roles,
    standardize_product_table,
    write_product_bundle,
)
from esp_lab.diagnostics.store import config_fingerprint


# ===========================================================================
# Section 1 — Configuration dataclasses
# ===========================================================================

def _identity(da):
    """Return da unchanged (default no-op unit conversion)."""
    return da


@dataclass
class ExperimentSpec:
    """Metadata for a single hindcast experiment.

    Parameters
    ----------
    label:
        Short human-readable name used in plot legends and filenames.
    case_prefix:
        E3SM case-prefix string, e.g.
        ``"WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL"``.
    init_description:
        Optional free-text describing the initialization method.

    Notes
    -----
    No default experiments are provided here.  Always define EXPERIMENTS
    explicitly at the top of your workflow notebook, e.g.::

        EXPERIMENTS = {
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

    For a Reanalysis–DART comparison, update the labels and case_prefix
    values above.  No other changes to the analysis code are needed.
    """

    label: str
    case_prefix: str
    init_description: str = ""

    def __post_init__(self):
        if not self.label:
            raise ValueError("ExperimentSpec.label must not be empty.")
        if not self.case_prefix:
            raise ValueError("ExperimentSpec.case_prefix must not be empty.")


@dataclass
class VariableSpec:
    """Metadata for a single model/observation variable.

    Parameters
    ----------
    native_field:
        E3SM output variable name (e.g. ``"TS"`` for surface temperature).
    plot_name:
        Human-readable label used in figure titles (e.g. ``"SST"``).
    plot_units:
        Physical units after conversion (e.g. ``"degC"``, ``"mm/day"``).
    obs_product:
        Observation product key understood by ``obs_access``
        (e.g. ``"HadISST2"``).  ``None`` disables obs-based diagnostics.
    obs_var:
        Variable name inside the observation dataset.
    obs_yrs:
        First year of the observation record.
    obs_yre:
        Last year of the observation record.
    mask:
        Spatial mask applied when area-averaging observations.
        One of ``"ocean"``, ``"land"``, or ``"none"``.
    model_convert:
        Callable applied to model DataArrays after loading (e.g. K → °C).
        Defaults to identity.
    obs_convert:
        Callable applied to observation DataArrays after loading.
        Defaults to identity.
    attractor_component, attractor_variable, attractor_grid, attractor_frequency:
        Metadata locating the prepared model-attractor reference.  These are
        configuration only; diagnostic functions contain no component-specific
        branching.
    """

    native_field: str
    plot_name: str
    plot_units: str
    obs_product: Optional[str]
    obs_var: Optional[str]
    obs_yrs: int = 1979
    obs_yre: int = 2022
    mask: str = "ocean"
    model_convert: Callable = field(default=_identity, repr=False)
    obs_convert: Callable = field(default=_identity, repr=False)
    attractor_component: Optional[str] = None
    attractor_variable: Optional[str] = None
    attractor_grid: str = "180x360_aave"
    attractor_frequency: str = "monthly"

    VALID_MASKS = {"ocean", "land", "none"}

    def __post_init__(self):
        if self.mask not in self.VALID_MASKS:
            raise ValueError(
                f"VariableSpec.mask={self.mask!r} is not valid; "
                f"choose from {self.VALID_MASKS}."
            )
        if (self.obs_product is None) != (self.obs_var is None):
            raise ValueError(
                "VariableSpec: obs_product and obs_var must both be set or "
                "both be None."
            )
        if self.attractor_frequency not in {"monthly", "daily"}:
            raise ValueError(
                "VariableSpec.attractor_frequency must be 'monthly' or 'daily'."
            )
        if (self.attractor_component is None) != (self.attractor_variable is None):
            raise ValueError(
                "VariableSpec: attractor_component and attractor_variable must "
                "both be set or both be None."
            )


@dataclass
class DriftConfig:
    """Global knobs for the paired drift / skill analysis.

    Parameters
    ----------
    init_years:
        List of initialization years (e.g. ``[1980, ..., 1986]``).
    leads:
        List of 1-based lead indices (e.g. ``list(range(1, 25))``).
    members:
        List of ensemble member identifiers (e.g. ``["EN00", ..., "EN09"]``).
    init_months:
        List of initialization calendar months, 1-based (e.g. ``[5, 11]``).
    baseline_lead:
        Lead at which ``adjustment D(τ) = 0`` by definition.  Defaults to 1.
        Must be in ``leads``.
    n_bootstrap:
        Number of bootstrap resamples of start years for paired CI.
    bootstrap_seed:
        Random seed for reproducibility.
    window_defs:
        Named lead windows for the summary table.  Keys are labels; values
        are ``(first_lead, last_lead)`` tuples (1-based, inclusive).
    freq_tag:
        Cache frequency tag: ``"mon"`` or ``"seas"``.
    region:
        ``[lon_min, lon_max, lat_min, lat_max]`` bounding box.
    region_name:
        Human-readable region label (e.g. ``"Nino3.4"``).
    reference_years:
        Inclusive historical period used upstream to prepare the attractor.
    distance_tolerance:
        Neutral band around zero for two-reference regime classification.
    """

    # Required
    init_years: List[int]
    leads: List[int]
    members: List[str]
    init_months: List[int]

    # Optional with defaults
    baseline_lead: int = 1
    n_bootstrap: int = 1000
    bootstrap_seed: int = 42
    window_defs: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "early_adj":    (1, 3),
            "first_season": (1, 6),
            "first_year":   (1, 12),
            "second_year":  (13, 24),
        }
    )
    freq_tag: str = "mon"
    region: List[float] = field(default_factory=lambda: [-170.0, -120.0, -5.0, 5.0])
    region_name: str = "Nino3.4"
    reference_years: Tuple[int, int] = (1981, 2010)
    distance_tolerance: float = 1.0e-6

    def __post_init__(self):
        if not self.init_years:
            raise ValueError("DriftConfig.init_years must not be empty.")
        if not self.leads:
            raise ValueError("DriftConfig.leads must not be empty.")
        if not self.members:
            raise ValueError("DriftConfig.members must not be empty.")
        if not self.init_months:
            raise ValueError("DriftConfig.init_months must not be empty.")
        if self.baseline_lead not in self.leads:
            raise ValueError(
                f"DriftConfig.baseline_lead={self.baseline_lead} is not in "
                f"leads={self.leads}."
            )
        if self.freq_tag not in {"mon", "seas"}:
            raise ValueError(
                f"DriftConfig.freq_tag={self.freq_tag!r}; expected 'mon' or 'seas'."
            )
        if len(self.region) != 4:
            raise ValueError(
                "DriftConfig.region must be [lon_min, lon_max, lat_min, lat_max]."
            )
        if self.reference_years[0] > self.reference_years[1]:
            raise ValueError("DriftConfig.reference_years must be increasing.")
        if self.distance_tolerance < 0:
            raise ValueError("DriftConfig.distance_tolerance must be non-negative.")

    @property
    def n_years(self) -> int:
        return len(self.init_years)

    @property
    def n_leads(self) -> int:
        return len(self.leads)

    @property
    def n_members(self) -> int:
        return len(self.members)


# ===========================================================================
# Section 2 — Data contract: CaseArray
# ===========================================================================

class CaseArray:
    """A validated hindcast DataArray with dimensions ``(Y, M, L)``.

    Validation
    ----------
    * Exactly three dimensions in the order ``Y``, ``M``, ``L``.
    * Coordinate sizes match data shape.
    * Y (init years) and L (leads) are unique and monotonically increasing.
    * At least one finite value exists.

    Attributes
    ----------
    data : xr.DataArray
        The underlying array with dims ``(Y, M, L)``.
    name : str
        Human-readable experiment label (used in error messages).
    """

    REQUIRED_DIMS = ("Y", "M", "L")

    def __init__(self, da: xr.DataArray, name: str = ""):
        self._validate(da, name)
        self.data = da
        self.name = name

    @classmethod
    def _validate(cls, da: xr.DataArray, name: str) -> None:
        tag = f"CaseArray({name!r})" if name else "CaseArray"

        if tuple(da.dims) != cls.REQUIRED_DIMS:
            raise ValueError(
                f"{tag}: expected dims {cls.REQUIRED_DIMS}, got {tuple(da.dims)}. "
                "Transpose the array before creating a CaseArray."
            )
        for dim in cls.REQUIRED_DIMS:
            if dim not in da.coords:
                raise ValueError(f"{tag}: missing coordinate '{dim}'.")
            if da.coords[dim].size != da.sizes[dim]:
                raise ValueError(
                    f"{tag}: coordinate '{dim}' size "
                    f"({da.coords[dim].size}) ≠ data size ({da.sizes[dim]})."
                )

        y_vals = np.asarray(da.coords["Y"].values, dtype=int)
        if len(y_vals) != len(np.unique(y_vals)):
            raise ValueError(f"{tag}: Y coordinate has duplicate values: {y_vals}.")
        if not np.all(np.diff(y_vals) > 0):
            raise ValueError(
                f"{tag}: Y coordinate is not monotonically increasing: {y_vals}."
            )

        l_vals = np.asarray(da.coords["L"].values, dtype=int)
        if len(l_vals) != len(np.unique(l_vals)):
            raise ValueError(f"{tag}: L coordinate has duplicate values: {l_vals}.")
        if not np.all(np.diff(l_vals) > 0):
            raise ValueError(
                f"{tag}: L coordinate is not monotonically increasing: {l_vals}."
            )

        try:
            arr = da.values
        except Exception:
            arr = np.array(da)
        if not np.any(np.isfinite(arr)):
            raise ValueError(f"{tag}: no finite values found in the data.")

    @property
    def init_years(self) -> np.ndarray:
        return np.asarray(self.data.coords["Y"].values, dtype=int)

    @property
    def members(self):
        return list(self.data.coords["M"].values)

    @property
    def leads(self) -> np.ndarray:
        return np.asarray(self.data.coords["L"].values, dtype=int)

    @property
    def n_years(self) -> int:
        return self.data.sizes["Y"]

    @property
    def n_members(self) -> int:
        return self.data.sizes["M"]

    @property
    def n_leads(self) -> int:
        return self.data.sizes["L"]

    def ens_mean(self) -> xr.DataArray:
        """Ensemble mean over M; result has dims ``(Y, L)``."""
        return self.data.mean("M", skipna=True)

    def __repr__(self) -> str:
        return (
            f"CaseArray({self.name!r}) "
            f"Y={list(self.init_years)} M={self.n_members} L={list(self.leads)}"
        )


# ===========================================================================
# Section 3 — Calendar utilities
# ===========================================================================

def valid_year_month(init_year: int, init_month: int, lead: int) -> Tuple[int, int]:
    """Return the ``(year, month)`` of a forecast valid time.

    Uses exact modular arithmetic — no nearest-neighbour selection, no risk
    of silently picking the wrong month when observation dates use different
    calendar conventions.

    Parameters
    ----------
    init_year:
        Calendar year of initialization.
    init_month:
        Calendar month of initialization (1–12).
    lead:
        1-based lead index.  Lead 1 verifies in the initialization month.

    Returns
    -------
    (year, month) : tuple[int, int]

    Examples
    --------
    >>> valid_year_month(1980, 11, 3)   # Nov init, lead 3
    (1981, 1)
    >>> valid_year_month(1980, 5, 12)   # May init, lead 12
    (1981, 4)
    """
    month_index = (init_month - 1) + (int(lead) - 1)
    year  = init_year + month_index // 12
    month = month_index % 12 + 1
    return int(year), int(month)


def lead_to_cal_month(init_month: int, lead: int) -> int:
    """Calendar month (1–12) for a given init month and 1-based lead."""
    _, month = valid_year_month(2000, init_month, lead)
    return month


# ===========================================================================
# Section 4 — Observation lookup (exact, no nearest-neighbour)
# ===========================================================================

def build_obs_lookup(
    obs_series: xr.DataArray,
    init_years: Sequence[int],
    init_month: int,
    leads: Sequence[int],
) -> Dict[Tuple[int, int], float]:
    """Build an exact ``{(year, month): float}`` lookup from an obs time-series.

    Every required valid ``(year, month)`` must be present; an informative
    ``KeyError`` is raised when any are missing.

    Parameters
    ----------
    obs_series:
        Monthly observed regional index with a ``time`` coordinate
        parseable as cftime or numpy.datetime64.
    init_years:
        Initialization years for all hindcasts.
    init_month:
        Initialization calendar month (1–12).
    leads:
        1-based lead indices.

    Returns
    -------
    dict mapping ``(year, month)`` → float scalar.
    """
    available: Dict[Tuple[int, int], float] = {}
    for t in obs_series.time.values:
        try:
            y, m = int(t.year), int(t.month)
        except AttributeError:
            ts = pd.Timestamp(t)
            y, m = ts.year, ts.month
        available[(y, m)] = float(obs_series.sel(time=t))

    lookup: Dict[Tuple[int, int], float] = {}
    missing = []
    for yr in init_years:
        for L in leads:
            ym = valid_year_month(yr, init_month, L)
            if ym not in available:
                missing.append(ym)
            else:
                lookup[ym] = available[ym]

    if missing:
        raise KeyError(
            f"Observation coverage is incomplete for init_month={init_month}, "
            f"init_years={list(init_years)}, leads={list(leads)}.\n"
            f"Missing valid (year, month) pairs: {sorted(missing)}"
        )
    return lookup


def _obs_matrix(
    obs_lookup: Dict[Tuple[int, int], float],
    init_years: Sequence[int],
    init_month: int,
    leads: Sequence[int],
) -> xr.DataArray:
    """Build a ``(Y, L)`` DataArray from a pre-validated obs lookup."""
    rows = [
        [obs_lookup[valid_year_month(yr, init_month, L)] for L in leads]
        for yr in init_years
    ]
    return xr.DataArray(
        np.asarray(rows, dtype=float),
        dims=("Y", "L"),
        coords={"Y": list(init_years), "L": list(leads)},
        name="obs_valid_by_year_lead",
    )


# ===========================================================================
# Section 5 — Core drift diagnostics
# ===========================================================================

def absolute_bias(
    case_arr: CaseArray,
    obs_lookup: Dict[Tuple[int, int], float],
    init_month: int,
) -> xr.DataArray:
    """Absolute forecast bias B(τ) = X̄_hindcast(τ) − X_obs(τ).

    The ensemble mean is averaged over all start years, then compared to
    the observation average over the same valid months.

    Returns
    -------
    xr.DataArray with dim ``L``, named ``"bias"``.
    """
    obs_mat    = _obs_matrix(obs_lookup, case_arr.init_years, init_month, case_arr.leads)
    model_clim = case_arr.ens_mean().mean("Y", skipna=True)
    obs_clim   = obs_mat.mean("Y", skipna=True)
    return (model_clim - obs_clim).rename("bias")


def adjustment(bias: xr.DataArray, baseline_lead: int) -> xr.DataArray:
    """Adjustment relative to baseline lead: D(τ) = B(τ) − B(τ₀).

    Removes the initial offset and isolates subsequent evolution.
    By construction D(τ₀) = 0.

    Returns
    -------
    xr.DataArray with dim ``L``, named ``"adjustment"``.
    """
    if baseline_lead not in bias.L.values:
        raise ValueError(
            f"baseline_lead={baseline_lead} not in bias.L={list(bias.L.values)}."
        )
    anchor = float(bias.sel(L=baseline_lead))
    return (bias - anchor).rename("adjustment")


def paired_difference(
    adj_test: xr.DataArray,
    adj_ref: xr.DataArray,
) -> xr.DataArray:
    """Paired method difference ΔD(τ) = D_test(τ) − D_ref(τ).

    Returns
    -------
    xr.DataArray with dim ``L``, named ``"paired_diff"``.
    """
    common_leads = np.intersect1d(adj_test.L.values, adj_ref.L.values)
    return (
        adj_test.sel(L=common_leads) - adj_ref.sel(L=common_leads)
    ).rename("paired_diff")


def paired_difference_by_start(
    case_test: CaseArray,
    case_ref: CaseArray,
    baseline_lead: int,
) -> xr.DataArray:
    """Paired adjustment per initialization year, retaining the ``Y`` axis."""
    test, ref = xr.align(case_test.data, case_ref.data, join="inner")
    state_difference = test.mean("M", skipna=True) - ref.mean("M", skipna=True)
    result = (
        state_difference - state_difference.sel(L=baseline_lead)
    ).rename("paired_diff_by_start")
    result.attrs["sign_convention"] = PAIRED_SIGN_CONVENTION
    result.attrs["baseline"] = f"lead {baseline_lead} (1-based valid month)"
    return result


# ===========================================================================
# Section 6 — Skill diagnostics (separate from drift)
# ===========================================================================

def compute_skill(
    case_arr: CaseArray,
    obs_lookup: Dict[Tuple[int, int], float],
    init_month: int,
) -> Dict[str, xr.DataArray]:
    """RMSE, ACC, ensemble spread, and spread/RMSE at each lead.

    All metrics are returned as ``xr.DataArray`` with dim ``L``.

    Returns
    -------
    dict with keys: ``rmse``, ``acc``, ``spread``, ``spread_rmse_ratio``.
    """
    obs_mat  = _obs_matrix(obs_lookup, case_arr.init_years, init_month, case_arr.leads)
    ens_mean = case_arr.ens_mean()                                   # (Y, L)

    error  = ens_mean - obs_mat
    rmse   = np.sqrt((error ** 2).mean("Y", skipna=True)).rename("rmse")

    model_anom = ens_mean - ens_mean.mean("Y", skipna=True)
    obs_anom   = obs_mat  - obs_mat.mean("Y",  skipna=True)
    acc    = xr.corr(model_anom, obs_anom, dim="Y").rename("acc")

    spread = case_arr.data.std("M", skipna=True).mean("Y", skipna=True).rename("spread")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ratio = (spread / rmse).rename("spread_rmse_ratio")

    return {"rmse": rmse, "acc": acc, "spread": spread, "spread_rmse_ratio": ratio}


# ===========================================================================
# Section 7 — Paired bootstrap (resampling start years)
# ===========================================================================

def bootstrap_paired_ci(
    case_test: CaseArray,
    case_ref: CaseArray,
    obs_lookup_test: Dict[Tuple[int, int], float],
    obs_lookup_ref: Dict[Tuple[int, int], float],
    init_month: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    baseline_lead: Optional[int] = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Bootstrap paired confidence interval for ΔD(τ), resampling start years.

    Algorithm
    ---------
    1. Find common initialization years across both experiments.
    2. For each replicate: draw start years with replacement, compute bias
       and adjustment for each experiment, record ΔD = D_test − D_ref.
    3. Return alpha/2 and 1−alpha/2 quantiles.

    Ensemble members contribute to spread but are **not** treated as
    independent start years.

    Returns
    -------
    (lower, upper) : tuple of xr.DataArray with dim ``L``.
    """
    rng = np.random.default_rng(seed)

    common_years = np.intersect1d(case_test.init_years, case_ref.init_years)
    common_leads = np.intersect1d(case_test.leads, case_ref.leads)
    if len(common_years) < 2:
        raise ValueError(
            f"bootstrap_paired_ci requires ≥ 2 common init years; "
            f"found {len(common_years)}: {common_years}."
        )

    baseline = (
        int(case_test.leads[0])
        if baseline_lead is None
        else int(baseline_lead)
    )
    if baseline not in common_leads:
        raise ValueError(
            f"baseline_lead={baseline} is not in the common leads "
            f"{list(common_leads)}."
        )

    def _year_means(ca, years, leads):
        return ca.data.sel(Y=list(years), L=list(leads)).mean("M", skipna=True).values

    def _obs_vec(lookup, years, leads, month):
        return np.array(
            [[lookup[valid_year_month(y, month, L)] for L in leads] for y in years],
            dtype=float,
        )

    test_mod = _year_means(case_test, common_years, common_leads)
    ref_mod  = _year_means(case_ref,  common_years, common_leads)
    test_obs = _obs_vec(obs_lookup_test, common_years, common_leads, init_month)
    ref_obs  = _obs_vec(obs_lookup_ref,  common_years, common_leads, init_month)

    n_y, n_l = len(common_years), len(common_leads)
    bl_idx   = list(common_leads).index(baseline)

    boot_diffs = np.empty((n_boot, n_l), dtype=float)
    for b in range(n_boot):
        idx    = rng.integers(0, n_y, size=n_y)
        t_bias = test_mod[idx].mean(axis=0) - test_obs[idx].mean(axis=0)
        r_bias = ref_mod[idx].mean(axis=0)  - ref_obs[idx].mean(axis=0)
        t_adj  = t_bias - t_bias[bl_idx]
        r_adj  = r_bias - r_bias[bl_idx]
        boot_diffs[b] = t_adj - r_adj

    lo = np.quantile(boot_diffs, alpha / 2, axis=0)
    hi = np.quantile(boot_diffs, 1 - alpha / 2, axis=0)
    coords = {"L": list(common_leads)}
    return (
        xr.DataArray(lo, dims=("L",), coords=coords, name="ci_lower"),
        xr.DataArray(hi, dims=("L",), coords=coords, name="ci_upper"),
    )


# ===========================================================================
# Section 8 — Window-level summary table
# ===========================================================================

def window_summary(
    bias: xr.DataArray,
    adj: xr.DataArray,
    skill: Dict[str, xr.DataArray],
    window_defs: Dict[str, Tuple[int, int]],
    baseline_lead: int,
    experiment: str,
    init_month: int,
) -> pd.DataFrame:
    """One-row-per-window summary DataFrame.

    Columns
    -------
    window, lead_first, lead_last, n_leads,
    mean_bias, mean_abs_adj, adj_slope,
    mean_rmse, mean_acc, mean_spread, mean_spread_rmse,
    experiment, init_month, baseline_lead
    """
    rows = []
    all_leads = bias.L.values

    for win_name, (l_first, l_last) in window_defs.items():
        win_leads = [L for L in all_leads if l_first <= L <= l_last]
        if not win_leads:
            continue

        b_win  = bias.sel(L=win_leads)
        a_win  = adj.sel(L=win_leads)

        x = np.asarray(win_leads, dtype=float)
        y = np.asarray(a_win.values, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        slope = float(np.polyfit(x[valid], y[valid], 1)[0]) if valid.sum() >= 2 else np.nan

        rows.append({
            "experiment":     experiment,
            "init_month":     int(init_month),
            "window":         win_name,
            "lead_first":     int(l_first),
            "lead_last":      int(l_last),
            "n_leads":        len(win_leads),
            "mean_bias":      float(b_win.mean("L", skipna=True)),
            "mean_abs_adj":   float(np.abs(a_win).mean("L", skipna=True)),
            "adj_slope":      slope,
            "mean_rmse":      float(skill["rmse"].sel(L=win_leads).mean("L", skipna=True)),
            "mean_acc":       float(skill["acc"].sel(L=win_leads).mean("L", skipna=True)),
            "mean_spread":    float(skill["spread"].sel(L=win_leads).mean("L", skipna=True)),
            "mean_spread_rmse": float(skill["spread_rmse_ratio"].sel(L=win_leads).mean("L", skipna=True)),
            "baseline_lead":  int(baseline_lead),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Section 9 — Provenance manifest
# ===========================================================================

def write_manifest(
    outdir: Path,
    prefix: str,
    experiments: Dict[str, ExperimentSpec],
    var_spec: VariableSpec,
    drift_cfg: DriftConfig,
    region: Optional[List[float]] = None,
    region_name: str = "",
    input_hashes: Optional[Dict[str, str]] = None,
    sample_years: Optional[Dict[int, Sequence[int]]] = None,
    sample_leads: Optional[Dict[int, Sequence[int]]] = None,
) -> Path:
    """Write a JSON provenance manifest alongside output figures/tables.

    Returns
    -------
    Path to the written ``{prefix}_manifest.json`` file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _to_dict(spec) -> dict:
        d = {}
        for k, v in vars(spec).items():
            if callable(v):
                d[k] = getattr(v, "__name__", repr(v))
            elif isinstance(v, (list, tuple, dict, str, int, float, bool, type(None))):
                d[k] = v
            else:
                d[k] = str(v)
        return d

    manifest = {
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0.0",
        "workflow": "5a_regional_drift",
        "sign_convention": PAIRED_SIGN_CONVENTION,
        "baseline": f"lead {drift_cfg.baseline_lead} (1-based valid month)",
        "python_version": sys.version,
        "region":         region if region is not None else drift_cfg.region,
        "region_name":    region_name or drift_cfg.region_name,
        "experiments":    {lbl: _to_dict(spec) for lbl, spec in experiments.items()},
        "variable":       _to_dict(var_spec),
        "drift_config": {
            "init_years":     list(drift_cfg.init_years),
            "leads":          list(drift_cfg.leads),
            "members":        list(drift_cfg.members),
            "init_months":    list(drift_cfg.init_months),
            "baseline_lead":  drift_cfg.baseline_lead,
            "n_bootstrap":    drift_cfg.n_bootstrap,
            "bootstrap_seed": drift_cfg.bootstrap_seed,
            "window_defs":    {k: list(v) for k, v in drift_cfg.window_defs.items()},
            "freq_tag":       drift_cfg.freq_tag,
        },
        "input_hashes": input_hashes or {},
        "analysis_samples": {
            str(init_month): {
                "years": [int(value) for value in years],
                "leads": [
                    int(value)
                    for value in (sample_leads or {}).get(init_month, [])
                ],
            }
            for init_month, years in (sample_years or {}).items()
        },
    }

    path = outdir / f"{prefix}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


# ===========================================================================
# Section 10 — Unit conversions
# ===========================================================================

def convert_kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    out = da - 273.15
    out.attrs.update(da.attrs)
    out.attrs["units"] = "degC"
    return out


def _temperature_units_kind(da: xr.DataArray) -> str:
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin", "degrees_k", "degree_k"} or "kelvin" in units:
        return "kelvin"
    if any(t in units for t in ("degc", "degree_c", "degrees_c", "celsius", "°c")):
        return "celsius"
    return "unknown"


def convert_kelvin_to_celsius_if_needed(da: xr.DataArray) -> xr.DataArray:
    """Convert Kelvin → °C if units metadata or sample value suggests Kelvin."""
    kind = _temperature_units_kind(da)
    if kind == "kelvin":
        return convert_kelvin_to_celsius(da)
    if kind == "celsius":
        return da
    try:
        indexers = {dim: 0 for dim in da.dims if da.sizes.get(dim, 0) > 0}
        sample = float(da.isel(indexers).mean(skipna=True))
    except Exception:
        return da
    if np.isfinite(sample) and sample > 100.0:
        return convert_kelvin_to_celsius(da)
    return da


def convert_precip_mps_to_mmday(da: xr.DataArray) -> xr.DataArray:
    if _units_match(da.attrs.get("units"), "mm/day"):
        return da
    out = da * 1000.0 * 86400.0
    out.attrs.update(da.attrs)
    out.attrs["units"] = "mm/day"
    return out


def convert_pa_to_hpa(da: xr.DataArray) -> xr.DataArray:
    if _units_match(da.attrs.get("units"), "hPa"):
        return da
    out = da * 1.0e-2
    out.attrs.update(da.attrs)
    out.attrs["units"] = "hPa"
    return out


def no_conversion(da: xr.DataArray) -> xr.DataArray:
    """Identity unit conversion (no-op)."""
    return da


def _normalise_units(units: object) -> str:
    """Return a compact unit token for idempotent cache conversions."""
    return re.sub(r"[\s_^{-}]", "", str(units or "").strip().lower())


def _units_match(actual: object, expected: object) -> bool:
    actual_token = _normalise_units(actual)
    expected_token = _normalise_units(expected)
    aliases = {
        "mm/day": {"mm/day", "mmday-1", "mmd-1", "mmperday"},
        "hpa": {"hpa", "hectopascal", "hectopascals"},
        "degc": {"degc", "degreec", "degreesc", "celsius", "°c"},
    }
    expected_aliases = aliases.get(expected_token, {expected_token})
    return actual_token in {_normalise_units(value) for value in expected_aliases}


# ===========================================================================
# Section 11 — ESP-Lab I/O (only section with external dependencies)
# ===========================================================================

def _safe_token(value: str) -> str:
    """Convert a string to a filesystem-safe token while preserving case."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def regional_cache_path(
    outdir: Path,
    exp_name: str,
    init_month: int,
    freq_tag: str,
    field_key: str,
    region_name: str,
) -> Path:
    """Return the canonical path to a regional-index cache file."""
    return (
        outdir
        / _safe_token(exp_name)
        / "leadtime_drift"
        / "atm"
        / (
            f"{_safe_token(exp_name)}"
            f"_init{init_month:02d}"
            f"_{_safe_token(field_key)}"
            f"_{_safe_token(region_name)}"
            f"_{freq_tag}.nc"
        )
    )


def open_regional_cache(
    path: Path,
    var_spec: VariableSpec,
    reg_chunk: Optional[dict] = None,
    expected_attrs: Optional[Dict[str, object]] = None,
) -> xr.DataArray:
    """Open a regional-index cache file, validate metadata, convert units.

    Parameters
    ----------
    path:
        Path to a ``.nc`` file written by ``S2DDiagnostics``.
    var_spec:
        VariableSpec; used for unit conversion.
    reg_chunk:
        Optional dask chunking dict.

    Returns
    -------
    xr.DataArray with units set to ``var_spec.plot_units``.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Regional cache not found: {path}")
    da = xr.open_dataarray(path, chunks=reg_chunk or {})
    mismatches = []
    for key, expected in (expected_attrs or {}).items():
        actual = da.attrs.get(key)
        if str(actual) != str(expected):
            mismatches.append(f"{key}: expected {expected!r}, found {actual!r}")
    if mismatches:
        da.close()
        raise ValueError(
            f"Regional cache metadata does not match the requested analysis: "
            f"{path}. " + "; ".join(mismatches)
        )
    if not _units_match(da.attrs.get("units"), var_spec.plot_units):
        da = var_spec.model_convert(da)
    da.attrs["units"] = var_spec.plot_units
    return da


def cache_to_case_array(
    da: xr.DataArray,
    exp_name: str,
    drift_cfg: DriftConfig,
) -> CaseArray:
    """Convert a regional-cache DataArray to a validated CaseArray.

    Normalises the Y coordinate to integer years, filters to
    ``drift_cfg.init_years`` and ``drift_cfg.leads``, and validates
    via ``CaseArray.__init__``.

    Parameters
    ----------
    da:
        Open regional-index DataArray with expected dims ``(Y, M, L)``.
    exp_name:
        Human-readable label (used in error messages).
    drift_cfg:
        DriftConfig supplying init_years, members, and leads.
    """
    missing_dims = [dim for dim in ("Y", "M", "L") if dim not in da.dims]
    if missing_dims:
        raise ValueError(
            f"Experiment {exp_name!r}: cache is missing dimensions {missing_dims}; "
            f"found {tuple(da.dims)}."
        )

    y_raw = da.coords["Y"].values
    y_int = []
    for yv in y_raw:
        if hasattr(yv, "year"):
            y_int.append(int(yv.year))
        else:
            m = re.search(r"(\d{4})", str(yv))
            y_int.append(int(m.group(1)) if m else int(yv))
    da = da.assign_coords(Y=("Y", y_int))

    req_years = [y for y in drift_cfg.init_years if y in y_int]
    missing_years = [y for y in drift_cfg.init_years if y not in y_int]
    available_leads = [int(value) for value in da.coords["L"].values]
    if drift_cfg.freq_tag == "mon":
        expected_leads = list(drift_cfg.leads)
    else:
        expected_leads = [L for L in drift_cfg.leads if L in available_leads]
    missing_leads = [L for L in expected_leads if L not in available_leads]
    available_members = [str(value) for value in da.coords["M"].values]
    missing_members = [m for m in drift_cfg.members if m not in available_members]
    if not req_years:
        raise ValueError(
            f"Experiment {exp_name!r}: none of the requested init_years "
            f"{drift_cfg.init_years} are in the cache (Y={y_int})."
        )
    if missing_years:
        warnings.warn(
            f"Experiment {exp_name!r}: cache is missing requested initialization "
            f"years {missing_years}; available requested years are {req_years}. "
            "Comparative diagnostics will use the common-year intersection.",
            RuntimeWarning,
            stacklevel=2,
        )
    if missing_leads:
        raise ValueError(
            f"Experiment {exp_name!r}: cache is missing requested leads "
            f"{missing_leads}; available leads are {available_leads}."
        )
    if drift_cfg.baseline_lead not in expected_leads:
        raise ValueError(
            f"Experiment {exp_name!r}: baseline_lead={drift_cfg.baseline_lead} "
            f"is not available for freq_tag={drift_cfg.freq_tag!r}; "
            f"available requested leads are {expected_leads}."
        )
    if missing_members:
        raise ValueError(
            f"Experiment {exp_name!r}: cache is missing requested members "
            f"{missing_members}; available members are {available_members}."
        )
    da = da.sel(Y=req_years, M=drift_cfg.members, L=expected_leads)
    if tuple(da.dims) != ("Y", "M", "L"):
        da = da.transpose("Y", "M", "L")
    return CaseArray(da, name=exp_name)


def build_obs_regional(
    obs_access,
    var_spec: VariableSpec,
    drift_cfg: DriftConfig,
    obs_dir: str,
    spatial,
) -> Optional[xr.DataArray]:
    """Load, area-weight, and return the regional observation time-series.

    Parameters
    ----------
    obs_access:
        ESP-Lab ``data_access_obs`` module.
    var_spec:
        VariableSpec for the primary field.
    drift_cfg:
        DriftConfig (provides region, region_name, freq_tag, init_years).
    obs_dir:
        Root path to the observation directory.
    spatial:
        ESP-Lab ``spatial_utils`` module (for land/sea masking).

    Returns
    -------
    Monthly (or 3-month rolling) regional ``xr.DataArray`` with a ``time``
    coordinate, or ``None`` if ``var_spec.obs_product is None``.
    """
    if var_spec.obs_product is None:
        return None

    lat_name, lon_name = "lat", "lon"
    obs_chunk = {"time": 120, "lat": 90, "lon": 180}

    ds = obs_access.get_monthly_data(
        obs_dir=obs_dir,
        field=var_spec.native_field,
        field_map={var_spec.native_field: var_spec.obs_var},
        product=var_spec.obs_product,
        start_year=var_spec.obs_yrs,
        end_year=var_spec.obs_yre,
        chunks=obs_chunk,
        verbose=True,
    )
    if var_spec.obs_var not in ds.data_vars:
        raise KeyError(
            f"Observation variable {var_spec.obs_var!r} not found in "
            f"{var_spec.obs_product}. Available: {list(ds.data_vars)}"
        )

    obs_da = var_spec.obs_convert(ds[var_spec.obs_var])
    ds[var_spec.obs_var] = obs_da

    obs_sel_end = min(var_spec.obs_yre, max(drift_cfg.init_years) + 3)
    ds = ds.sel(time=slice(str(min(drift_cfg.init_years)), str(obs_sel_end)))

    if var_spec.mask == "ocean":
        landmask = spatial.create_land_sea_mask(
            ds[var_spec.obs_var], lat_key=lat_name, lon_key=lon_name
        ).astype(bool)
        mask = (~landmask).chunk({lat_name: 90, lon_name: 180})
    elif var_spec.mask == "land":
        landmask = spatial.create_land_sea_mask(
            ds[var_spec.obs_var], lat_key=lat_name, lon_key=lon_name
        ).astype(bool)
        mask = landmask.chunk({lat_name: 90, lon_name: 180})
    else:
        mask = None

    weights = obs_access.obs_regional_weights(
        ds[var_spec.obs_var],
        drift_cfg.region,
        lat_name=lat_name,
        lon_name=lon_name,
        mask=mask,
    )
    obs_regional = (
        ds[var_spec.obs_var].weighted(weights).mean((lat_name, lon_name)).load()
    )
    obs_regional.name = "obs_regional_index"
    obs_regional.attrs["units"] = var_spec.plot_units

    if drift_cfg.freq_tag == "mon":
        return obs_regional
    if drift_cfg.freq_tag == "seas":
        result = (
            obs_regional.rolling(time=3, min_periods=3, center=True)
            .mean()
            .dropna("time", how="all")
        )
        result.name = "obs_regional_index_seasonal"
        result.attrs["units"] = var_spec.plot_units
        return result

    raise ValueError(
        f"Unsupported freq_tag={drift_cfg.freq_tag!r}; expected 'mon' or 'seas'."
    )


# ===========================================================================
# Section 12 — Pipeline runner
# ===========================================================================

def run_pipeline(
    experiment_specs: Dict[str, ExperimentSpec],
    var_spec: VariableSpec,
    drift_cfg: DriftConfig,
    case_arrays: Dict[str, Dict[int, CaseArray]],
    obs_series: xr.DataArray,
    outdir: Optional[Path] = None,
    figure_outdir: Optional[Path] = None,
    prefix: str = "drift",
    write_outputs: bool = True,
) -> Dict:
    """Run the full paired drift and skill analysis.

    Parameters
    ----------
    experiment_specs:
        Dict mapping experiment label → ExperimentSpec.
    var_spec:
        VariableSpec for the primary field.
    drift_cfg:
        Global drift configuration.
    case_arrays:
        Pre-loaded validated hindcasts:
        ``{exp_label: {init_month: CaseArray}}``.
    obs_series:
        Regional observation time-series (from ``build_obs_regional``).
    outdir:
        Directory for CSV / JSON output.  ``None`` → no files written.
    figure_outdir:
        Directory for figure output (reserved for future use).
    prefix:
        Common filename prefix for outputs.
    write_outputs:
        ``False`` suppresses all file writes (useful for testing).

    Returns
    -------
    dict with keys:
        ``bias``         : {exp_label: {init_month: DataArray}}
        ``adjustment``   : {exp_label: {init_month: DataArray}}
        ``skill``        : {exp_label: {init_month: dict}}
        ``paired_diff``  : {init_month: DataArray}  (2-experiment case)
        ``ci_lower``     : {init_month: DataArray}  (2-experiment case)
        ``ci_upper``     : {init_month: DataArray}  (2-experiment case)
        ``window_tables``: list[pd.DataFrame]
    """
    exp_labels = list(experiment_specs.keys())
    results: Dict = {
        "bias": {}, "adjustment": {}, "skill": {},
        "paired_diff": {}, "paired_diff_by_start": {},
        "ci_lower": {}, "ci_upper": {},
        "window_tables": [], "sample_years": {}, "sample_leads": {},
    }

    for init_month in drift_cfg.init_months:
        missing_labels = [
            label for label in exp_labels
            if label not in case_arrays or init_month not in case_arrays[label]
        ]
        if missing_labels:
            raise KeyError(
                f"Missing CaseArray data for init_month={init_month}: "
                f"{missing_labels}."
            )

        common_years = sorted(set.intersection(*(
            set(case_arrays[label][init_month].init_years) for label in exp_labels
        )))
        common_leads = sorted(set.intersection(*(
            set(case_arrays[label][init_month].leads) for label in exp_labels
        )))
        if not common_years:
            raise ValueError(
                f"Experiments have no common initialization years for "
                f"init_month={init_month}."
            )
        if not common_leads:
            raise ValueError(
                f"Experiments have no common leads for init_month={init_month}."
            )
        if drift_cfg.baseline_lead not in common_leads:
            raise ValueError(
                f"baseline_lead={drift_cfg.baseline_lead} is not in the common "
                f"leads {common_leads} for init_month={init_month}."
            )
        if len(exp_labels) > 1 and len(common_years) < 2:
            raise ValueError(
                f"Comparative diagnostics require at least two common initialization "
                f"years for init_month={init_month}; found {common_years}."
            )

        results["sample_years"][init_month] = common_years
        results["sample_leads"][init_month] = common_leads
        aligned_cases = {
            label: CaseArray(
                case_arrays[label][init_month].data.sel(
                    Y=common_years, L=common_leads
                ),
                name=label,
            )
            for label in exp_labels
        }

        obs_lookups = {
            label: build_obs_lookup(
                obs_series,
                common_years,
                init_month,
                common_leads,
            )
            for label in exp_labels
        }

        for label in exp_labels:
            ca = aligned_cases[label]
            results["bias"].setdefault(label, {})[init_month] = absolute_bias(
                ca, obs_lookups[label], init_month
            )
            results["adjustment"].setdefault(label, {})[init_month] = adjustment(
                results["bias"][label][init_month], drift_cfg.baseline_lead
            )
            results["skill"].setdefault(label, {})[init_month] = compute_skill(
                ca, obs_lookups[label], init_month
            )

        if len(exp_labels) == 2:
            ref_lbl, test_lbl = resolve_experiment_roles(exp_labels)
            results["paired_diff"][init_month] = paired_difference(
                results["adjustment"][test_lbl][init_month],
                results["adjustment"][ref_lbl][init_month],
            )
            results["paired_diff_by_start"][init_month] = paired_difference_by_start(
                aligned_cases[test_lbl], aligned_cases[ref_lbl],
                drift_cfg.baseline_lead,
            )
            try:
                lo, hi = bootstrap_paired_ci(
                    aligned_cases[test_lbl],
                    aligned_cases[ref_lbl],
                    obs_lookups[test_lbl],
                    obs_lookups[ref_lbl],
                    init_month,
                    n_boot=drift_cfg.n_bootstrap,
                    seed=drift_cfg.bootstrap_seed,
                    baseline_lead=drift_cfg.baseline_lead,
                )
                results["ci_lower"][init_month] = lo
                results["ci_upper"][init_month] = hi
            except ValueError as exc:
                warnings.warn(
                    f"bootstrap_paired_ci skipped for init_month={init_month}: {exc}"
                )

        for label in exp_labels:
            window_table = window_summary(
                    bias=results["bias"][label][init_month],
                    adj=results["adjustment"][label][init_month],
                    skill=results["skill"][label][init_month],
                    window_defs=drift_cfg.window_defs,
                    baseline_lead=drift_cfg.baseline_lead,
                    experiment=label,
                    init_month=init_month,
                )
            window_table["n_years"] = len(common_years)
            window_table["sample_years"] = ",".join(
                str(value) for value in common_years
            )
            results["window_tables"].append(window_table)

    if write_outputs and outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        pd.concat(results["window_tables"], ignore_index=True).to_csv(
            outdir / f"{prefix}_window_summary.csv", index=False
        )

        if results["paired_diff"]:
            rows = []
            for init_month, da in results["paired_diff"].items():
                for L, val in zip(da.L.values, da.values):
                    rows.append({
                        "init_month":  int(init_month),
                        "lead":        int(L),
                        "paired_diff": float(val),
                        "ci_lower": (
                            float(results["ci_lower"][init_month].sel(L=L))
                            if init_month in results["ci_lower"] else float("nan")
                        ),
                        "ci_upper": (
                            float(results["ci_upper"][init_month].sel(L=L))
                            if init_month in results["ci_upper"] else float("nan")
                        ),
                    })
            pd.DataFrame(rows).to_csv(
                outdir / f"{prefix}_paired_diff.csv", index=False
            )
            start_rows = []
            for init_month, da in results["paired_diff_by_start"].items():
                for year in da.Y.values:
                    for lead in da.L.values:
                        start_rows.append({
                            "init_month": int(init_month),
                            "init_year": int(year),
                            "lead": int(lead),
                            "paired_diff": float(da.sel(Y=year, L=lead)),
                            "sign_convention": PAIRED_SIGN_CONVENTION,
                            "baseline_lead": int(drift_cfg.baseline_lead),
                        })
            pd.DataFrame(start_rows).to_csv(
                outdir / f"{prefix}_paired_diff_by_start.csv", index=False
            )
            product_rows = [
                {
                    "init_month": row["init_month"],
                    "component": "atm", "variable": var_spec.native_field,
                    "units": var_spec.plot_units, "native_grid_id": "regional_mean",
                    "region": drift_cfg.region_name, "lead_units": "month",
                    "lead_start": row["lead"], "lead_end": row["lead"],
                    "baseline": (
                        f"lead {drift_cfg.baseline_lead} (1-based valid month)"
                    ),
                    "metric_name": "paired_adjustment", "value": row["paired_diff"],
                    "uncertainty_lower": row["ci_lower"],
                    "uncertainty_upper": row["ci_upper"],
                    "reference_product": var_spec.obs_product,
                    "sample_count": len(results["sample_years"][row["init_month"]]),
                    "significance_method": "paired initialization-year bootstrap",
                }
                for row in rows
            ]
            product_rows.extend({
                "start_date": f"{row['init_year']:04d}-{row['init_month']:02d}-01",
                "init_year": row["init_year"], "init_month": row["init_month"],
                "component": "atm", "variable": var_spec.native_field,
                "units": var_spec.plot_units, "native_grid_id": "regional_mean",
                "region": drift_cfg.region_name, "lead_units": "month",
                "lead_start": row["lead"], "lead_end": row["lead"],
                "baseline": f"lead {drift_cfg.baseline_lead} (1-based valid month)",
                "metric_name": "paired_adjustment_by_start",
                "value": row["paired_diff"], "reference_product": var_spec.obs_product,
                "sample_count": 1,
            } for row in start_rows)
            fingerprint = config_fingerprint(
                drift_cfg, variable=var_spec.native_field,
                context={"experiments": list(experiment_specs)},
            )
            product_table = standardize_product_table(
                product_rows, workflow="5a_regional_drift",
                configuration_hash=fingerprint,
            )
            write_product_bundle(
                outdir / f"{prefix}_products", product_table,
                workflow="5a_regional_drift", configuration_hash=fingerprint,
                metadata={
                    "region_bounds": drift_cfg.region,
                    "baseline_lead": drift_cfg.baseline_lead,
                    "lead_numbering": "lead 1 is the initialization calendar month",
                },
            )

        write_manifest(
            outdir=outdir, prefix=prefix,
            experiments=experiment_specs,
            var_spec=var_spec, drift_cfg=drift_cfg,
            sample_years=results["sample_years"],
            sample_leads=results["sample_leads"],
        )

    return results


# ===========================================================================
# Section 13 — Legacy compatibility (used by s2d.py)
# ===========================================================================

def remove_model_drift(stats, da, time, climy0, climy1):
    """Thin wrapper around stats.remove_drift kept for backward compatibility.

    Used internally by S2DDiagnostics.  Not part of the new drift pipeline.
    """
    return stats.remove_drift(da, time, climy0, climy1)
