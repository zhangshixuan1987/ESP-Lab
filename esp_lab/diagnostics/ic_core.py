"""
ic_core.py
==========
Initial-condition (IC) analysis for S2D hindcast campaigns — pure core module.

Design principles
-----------------
* Sections 1–6 contain NO I/O and NO ESP-Lab dependency.
  They can be imported and unit-tested without access to NERSC or real data.
* Section 7 is the only bridge to xarray / numpy computation.
  It converts environment-specific structures into plain dicts / DataFrames.
* No default experiments or paths are provided here.
  Always define ICConfig explicitly in your workflow script or notebook.

Sections
--------
1. Configuration dataclasses   (ICConfig, ComponentSpec, ExperimentPair)
2. File-audit types            (FileRecord, AuditResult, compare_file_records)
3. NetCDF schema comparison    (StructureDiff, compare_nc_schema)
4. Variable-level statistics   (ic_variable_stats, classify_variable)
5. Campaign aggregation        (aggregate_campaign_stats)
6. Physical consistency        (ConsistencyReport, check_atm_surface_vs_land,
                                check_ocean_ice_consistency)
7. Provenance helpers          (write_manifest_json)

Usage
-----
See ``workflows/diagnostics/initial_conditions/`` for workflow orchestration and
``tests/test_ic_core.py`` for offline smoke tests.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# ===========================================================================
# Section 1 — Configuration dataclasses
# ===========================================================================

# Canonical component names used throughout the workflow
COMPONENT_NAMES = ("atm", "lnd", "ocn", "ice", "rof", "cpl")

# Default variable priority categories per component.
# Scripts discover actual variable names from the files; this map is used to
# classify them afterwards.  Keys are component names; values are
# (priority_keywords, metadata_keywords) tuples of lowercase substring lists.
_DEFAULT_PRIORITY_KEYWORDS: Dict[str, List[str]] = {
    # Use longer / more specific tokens for atmosphere to avoid false
    # positive substring matches on short letters like 'u', 'v', 'q'.
    "atm":  ["_t_", "_u_", "_v_", "_q_", "tbot", "trefht", "ts_",
              "temp", "wind", "humid", "surf_pres", "psurf", "ps_",
              "ubot", "vbot", "qbot", "tsurf"],
    "lnd":  ["tsoi", "h2osoi", "h2ocan", "snow", "frac_sno", "t_grnd",
              "wa", "zwt", "qcharge"],
    "ocn":  ["temperature", "salinity", "ssh", "layerthickness", "velocity",
              "normalvelocity", "zmid", "activetracer"],
    "ice":  ["aice", "vice", "vsno", "tice", "uvel", "vvel", "conc", "thick"],
    "rof":  ["volr", "rof", "dvolrdt", "qavg", "storage", "flood"],
    "cpl":  ["flux", "state", "import", "export", "accum", "x2", "a2x", "o2x"],
}

_DEFAULT_METADATA_KEYWORDS: List[str] = [
    "time", "date", "dtime", "nstep", "kdt", "nday", "counter",
    "restart", "history", "ndens", "nyr", "nmo", "nhr",
]


@dataclass
class ComponentSpec:
    """Specification for a single coupled component's restart files.

    Parameters
    ----------
    name:
        Short component name: ``"atm"``, ``"lnd"``, ``"ocn"``, ``"ice"``,
        ``"rof"``, or ``"cpl"``.
    file_glob:
        Glob pattern relative to the start-date directory, e.g.
        ``"*.eam.i.*.nc"`` or ``"*.mpaso.rst.*.nc"``.
    per_member:
        If True, one file exists per ensemble member; hash every member.
        If False, a single (non-atmospheric) restart file is expected and
        one hash per start date is sufficient.
    member_pattern:
        Regex to extract member tag from filename, e.g. ``r"EN\\d{2}"``.
        Only used when ``per_member=True``.
    priority_keywords:
        Lowercase substrings used to classify a variable as *physical state*.
        Defaults to ``_DEFAULT_PRIORITY_KEYWORDS[name]``.
    """

    name: str
    file_glob: str
    per_member: bool = False
    member_pattern: str = r"EN\d{2}"
    priority_keywords: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.name not in COMPONENT_NAMES:
            raise ValueError(
                f"ComponentSpec.name={self.name!r} is not one of {COMPONENT_NAMES}."
            )
        if not self.file_glob:
            raise ValueError("ComponentSpec.file_glob must not be empty.")
        if not self.priority_keywords:
            self.priority_keywords = list(
                _DEFAULT_PRIORITY_KEYWORDS.get(self.name, [])
            )


@dataclass
class ExperimentPair:
    """A reference/test pair of experiment labels.

    Parameters
    ----------
    ref_label:
        Short label for the reference experiment (e.g. ``"BruteForce"``).
    test_label:
        Short label for the test experiment (e.g. ``"JRA55-FOSIRL"``).
    ref_root:
        Root directory under which ``{YYYY-MM-DD-00000}/`` directories live
        for the reference experiment.  Set as a string so this dataclass
        has no filesystem dependency; convert to Path in ic_io.
    test_root:
        Equivalent for the test experiment.
    """

    ref_label: str
    test_label: str
    ref_root: str
    test_root: str

    def __post_init__(self) -> None:
        for attr in ("ref_label", "test_label", "ref_root", "test_root"):
            if not getattr(self, attr):
                raise ValueError(f"ExperimentPair.{attr} must not be empty.")


@dataclass
class ICConfig:
    """Global configuration for the IC analysis workflow.

    Parameters
    ----------
    experiment_pair:
        Defines the two experiments and their filesystem roots.
    start_dates:
        List of start-date strings in ``"YYYY-MM-DD-00000"`` format.
    seasons:
        List of season labels (``"May"`` and/or ``"November"``); used for
        stratified campaign aggregation.
    members:
        Ensemble member tags that share the same atmospheric IC (e.g.
        ``["EN00", ..., "EN09"]``).
    components:
        List of ``ComponentSpec`` objects (one per coupled component).
    pilot_date:
        The single date to process when ``pilot_only=True``.
        Defaults to ``"1980-05-01-00000"``.
    pilot_only:
        If ``True``, only process ``pilot_date``.
    output_root:
        Root directory for all output sub-directories.  Relative paths are
        resolved against the calling script's working directory.
    atm_hash_all_members:
        If ``True``, compute SHA-256 for every atmospheric EN?? member file
        (needed for the control check).  Default ``True``.
    non_atm_hash_per_member:
        If ``True``, also hash every member for non-atmospheric components.
        Default ``False`` (one hash per start date is sufficient for the
        first-pass audit).
    """

    experiment_pair: ExperimentPair
    start_dates: List[str]
    seasons: List[str]
    members: List[str]
    components: List[ComponentSpec]

    pilot_date: str = "1980-05-01-00000"
    pilot_only: bool = True
    output_root: str = "output"

    atm_hash_all_members: bool = True
    non_atm_hash_per_member: bool = False

    _DATE_RE: str = field(
        default=r"^\d{4}-\d{2}-\d{2}-\d{5}$", init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.start_dates:
            raise ValueError("ICConfig.start_dates must not be empty.")
        if not self.members:
            raise ValueError("ICConfig.members must not be empty.")
        if not self.components:
            raise ValueError("ICConfig.components must not be empty.")
        date_re = re.compile(self._DATE_RE)
        for d in self.start_dates:
            if not date_re.match(d):
                raise ValueError(
                    f"ICConfig: start date {d!r} does not match "
                    "expected format YYYY-MM-DD-00000."
                )
        if self.pilot_date and not date_re.match(self.pilot_date):
            raise ValueError(
                f"ICConfig.pilot_date={self.pilot_date!r} does not match "
                "expected format YYYY-MM-DD-00000."
            )

    @property
    def active_dates(self) -> List[str]:
        """Return only the pilot date when ``pilot_only=True``."""
        if self.pilot_only:
            return [self.pilot_date]
        return list(self.start_dates)

    def date_season(self, date_str: str) -> str:
        """Return ``'May'`` or ``'November'`` based on the date string."""
        month = int(date_str.split("-")[1])
        if month == 5:
            return "May"
        if month == 11:
            return "November"
        return f"Month{month:02d}"


# Default component specs — can be overridden in config.yaml
DEFAULT_COMPONENTS: List[ComponentSpec] = [
    ComponentSpec(
        name="atm",
        file_glob="*.eam.i.*.nc",
        per_member=True,
        member_pattern=r"EN\d{2}",
    ),
    ComponentSpec(
        name="lnd",
        file_glob="*.elm.r.*.nc",
        per_member=False,
    ),
    ComponentSpec(
        name="ocn",
        file_glob="*.mpaso.rst.*.nc",
        per_member=False,
    ),
    ComponentSpec(
        name="ice",
        file_glob="*.mpassi.rst.*.nc",
        per_member=False,
    ),
    ComponentSpec(
        name="rof",
        file_glob="*.mosart.r.*.nc",
        per_member=False,
    ),
    ComponentSpec(
        name="cpl",
        file_glob="*.cpl.r.*.nc",
        per_member=False,
    ),
]


# ===========================================================================
# Section 2 — File-audit types
# ===========================================================================


@dataclass
class FileRecord:
    """Metadata record for a single restart file or rpointer.

    Parameters
    ----------
    path:
        Absolute path as a string (not Path, for JSON serialisability).
    component:
        Component name (``"atm"``, ``"lnd"``, etc.) or ``"rpointer"``.
    experiment:
        Experiment label (e.g. ``"BruteForce"``).
    start_date:
        Start-date string in ``"YYYY-MM-DD-00000"`` format.
    member:
        Ensemble member tag (e.g. ``"EN00"``) or ``""`` for non-member files.
    size_bytes:
        File size in bytes; ``-1`` if the file does not exist.
    sha256:
        Hex SHA-256 digest; ``""`` if not computed or file missing.
    nc_dims:
        Dict mapping dimension names to sizes; ``{}`` for non-NetCDF files.
    sim_timestamp:
        Simulation timestamp extracted from the file's ``time`` variable or
        global attribute; ``""`` if not available.
    global_attrs:
        Dict of global attributes from the NetCDF file; ``{}`` otherwise.
    pointer_target:
        Target filename from an rpointer file; ``""`` otherwise.
    """

    path: str
    component: str
    experiment: str
    start_date: str
    member: str = ""
    size_bytes: int = -1
    sha256: str = ""
    nc_dims: Dict[str, int] = field(default_factory=dict)
    sim_timestamp: str = ""
    global_attrs: Dict = field(default_factory=dict)
    pointer_target: str = ""

    def to_dict(self) -> dict:
        """Return a flat dict suitable for a pandas DataFrame row."""
        return {
            "path": self.path,
            "component": self.component,
            "experiment": self.experiment,
            "start_date": self.start_date,
            "member": self.member,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "nc_dims": json.dumps(self.nc_dims),
            "sim_timestamp": self.sim_timestamp,
            "pointer_target": self.pointer_target,
        }


class AuditResult(enum.Enum):
    """Outcome of a file-level comparison between two ``FileRecord`` objects."""

    IDENTICAL = "IDENTICAL"
    DIFFERENT = "DIFFERENT"
    MISSING = "MISSING"
    INCOMPATIBLE_STRUCTURE = "INCOMPATIBLE_STRUCTURE"


def compare_file_records(ref: FileRecord, test: FileRecord) -> AuditResult:
    """Compare two ``FileRecord`` objects and return an ``AuditResult``.

    Rules
    -----
    * If either record has ``size_bytes == -1`` → ``MISSING``
    * If dimension sets are both non-empty but incompatible → ``INCOMPATIBLE_STRUCTURE``
    * If both SHA-256 digests are non-empty and identical → ``IDENTICAL``
    * If both SHA-256 digests are non-empty and different → ``DIFFERENT``
    * If either digest is empty (not computed) → ``DIFFERENT`` (conservative)

    Parameters
    ----------
    ref:
        Reference experiment file record.
    test:
        Test experiment file record.

    Returns
    -------
    AuditResult
    """
    if ref.size_bytes == -1 or test.size_bytes == -1:
        return AuditResult.MISSING

    # Check structural compatibility when both have dimension info
    if ref.nc_dims and test.nc_dims:
        ref_keys = set(ref.nc_dims.keys())
        test_keys = set(test.nc_dims.keys())
        # If dimension names differ significantly, flag as incompatible
        shared = ref_keys & test_keys
        if not shared and (ref_keys or test_keys):
            return AuditResult.INCOMPATIBLE_STRUCTURE
        # Check that shared dimension sizes match
        for dim in shared:
            if ref.nc_dims[dim] != test.nc_dims[dim]:
                return AuditResult.INCOMPATIBLE_STRUCTURE

    if ref.sha256 and test.sha256:
        if ref.sha256 == test.sha256:
            return AuditResult.IDENTICAL
        return AuditResult.DIFFERENT

    # No checksums available — conservative default
    return AuditResult.DIFFERENT


# ===========================================================================
# Section 3 — NetCDF schema comparison
# ===========================================================================


@dataclass
class StructureDiff:
    """Result of a schema-level comparison between two xr.Datasets.

    Attributes
    ----------
    common_vars:
        Variables present in both datasets.
    only_in_ref:
        Variables present only in the reference dataset.
    only_in_test:
        Variables present only in the test dataset.
    dim_mismatches:
        Dict of dimension name → (ref_size, test_size) for mismatched dims.
    dtype_mismatches:
        Dict of variable name → (ref_dtype, test_dtype) for type mismatches.
    coord_mismatches:
        Dict of coordinate name → description of mismatch.
    """

    common_vars: List[str] = field(default_factory=list)
    only_in_ref: List[str] = field(default_factory=list)
    only_in_test: List[str] = field(default_factory=list)
    dim_mismatches: Dict[str, Tuple] = field(default_factory=dict)
    dtype_mismatches: Dict[str, Tuple] = field(default_factory=dict)
    coord_mismatches: Dict[str, str] = field(default_factory=dict)
    variable_attr_mismatches: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_common": len(self.common_vars),
            "n_only_ref": len(self.only_in_ref),
            "n_only_test": len(self.only_in_test),
            "n_dim_mismatch": len(self.dim_mismatches),
            "n_dtype_mismatch": len(self.dtype_mismatches),
            "n_coord_mismatch": len(self.coord_mismatches),
            "n_variable_attr_mismatch": len(self.variable_attr_mismatches),
            "common_vars": ",".join(sorted(self.common_vars)),
            "only_ref": ",".join(sorted(self.only_in_ref)),
            "only_test": ",".join(sorted(self.only_in_test)),
            "dim_mismatches": json.dumps(self.dim_mismatches, sort_keys=True),
            "dtype_mismatches": json.dumps(self.dtype_mismatches, sort_keys=True),
            "coord_mismatches": json.dumps(self.coord_mismatches, sort_keys=True),
            "variable_attr_mismatches": json.dumps(
                self.variable_attr_mismatches, sort_keys=True
            ),
        }


def compare_nc_schema(
    ds_ref: xr.Dataset,
    ds_test: xr.Dataset,
) -> StructureDiff:
    """Compare the schema (variables, dimensions, dtypes) of two datasets.

    This function reads only metadata — it does *not* load data values into
    memory.  Use it to separate structural differences from value differences.

    Parameters
    ----------
    ds_ref:
        Reference dataset (e.g. BruteForce restart).
    ds_test:
        Test dataset (e.g. JRA55-FOSIRL restart).

    Returns
    -------
    StructureDiff
    """
    ref_vars = set(ds_ref.data_vars)
    test_vars = set(ds_test.data_vars)

    common = sorted(ref_vars & test_vars)
    only_ref = sorted(ref_vars - test_vars)
    only_test = sorted(test_vars - ref_vars)

    # Dimension size mismatches
    ref_dims = dict(ds_ref.sizes)
    test_dims = dict(ds_test.sizes)
    all_dims = set(ref_dims) | set(test_dims)
    dim_mismatches: Dict[str, Tuple] = {}
    for d in all_dims:
        r = ref_dims.get(d, None)
        t = test_dims.get(d, None)
        if r != t:
            dim_mismatches[d] = (r, t)

    # Dtype mismatches for common variables
    dtype_mismatches: Dict[str, Tuple] = {}
    for v in common:
        r_dt = str(ds_ref[v].dtype)
        t_dt = str(ds_test[v].dtype)
        if r_dt != t_dt:
            dtype_mismatches[v] = (r_dt, t_dt)

    # Coordinate identity includes values and ordering, not only shape.  This
    # prevents cell-by-cell subtraction on a permuted native grid.
    ref_coords = set(ds_ref.coords)
    test_coords = set(ds_test.coords)
    coord_mismatches: Dict[str, str] = {}
    for c in ref_coords & test_coords:
        if ds_ref.coords[c].shape != ds_test.coords[c].shape:
            coord_mismatches[c] = (
                f"shape ref={ds_ref.coords[c].shape} "
                f"test={ds_test.coords[c].shape}"
            )
        elif not ds_ref.coords[c].equals(ds_test.coords[c]):
            coord_mismatches[c] = "values or ordering differ"

    # MPAS grid descriptors are commonly stored as data variables rather than
    # coordinates.  Equality here is a direct mesh/cell-ordering check.
    grid_descriptors = {
        "latCell", "lonCell", "xCell", "yCell", "zCell", "areaCell",
        "cellsOnCell", "verticesOnCell", "edgesOnCell", "indexToCellID",
    }
    for name in sorted(grid_descriptors & set(common)):
        if not ds_ref[name].equals(ds_test[name]):
            coord_mismatches[f"data_var:{name}"] = "values or ordering differ"

    attrs_to_compare = ("units", "standard_name", "long_name", "positive", "axis")
    variable_attr_mismatches: Dict[str, str] = {}
    for v in common:
        mismatched = {
            attr: (ds_ref[v].attrs.get(attr), ds_test[v].attrs.get(attr))
            for attr in attrs_to_compare
            if ds_ref[v].attrs.get(attr) != ds_test[v].attrs.get(attr)
        }
        if mismatched:
            variable_attr_mismatches[v] = json.dumps(mismatched, default=str)

    return StructureDiff(
        common_vars=common,
        only_in_ref=only_ref,
        only_in_test=only_test,
        dim_mismatches=dim_mismatches,
        dtype_mismatches=dtype_mismatches,
        coord_mismatches=coord_mismatches,
        variable_attr_mismatches=variable_attr_mismatches,
    )


# ===========================================================================
# Section 4 — Variable-level statistics
# ===========================================================================

# Metadata / bookkeeping variable keywords (lowercase)
_METADATA_KEYWORDS: Tuple[str, ...] = tuple(_DEFAULT_METADATA_KEYWORDS)


def classify_variable(
    var_name: str,
    component: str,
    priority_keywords: Optional[List[str]] = None,
) -> str:
    """Classify a variable as ``'physical'``, ``'metadata'``, or ``'unknown'``.

    Parameters
    ----------
    var_name:
        Variable name as it appears in the NetCDF file.
    component:
        Component name (``"atm"``, ``"lnd"``, etc.).
    priority_keywords:
        Override the default priority keyword list for this component.

    Returns
    -------
    str
        One of ``"physical"``, ``"metadata"``, or ``"unknown"``.
    """
    lower = var_name.lower()

    # Check metadata first (more specific)
    if any(kw in lower for kw in _METADATA_KEYWORDS):
        return "metadata"

    keys = priority_keywords or _DEFAULT_PRIORITY_KEYWORDS.get(component, [])
    if any(kw in lower for kw in keys):
        return "physical"

    return "unknown"


def ic_variable_stats(
    da_ref: xr.DataArray,
    da_test: xr.DataArray,
    area_weights: Optional[xr.DataArray] = None,
    meaningful_threshold: Optional[float] = None,
    eps: float = 1e-10,
) -> dict:
    """Compute IC difference statistics between two DataArrays.

    Both arrays must share the same shape.  Missing values are handled via
    xarray's ``skipna=True`` convention.

    Parameters
    ----------
    da_ref:
        Reference IC field (BruteForce).
    da_test:
        Test IC field (JRA55-FOSIRL).
    area_weights:
        Optional spatial area weights (same spatial dimensions as the arrays).
        If provided, RMSE, mean diff, and MAD are area-weighted.
        For MPAS components, pass the native cell-area array.
    meaningful_threshold:
        Absolute-difference threshold in the variable's native units.  The
        exceedance fraction uses this value; when omitted, ``eps`` is used.
    eps:
        Small value used to guard against division by zero in relative RMSE.

    Returns
    -------
    dict with keys:
        min_diff, max_diff, max_abs_diff, p05_diff, p50_diff, p95_diff,
        mean_diff, mad, rmse, reference_std, nrmse, pattern_corr,
        meaningful_threshold, frac_exceeding_threshold, integral_ref,
        integral_test, integral_diff, integral_pct_diff, n_valid
    """
    diff = da_test - da_ref
    n_valid = int(diff.notnull().sum().values)

    if n_valid == 0:
        return _empty_stats()

    if area_weights is not None:
        # Broadcast weights to match diff's spatial dims
        try:
            w = area_weights.broadcast_like(diff)
            w = w.where(diff.notnull(), other=0.0)
            w_sum = float(w.sum())
            if w_sum <= 0:
                area_weights = None
        except Exception:
            area_weights = None

    def _wavg(x: xr.DataArray) -> float:
        if area_weights is not None:
            w = area_weights.broadcast_like(x)
            w = w.where(x.notnull(), other=0.0)
            w_sum = float(w.sum())
            if w_sum <= 0:
                return float(x.mean(skipna=True).values)
            return float((x * w).sum(skipna=True).values) / w_sum
        return float(x.mean(skipna=True).values)

    mean_diff = _wavg(diff)
    mad = _wavg(np.abs(diff))
    rmse = float(np.sqrt(_wavg(diff ** 2)))

    spatial_std = float(diff.std(skipna=True).values)

    # Reference scale for NRMSE: weighted spatial standard deviation.
    ref_mean = _wavg(da_ref)
    ref_std = float(np.sqrt(max(_wavg((da_ref - ref_mean) ** 2), 0.0)))
    nrmse = rmse / ref_std if ref_std > eps else float("nan")

    # Pattern correlation (Pearson, area-weighted if available)
    try:
        ref_anom = da_ref - _wavg(da_ref)
        test_anom = da_test - _wavg(da_test)
        num = _wavg(ref_anom * test_anom)
        denom = float(
            np.sqrt(abs(_wavg(ref_anom ** 2)) * abs(_wavg(test_anom ** 2)))
        )
        pattern_corr = num / (denom + eps)
    except Exception:
        pattern_corr = float("nan")

    threshold = eps if meaningful_threshold is None else float(meaningful_threshold)
    if threshold < 0:
        raise ValueError("meaningful_threshold must be non-negative")
    frac_exceeding = float(
        (np.abs(diff) > threshold).sum(skipna=True).values
    ) / max(n_valid, 1)

    # A physical area integral when native cell areas are available; otherwise
    # retain the explicitly unweighted discrete sum for backward compatibility.
    if area_weights is not None:
        ref_w = area_weights.broadcast_like(da_ref).where(da_ref.notnull(), other=0.0)
        test_w = area_weights.broadcast_like(da_test).where(da_test.notnull(), other=0.0)
        integral_diff_ref = float((da_ref * ref_w).sum(skipna=True).values)
        integral_diff_test = float((da_test * test_w).sum(skipna=True).values)
    else:
        integral_diff_ref = float(da_ref.sum(skipna=True).values)
        integral_diff_test = float(da_test.sum(skipna=True).values)

    integral_diff = integral_diff_test - integral_diff_ref
    integral_pct_diff = (
        100.0 * integral_diff / integral_diff_ref
        if abs(integral_diff_ref) > eps else float("nan")
    )
    quantiles = diff.quantile([0.05, 0.50, 0.95], skipna=True).values

    return {
        "min_diff": float(diff.min(skipna=True).values),
        "max_diff": float(diff.max(skipna=True).values),
        "max_abs_diff": float(np.abs(diff).max(skipna=True).values),
        "p05_diff": float(quantiles[0]),
        "p50_diff": float(quantiles[1]),
        "p95_diff": float(quantiles[2]),
        "mean_diff": mean_diff,
        "mad": mad,
        "rmse": rmse,
        "spatial_std": spatial_std,
        "reference_std": ref_std,
        "nrmse": nrmse,
        "rel_rmse": nrmse,
        "pattern_corr": pattern_corr,
        "meaningful_threshold": threshold,
        "frac_exceeding_threshold": frac_exceeding,
        "frac_differing": frac_exceeding,
        "integral_ref": integral_diff_ref,
        "integral_test": integral_diff_test,
        "integral_diff": integral_diff,
        "integral_pct_diff": integral_pct_diff,
        "n_valid": n_valid,
    }


def _empty_stats() -> dict:
    """Return an all-NaN stats dict when no valid data exists."""
    return {
        k: float("nan")
        for k in (
            "min_diff", "max_diff", "max_abs_diff", "p05_diff", "p50_diff",
            "p95_diff", "mean_diff", "mad", "rmse", "spatial_std",
            "reference_std", "nrmse", "rel_rmse", "pattern_corr",
            "meaningful_threshold", "frac_exceeding_threshold",
            "frac_differing", "integral_ref", "integral_test",
            "integral_diff", "integral_pct_diff",
        )
    } | {"n_valid": 0}


# ===========================================================================
# Section 5 — Campaign aggregation
# ===========================================================================


def aggregate_campaign_stats(
    records: List[dict],
    group_by: str = "season",
) -> pd.DataFrame:
    """Aggregate per-start-date variable statistics across a full campaign.

    Parameters
    ----------
    records:
        List of dicts, each containing at minimum:
            ``start_date``, ``component``, ``variable``, ``season``,
            ``rmse``, ``mean_diff``, ``mad``, ``pattern_corr``,
            ``frac_differing``.
    group_by:
        One of ``"season"``, ``"component"``, or ``"variable"`` (or a comma-
        separated combination).  Controls the grouping used for summary stats.

    Returns
    -------
    pd.DataFrame with aggregated statistics:
        mean_rmse, std_rmse, min_rmse, max_rmse,
        mean_abs_mean_diff, mean_pattern_corr,
        mean_frac_differing, n_starts,
        sign_agreement_frac  (fraction of starts with mean_diff > 0)
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    required = {"start_date", "component", "variable", "rmse", "mean_diff"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"aggregate_campaign_stats: records missing columns {missing_cols}"
        )

    # Derive season column if not present
    if "season" not in df.columns:
        df["season"] = df["start_date"].apply(_date_to_season)

    # Build group keys
    valid_groups = {"season", "component", "variable"}
    requested = {g.strip() for g in group_by.split(",")}
    invalid = requested - valid_groups
    if invalid:
        raise ValueError(
            f"aggregate_campaign_stats: invalid group_by keys {invalid}; "
            f"choose from {valid_groups}."
        )
    group_keys = sorted(requested)

    agg = (
        df.groupby(group_keys)
        .apply(_agg_group, include_groups=False)
        .reset_index()
    )
    return agg


def _date_to_season(date_str: str) -> str:
    """Return 'May' or 'November' from a YYYY-MM-DD-00000 string."""
    try:
        month = int(date_str.split("-")[1])
        return {5: "May", 11: "November"}.get(month, f"Month{month:02d}")
    except Exception:
        return "Unknown"


def _agg_group(g: pd.DataFrame) -> pd.Series:
    """Aggregate one group of per-start-date stats."""
    rmse_vals = g["rmse"].dropna()
    nrmse_vals = g["nrmse"].dropna() if "nrmse" in g else pd.Series(dtype=float)
    md_vals = g["mean_diff"].dropna()
    pc_vals = g["pattern_corr"].dropna() if "pattern_corr" in g else pd.Series(dtype=float)
    fd_vals = g["frac_differing"].dropna() if "frac_differing" in g else pd.Series(dtype=float)

    return pd.Series(
        {
            "n_starts": len(g),
            "mean_rmse": rmse_vals.mean() if len(rmse_vals) else float("nan"),
            "median_rmse": rmse_vals.median() if len(rmse_vals) else float("nan"),
            "std_rmse": rmse_vals.std() if len(rmse_vals) > 1 else float("nan"),
            "p25_rmse": rmse_vals.quantile(0.25) if len(rmse_vals) else float("nan"),
            "p75_rmse": rmse_vals.quantile(0.75) if len(rmse_vals) else float("nan"),
            "min_rmse": rmse_vals.min() if len(rmse_vals) else float("nan"),
            "max_rmse": rmse_vals.max() if len(rmse_vals) else float("nan"),
            "median_nrmse": nrmse_vals.median() if len(nrmse_vals) else float("nan"),
            "max_start_rmse_fraction": (
                rmse_vals.max() / rmse_vals.sum()
                if len(rmse_vals) and rmse_vals.sum() > 0 else float("nan")
            ),
            "mean_abs_mean_diff": md_vals.abs().mean() if len(md_vals) else float("nan"),
            "mean_pattern_corr": pc_vals.mean() if len(pc_vals) else float("nan"),
            "mean_frac_differing": fd_vals.mean() if len(fd_vals) else float("nan"),
            "sign_agreement_frac": (
                float((md_vals > 0).sum()) / len(md_vals)
                if len(md_vals) else float("nan")
            ),
            "sign_consistency": (
                max(float((md_vals > 0).sum()), float((md_vals < 0).sum()))
                / len(md_vals) if len(md_vals) else float("nan")
            ),
        }
    )


# ===========================================================================
# Section 6 — Physical consistency checks
# ===========================================================================


@dataclass
class ConsistencyReport:
    """Result of a cross-component physical consistency check.

    Attributes
    ----------
    check_name:
        Human-readable name for the check (e.g. ``"atm_vs_land_surface"``).
    n_points:
        Number of spatial points examined.
    mean_abs_mismatch:
        Area-weighted mean absolute mismatch in the overlapping quantity.
    max_abs_mismatch:
        Spatial maximum absolute mismatch.
    frac_inconsistent:
        Fraction of points exceeding ``threshold``.
    threshold:
        Threshold used to define *inconsistent* (same units as the mismatch).
    notes:
        Free-text notes about the check.
    warnings:
        List of warning strings raised during the check.
    """

    check_name: str
    n_points: int = 0
    mean_abs_mismatch: float = float("nan")
    max_abs_mismatch: float = float("nan")
    frac_inconsistent: float = float("nan")
    threshold: float = 1.0
    notes: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "n_points": self.n_points,
            "mean_abs_mismatch": self.mean_abs_mismatch,
            "max_abs_mismatch": self.max_abs_mismatch,
            "frac_inconsistent": self.frac_inconsistent,
            "threshold": self.threshold,
            "notes": self.notes,
            "n_warnings": len(self.warnings),
        }


def check_atm_surface_vs_land(
    da_atm_ts: xr.DataArray,
    da_lnd_tgrnd: xr.DataArray,
    threshold_k: float = 2.0,
    area_weights: Optional[xr.DataArray] = None,
) -> ConsistencyReport:
    """Check whether atmospheric surface temperature and land surface temperature
    are mutually consistent (within ``threshold_k`` kelvin).

    Both arrays should be on compatible spatial grids; if they differ, the
    function will attempt to align on common coordinates.  No regridding is
    performed — this is a schema-level sanity check.

    Parameters
    ----------
    da_atm_ts:
        Atmospheric surface temperature DataArray (K).
    da_lnd_tgrnd:
        Land surface temperature DataArray (K).
    threshold_k:
        Mismatch threshold in kelvin.
    area_weights:
        Optional spatial weights.

    Returns
    -------
    ConsistencyReport
    """
    report = ConsistencyReport(
        check_name="atm_surface_vs_land_surface",
        threshold=threshold_k,
    )

    try:
        spatial_names = {"ncol", "nCells", "gridcell", "lndgrid", "lat", "lon", "x", "y"}
        shared_dims = set(da_atm_ts.dims) & set(da_lnd_tgrnd.dims) & spatial_names
        if not shared_dims:
            report.notes = "Native grids differ; model mapping weights are required."
            report.warnings.append(
                "No shared spatial dimension — cross-component comparison skipped."
            )
            return report
        # Align on shared coordinates (inner join → only overlap)
        atm_al, lnd_al = xr.align(da_atm_ts, da_lnd_tgrnd, join="inner")
        mismatch = np.abs(atm_al - lnd_al)
        n_pts = int(mismatch.notnull().sum().values)
        report.n_points = n_pts

        if n_pts == 0:
            report.notes = "No overlapping grid points found after alignment."
            report.warnings.append("No overlapping points — cannot check consistency.")
            return report

        def _wavg(x):
            if area_weights is not None:
                try:
                    w = area_weights.broadcast_like(x)
                    w = w.where(x.notnull(), other=0.0)
                    w_sum = float(w.sum())
                    if w_sum > 0:
                        return float((x * w).sum(skipna=True).values) / w_sum
                except Exception:
                    pass
            return float(x.mean(skipna=True).values)

        report.mean_abs_mismatch = _wavg(mismatch)
        report.max_abs_mismatch = float(mismatch.max(skipna=True).values)
        report.frac_inconsistent = float(
            (mismatch > threshold_k).sum(skipna=True).values
        ) / max(n_pts, 1)

        if report.frac_inconsistent > 0.1:
            report.warnings.append(
                f"More than 10% of land points show |atm_TS - lnd_T_grnd| > "
                f"{threshold_k} K (frac={report.frac_inconsistent:.3f}).  "
                "This may indicate an initial atmosphere–land imbalance."
            )

    except Exception as exc:
        report.warnings.append(f"check_atm_surface_vs_land failed: {exc}")

    return report


def check_ocean_ice_consistency(
    da_sst: xr.DataArray,
    da_aice: xr.DataArray,
    freezing_point_k: float = 271.35,
    threshold_frac: float = 0.01,
    area_weights: Optional[xr.DataArray] = None,
) -> ConsistencyReport:
    """Check consistency between ocean SST and sea-ice concentration.

    Points where SST < freezing_point_k but aice < threshold_frac (no ice
    at a below-freezing SST) are flagged as inconsistent.

    Parameters
    ----------
    da_sst:
        Sea-surface temperature DataArray (K).
    da_aice:
        Sea-ice concentration DataArray (fraction 0–1).
    freezing_point_k:
        Seawater freezing point in kelvin.
    threshold_frac:
        Minimum ice concentration to consider a cell *ice-covered*.
    area_weights:
        Optional spatial weights.

    Returns
    -------
    ConsistencyReport
    """
    report = ConsistencyReport(
        check_name="ocean_sst_vs_sea_ice_concentration",
        threshold=threshold_frac,
        notes=f"freezing_point_k={freezing_point_k}",
    )

    try:
        spatial_names = {"ncol", "nCells", "gridcell", "lat", "lon", "x", "y"}
        shared_dims = set(da_sst.dims) & set(da_aice.dims) & spatial_names
        if not shared_dims:
            report.notes += " | Native grids differ; model mapping weights are required."
            report.warnings.append(
                "No shared spatial dimension — ocean/ice comparison skipped."
            )
            return report
        sst_al, aice_al = xr.align(da_sst, da_aice, join="inner")
        n_pts = int(sst_al.notnull().sum().values)
        report.n_points = n_pts

        if n_pts == 0:
            report.notes += " | No overlapping grid points."
            report.warnings.append("No overlapping points — cannot check ocean/ice consistency.")
            return report

        # Inconsistent: below freezing but no ice
        below_freezing = sst_al < freezing_point_k
        no_ice = aice_al < threshold_frac
        inconsistent = (below_freezing & no_ice).where(sst_al.notnull())

        n_inconsistent = int(inconsistent.sum(skipna=True).values)
        report.frac_inconsistent = n_inconsistent / max(n_pts, 1)
        report.mean_abs_mismatch = float((sst_al - freezing_point_k).where(
            inconsistent == 1
        ).mean(skipna=True).values)
        report.max_abs_mismatch = float((freezing_point_k - sst_al).where(
            inconsistent == 1
        ).max(skipna=True).values)

        if report.frac_inconsistent > 0.01:
            report.warnings.append(
                f"SST–ice inconsistency: {n_inconsistent} cells below "
                f"{freezing_point_k} K have sea-ice fraction < {threshold_frac} "
                f"(frac={report.frac_inconsistent:.4f})."
            )

    except Exception as exc:
        report.warnings.append(f"check_ocean_ice_consistency failed: {exc}")

    return report


# ===========================================================================
# Section 7 — Provenance helpers
# ===========================================================================


def write_manifest_json(
    records: List[dict],
    outpath: str | Path,
    extra_meta: Optional[dict] = None,
) -> None:
    """Write an IC analysis manifest as a JSON file.

    Parameters
    ----------
    records:
        List of flat dicts (e.g. from ``FileRecord.to_dict()``).
    outpath:
        Destination JSON file path.
    extra_meta:
        Optional dict of additional top-level metadata keys.
    """
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "records": records,
    }
    if extra_meta:
        manifest.update(extra_meta)

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)


__all__ = [
    # Section 1 — Configuration
    "COMPONENT_NAMES",
    "DEFAULT_COMPONENTS",
    "ComponentSpec",
    "ExperimentPair",
    "ICConfig",
    # Section 2 — File audit
    "FileRecord",
    "AuditResult",
    "compare_file_records",
    # Section 3 — Schema comparison
    "StructureDiff",
    "compare_nc_schema",
    # Section 4 — Variable statistics
    "classify_variable",
    "ic_variable_stats",
    # Section 5 — Campaign aggregation
    "aggregate_campaign_stats",
    # Section 6 — Physical consistency
    "ConsistencyReport",
    "check_atm_surface_vs_land",
    "check_ocean_ice_consistency",
    # Section 7 — Provenance
    "write_manifest_json",
]
