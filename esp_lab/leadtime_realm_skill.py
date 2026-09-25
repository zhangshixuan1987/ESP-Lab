"""Shared lead-time ACC skill-map helpers for the atmosphere and ocean realms.

``jupyter/1a_atm_leadtime_acc_skill_map.ipynb`` and
``jupyter/1a_ocn_leadtime_acc_skill_map.ipynb`` are, apart from their
``VAR_CONFIG``/``E3SM_CASES`` data and a handful of labels, the same workflow:
both build provenance-tracked, drift-removed E3SM and CESM-SMYLE anomaly
bundles from :func:`esp_lab.stats.remove_drift`, then compute and cache ACC
skill with :func:`esp_lab.leadtime_workflow.compute_skill_lead_range`. Before
this module existed, that orchestration was duplicated verbatim in both
notebooks (and duplicated again between the "E3SM case" and "CESM-SMYLE
benchmark" loops within each notebook) -- exactly the kind of duplication
that risks a silently wrong drift-removal formula or skill computation in one
copy but not the others.

This module factors out only the two steps where that risk concentrates:

* :func:`prepare_drift_removed_anomaly` -- the "load a compatible cached
  anomaly bundle, or remove drift and build/write/reopen one" step used for
  both the E3SM hindcast and the CESM-SMYLE benchmark.
* :func:`compute_and_cache_skill` -- the "load a compatible cached skill
  Dataset, or compute it over an inclusive seasonal lead range and cache it"
  step used for every case/month (and the CESM-SMYLE full-record and
  E3SM-overlap variants).
* :func:`common_finite_ocean_mask` -- the small "common all-finite domain
  across several unmasked prepared bundles" helper used by ocean-only fields
  (e.g. SST) to build one shared ocean mask before scoring.

These mirror the ``land_input_cache.prepare_model_cache`` /
``land_skill.compute_land_acc_skill`` split used by the already-refactored
land notebooks, adapted to the atm/ocn workflow's existing
``leadtime_prepared_cache`` / ``leadtime_skill_cache`` / ``leadtime_workflow``
cache-contract infrastructure (which the land modules do not use, since land
has its own simpler two-source-kind -- reference vs. model -- cache
contract).

Scope note
----------
This module intentionally does **not** yet cover the rest of the 1a
notebooks (raw E3SM/CESM-SMYLE loading and regridding, observation
preparation, prepared-input source-identity/cache-spec planning, the finite-
ensemble CESM-SMYLE-vs-E3SM significance comparison, or any of the
multi-panel plotting cells). Those remain inline and duplicated between the
atm and ocn notebooks. See the migration notes in this repository's refactor
history for why: the remaining cells are each tens of thousands of
characters of bespoke multi-panel plotting layout or large, mostly
self-contained statistical-resampling blocks, and porting them faithfully
needs the same careful, dedicated attention given here to the
drift-removal and skill-computation steps, which is why they were left for a
follow-up pass instead of being rushed.

A follow-up pass extended the module to also cover the 1b (RMSE skill map)
atm/ocn notebooks' drift-removal step (:func:`prepare_drift_removed_anomaly`
is directly reusable there -- verified line-by-line against both notebooks'
``prepared_cache_spec``/``e3sm_seasonal_cache_spec`` helpers, which build
``expected_attrs`` dicts with the exact same
``ensemble_member_count``/``lead_count``/``unit_conversion_version`` shape
1a uses) and skill-computation step via the new
:func:`compute_and_cache_rmse_skill`. That new function is deliberately
**not** a thin wrapper around :func:`compute_and_cache_skill`: 1b calls
``stats.compute_skill_seasonal`` directly with ``lead_start``/``lead_end``
passed straight through as its ``nleadavg``/``nleads`` positional
parameters (no prior :func:`esp_lab.leadtime_workflow.select_lead_range`
slicing step), which is a different calling convention than 1a's
:func:`esp_lab.leadtime_workflow.compute_skill_lead_range` wrapper (which
slices to the lead range first, then always passes ``nleadavg=1`` and
``nleads=<post-slice lead count>``). In the notebooks' current
configuration (``lead_start=1``, ``lead_end=case_nlead // 3``) these two
calling conventions happen to produce the same result, but they are not
equivalent in general, so :func:`compute_and_cache_rmse_skill` preserves
1b's exact original call shape verbatim rather than reusing
:func:`compute_and_cache_skill`. It also preserves 1b's narrower
``required_variables=('rmse', 'sig_obs')`` cache-compatibility check (1a's
:func:`compute_and_cache_skill` defaults to requiring all
:data:`ACC_SKILL_REQUIRED_VARIABLES`) and its explicit
``xr.Dataset({...})`` reconstruction from the named ``skill.<var>`` fields
returned by ``compute_skill_seasonal`` (rather than 1a's
``skill.assign_attrs(...)`` on the whole returned Dataset), since
``compute_skill_seasonal`` returns extra fields (e.g. sample-count/target-
year bookkeeping) that 1b's cache files have never included.

The 1c (RMSE compare) atm/ocn notebooks were read in full, cell by cell, and
found to need no equivalent extraction: they compute a fundamentally
different metric (direct, non-anomaly RMSE against raw values, explicitly
*not* the drift-removed/``compute_skill_seasonal`` skill pipeline -- see
1c's own "The original full skill-metric calculation is intentionally
skipped in this direct-RMSE notebook" comment) and that computation already
lives entirely in the shared ``workflows.leadtime_skill.rmse_comparison``
helper module (imported as ``rmse_compare_helper``), not duplicated inline.
Diffing 1c's atm and ocn notebooks cell-by-cell confirms this: every code
cell touching drift/RMSE computation, caching, or helper wiring (e.g. the
"Compute and save direct RMSE" and "Helper functions for direct RMSE"
cells) is byte-identical between the two notebooks already; only the title
cell, one heading's field-list text, and the ``VAR_CONFIG``/period-setup
cell (genuinely realm-specific field metadata) differ. There is no
atm/ocn-duplicated formula left in 1c for this module to absorb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import xarray as xr

from . import stats
from .leadtime_prepared_cache import (
    build_prepared_skill_dataset,
    cache_status,
    open_prepared_skill_dataset,
    prepared_skill_cache_status,
    write_prepared_skill_dataset,
)
from .leadtime_workflow import compute_skill_lead_range
from .utils.netcdf_utils import atomic_to_netcdf, load_netcdf


ACC_SKILL_REQUIRED_VARIABLES = (
    "corr",
    "pval",
    "rmse",
    "msss",
    "rpc",
    "sig_obs",
    "sig_sig",
    "sig_tot",
    "s2t",
)


def prepare_drift_removed_anomaly(
    da_fn,
    valid_time_fn,
    *,
    prepared_path: str | Path,
    expected_attrs: Mapping,
    source: str,
    component: str,
    field: str,
    init_month: int,
    climatology_years: tuple[int, int],
    requested_years: Sequence[int],
    prepared_chunks: Mapping[str, int],
    reuse: bool = True,
    force_recompute: bool = False,
) -> xr.Dataset:
    """Load or build one provenance-tracked, drift-removed anomaly bundle.

    ``da_fn`` and ``valid_time_fn`` are zero-argument callables (typically a
    ``lambda`` closing over a case/month dict lookup) rather than plain
    values. This matters: on a compatible cache hit (``reuse`` and not
    ``force_recompute``), the prepared bundle is opened directly and neither
    callable is ever invoked. The pre-refactor inline code relied on exactly
    this laziness -- the raw regridded input for a case/month is only loaded
    upstream when that case/month's prepared-skill cache is stale (see each
    notebook's ``e3sm_months_to_prepare``/``smyle_months_to_prepare``
    filtering), so a case/month with a valid cache is never present in
    ``e3sm_da_by_case_month``/``smyle_da_by_month`` at all. Passing
    ``e3sm_da_by_case_month[case_key][init_month]`` as a plain positional
    argument (as an earlier version of this function did) evaluates that
    dict lookup eagerly at the call site, before this function's cache check
    runs, raising a ``KeyError`` on every fully-cached case/month -- even
    though the value would never have been used. Callables restore the
    original lazy evaluation.

    On a miss, :func:`esp_lab.stats.remove_drift` is applied over
    ``climatology_years`` and the result is atomically written then reopened
    from disk, so downstream cells operate on a small graph instead of the
    full upstream lazy-processing chain.

    This is the exact "load-or-build" step used, identically, for both the
    E3SM hindcast anomaly and the CESM-SMYLE benchmark anomaly in the atm and
    ocn lead-time ACC notebooks; calling it from a per-case/month loop in the
    notebook (as :func:`esp_lab.land_input_cache.prepare_model_cache` is
    called from land's per-case/month loop) keeps the orchestration visible
    while sharing the drift-removal and cache-write/read logic itself.
    """
    climy0, climy1 = map(int, climatology_years)
    cache_compatible, _ = prepared_skill_cache_status(
        prepared_path, expected_attrs=expected_attrs
    )
    if reuse and cache_compatible and not force_recompute:
        print(f"Loading prepared input: {prepared_path}")
        return open_prepared_skill_dataset(
            prepared_path, expected_attrs=expected_attrs, chunks=prepared_chunks
        )

    print(f"Creating prepared input: {prepared_path}")
    da = da_fn()
    valid_time = valid_time_fn()
    anomaly, climatology = stats.remove_drift(da, valid_time, climy0, climy1)
    prepared = build_prepared_skill_dataset(
        anomaly,
        climatology,
        valid_time,
        source=source,
        component=component,
        variable=field,
        init_month=init_month,
        climatology_years=(climy0, climy1),
        source_data_identity=expected_attrs["source_data_identity"],
        case_prefix=expected_attrs["case_prefix"],
        requested_years=requested_years,
        target_grid=expected_attrs["target_grid"],
        regridding_method=expected_attrs["regridding_method"],
        ensemble_member_count=expected_attrs["ensemble_member_count"],
        lead_count=expected_attrs["lead_count"],
        unit_conversion_version=expected_attrs["unit_conversion_version"],
    )
    write_prepared_skill_dataset(prepared, prepared_path)
    return open_prepared_skill_dataset(
        prepared_path, expected_attrs=expected_attrs, chunks=prepared_chunks
    )


def compute_and_cache_skill(
    model_anom: xr.DataArray,
    model_time: xr.DataArray,
    observations: xr.DataArray,
    clim_start,
    clim_end,
    lead_start: int,
    lead_end: int,
    *,
    outfile: str | Path,
    expected_attrs: Mapping,
    netcdf_write_options: Mapping,
    detrend: bool,
    force_compute: bool = False,
    required_variables: Sequence[str] = ACC_SKILL_REQUIRED_VARIABLES,
) -> xr.Dataset:
    """Load a compatible cached skill Dataset, or compute and cache one.

    Wraps :func:`esp_lab.leadtime_workflow.compute_skill_lead_range` (an
    inclusive, one-based seasonal lead range, no multi-year averaging) with
    the same "check compatibility, reuse or recompute, write atomically"
    contract used identically for the E3SM per-case/month skill, the
    CESM-SMYLE full-record skill, and the CESM-SMYLE E3SM-overlap skill in
    the atm and ocn lead-time ACC notebooks.
    """
    outfile = Path(outfile)
    compatible, reason = cache_status(
        outfile, expected_attrs=expected_attrs, required_variables=required_variables
    )
    if compatible and not force_compute:
        return load_netcdf(outfile)

    if outfile.exists() and not force_compute:
        print(f"Ignoring incompatible skill cache {outfile}: {reason}")

    skill = compute_skill_lead_range(
        model_anom,
        model_time,
        observations,
        clim_start,
        clim_end,
        lead_start,
        lead_end,
        resamp=0,
        detrend=detrend,
    )
    skill_ds = skill.assign_attrs(expected_attrs).compute()
    atomic_to_netcdf(skill_ds, outfile, **dict(netcdf_write_options))
    return skill_ds


def compute_and_cache_rmse_skill(
    model_anom: xr.DataArray,
    model_time: xr.DataArray,
    observations: xr.DataArray,
    clim_start,
    clim_end,
    lead_start: int,
    lead_end: int,
    *,
    outfile: str | Path,
    expected_attrs: Mapping,
    netcdf_write_options: Mapping,
    detrend: bool,
    force_compute: bool = False,
    required_variables: Sequence[str] = ("rmse", "sig_obs"),
) -> xr.Dataset:
    """Load a compatible cached RMSE skill Dataset, or compute and cache one.

    This is the 1b (RMSE skill map) atm/ocn notebooks' "check compatibility,
    reuse or recompute, write atomically" step for the E3SM per-case/month
    skill, the CESM-SMYLE full-record skill, and the CESM-SMYLE overlap
    skill. Unlike :func:`compute_and_cache_skill` (1a's ACC equivalent),
    this calls ``stats.compute_skill_seasonal`` directly with ``lead_start``
    and ``lead_end`` passed straight through as its ``nleadavg``/``nleads``
    positional parameters -- exactly matching 1b's original inline call --
    rather than routing through
    :func:`esp_lab.leadtime_workflow.compute_skill_lead_range`'s
    slice-then-``nleadavg=1`` convention. See the module docstring's "Scope
    note" for why these two calling conventions are not interchangeable in
    general even though they agree for 1b's current configuration.

    It also matches 1b's narrower default ``required_variables`` cache
    check and its explicit reconstruction of the cached Dataset from only
    the named ``corr``/``pval``/``rmse``/``msss``/``rpc``/``sig_obs``/
    ``sig_sig``/``sig_tot``/``s2t`` fields (dropping any other fields
    ``compute_skill_seasonal`` returns), and 1b's unconditional "Recomputing
    ..." print message on a cache miss (1a only prints when the stale file
    already exists and a recompute was not explicitly forced).
    """
    outfile = Path(outfile)
    compatible, reason = cache_status(
        outfile, expected_attrs=expected_attrs, required_variables=required_variables
    )
    if compatible and not force_compute:
        return load_netcdf(outfile)

    print(f"Recomputing {outfile}: {reason}")
    skill = stats.compute_skill_seasonal(
        model_anom,
        model_time,
        observations,
        clim_start,
        clim_end,
        lead_start,
        lead_end,
        resamp=0,
        detrend=detrend,
    )
    skill_ds = xr.Dataset(
        {
            "corr": skill.corr,
            "pval": skill.pval,
            "rmse": skill.rmse,
            "msss": skill.msss,
            "rpc": skill.rpc,
            "sig_obs": skill.sig_obs,
            "sig_sig": skill.sig_sig,
            "sig_tot": skill.sig_tot,
            "s2t": skill.s2t,
        }
    ).compute()
    skill_ds.attrs.update(expected_attrs)
    atomic_to_netcdf(skill_ds, outfile, **dict(netcdf_write_options))
    return skill_ds


def common_finite_ocean_mask(samples: Sequence[xr.DataArray]) -> xr.DataArray:
    """Return the common all-finite domain across several unmasked samples.

    Each entry of ``samples`` should be one representative (e.g. a single
    ``Y``/``M``/``L`` slice) finite/non-finite mask sharing identical
    coordinates; the result keeps only cells finite in every sample. Used by
    ocean-only fields (e.g. SST) to build one common ocean mask across every
    configured E3SM case and the CESM-SMYLE benchmark before applying it to
    every prepared anomaly bundle and the observations, so every downstream
    skill product shares an identical spatial domain.
    """
    if not samples:
        raise ValueError("samples must contain at least one mask")
    aligned = xr.align(*samples, join="exact")
    return xr.concat(aligned, dim="mask_source").all("mask_source").compute()


__all__ = [
    "ACC_SKILL_REQUIRED_VARIABLES",
    "common_finite_ocean_mask",
    "compute_and_cache_rmse_skill",
    "compute_and_cache_skill",
    "prepare_drift_removed_anomaly",
]
