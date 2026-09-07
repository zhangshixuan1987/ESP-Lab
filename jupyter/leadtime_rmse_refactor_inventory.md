# Refactor review inventory

Reviewed 2026-09-07 at commit `df3999e`.

This file was absent from the checkout and has been recreated for the requested
`3_refactor*` review. It does not reconstruct earlier RMSE inventory entries or
certify the `2*` workflows. The only matching notebook is
[`3_refactor_sst_skill_ts.ipynb`](3_refactor_sst_skill_ts.ipynb).

**Status: review and targeted validation complete; scientific sign-off pending
the findings below.** This review updates documentation only; findings remain
unfixed in the implementation.

## Workflow inventory

Cell numbers below are zero-based notebook cell indices.

| Stage | Implementation | Contract/status |
| --- | --- | --- |
| Configuration | Cells 1, 5 | AtlMDR default; May/November; three E3SM cases; CESM-SMYLE; optional NMME; 1981–2010 climatology; requested 1981–2011 initialization cohort |
| E3SM/SMYLE/HadISST2 inputs | Cell 6; `scripts/run_process_sst_index.py` | Auto/require/rebuild modes; compact regional monthly and seasonal files |
| NMME inputs | Cell 7; `scripts/run_process_nmme_sst_index.py` | Regional drift-corrected products; cache provenance is incomplete (finding 2) |
| E3SM skill | Cells 8–14; `esp_lab/psl_skill.py`; `esp_lab/stats.py` | Lead-dependent drift removal, ACC, normalized RMSE, optional PSL reference sensitivity; cohort defects (findings 1 and 3) |
| SMYLE benchmark | Cells 16–17 | Monthly/seasonal skill cache with input digest and settings metadata; incomplete required-variable validation (finding 4) |
| NMME benchmark | Cell 18 | Fixed model intersection across requested months/leads; equal weighting of model ensemble means; skill recomputed and written on every enabled run |
| Figures | Cells 19–22 | Skill curves and DJF anomaly time series; seasonal cohort labels are incorrect under finding 1 |
| Resources | Cells 3, 23 | Notebook-owned Dask setup and cleanup through shared resource helpers |

## Findings

### 1. P1 — November seasonal selection shifts the initialization cohort by one year

[`subset_hindcast_initialization_years`](../esp_lab/psl_skill.py#L184) derives
initialization year from the first retained verification timestamp. For monthly
November forecasts that timestamp is November of the initialization year; for
seasonal forecasts the first timestamp is January of the following year at
`L=3`. Calls in cells 13, 16, and 18 therefore interpret seasonal verification
years as initialization years.

Reproduced with the existing AtlMDR caches for FOSIRL, 4DEnVarOcn, and SMYLE:

| Requested cohort | Monthly selected `Y` endpoints | Seasonal selected `Y` endpoints |
| --- | --- | --- |
| 1981–2011 | `1981110100`, `2011110100` | `1980110100`, `2010110100` |

For FOSIRL November `L=3`, using the existing seasonal input and HadISST2,
1981–2010 drift climatology, and detrending:

| Selection | ACC | nRMSE | Target years | Samples |
| --- | --- | --- | --- | --- |
| Current helper | 0.66708868 | 0.75985461 | 1981–2011 | 31 |
| Explicit initialization `Y` selection | 0.68110937 | 0.74764032 | 1982–2012 | 31 |

The sample-count check cannot detect this shift because both selections have
31 samples. Select from explicit initialization metadata or derive initialization
time using the lead offset, preserving the original verification timestamps.
Add a November seasonal regression and invalidate affected saved skill products
when fixed. The existing helper test starts at monthly `L=1`, so it misses this
case.

### 2. P2 — NMME cache reuse ignores the requested model set and mask settings

Cell 7's `_nmme_timeseries_file_is_current` checks region, climatology, nonmissing
data, and the seasonal lead count, but not requested models, model-set selection,
mask settings/version, or complete monthly coverage. Model selection is resolved
only inside the stale-file processing branch. Thus changing `models`,
`model_set`, or `sst_land_mask` can leave existing files classified as current,
and cell 18 uses the models present in those files.

Resolve the requested model contract before deciding reuse and validate saved
provenance against it. Preserve sufficient provenance in the combined time-series
writer. Add a test that changes the model/mask request while keeping the same
output filename. A finite monthly fixture with an unrelated model, false mask
metadata, and matching region/climatology is accepted by the notebook validator.

### 3. P2 — E3SM silently shortens a supposedly common verification cohort

Cell 13 clamps the requested endpoints independently to each case's available
dates before calling the strict cohort helper (`case_skill_y0/y1` and
`case_seas_skill_y0/y1`). For example, changing the configured end year to 2018
leaves 4DEnVarOcn on its shorter archive while longer cases use the full request;
figure captions still report the configured common period. This is separate
from the seasonal offset in finding 1.

Validate the exact requested initialization cohort for every case, or compute
one explicit intersection shared by all systems and label that actual cohort.
Test a shorter case against a longer requested period. The default monthly
1981–2011 request fits the inspected archives.

### 4. P2 — Incomplete SMYLE skill files can pass validation and then fail loading

Cell 16's `required_vars` includes correlation and cohort metadata, but omits
monthly/seasonal `pval` and `rmse`, which cells 16–17 access unconditionally.
It also does not require the reference-specific metrics used for optional PSL
curves. A cache retaining the checked variables and provenance but missing
`smyle_seas_skill_pval` passes the gate and raises `KeyError` at reconstruction.

Validate every consumed metric and required reference before reuse; test a
saved cache with an omitted metric and verify that it triggers recomputation.

## Validation evidence

- All 21 code cells parse after removing IPython magic lines. Notebook JSON is
  readable; no saved error outputs are present. This is not an execution test.
- **37 tests passed** across `test_psl_skill.py`, `test_stats_common_samples.py`,
  `test_sst_utils.py`, `test_nmme_sst_index.py`,
  `test_run_process_sst_index.py`, and `test_notebook_resources.py`.
- Read all 16 AtlMDR monthly/seasonal files for May/November across FOSIRL,
  Reanalysis, 4DEnVarOcn, and SMYLE under
  `/global/cfs/cdirs/e3sm/S2S2D/s2d_diag`. Each has nonmissing values and 24 monthly
  leads or seven retained seasonal leads. FOSIRL/Reanalysis/SMYLE have 39 rows;
  4DEnVarOcn has 32. This check does not assert every member/sample is finite.
- Recomputed the FOSIRL first November DJF skill against real HadISST2 inputs
  to quantify finding 1; source diagnostic files were not modified.
- Reviewed the upstream processors and the legacy tutorial
  `docs/source/tutorials/Regional-Average_SST-Index_Skill.ipynb`. The tutorial
  uses a different climatology (1972–2018) and configuration, so numerical
  equivalence to that workflow has **not** been established.

Test command, from the repository root:

```bash
/global/common/software/e3sm/anaconda_envs/e3smu_1_13_0/pm-cpu/pixi_login/.pixi/envs/default/bin/python \
  -m pytest -o addopts='' -q \
  tests/test_psl_skill.py tests/test_stats_common_samples.py \
  tests/test_sst_utils.py tests/test_nmme_sst_index.py \
  tests/test_run_process_sst_index.py tests/test_notebook_resources.py
```

The shell's default Python 3.8 lacks scientific dependencies and pytest-cov;
validation used the installed E3SM Unified login Python 3.13 environment.
`addopts` was cleared to avoid repository-wide coverage output. The passing run
reported dependency/runtime warnings, including a NumPy binary-size warning.

## Remaining acceptance checks

- Fix findings 1–4 and add behavioral regression coverage for their triggers.
- Regenerate/invalidate skill caches affected by the cohort correction.
- Execute the full notebook with controlled output directories, including
  cached/recomputed SMYLE paths, NMME enabled/disabled, and a PSL-supported region.
- Check saved figures visually and compare numerical outputs to an agreed
  baseline using identical input files, climatology, and initialization cohorts.

Full notebook execution, raw-archive preprocessing, all-region validation, and
figure rendering were not performed in this review.
