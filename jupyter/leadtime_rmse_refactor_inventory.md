# Refactor review inventory

Reviewed 2026-09-07 at commit `df3999e`.

Rechecked 2026-09-07 at `570c42c`: the working tree was clean and the
`3_refactor_sst_skill_ts.ipynb` notebook, supporting `psl_skill.py`/`stats.py`,
both SST processors, and relevant cohort/processor tests were unchanged from
`df3999e`. All four findings below remain open. The previous validation results
still describe this implementation; tests were not rerun for this recheck.

This file was absent from the checkout and has been recreated for the requested
`3_refactor*` review. It does not reconstruct earlier RMSE inventory entries or
certify the `2*` workflows. The only matching notebook is
[`3_refactor_sst_skill_ts.ipynb`](3_refactor_sst_skill_ts.ipynb).

**Current status: findings 1–6 addressed in the working tree.** The review
sections below retain the original failure evidence; they describe the code
before this refinement.

## Refinement and current validation (2026-09-07)

- Initialization years are derived from verification time minus the one-based
  monthly lead offset. November seasonal L=3 now selects the requested
  initialization years. E3SM no longer clamps the cohort per case.
- Skill cache version is 5, invalidating old SMYLE results affected by the
  seasonal selection bug. Required cached metrics include monthly/seasonal
  correlation, p-value, RMSE, and reference metrics/coordinates.
- NMME model selection is resolved before cache validation. Combined products
  store and validate requested models, loaded model coordinates, mask choice,
  and mask version as well as region/climatology. Older products without this
  provenance are intentionally stale. For cache-only operation without the raw
  directory, configure explicit models or the named Yeager set rather than
  discovery of all models.
- SST processor write gates reject missing/nonfinite regional index variables,
  matching the notebook's invalid-content rejection and allowing auto repair.
- SMYLE regional processing ensures monthly and seasonal gridded TS benchmarks,
  checks their requested years/members/leads/timestamps, generates missing or
  invalid frequencies, and validates generated products. Its raw archive and
  benchmark directories are configurable in cell 6 and through processor CLI
  options. Existing valid benchmarks are reused.

Validation: 52 distinct tests passed across the original six modules,
`test_cesm_smyle_benchmark_notebook.py`, and the new
`test_sst_notebook_orchestration.py`. Coverage includes November seasonal year
selection, missing benchmark frequency generation/reuse, nonfinite cache
rejection, NMME provenance checks and writer round-trip, missing SMYLE skill
metrics, and auto/require/rebuild dispatch. Processor execution is mocked in
orchestration tests; these are not raw-archive integration runs.

Real-data downstream smoke runs executed the current notebook's loading,
drift removal, skill computation, and plotting cells for AtlMDR with NMME
both disabled and enabled. The enabled run retained six NMME models. Figures
and NMME skill output were redirected to `/tmp`; SMYLE skill saving was
disabled. The runs recomputed SMYLE skill, bypassed upstream ensure cells and
Dask-cluster startup, and used an Agg figure writer instead of publication
metadata. Both runs completed. The DJF figure was visually inspected. No
production diagnostics were regenerated or overwritten.

Remaining limits: full raw-archive processing, production publication, and
all-region/PSL integration have not been exercised. Legacy numerical parity
still requires an agreed configuration and baseline.

### Notebook organization

The optional shared and NMME SST-index drivers now live under
`jupyter/preprocessing/sst/`, with a README describing their scope and their
relationship to `3_refactor_sst_skill_ts.ipynb`. Their repository-root discovery
searches all parents, so launching them from the nested directory still resolves
the active checkout. The required CESM-SMYLE benchmark driver remains at
`jupyter/0_run_cesm_smyle_benchmark.ipynb` because `1a_refactor*` consumes its
pre-generated atmospheric benchmark products. Analysis notebooks and production
processing scripts retain their existing locations.

### Modes-of-variability standalone refactor (2026-09-07)

`4_refactor_mov_analysis.ipynb` now ensures the selected mode through
`workflows/modes_of_variability/orchestration.py`. The `auto`, `require`, and
`rebuild` policy covers the observation reference, CESM-SMYLE, both configured
E3SM cases, and optional NMME products. It also prepares the required gridded
CESM-SMYLE PSL or TS benchmark for the selected mode before invoking the mode
processor.

Global teleconnection maps were the last notebook-only preprocessing step.
They now live in `workflows/modes_of_variability/teleconnections.py`, use
atomic NetCDF replacement, and record input size/mtime fingerprints plus the
regression settings. The ensure step rejects missing, unreadable, all-NaN,
undersampled, or stale maps. The MOV analysis therefore no longer depends on a
saved execution of either legacy `0_run*emov` notebook.

The optional batch drivers moved to `jupyter/preprocessing/mov/`. Their README
describes when to use them, and both resolve the repository root from the nested
directory. The older notebook-side teleconnection pass is disabled by default
so it cannot overwrite products managed by the reusable workflow.

Validation: 41 focused tests passed across MOV orchestration, teleconnection
generation and stale-input detection, processor cache contracts, and map
utilities. Processor calls are mocked in orchestration tests; the
teleconnection tests write and reopen synthetic reference NetCDF products.
Existing production NAM base products were also validated in cache-only mode
before teleconnection fingerprinting was added. Production teleconnection
files predate the new provenance signature and will be regenerated on the next
`auto` run or reported as incompatible by `require`.

## Original workflow inventory

### Standalone orchestration follow-up (2026-09-07)

**The refactor removes the manual SST-index notebook step, but is not yet robust
from missing upstream products.** Direct processor calls replace running
the former top-level `0_run_sigmod_sst_index.ipynb` and
`0_run_nmme_sst_index.ipynb`. They did not
ensure every transitive preprocessing dependency. Findings 5–6 below are new
and specifically concern this standalone behavior; findings 1–4 concern skill
correctness and cache integrity separately.

Six targeted fault-injection checks passed using temporary files and the actual
cell-6 orchestration/function source: reuse in `auto`, missing-input dispatch in
`auto`, failure without processing in `require`, forced dispatch in `rebuild`,
the inconsistent all-NaN validators, and failure on missing SMYLE benchmark
inputs. The four mode checks mock processor execution; the other two use the
real processor validator/SMYLE entry point. Passing the latter two checks
**confirms the defects**, not successful recovery. These checks were run with
the E3SM Unified login Python and `/tmp/test_sst_standalone_review.py`.
No production diagnostics were written or rebuilt.

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

## Original findings (addressed above)

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

### 5. P1 — Missing SMYLE benchmark products still require a separate upstream step

Cell 6 calls `process_smyle` to create missing regional indices. That processor
[`loads gridded benchmark files`](../scripts/run_process_sst_index.py#L650)
from the fixed `CESM_SMYLE_DIAG_DIR` using `smyle_access.load_benchmark` for
both seasonal and monthly frequencies. The loader raises `FileNotFoundError`
when these files are missing and instructs the user to run
`scripts/run_process_cesm_smyle_benchmark.py`, the processor driven by
the former top-level `0_run_cesm_smyle_benchmark.ipynb`. At the time of the
finding, neither cell 6 nor `process_smyle` invoked
that missing stage.

Reproduced by redirecting the processor's benchmark root to an empty temporary
directory and calling the real `process_smyle` entry point. It fails at the
first seasonal benchmark load. The notebook can generate regional indices only
when these gridded benchmark products already exist. Moreover, `smyle_outdir`
redirects regional output, not the fixed benchmark input root.

Ensure the required gridded TS benchmarks before reducing regional indices, or
explicitly document and preflight this prerequisite before processing the other
systems. Test missing monthly and seasonal benchmark inputs independently.

### 6. P1 — Auto mode cannot repair some files it classifies as stale

Cell 6's `_sst_index_file_is_current` rejects a missing SST variable or an
all-NaN SST variable. Its auto-processing calls pass `force=False`. However,
[`_output_is_current`](../scripts/run_process_sst_index.py#L236), used at the
processor's write gates, checks only version/mask metadata for ordinary regional
indices (with a special nonmissing check for RONI). A correctly versioned
all-NaN AtlMDR file therefore triggers processing in the notebook but is
considered reusable by the processor. After expensive raw-data computation,
the write is skipped and the notebook's postprocessing check fails again.

Reproduced using a temporary HadISST2 AtlMDR file containing only NaN SST and
current version/mask attributes: the notebook validator returns false while
the actual processor validator returns true. This also affects E3SM and SMYLE
write gates that use the same validator.

Use a shared validity contract in orchestration and processor write decisions,
or explicitly force replacement of the rejected outputs. Test that auto mode
replaces an invalid file and then passes postvalidation. Manual `rebuild` is a
workaround, but unnecessarily recomputes the requested groups.

## Validation evidence (initial review)

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

## Remaining acceptance checks after refinement

- Run the refined workflow against production paths when ready; version 5 invalidates
  old SMYLE skill caches and NMME provenance checks invalidate unverified inputs.
- Execute the full notebook with controlled output directories, including
  cached/recomputed SMYLE paths, NMME enabled/disabled, and a PSL-supported region.
- Check saved figures visually and compare numerical outputs to an agreed
  baseline using identical input files, climatology, and initialization cohorts.

Full notebook execution from raw archives and all-region validation remain
unperformed. Downstream figure rendering was exercised during refinement as
described above.
