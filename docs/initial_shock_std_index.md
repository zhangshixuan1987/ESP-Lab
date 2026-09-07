# Initial-shock variability index

`esp_lab.diagnostics.initial_shock` implements the active calculation in
`temp/Compute_Std_Index_Initial_Shock_share.ncl`. The independent workflow is
`workflows.diagnostics.initial_shock`; it does not use or overwrite ACC caches.

## Archive-backed driver notebook

Open [`jupyter/6a_initial_shock_std_index.ipynb`](../jupyter/6a_initial_shock_std_index.ipynb).
It follows the `1a_refactor` workflow: centralized field/case settings, full source
inventory, optional Dask cluster, E3SM monthly archive loading, CESM-SMYLE monthly
benchmarks, observation loading and common-grid regridding, compact cached metrics,
and separate May/November comparison figures. No hand-prepared file paths are
needed. The archive bridge is `workflows.diagnostics.initial_shock_archive`.

The notebook starts with TREFHT, 1980–2011, May/November initializations and the
three E3SM cases plus CESM-SMYLE. It explicitly uses the exploratory 24-month,
12-month-block variant for the current forecast length. `smoke_mode=True` selects
two years and one E3SM case plus CESM-SMYLE. `cache.mode` supports auto, rebuild,
and require. Missing members or requested starts stop the run rather than silently
changing the cohort. Only small global-index/metric products are written.

The input-file workflow below remains available when normalized monthly files
already exist; its explicit input contract also describes what the archive bridge
prepares internally.

## Definition and interpretation

For each initialization, the NCL script uses 60 represented months, averages
members first, computes five unweighted annual means at each grid cell, takes
area-weighted global means, and calculates

`std_ratio = sample_std(model_global_index) / sample_std(observed_global_index)`.

Both standard deviations use ddof=1 and retain trends. A ratio of 1 means equal
variability; greater than 1 means excess model variability. This statistic alone
does not attribute excess variability to initialization shock or measure an
initial discontinuity. It is not ensemble spread. The trend/sign functions in
the NCL file are unused, so this module does not invent a trend index.

The NCL script subtracts a scalar 1981–2010 mean from each global series. This
cannot change its standard deviation and is omitted here. Do not substitute
ACC prepared anomalies: removing a lead-dependent climatology or detrending
can suppress the signal this diagnostic is intended to inspect.

The reference defaults follow NCL's [month_to_annual](https://www.ncl.ucar.edu/Document/Functions/Contributed/month_to_annual.shtml)
(unweighted, all 12 months required) and [stddev](https://www.ncl.ucar.edu/Document/Functions/Built-in/stddev.shtml)
(sample standard deviation). Missing-value handling is deliberately stricter:
model and observation must use paired valid blocks, with all requested blocks
required by default. `min_samples` can explicitly relax that requirement.

## Input and workflow contract

1. Open monthly model fields with the existing E3SM/SMYLE loaders. Normalize
   represented months using the existing time-bound/filename logic. Convert units
   and regrid observations/model to the same lat/lon coordinates upstream.
2. Supply model `(Y,L,M,lat,lon)` or ensemble mean `(Y,L,lat,lon)`, numeric monthly
   lead coordinates, and `verification_time(Y,L)`. Each initialization's time
   row must contain consecutive represented months. May/November starts remain
   separate initializations; do not average start months as ensemble members.
3. Select observations with `align_observation_months(obs, verification_time)`.
   This matches year/month across calendars and rejects missing/duplicate months.
4. Compute compact global-index and metric outputs; cache those outputs and plot
   from the cache. Regular grids default to cosine-latitude weights. Supply
   explicit `area(lat,lon)` for other grids. A region may be represented with
   zero area outside it; areas must be finite and non-negative.
5. Diagnose each ratio with `model_index`, `observation_index`, their standard
   deviations, `paired_sample_count`, `valid_ratio`, and spatial coverage.
   Independent finite-cell spatial masks match the NCL strategy, but differing
   masks can affect comparisons; `min_area_fraction` allows a coverage threshold.

No automatic archive search, regridding, unit inference, drift removal, or
calendar-year relabeling occurs in this module. Gridded input caches remain
owned by their preprocessing workflow. The small metric products go under
`<output_root>/<case>/initial_shock/metrics/std_index_<identity>.nc`. Identity
includes source paths, sizes, modification times, conversions, metric settings,
and algorithm version. Atomic writes avoid partially published NetCDF caches.
Modes are `auto`, `rebuild`, and `require`; `require` still needs source metadata.

## Notebook use

```python
from esp_lab.diagnostics.initial_shock import (
    align_observation_months, compute_initial_shock_index, plot_std_ratio,
)

# model_monthly and obs_monthly: compatible physical units, same grid.
# verification_time: represented calendar month, shape (Y,L).
obs_aligned = align_observation_months(obs_monthly, verification_time)
metrics = compute_initial_shock_index(
    model_monthly, obs_aligned,
    window_months=60, block_months=12, min_area_fraction=0.9,
).compute()
fig = plot_std_ratio(metrics)
```

**Current 24-month S2D forecasts cannot reproduce the NCL five-year statistic.**
The default call rejects them. Setting `window_months=24, block_months=12`
computes a two-block annual-scale ratio with only two temporal samples. Treat
this as exploratory. May-start blocks are May–April, not calendar years.
Alternatively, `window_months=24, block_months=3` gives eight seasonal blocks;
this is a different metric and includes seasonal-cycle variability. Neither
should be labeled the original five-year index. Keep an explicit averaging
choice and compare like windows across models.

## Separate cached execution

Use `python -m workflows.diagnostics.initial_shock config.json` from ESP-Lab.
The example configuration uses placeholder paths to normalized monthly inputs;
replace them before running. `scale` and `offset` apply explicit unit conversion.
`rename` optionally maps source dimensions to the above input contract.

```json
{
  "output_root": "/path/to/diagnostics",
  "figure_path": "/path/to/figures/initial_shock_std_ratio.png",
  "cache_mode": "auto",
  "settings": {"window_months": 60, "block_months": 12, "min_area_fraction": 0.9},
  "cases": {
    "E3SM-FOSIRL": {
      "path": "/path/to/model_monthly.nc",
      "variable": "TREFHT",
      "verification_time": "verification_time",
      "units": "degC", "offset": -273.15
    }
  },
  "observation": {
    "path": "/path/to/observation_monthly.nc",
    "variable": "tas", "units": "degC", "offset": -273.15
  }
}
```

Add Reanalysis, 4DEnVarOcn, and CESM-SMYLE to `cases` with the same initialization
labels, represented months and units. The comparison rejects differing Y labels.
Ensemble sizes may differ, but affect ensemble-mean variability: these ratios do
not include the finite-ensemble resampling correction from the ACC workflow.

Validation uses analytic synthetic inputs and an end-to-end NetCDF/cache/plot
test. Numerical parity with the original NCL data requires those input files;
the module does not claim that archive-backed comparison has been run.
