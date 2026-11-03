# Workflow package layout

The `workflows` package contains high-level orchestration modules directly supporting the `jupyter/s2d_skill/` analysis suite (0 through 8). Reusable calculations and data access primitives belong in `esp_lab`.

| Package | Supported Notebooks | Purpose |
|---|---|---|
| `leadtime_skill` | `1a_*`, `1b_*`, `1c_*`, `2a` | Monthly skill preparation, RMSE comparison, and regional ACC summaries |
| `modes_of_variability` | `4a`, `4b` | Modes-of-variability processing, EOF projection, and teleconnection analysis |
| `diagnostics` | `3b`, `4b`, `5c`, `6a`, `6b` | Teleconnections (`sst_teleconnections`, `mov_teleconnections`, `teleconnection_inputs`) and initial-shock archive runners |

Each workflow package maintains configuration, discovery, preprocessing, diagnostics, and plotting together. Generated data, figures, and notebook checkpoints do not belong under this directory.

## Drift, initialization, and S2S packages (`jupyter/s2d_drift/`, `jupyter/s2s_skill/`)

| Package / module | Purpose |
|---|---|
| `diagnostics/{inventory_and_hash,compare_netcdf_structure,compute_ic_statistics,plot_component_differences,check_physical_consistency,summarize_campaign}` | Initial-condition audit steps configured by `diagnostics/ic_config.yaml` (`s2d_drift/0_refactor_ic_analysis`) |
| `diagnostics/drift_inputs`, `diagnostics/two_reference_drift`, `diagnostics/drift_summary` | Two-reference drift inputs, pipeline, and summaries (`s2d_drift/1a`–`1e`) |
| `diagnostics/daily_drift`, `diagnostics/monthly_drift` | Daily- and monthly-frequency two-reference drift: discovery, preprocessing, diagnostics, plotting, and CLI runners |
| `diagnostics/physical_consistency` | Flux partitioning, land–atmosphere coupling, precipitation–soil-moisture response, ocean coupling, and energy/water budgets (`s2d_drift/2a`) |
| `diagnostics/unified` | Unified S2D orchestrator: field drift, physical consistency, model attractor, and IC-to-drift attribution (`s2d_drift/2b`) |
| `tropical_cyclones` | TC track density, lead-time diagnostics, and method comparisons (`s2d_skill/7a`, `7b`) |
| `esp_lab/diagnostics/s2s_io`, `esp_lab/diagnostics/s2s_core` | S2S weekly (weeks 1–8) loading, caching, anomalies, ACC/RMSE (`s2s_skill/1a`–`4a`) |

The drift modules follow the `s2d_skill` I/O conventions. Roots come from `esp_lab.env_paths`.
Per-source products live under `<s2d_diag>/<source>/leadtime_drift/{references,diagnostics,regional,spatial_maps}/`
with `<source>_init<MM>_<FIELD>_…` names. Cross-source products and tables live under
`<s2d_diag>/multimodel/<diagnostic>/`, and IC audit products under `<s2d_diag>/multimodel/initial_conditions/`.
Figures are written flat to `env_paths.drift_figure_root()` through `figure_filename`, as
`fig_drift_<diagnostic>_<field>_init<MM>…png` or `fig_ic_…png`. The monthly and daily drift
stores live in the fixed folder `diagnostics/<workflow>/<VAR>/` as `<VAR>_init<MM>_<workflow>.nc`.
Their configuration fingerprint is kept in `manifest.json`, and a stale store is rebuilt in place.
`s2d_drift/3_run_viewer_webpage.ipynb` builds the gallery, which groups figures as Lead-time Drift and Initial-condition Audit.

The S2S suite follows the same conventions. `ensure_s2s_campaign_weekly` caches weekly hindcasts under
`<s2d_diag>/<source>/s2s_weekly/inputs/<comp>/<FIELD>/<source>_init<MM>_<FIELD>_y<a>-<b>_ny<n>_N<members>_W<weeks>_week.nc`,
using the notebooks' `derivation_mode` (`auto`/`require`/`rebuild`). Figures go flat to `env_paths.s2s_figure_root()` as
`fig_s2s_<comp>_<diagnostic>_<field>_<source>_init<MM>_w1_w8.png`, and `s2s_skill/5_run_viewer_webpage.ipynb` builds their gallery.

