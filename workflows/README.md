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
