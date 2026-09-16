# Workflow package layout

The `workflows` package contains high-level orchestration modules directly supporting the `jupyter/` analysis suite (0 through 7). Reusable calculations and data access primitives belong in `esp_lab`.

| Package | Supported Notebooks | Purpose |
|---|---|---|
| `leadtime_skill` | `2b` | Lead-time RMSE comparison and multi-model metrics |
| `modes_of_variability` | `4a`, `4b` | Modes-of-variability processing, EOF projection, and teleconnection analysis |
| `diagnostics` | `3b`, `4b`, `5c`, `6a`, `6b` | Teleconnections (`sst_teleconnections`, `mov_teleconnections`, `teleconnection_inputs`) and initial-shock archive runners |

Each workflow package maintains configuration, discovery, preprocessing, diagnostics, and plotting together. Generated data, figures, and notebook checkpoints do not belong under this directory.

## Additional Workflow Packages (Current Branch)

| Package | Purpose |
|---|---|
| `diagnostics/initial_conditions` | IC hash audit (SHA-256), variable statistics, spatial plots, cross-component physical consistency, and campaign summary across all start dates |
| `diagnostics/physical_consistency` | Flux partitioning (EF & Bowen ratio), land-atmosphere coupling, precip–SM lag response, ocean coupling, and apparent surface energy residual |
| `diagnostics/unified` | Master S2D orchestrator: field drift (Branch A), physical consistency (Branch B), model attractor (Branch C), and IC-to-drift attribution (Branch D) |
| `diagnostics/daily_drift` | Daily-frequency two-reference drift — discovery, preprocessing, diagnostics, plotting, and CLI runner |
| `diagnostics/monthly_drift` | Monthly-frequency two-reference drift — same structure as `daily_drift` |
| `e3sm_analysis` | E3SM time-series diagnostics and multi-experiment comparisons |
| `tropical_cyclones` | TC track density, lead-time diagnostics, and multi-method comparisons |

Corresponding Jupyter notebooks: `5i` (IC analysis), `5j` (physical consistency), `5k` (unified diagnostics), `8a–8b` (tropical cyclones).
