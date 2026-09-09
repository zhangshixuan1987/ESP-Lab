# Workflow package layout

The `workflows` package contains orchestration code. Reusable calculations and
I/O primitives belong in `esp_lab`.

| Package | Purpose |
|---|---|
| `diagnostics` | Daily/monthly drift, initial-condition, physical-consistency, and integrated diagnostic pipelines |
| `e3sm_analysis` | E3SM diagnostics and time-series comparisons |
| `leadtime_skill` | Lead-time skill and RMSE comparison helpers |
| `modes_of_variability` | Modes-of-variability processing support |
| `tropical_cyclones` | Tropical-cyclone diagnostics |

Each workflow package should keep configuration, discovery, preprocessing,
diagnostics, plotting, and its command-line runner together when those layers
are needed. Generated data, figures, notebook checkpoints, and Python bytecode
do not belong under this directory.

## S2D Drift Analysis Notebook Suite (5a--5e)

The interactive analysis notebooks in `jupyter/` provide sequential, publication-ready two-reference drift diagnostics:

| Notebook | Focus | Primary Products & Methods |
|---|---|---|
| `5a_refactor_drift_map.ipynb` | Global spatial maps | 2D global maps of distance change to observations ($X_{\text{obs}}$) vs model attractor ($X_{\text{att}}$) |
| `5b_refactor_drift_region.ipynb` | Regional drift & skill | Lead-time drift trajectories and RMSE/spread curves for Niño3.4, North Atlantic, and global land H2OSOI |
| `5c_refactor_drift_regime.ipynb` | Drift-regime frequency | Fraction of regional area in Regimes 1--4 (converging to obs vs drifting to model climate) |
| `5d_refactor_drift_skill_relationship.ipynb` | Drift-to-skill attribution | Unified Method 1 (unconditional early drift vs later RMSE) and Method 2 (conditional error-growth rate controlling for initial $L=1$ error with block bootstrap), plus paired strategy contrasts |
| `5e_refactor_drift_summary.ipynb` | Integrated S2D summary | Lead-time bias/spread, calendar-month bias, SST vs 2m air temp consistency, ENSO event frequency, and summary scorecards |

Input preparation is managed on-demand via `workflows.diagnostics.drift_inputs`, or batch-precomputed via `jupyter/preprocessing/drift/` drivers (`0_run_drift_input.ipynb`, `0_run_drift_diag.ipynb`).
Paired contrasts always use `JRA55_FOSIRL - Reanalysis`. Detailed modular subpackages (`daily_drift`, `monthly_drift`, `physical_consistency`, `unified`, `initial_conditions`) support underlying spatial and process calculations.
