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
