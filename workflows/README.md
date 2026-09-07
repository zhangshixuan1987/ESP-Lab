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

## S2D 5a--5f product flow

The notebooks are thin workflow clients with one responsibility each:

| Notebook | Product responsibility |
|---|---|
| 5a | Regional bias, adjustment, paired adjustment, and skill |
| 5b | Time-zero native-grid IC fingerprints and candidate drivers |
| 5c | Monthly spatial bias and paired adjustment |
| 5d | Daily rapid adjustment and weeks 1--12 evolution |
| 5e | Physical pathways and budget consistency from validated 5c/5d fields |
| 5f | Read 5b--5e products, synthesize them, and perform IC--drift attribution |

Producers write a tidy `products.csv`, a `product_manifest.json`, and links to
native-grid NetCDF products using `esp_lab.diagnostics.products`. Paired values
always use `JRA55_FOSIRL - Reanalysis`. Monthly lead 1 is the initialization
calendar month; daily production adjustment uses the days 1--3 mean when all
three days are available. Initialization years are the bootstrap blocks.
