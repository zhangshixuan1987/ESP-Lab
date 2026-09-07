# Lead-time climatology preprocessing driver

This notebook is an optional interactive batch driver for generating reusable monthly lead-dependent climatology products.

| Notebook | Purpose |
| --- | --- |
| `0_run_leadtime_mon_clim.ipynb` | Compute and save monthly lead-dependent climatologies for selected variables across hindcast models. |

Outputs use the component-aware layout:
`<S2D_DIAG_ROOT>/<source>/leadtime_acc/climatology/<component>/<variable>/`

The active lead-time skill-map and comparison notebooks (`1a`, `1b`, `2a`, `2b`) derive anomaly baselines and seasonal aggregations through their own self-contained orchestration pipelines or unified references, so running this driver beforehand is optional.
