# Two-reference drift preprocessing drivers

These notebooks are optional interactive batch drivers for the two-reference drift diagnostics. Core diagnostic logic lives in `esp_lab/diagnostics/two_reference_drift.py` and `workflows/diagnostics/`.

| Notebook | Purpose |
| --- | --- |
| `0_run_drift_input.ipynb` | Precompute observation (`X_obs`), E3SM-LE historical attractor (`X_att`), and spread (`sigma_att`) references, plus monthly hindcast caches. |
| `0_run_drift_diag.ipynb` | Bulk-generate complete multi-region, 2D spatial drift map, and bootstrap paired-difference diagnostic NetCDF files. |

The refactored analysis notebooks (`5a_refactor_drift_map.ipynb`, `5b_refactor_drift_region.ipynb`, `5c_refactor_drift_regime.ipynb`, `5d_refactor_drift_analysis.ipynb`, and `5d_refactor_drift_errcorr.ipynb`) do not require running either notebook first. Their input preparation cells automatically ensure the needed regional products or reference inputs on demand via `workflows.diagnostics.drift_inputs`. Set `DRIFT_INPUT_MODE` to `auto`, `require`, or `rebuild` to control cache behavior.

Use these drivers when precomputing the entire multi-variable matrix of diagnostics across all regions, full 2D spatial grids, and bootstrap iterations in advance. Both notebooks resolve the repository root when launched from this nested directory.
