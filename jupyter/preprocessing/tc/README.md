# Tropical cyclone track preprocessing driver

This notebook is an optional interactive batch driver for detecting, stitching, and histogramming E3SM hindcast tropical cyclones using TempestExtremes.

| Notebook | Purpose |
| --- | --- |
| `0_run_tc_track_diag.ipynb` | Run TempestExtremes (`DetectNodes`, `StitchNodes`, `HistogramNodes`) across E3SM cases, ensemble members, and warm-core parameter sets (set1–set5). |

Downstream diagnostic notebooks (`7a_refactor_tc_method_analysis.ipynb` and `7b_refactor_tc_leadtime_analysis.ipynb`) can run self-contained TC track preparation directly via `workflows.tropical_cyclones.inputs.ensure_tracks`, so running this batch driver beforehand is optional.
