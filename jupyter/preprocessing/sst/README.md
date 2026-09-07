# SST preprocessing drivers

These notebooks are optional interactive drivers for creating reusable regional
SST-index diagnostics. Production logic lives in `scripts/`; the refactored SST
analysis calls those scripts or their functions directly.

| Notebook | Purpose |
| --- | --- |
| `0_run_sigmod_sst_index.ipynb` | Build regional SST indices for E3SM, CESM-SMYLE, and HadISST2, including optional native ELI products. |
| `0_run_nmme_sst_index.ipynb` | Build regional, drift-corrected NMME SST-index time series for selected regions and models. |

`jupyter/3_refactor_sst_skill_ts.ipynb` does not require these notebooks to be
run first. It ensures the products needed for its selected region through
`scripts/run_process_sst_index.py` and `scripts/run_process_nmme_sst_index.py`.

Use these drivers when precomputing many regions, inspecting archive coverage,
or running preprocessing separately from the analysis notebook. They resolve
the repository root from nested working directories, so they can be launched
from the repository root or from this directory.

The gridded CESM-SMYLE benchmark driver remains at
`jupyter/0_run_cesm_smyle_benchmark.ipynb` because its outputs are required by
`1a_refactor_atm_leadtime_acc_skill_map.ipynb`. The SST analysis can ensure its
own TS benchmark inputs, but the generalized atmospheric skill-map workflow
still consumes pre-generated benchmark files.
