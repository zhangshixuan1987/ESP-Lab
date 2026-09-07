# Modes-of-variability preprocessing drivers

These notebooks are optional interactive batch drivers. The reusable processing
code lives in `scripts/run_process_modes_of_variability.py` and
`workflows/modes_of_variability/`.

| Notebook | Purpose |
| --- | --- |
| `0_run_sigmod_emov.ipynb` | Precompute observation, E3SM, and CESM-SMYLE products for many modes. |
| `0_run_nmme_emov.ipynb` | Precompute gridded NMME products for selected modes and models. |

`jupyter/4_refactor_mov_analysis.ipynb` does not require either notebook to be
run first. Its input cell directly ensures the selected mode, the required
CESM-SMYLE benchmark, and global teleconnection products. Set
`mov_input_mode` to `auto`, `require`, or `rebuild` to control cache behavior.

Use these drivers to precompute several modes in a batch, inspect archive
coverage, or run preprocessing separately. Both notebooks discover the
repository root when launched from this nested directory.
