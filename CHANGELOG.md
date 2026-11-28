# Changelog

## Unreleased — November 2026
- **Production setup**: FOSIRL and Reanalysis hindcasts, May/November starts 1980–2018 and 1981–2010 climatology in every notebook and the gallery. February/August starts (Reanalysis only, every five years) are still evaluation work in progress: they remain an option (`init_months = [2, 5, 8, 11]`) but are off by default; 4DEnVarOcn/Lnd stay switched off in the production notebooks until their runs are complete. CESM-SMYLE benchmarks extend to 2018.
- **NMME, one method**: Sha Feng's cross-source workflow (`workflows/modes_of_variability/cross_source.py`) is the only NMME implementation. Each model's drift is removed with its own climatology; the NMME line is the skill of the equal-weight multi-model forecast and the band is the min–max of the individual models' skill. Default cohort: May/November starts 1982–2010 with every model covering them (20 of 23); years, model list, or a single model are options. `scripts/slurm/nmme_cross_source_test.slurm` with `scripts/render_nmme_comparison.py` runs the comparison end to end. The old NMME SST-index processor (`scripts/run_process_nmme_sst_index.py`), the old NMME paths in `3a`/`5a`/`5b`, and the two earlier Slurm templates are removed. `3a`, `4a`, and `5a` each add a second figure (`..._nmme1982_2010`) that re-scores E3SM and CESM-SMYLE over the NMME start years next to the NMME line and band (`include_nmme`, `NMME_YEARS`, `NMME_MODELS`, `NMME_SINGLE_MODEL`); the main figures are unchanged. Seasonal NMME ELI comes from three-month-mean SST (`ELI_seas`), and each model is scored only at the leads its hindcast covers. 4a reads the NMME mode products and never builds them (hours per start month; build in batch).
- **Sea ice (`1a_ice`)**: sea-ice concentration ACC against HadISST2 on Northern and Southern Hemisphere polar-cap maps, with FOSIRL–Reanalysis difference maps.
- **NMME data checks**: all-zero archive maps (CanSIPS-IC4 member M015 and one 2014 start of M027 store missing PSL as zeros) are treated as missing; mode PCs leave out (year, member) samples with no data instead of scoring them as 0; a pressure-mode gap of at most 1% of the reference area (CanSIPS-IC4 has no 90°N row on the 5° grid) is zero-filled before the fixed-EOF projection.
- **7b**: the ENSO–TC regression figure uses fixed-size map panels with the observed panel centred and smaller basin labels.
- **Robustness**: file locks fall back to a lock directory where the file system refuses `flock` (CFS compute nodes); notebook runs have a per-cell timeout; cache names carry the analysis grid; `1c` comparisons skip start months a case lacks; `6a`/`6b` use ERA5 1979–2025 and end PRECT starts with GPCP coverage; CPC soil moisture is read from `CPC_SOM/1x1/monthly`.

## Version 1.4.0 — September 2026
Final-pass consolidation of the E3SM S2D diagnostic suite:
- **Run controls everywhere**: every workflow notebook starts with `derivation_mode` and `recompute_*` flags, and forcing now reaches every cache it should (1b/1c observations, 1c bootstrap, 2a monthly skill, 3a NMME skill, 5a/5b native ELI, 7b ENSO regression).
- **Ocean fields**: `SSS` (EN4) and `OHC700` (EN4 0–700 m heat content, MPAS field converted to J m-2 before regridding) across `1a`/`1b`/`1c`/`2a`; fields without a CESM-SMYLE benchmark get E3SM-only maps.
- **Consistent comparisons**: ACC skill maps show the full field with significance dots; every `1a`/`1b` field has an E3SM case-difference figure; `1c` keeps only the significance-tested RMSE difference; `1c_lnd` skips direct RMSE for anomaly-only references (TWS) instead of failing.
- **Regional skill (`2a`)**: separate `OCN_MONTHLY_REFERENCES` and `LND_MONTHLY_REFERENCES`, realm-appropriate regions, correct season labels, and nRMSE that ignores cells without observed variability (e.g. SST under sea ice).
- **Figures**: ACC and difference color maps give each color-bar interval exactly one color, so white spans exactly -0.1 to 0.1 and the 0.5 ACC cutoff is a clear step (all ACC, difference, and teleconnection maps); `1b_lnd` and `1c_lnd` add a CONUS zoom in the `1b_atm` layout; `1c_lnd` draws one RMSE difference figure per start month.
- **Web gallery**: descriptive, fixed-order Lead-time RMSE buttons, Case Difference buttons, SSS/OHC700 rows, PSL initial-shock figures, and Niño driver names; no two figures share a button.
- **Robustness**: large dask-backed NetCDF writes are computed before writing (fixes a hang on the HDF5 lock); native ELI refuses to write empty caches; scripts and workflows resolve roots through `esp_lab.env_paths`.
- **Packaging**: version 1.4.0; unused `eofs`, `xcdat`, and `cmocean` removed from the runtime requirements; `global-land-mask` added to `environment.yml`.

## Version 1.3.0 — September 2024
This version delivers major enhancements for E3SM subseasonal-to-decadal (S2D) hindcast evaluation, interactive visualization, and distributed workflow resilience:
- **Streamlined `jupyter/` Analysis & Diagnostic Suite**: Refactored evaluation workflows into a clean sequential suite (Notebooks 0 through 7) covering distributed CESM-SMYLE benchmark preprocessing, atmospheric and land lead-time ACC maps, multi-model RMSE comparisons, SST index skill and teleconnections, modes of variability (PDO, AMO, NAO), native and regridded ELI analysis, and initialization shock diagnostics.
- **Interactive Diagnostics Viewer**: Added `esp_lab.diagnostics.web` to automatically generate a standalone responsive HTML diagnostics portal with a one-page Quick Buttons matrix view, driver mode filter pills (`init05`, `init11`, etc.), and pop-out lightbox modals.
- **Hardened Distributed Dask Engine**: Enhanced `esp_lab.utils.dask_util.py` and benchmark runner with network filesystem HDF5 lock bypass (`HDF5_USE_FILE_LOCKING=FALSE`), dynamic ephemeral dashboard port binding (`:0`) to avoid conflicts on multi-user nodes, worker memory garbage collection, and pre-loop cluster health validation.
- **Cleaned Module Architecture**: Consolidated diagnostic modules under `esp_lab.diagnostics` (`s2d`, `web`, `config`, `sst_index`, `native_eli`, `initial_shock`), retired obsolete legacy drift and physical consistency modules, and purged untracked checkpoint files.
- **Modernized Packaging & CI**: Updated Python runtime support to Python 3.10+ (tested through 3.12/3.13), upgraded ReadTheDocs build environment to `ubuntu-22.04`, and streamlined installation dependencies.

## Version 1.2.0 (E3SM Extensions) — May 20, 2024
This version builds on the original ESP-Lab package and introduces refactoring to support its application to the E3SM model. The updates focus on reorganizing the code, improving modularity, and preparing the package for E3SM-specific workflows and future development.

## Version 1.1 - Aug 4, 2022
Stats.py has been updated to include new skill score wrapper functions, allowing users to compute annual and seasonal skill with data arrays that have or have not been resampled. Dependabot features have been added. Documentation was updated. Parameters have been adjusted in data_access.get_monthly_data() to include a list of years instead of start/end year. Package updates were made to avoid dask incompatabilities and generally update packages. Jupyter notebook examples were provided as a tutorial for users.

## Version 1.0 - June 2, 2022
Updates include module documentation, testing capabilities, and various package setup requirements. ReadTheDocs and sphinx automodules have been implemented and include some FAQs, installation documentation, changelogs, and a how-to guide. Codecov has been implemented and testing increased from 0% to 42%. Continuous Integration and workflow requirements were updated.

## Version 0.1 - April 28, 2022
Updates have been made to include docstrings and comments, add readthedocs, adjust directory structure, improve hard coded sections of the code, clarify parameters, make functions directly accessible in `__init__.py`

## Version 0.0 - March 25, 2022
Original version of code from Steve Yeager
