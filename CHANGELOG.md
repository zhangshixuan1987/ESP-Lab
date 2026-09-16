# Changelog

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
