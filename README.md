# ESP-Lab

- [ESP-Lab](#esp-lab)
  - [Badges](#badges)
  - [Overview](#overview)
  - [E3SM S2D Extensions](#e3sm-s2d-extensions)
  - [Analysis & Diagnostic Suite (`jupyter/`)](#analysis--diagnostic-suite-jupyter)
  - [Interactive Web Viewer](#interactive-web-viewer)
  - [Installation](#installation)
    - [Developer Installation (Conda Environment)](#developer-installation-conda-environment)
    - [Installation into Existing Conda Environment](#installation-into-existing-conda-environment)
    - [Pip Installation](#pip-installation)
  - [Documentation & Links](#documentation--links)

## Badges
| CI          |                                                               [![Code Coverage Status][codecov-badge]][codecov-link] |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Docs**    |                                                                     [![Documentation Status][rtd-badge]][rtd-link]                                                                     |
| **Package** |                                                                             [![PyPI][pypi-badge]][pypi-link]                                                                           |
| **License** |                                                                         [![License][license-badge]][repo-link]                                                                         |

## Overview
ESP-Lab is an Earth System Predictions Python package originally designed to enable users to effectively perform I/O operations and statistics on [SMYLE (The Seasonal-to-Multiyear Large Ensemble)](https://doi.org/10.5194/gmd-2022-60) data. It provides a foundational toolkit for analyzing subseasonal-to-multiyear predictions of climate variability and environmental change.

Key capabilities include:
- Efficient lead-time slicing and indexing across hindcast ensembles spanning 1 month to multiple years.
- Dask-distributed data processing and memory-optimized preprocessors.
- Statistical verification metrics (deterministic skill, ACC, RMSE, ensemble distributions, and detrending).

## E3SM S2D Extensions
This fork extends ESP-Lab to fully support analysis and verification of **E3SM (Energy Exascale Earth System Model)** subseasonal-to-decadal (S2D) hindcasts alongside CESM-SMYLE, NMME, and observational reference benchmarks:

- **E3SM Hindcast Workflows**: Native support for E3SM file naming, spatial grids, variable structures, and multi-start initialization cycles (e.g., February, May, August, November).
- **Multi-Model Intercomparisons**: Standardized diagnostic pipelines evaluating E3SM against CESM-SMYLE, NMME models, and observational datasets (e.g., ERA5, GPCP, HadISST).
- **Streamlined Diagnostics Package**: Modular diagnostic workflows under `esp_lab.diagnostics` covering SST indices, ENSO teleconnections, modes of variability (PDO, AMO, NAO), native equatorial Pacific longitudinal index (ELI), and initialization shock.
- **Robust Dask Automation**: Resilient Dask distributed cluster lifecycle, file locking safeguards on network filesystems (GPFS/CFS), and worker memory management.

## Analysis & Diagnostic Suite (`jupyter/`)
The evaluation workflows are categorized under the [`jupyter/`](jupyter/) directory by timescale and diagnostic focus:

### S2D Prediction Skill Suite (`jupyter/s2d_skill/`)

| Notebook | Focus Area | Description |
|---|---|---|
| [`0_run_cesm_smyle_benchmark.ipynb`](jupyter/s2d_skill/0_run_cesm_smyle_benchmark.ipynb) | Benchmark Data | Dask-distributed preprocessing of CESM-SMYLE hindcasts |
| [`1a_atm_leadtime_acc_skill_map.ipynb`](jupyter/s2d_skill/1a_atm_leadtime_acc_skill_map.ipynb) | Atmospheric Skill | Lead-time anomaly correlation coefficient (ACC) maps |
| [`1b_lnd_leadtime_acc_skill_map.ipynb`](jupyter/s2d_skill/1b_lnd_leadtime_acc_skill_map.ipynb) | Land Skill | Land surface lead-time ACC maps (soil moisture, runoff, etc.) |
| [`2a_leadtime_rmse_skill_map.ipynb`](jupyter/s2d_skill/2a_leadtime_rmse_skill_map.ipynb) | Error Maps | Spatial root mean square error (RMSE) skill maps |
| [`2b_leadtime_rmse_compare.ipynb`](jupyter/s2d_skill/2b_leadtime_rmse_compare.ipynb) | Model Comparison | Multi-model RMSE comparison and model difference metrics |
| [`3a_sst_skill_ts.ipynb`](jupyter/s2d_skill/3a_sst_skill_ts.ipynb) | Ocean Skill | SST index skill time series (E3SM, CESM-SMYLE, NMME) |
| [`3b_sst_telecon.ipynb`](jupyter/s2d_skill/3b_sst_telecon.ipynb) | Teleconnections | Sea surface temperature teleconnection diagnostics |
| [`4a_mov_analysis.ipynb`](jupyter/s2d_skill/4a_mov_analysis.ipynb) | Modes of Variability | EOF projection and index calculation (PDO, AMO, NAO) |
| [`4b_mov_telecon.ipynb`](jupyter/s2d_skill/4b_mov_telecon.ipynb) | Teleconnections | Modes of variability climate teleconnection patterns |
| [`5a_eli_skill_ts.ipynb`](jupyter/s2d_skill/5a_eli_skill_ts.ipynb) | Tropical Pacific | Equatorial Longitude Index (ELI) skill time series |
| [`5b_eli_diagnostics.ipynb`](jupyter/s2d_skill/5b_eli_diagnostics.ipynb) | ELI Diagnostics | Native & regridded ELI diagnostics across starts |
| [`5c_eli_telecon.ipynb`](jupyter/s2d_skill/5c_eli_telecon.ipynb) | Teleconnections | ELI precipitation and temperature teleconnections |
| [`6a_shock_ts.ipynb`](jupyter/s2d_skill/6a_shock_ts.ipynb) | Initialization Shock | Lead-dependent drift and initialization shock time series |
| [`6b_shock_index.ipynb`](jupyter/s2d_skill/6b_shock_index.ipynb) | Shock Indices | Initialization shock metrics and multi-model indices |
| [`7a_tc_method_analysis.ipynb`](jupyter/s2d_skill/7a_tc_method_analysis.ipynb) | TC Methods | Tropical cyclone tracking method comparison and sanity plots |
| [`7b_tc_leadtime_analysis.ipynb`](jupyter/s2d_skill/7b_tc_leadtime_analysis.ipynb) | TC Lead-time | TC lead-time diagnostics, IBTrACS companion, ENSO regression |
| [`8_run_viewer_webpage.ipynb`](jupyter/s2d_skill/8_run_viewer_webpage.ipynb) | Interactive Web | Generate interactive diagnostics viewer webpage |

### S2D Drift & Initialization Suite (`jupyter/s2d_drift/`)

| Notebook | Focus Area | Description |
|---|---|---|
| [`0_refactor_ic_analysis.ipynb`](jupyter/s2d_drift/0_refactor_ic_analysis.ipynb) | IC Audit | 6-step initial-condition audit: hash, structure, stats, plots, consistency, campaign summary |
| [`1a_refactor_drift_map.ipynb`](jupyter/s2d_drift/1a_refactor_drift_map.ipynb) | Drift Maps | Global two-reference distance-change maps (obs & model attractor) |
| [`1b_refactor_drift_region.ipynb`](jupyter/s2d_drift/1b_refactor_drift_region.ipynb) | Regional Drift | Lead-time drift trajectories and RMSE/spread curves by region |
| [`1c_refactor_drift_regime.ipynb`](jupyter/s2d_drift/1c_refactor_drift_regime.ipynb) | Drift Regimes | Fraction of area in each drift regime (converging vs drifting) |
| [`1d_refactor_drift_skill_relationship.ipynb`](jupyter/s2d_drift/1d_refactor_drift_skill_relationship.ipynb) | Drift–Skill | Early drift vs later skill attribution (Methods 1 & 2 with bootstrap) |
| [`1e_refactor_drift_summary.ipynb`](jupyter/s2d_drift/1e_refactor_drift_summary.ipynb) | S2D Summary | Integrated bias/drift/spread, ENSO frequency, and scorecards |
| [`2a_refactor_physical_consistency.ipynb`](jupyter/s2d_drift/2a_refactor_physical_consistency.ipynb) | Physical Consistency | Flux partitioning, land coupling, ocean coupling, energy & water budget |
| [`2b_refactor_unified_diagnostics.ipynb`](jupyter/s2d_drift/2b_refactor_unified_diagnostics.ipynb) | Unified S2D | Master orchestrator: field drift, physical consistency, model attractor, IC attribution |

### Interactive Diagnostics Viewer
- [`run_viewer_webpage.ipynb`](jupyter/run_viewer_webpage.ipynb): Generates a responsive, standalone HTML diagnostics portal with matrix quick buttons and pop-out lightbox modals.



## Interactive Web Viewer
ESP-Lab includes a responsive HTML web generator (`esp_lab.diagnostics.web`) that compiles all evaluation figures into a standalone, browsable diagnostics gallery.

- **Live Web Gallery**: [NERSC CFS ESP-Lab Diagnostics Portal](https://portal.nersc.gov/cfs/e3sm/zhan391/esp-lab_diag/index.html)
- **Features**:
  - **Quick Buttons View**: One-page clickable matrix organized by diagnostic category and figure type.
  - **Driver Mode Filter**: Instant filtering by initialization mode (`init05`, `init11`, `all`, etc.).
  - **Lightbox Modal**: Click any diagnostic button to pop out the full-resolution graphic with caption and download link.

## Installation

### Developer Installation (Conda Environment)
To install from source and set up a dedicated conda environment:

```bash
git clone -b e3sm-esp https://github.com/zhangshixuan1987/ESP-Lab.git
cd ESP-Lab
conda env create --file environment.yml
conda activate esp-lab
pip install -e .
```

### Installation into Existing Conda Environment
To install ESP-Lab into an existing environment (such as `e3sm_analysis` on NERSC Perlmutter):

```bash
conda activate e3sm_analysis
cd ESP-Lab
pip install -e .
```

### Pip Installation
ESP-Lab can also be installed directly from PyPI:

```bash
pip install esp-lab
```

## Documentation & Links
- **Documentation**: [esp-lab.readthedocs.io](https://esp-lab.readthedocs.io/)
- **Repository**: [github.com/zhangshixuan1987/ESP-Lab](https://github.com/zhangshixuan1987/ESP-Lab)
- **Issue Tracker**: [github.com/zhangshixuan1987/ESP-Lab/issues](https://github.com/zhangshixuan1987/ESP-Lab/issues)
- **Upstream Project**: [CESM-ESPWG/ESP-Lab](https://github.com/CESM-ESPWG/ESP-Lab)

[codecov-badge]: https://img.shields.io/codecov/c/github/zhangshixuan1987/ESP-Lab/e3sm-esp.svg?logo=codecov
[codecov-link]: https://codecov.io/gh/zhangshixuan1987/ESP-Lab
[rtd-badge]: https://img.shields.io/readthedocs/esp-lab/latest.svg
[rtd-link]: https://esp-lab.readthedocs.io/en/latest/?badge=latest
[pypi-badge]: https://img.shields.io/pypi/v/esp-lab?logo=pypi
[pypi-link]: https://pypi.org/project/esp-lab/
[license-badge]: https://img.shields.io/github/license/zhangshixuan1987/ESP-Lab
[repo-link]: https://github.com/zhangshixuan1987/ESP-Lab
