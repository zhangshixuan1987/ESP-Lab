# ESP-Lab

- [ESP-Lab](#esp-lab)
  - [Badges](#badges)
  - [Overview](#overview)
  - [E3SM S2D Extensions](#e3sm-s2d-extensions)
  - [Analysis & Diagnostic Suite (`jupyter/`)](#analysis--diagnostic-suite-jupyter)
  - [Interactive Web Viewer](#interactive-web-viewer)
  - [Data Organization & Output Conventions](#data-organization--output-conventions)
    - [Diagnostic Data Layout (`S2D_DIAG_ROOT`)](#diagnostic-data-layout-s2d_diag_root)
    - [NetCDF File Naming Conventions](#netcdf-file-naming-conventions)
    - [Figure Naming Conventions (`FIGURE_OUTDIR`)](#figure-naming-conventions-figure_outdir)
  - [Installation](#installation)
    - [Developer Installation (Conda Environment)](#developer-installation-conda-environment)
    - [Installation into Existing Conda Environment](#installation-into-existing-conda-environment)
    - [PyPI Package](#pypi-package)
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
The evaluation workflows live under [`jupyter/`](jupyter/) in three suites: S2D prediction skill (`s2d_skill/`), S2D drift and initialization (`s2d_drift/`), and S2S weekly skill (`s2s_skill/`). `jupyter/run_viewer_webpage.ipynb` builds a gallery across the suites.

### S2D Prediction Skill Suite (`jupyter/s2d_skill/`)

The refactored `1a_*`, `1b_*`, and `1c_*` families separate ACC, anomaly-based RMSE maps, and direct raw-value RMSE comparisons, respectively, with `atm`, `lnd`, and `ocn` variants. Atmosphere and ocean workflows can compare against CESM-SMYLE; land workflows compare E3SM cases against a configured land reference. Land `1b` reuses or computes the same normalized-RMSE skill caches as land `1a`. Land `1c` requires an absolute-value reference: the C3S TWSA anomaly product used for `TWS` is suitable for `1a`/`1b`, so for `TWS` `1c` prepares its inputs but skips the direct-RMSE cells with a message.

Fields without a CESM-SMYLE benchmark (ocean `SSS` and `OHC700`, and all land fields) show the E3SM cases only. Every `1a` and `1b` notebook also draws a **case-difference** figure: each E3SM experiment minus a selected control case (`ACC_DIFFERENCE_CONTROL` / `RMSE_DIFFERENCE_CONTROL`, default `E3SM-Reanalysis`).

| Notebook | Focus Area | Description |
|---|---|---|
| [`0_run_cesm_smyle_benchmark.ipynb`](jupyter/s2d_skill/0_run_cesm_smyle_benchmark.ipynb) | Benchmark Data | Dask-distributed preprocessing of CESM-SMYLE hindcasts |
| [`1a_atm_leadtime_acc_skill_map.ipynb`](jupyter/s2d_skill/1a_atm_leadtime_acc_skill_map.ipynb) | Atmospheric Skill | Lead-time anomaly correlation coefficient (ACC) maps |
| [`1a_lnd_leadtime_acc_skill_map.ipynb`](jupyter/s2d_skill/1a_lnd_leadtime_acc_skill_map.ipynb) | Land Skill | Land lead-time ACC maps (H2OSNO, H2OSOI, TWS) |
| [`1a_ocn_leadtime_acc_skill_map.ipynb`](jupyter/s2d_skill/1a_ocn_leadtime_acc_skill_map.ipynb) | Ocean Skill | Ocean lead-time ACC maps (SST, SSS, OHC700) |
| [`1b_atm_leadtime_rmse_skill_map.ipynb`](jupyter/s2d_skill/1b_atm_leadtime_rmse_skill_map.ipynb) | Error Maps | Anomaly RMSE skill maps (atmosphere: TREFHT, TS, PRECT, PSL) |
| [`1b_ocn_leadtime_rmse_skill_map.ipynb`](jupyter/s2d_skill/1b_ocn_leadtime_rmse_skill_map.ipynb) | Error Maps | Anomaly RMSE skill maps (ocean: SST, SSS, OHC700) |
| [`1b_lnd_leadtime_rmse_skill_map.ipynb`](jupyter/s2d_skill/1b_lnd_leadtime_rmse_skill_map.ipynb) | Error Maps | Normalized RMSE skill maps (land: H2OSNO, H2OSOI, TWS) |
| [`1c_atm_leadtime_rmse_compare.ipynb`](jupyter/s2d_skill/1c_atm_leadtime_rmse_compare.ipynb) | Model Comparison | Direct (bias-inclusive) RMSE and significance-tested model differences (atmosphere) |
| [`1c_ocn_leadtime_rmse_compare.ipynb`](jupyter/s2d_skill/1c_ocn_leadtime_rmse_compare.ipynb) | Model Comparison | Direct (bias-inclusive) RMSE and significance-tested model differences (ocean: SST, SSS, OHC700) |
| [`1c_lnd_leadtime_rmse_compare.ipynb`](jupyter/s2d_skill/1c_lnd_leadtime_rmse_compare.ipynb) | Model Comparison | Direct-RMSE comparison between E3SM land cases (absolute-value reference required; skipped for TWS) |
| [`2a_regional_acc_skill_ts.ipynb`](jupyter/s2d_skill/2a_regional_acc_skill_ts.ipynb) | Regional Skill | Global and regional ACC and nRMSE versus lead (seasonal from 1a, plus monthly all-start curves) for all atmosphere, ocean, and land fields |
| [`3a_sst_skill_ts.ipynb`](jupyter/s2d_skill/3a_sst_skill_ts.ipynb) | Ocean Skill | SST index skill time series (E3SM, CESM-SMYLE, NMME) |
| [`3b_sst_telecon.ipynb`](jupyter/s2d_skill/3b_sst_telecon.ipynb) | Teleconnections | Sea surface temperature teleconnection diagnostics |
| [`4a_mov_analysis.ipynb`](jupyter/s2d_skill/4a_mov_analysis.ipynb) | Modes of Variability | EOF projection and index calculation (PDO, AMO, NAO) |
| [`4b_mov_telecon.ipynb`](jupyter/s2d_skill/4b_mov_telecon.ipynb) | Teleconnections | Modes of variability climate teleconnection patterns |
| [`5a_eli_skill_ts.ipynb`](jupyter/s2d_skill/5a_eli_skill_ts.ipynb) | Tropical Pacific | Equatorial Longitude Index (ELI) skill time series |
| [`5b_eli_diagnostics.ipynb`](jupyter/s2d_skill/5b_eli_diagnostics.ipynb) | ELI Diagnostics | Native & regridded ELI diagnostics across starts |
| [`5c_eli_telecon.ipynb`](jupyter/s2d_skill/5c_eli_telecon.ipynb) | Teleconnections | ELI precipitation and temperature teleconnections |
| [`6a_shock_ts.ipynb`](jupyter/s2d_skill/6a_shock_ts.ipynb) | Initialization Shock | Lead-dependent drift and initialization shock time series |
| [`6b_shock_index.ipynb`](jupyter/s2d_skill/6b_shock_index.ipynb) | Shock Indices | Initialization shock metrics and multi-model indices |
| [`7a_tc_method_analysis.ipynb`](jupyter/s2d_skill/7a_tc_method_analysis.ipynb) | Tropical Cyclones | TempestExtremes tracking method and parameter comparison |
| [`7b_tc_leadtime_analysis.ipynb`](jupyter/s2d_skill/7b_tc_leadtime_analysis.ipynb) | Tropical Cyclones | TC lead-time density, IBTrACS comparison, and ENSO regression |
| [`8_run_viewer_webpage.ipynb`](jupyter/s2d_skill/8_run_viewer_webpage.ipynb) | Gallery Webpage | Interactive HTML diagnostics viewer generator |

### Run controls
Each workflow notebook starts with the same run-control block, so a rerun never needs edits deeper in the notebook:

```python
derivation_mode = "auto"   # "auto": reuse valid caches, rebuild stale ones; "rebuild": recompute; "require": fail if a cache is missing
recompute_skill = False    # force the notebook's own skill/comparison caches (names vary: recompute_compare, recompute_telecon, ...)
```

Caches are reused only when their stored provenance attributes match the current settings; otherwise they are rebuilt in place under the same fixed name.

### S2D Drift & Initialization Suite (`jupyter/s2d_drift/`)

| Notebook | Focus Area | Description |
|---|---|---|
| [`0_refactor_ic_analysis.ipynb`](jupyter/s2d_drift/0_refactor_ic_analysis.ipynb) | IC Audit | 6-step initial-condition audit: hash, structure, stats, plots, consistency, campaign summary (`workflows/diagnostics/*` IC modules, `ic_config.yaml`) |
| [`1a_refactor_drift_map.ipynb`](jupyter/s2d_drift/1a_refactor_drift_map.ipynb) | Drift Maps | Global two-reference distance-change maps (obs & model attractor) |
| [`1b_refactor_drift_region.ipynb`](jupyter/s2d_drift/1b_refactor_drift_region.ipynb) | Regional Drift | Lead-time drift trajectories and RMSE/spread curves by region |
| [`1c_refactor_drift_regime.ipynb`](jupyter/s2d_drift/1c_refactor_drift_regime.ipynb) | Drift Regimes | Fraction of area in each drift regime (converging vs drifting) |
| [`1d_refactor_drift_skill_relationship.ipynb`](jupyter/s2d_drift/1d_refactor_drift_skill_relationship.ipynb) | Drift–Skill | Early drift vs later skill attribution (Methods 1 & 2 with bootstrap) |
| [`1e_refactor_drift_summary.ipynb`](jupyter/s2d_drift/1e_refactor_drift_summary.ipynb) | S2D Summary | Integrated bias/drift/spread, ENSO frequency, and scorecards |
| [`2a_refactor_physical_consistency.ipynb`](jupyter/s2d_drift/2a_refactor_physical_consistency.ipynb) | Physical Consistency | Flux partitioning, land coupling, ocean coupling, energy & water budget |
| [`2b_refactor_unified_diagnostics.ipynb`](jupyter/s2d_drift/2b_refactor_unified_diagnostics.ipynb) | Unified S2D | Master orchestrator: field drift, physical consistency, model attractor, IC attribution |

### S2S Weekly Skill Suite (`jupyter/s2s_skill/`)

| Notebook | Focus Area | Description |
|---|---|---|
| [`1a_atm_leadtime_acc_skill_map.ipynb`](jupyter/s2s_skill/1a_atm_leadtime_acc_skill_map.ipynb) | Atmospheric Skill | Weekly (weeks 1–8) atmospheric ACC skill maps |
| [`1b_lnd_leadtime_acc_skill_map.ipynb`](jupyter/s2s_skill/1b_lnd_leadtime_acc_skill_map.ipynb) | Land Skill | Weekly land and hydrologic ACC skill maps |
| [`2a_leadtime_rmse_skill_map.ipynb`](jupyter/s2s_skill/2a_leadtime_rmse_skill_map.ipynb) | Error Maps | Weekly RMSE and bias skill maps |
| [`2b_leadtime_rmse_compare.ipynb`](jupyter/s2s_skill/2b_leadtime_rmse_compare.ipynb) | Model Comparison | Multi-experiment weekly skill and RMSE comparison |
| [`3a_s2s_telecon_modes.ipynb`](jupyter/s2s_skill/3a_s2s_telecon_modes.ipynb) | Teleconnections | Subseasonal NAO, PNA, and AO prediction |
| [`4a_soil_moisture_memory.ipynb`](jupyter/s2s_skill/4a_soil_moisture_memory.ipynb) | Land Memory | Soil-moisture memory and land–atmosphere coupling |

## Interactive Web Viewer
ESP-Lab includes a responsive HTML web generator (`esp_lab.diagnostics.web`) that compiles all evaluation figures into a standalone, browsable diagnostics gallery.

- **Live Web Gallery**: [NERSC CFS E3SM-S2D Diagnostics Portal](https://portal.nersc.gov/cfs/e3sm/zhan391/e3sm-s2d_diag/) — 944 figures across lead-time ACC and RMSE, regional skill, SST indices, modes of variability, ELI, initial shock, teleconnections, and tropical cyclones. See the [gallery overview](docs/source/gallery.md) for its sections and example figures.
- **Features**:
  - **Quick Buttons View**: One-page matrix with one row per field (or index/mode) and one button per figure type; greyed buttons mark figures that do not exist for that row (e.g. no CESM-SMYLE comparison for SSS).
  - **Lead-time RMSE labels** say what each figure shows: `RMSE Skill Map` (1b anomaly RMSE), `nRMSE Difference` (E3SM minus CESM-SMYLE), `Case Difference` (E3SM minus control), `Total RMSE` and `RMSE Difference` (1c, bias included).
  - **Driver Mode Filter**: In the Teleconnections section, filter by driving index or mode (Niño3.4, PDO, NAO, ...).
  - **Lightbox Modal**: Click any diagnostic button to pop out the full-resolution graphic with caption and download link.

## Data Organization & Output Conventions
Diagnostic outputs are structured in two complementary layers: analysis NetCDF datasets (`S2D_DIAG_ROOT`) and public web figures (`FIGURE_OUTDIR`).

All notebooks resolve their machine-specific roots (`S2D_DIAG_ROOT`, `FIGURE_OUTDIR`, the raw model archive, the obs archive, and the supplemental reference-data directory) through `esp_lab.env_paths`, which defaults to the paths documented below but can be overridden per account with `ESP_LAB_S2D_DIAG_ROOT`, `ESP_LAB_FIGURE_ROOT`, `ESP_LAB_RAW_MODEL_ROOT`, `ESP_LAB_OBS_ROOT`, and `ESP_LAB_DATA_ROOT` environment variables, so a different NERSC account can run the workflow without editing notebook source.

### Diagnostic Data Layout (`S2D_DIAG_ROOT`)
The analysis archive follows an **experiment-first** and **observation-first** canonical structure:

```
<S2D_DIAG_ROOT>/
├── 4DEnVarOcn/          # E3SM S2D with 4DEnVar ocean initial conditions
├── JRA55_FOSIRL/        # E3SM S2D with JRA55/FOSIRL ocean/sea-ice ICs
├── Reanalysis/          # E3SM S2D reanalysis-initialized hindcasts (formerly BruteForce)
├── CESM-SMYLE/          # CESM-SMYLE benchmark hindcasts
├── NMME/                # Multi-model NMME SST benchmark runs
├── observations/        # Observational reference products (ERA5, HadISST, C3S, GPCP)
├── multimodel/          # Cross-experiment combined metrics and teleconnections
└── tmp/                 # File inventories, manifests, and workflow logs
```

### NetCDF File Naming Conventions
Within each experiment or multi-model directory, subdirectories partition datasets by diagnostic type using deterministic naming patterns:

| Diagnostic Area | Directory Path | File Naming Pattern & Example |
|:---|:---|:---|
| **Lead-Time Skill (seasonal)** | `<source>/leadtime_acc/skill/{atm,ocn,lnd}/{FIELD}/` | `{source}_init{MM}_{FIELD}_skill_y{start}-{end}_ny{N}_clim_{c0}_{c1}_l1-8[_detrend].nc`<br>*(e.g., `JRA55_FOSIRL_init05_PRECT_skill_y1980-2011_ny32_clim_1981_2010_l1-8_detrend.nc`)* |
| **Lead-Time Skill (monthly, 2a)** | `<source>/leadtime_acc/skill_monthly/{realm}/{FIELD}/` | `{source}_init{MM}_{FIELD}_monthly_skill_y{start}-{end}_clim{c0}-{c1}_l1-24.nc` |
| **Prepared inputs** | `<source>/leadtime_acc/prepared_skill/{realm}/{FIELD}/` | `{source}_init{MM}_{FIELD}_seasonal_anomaly_y{start}-{end}_ny{N}_clim_{c0}_{c1}_m{members}_l{leads}_{grid}_{units}.nc` |
| **Direct RMSE (1c)** | `<source>/leadtime_acc/comparison/{realm}/direct_rmse/{FIELD}/`<br>`<source>/leadtime_rmse/inputs/{atm,ocn}/{FIELD}/` | `{source}_{FIELD}_direct_rmse_init{MM}_years_{start}-{end}_ny{N}.nc`<br>`{source}_minus_{reference}_{FIELD}_direct_rmse_diff_init{MM}_years_{start}-{end}_ny{N}.nc` |
| **Teleconnections** | `<source>/leadtime_telec/`<br>`observations/leadtime_telec/` | `{source}_teleconnection_{index}_{VAR}_verify{start}_{end}_{grid}.nc`<br>*(e.g., `JRA55_FOSIRL_teleconnection_AMO_PSL_verify1981_2011_5x5deg.nc`, `observations_teleconnection_AMO_PSL_verify1981_2011_5x5deg.nc`)* |
| **Modes of Var** | `<source>/modes_variability/{fields,modes,regression_patterns,skill}/` | Skill: `{source}_init{MM}_{MODE}_eof_skill_eofref{y0}_{y1}_init{y0}_{y1}_clim{c0}_{c1}_..._detrend.nc` |
| **SST & ELI indices** | `<source>/sst_index/timeseries/`<br>`multimodel/eli/` | `{source}_init{MM}_TS_y{start}-{end}_N{members}_M{leads}_{INDEX}SST_{mon,seas}.nc`<br>`{source}_init{MM}_ELI[_native]_y{start}-{end}_N{members}_M{leads}[_seas].nc`, `eli_multimodel_skill_{start}_{end}.nc` |
| **Initial Shock** | `<source>/initial_shock/metrics/{realm}/{FIELD}/`<br>`multimodel/initial_shock/tables/` | `{source}_init{MM}_{start}_{end}.nc` *(e.g., `JRA55_FOSIRL_init05_1980_2011.nc`)* |
| **Tropical Cyclones** | `<source>/tc_track/` | `{source}_tc_lead_track_density[_enso_regression]_{case_prefix}_set3_{start}_{end}[_hadisst2].nc` |

Every file name starts with its folder's source name (`JRA55_FOSIRL`, `Reanalysis`, `4DEnVarOcn`, `CESM-SMYLE`, `NMME`, or an observation product such as `ERA5`), the initialization month is its own `_init{MM}_` token, and names carry no provenance hashes: provenance lives in NetCDF attributes and a stale cache is rebuilt in place.

### Figure Naming Conventions (`FIGURE_OUTDIR`)
All diagnostic plots are published into a unified web-accessible root (e.g., `/global/cfs/cdirs/e3sm/www/zhan391/e3sm-s2d_diag/`) with semantic, self-describing scientific naming:

| Diagnostic Group | Notebook | Figure Filename Pattern | Example Figures |
|:---|:---:|:---|:---|
| **Atmospheric / Ocean ACC** | `1a_atm`, `1a_ocn` | `fig_{atm,ocn}_acc_{field}_acc[_{type}].png`, type ∈ `compare`, `difference`, `distribution`, `case_difference` | `fig_atm_acc_prect_acc.png`, `fig_ocn_acc_sss_acc_case_difference.png` |
| **Land ACC** | `1a_lnd` | `fig_lnd_acc_{field}_acc[_case_difference].png` | `fig_lnd_acc_tws_acc.png`, `fig_lnd_acc_h2osoi_acc_case_difference.png` |
| **Anomaly RMSE maps** | `1b_atm`, `1b_ocn` | `fig_{atm,ocn}_rmse_{field}_rmse_{global,conus,difference,case_difference}.png` | `fig_atm_rmse_prect_rmse_global.png`, `fig_ocn_rmse_ohc700_rmse_case_difference.png` |
| **Land normalized RMSE** | `1b_lnd` | `fig_lnd_rmse_{field}_rmse[_case_difference].png` | `fig_lnd_rmse_h2osoi_rmse.png` |
| **Direct RMSE comparison** | `1c_*` | `fig_rmse_compare_{field}_rmse_compare_{global,conus}.png`<br>`fig_rmse_compare_{field}_rmse_difference_compare_init{MM}.png` (atm/ocn), `..._rmse_difference_global.png` (land) | `fig_rmse_compare_prect_rmse_compare_global.png`, `fig_rmse_compare_prect_rmse_difference_compare_init05.png` |
| **Regional ACC / nRMSE** | `2a` | `fig_regional_acc_nrmse_{atm,ocn,lnd}_{field}_{region}.png` | `fig_regional_acc_nrmse_lnd_tws_global.png`, `fig_regional_acc_nrmse_ocn_sst_tropics.png` |
| **SST Indices** | `3a` | `fig_sst_index_{index}_{acc_skill,time_series}.png` | `fig_sst_index_nino3_4_acc_skill.png`, `fig_sst_index_roni_time_series.png` |
| **SST / MOV / ELI Teleconnections** | `3b`, `4b`, `5c` | `fig_teleconnection_{index}_{VAR}_{correlation_reference_comparison,summary,taylor_diagram}.png` | `fig_teleconnection_Nino34_TREFHT_summary.png`, `fig_teleconnection_NAO_PRECT_taylor_diagram.png` |
| **Modes of Var** | `4a` | `fig_mov_{mode}_{skill,pc_time_series,eof_patterns_year{1,2},global_teleconnection_patterns_init{MM}}.png` | `fig_mov_nam_skill.png`, `fig_mov_pdo_pc_time_series.png` |
| **ELI Diagnostics** | `5a`, `5b` | `fig_eli_{metric}.png` | `fig_eli_multimodel_acc_nrmse_skill.png`, `fig_eli_nmme_lead_time_benchmark.png` |
| **Initial Shock Evolution** | `6a` | `fig_shock_ts_{field}_{detail}.png` | `fig_shock_ts_prect_init05-init11_1980_2011_seasonal_absolute_normalized_change.png` |
| **Initial Shock Error** | `6b` | `fig_shock_error_{field}_{detail}.png` | `fig_shock_error_trefht_init1980-2011_clim1981-2010_monthly_normalized_rmse.png` |
| **Tropical Cyclones** | `7a`, `7b` | `fig_tc_{metric}[_{case}_{start}].png` | `fig_tc_leadtime_track_density_compare.png`, `fig_tc_track_density_enso_regression_enabled_e3sm_1980_2011.png` |

These figures are automatically cataloged by [`8_run_viewer_webpage.ipynb`](jupyter/s2d_skill/8_run_viewer_webpage.ipynb) into `figures.json` and rendered into the interactive web viewer `index.html`.

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

Install from source in editable mode (`pip install -e .`): the notebooks import the `workflows` package and read the reference index files under `external/` from the repository checkout. `7a_tc_method_analysis.ipynb` additionally needs `global-land-mask` (included in `environment.yml`; otherwise `pip install global-land-mask`).

Machine-specific data and output roots default to the NERSC E3SM paths; override them with the `ESP_LAB_*` environment variables described under [Data Organization](#data-organization--output-conventions).

### PyPI Package
The `esp-lab` package on PyPI is the upstream CESM ESP-Lab release (1.1.x); it does not contain the E3SM S2D extensions, workflows, or notebooks described here.

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
