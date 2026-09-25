# Analysis Notebooks

The analysis suites live in the [`jupyter/`](https://github.com/zhangshixuan1987/ESP-Lab/tree/esp-dev/jupyter) directory of the repository: `s2d_skill/` (S2D prediction skill), `s2d_drift/` (drift and initialization), and `s2s_skill/` (S2S weekly skill).
These notebooks cover the full E3SM S2D diagnostic workflow from data preprocessing through
skill evaluation, modes-of-variability analysis, and interactive diagnostics viewing.

## S2D Skill Suite (`jupyter/s2d_skill/`)

The refactored `1a_*`, `1b_*`, and `1c_*` families separate ACC, anomaly-based RMSE maps, and direct raw-value RMSE comparisons, respectively, with `atm`, `lnd`, and `ocn` variants. Atmosphere and ocean workflows can compare against CESM-SMYLE; land workflows compare E3SM cases against a configured land reference. Land `1b` reuses or computes the same normalized-RMSE skill caches as land `1a`. Land `1c` requires an absolute-value reference: the C3S TWSA anomaly product used for `TWS` is suitable for `1a`/`1b`, so for `TWS` `1c` prepares its inputs but skips the direct-RMSE cells with a message.

Fields without a CESM-SMYLE benchmark (ocean `SSS` and `OHC700`, and all land fields) show the E3SM cases only. Every `1a` and `1b` notebook also draws a **case-difference** figure: each E3SM experiment minus a selected control case (`ACC_DIFFERENCE_CONTROL` / `RMSE_DIFFERENCE_CONTROL`, default `E3SM-Reanalysis`).

| Notebook | Focus Area | Description |
|---|---|---|
| `0_run_cesm_smyle_benchmark.ipynb` | Benchmark Data | Dask-distributed preprocessing of CESM-SMYLE hindcasts |
| `1a_atm_leadtime_acc_skill_map.ipynb` | Atmospheric Skill | Lead-time anomaly correlation coefficient (ACC) maps |
| `1a_lnd_leadtime_acc_skill_map.ipynb` | Land Skill | Land lead-time ACC maps (H2OSNO, H2OSOI, TWS) |
| `1a_ocn_leadtime_acc_skill_map.ipynb` | Ocean Skill | Ocean lead-time ACC maps (SST, SSS, OHC700) |
| `1b_atm_leadtime_rmse_skill_map.ipynb` | Error Maps | Anomaly RMSE skill maps (atmosphere: TREFHT, TS, PRECT, PSL) |
| `1b_ocn_leadtime_rmse_skill_map.ipynb` | Error Maps | Anomaly RMSE skill maps (ocean: SST, SSS, OHC700) |
| `1b_lnd_leadtime_rmse_skill_map.ipynb` | Error Maps | Normalized RMSE skill maps (land: H2OSNO, H2OSOI, TWS) |
| `1c_atm_leadtime_rmse_compare.ipynb` | Model Comparison | Direct (bias-inclusive) RMSE and significance-tested model differences (atmosphere) |
| `1c_ocn_leadtime_rmse_compare.ipynb` | Model Comparison | Direct (bias-inclusive) RMSE and significance-tested model differences (ocean: SST, SSS, OHC700) |
| `1c_lnd_leadtime_rmse_compare.ipynb` | Model Comparison | Direct-RMSE comparison between E3SM land cases (absolute-value reference required; skipped for TWS) |
| `2a_regional_acc_skill_ts.ipynb` | Regional Skill | Global and regional ACC and nRMSE versus lead (seasonal from 1a, plus monthly all-start curves) for all atmosphere, ocean, and land fields |
| `3a_sst_skill_ts.ipynb` | Ocean Skill | SST index skill time series (E3SM, CESM-SMYLE, NMME) |
| `3b_sst_telecon.ipynb` | Teleconnections | Sea surface temperature teleconnection diagnostics |
| `4a_mov_analysis.ipynb` | Modes of Variability | EOF projection and index calculation (PDO, AMO, NAO) |
| `4b_mov_telecon.ipynb` | Teleconnections | Modes of variability climate teleconnection patterns |
| `5a_eli_skill_ts.ipynb` | Tropical Pacific | Equatorial Longitude Index (ELI) skill time series |
| `5b_eli_diagnostics.ipynb` | ELI Diagnostics | Native & regridded ELI diagnostics across starts |
| `5c_eli_telecon.ipynb` | Teleconnections | ELI precipitation and temperature teleconnections |
| `6a_shock_ts.ipynb` | Initialization Shock | Lead-dependent drift and initialization shock time series |
| `6b_shock_index.ipynb` | Shock Indices | Initialization shock metrics and multi-model indices |
| `7a_tc_method_analysis.ipynb` | Tropical Cyclones | TempestExtremes tracking method and parameter comparison |
| `7b_tc_leadtime_analysis.ipynb` | Tropical Cyclones | TC lead-time density, IBTrACS comparison, and ENSO regression |
| `8_run_viewer_webpage.ipynb` | Gallery Webpage | Interactive HTML diagnostics viewer generator |

## S2D Drift & Initialization Suite (`jupyter/s2d_drift/`)

| Notebook | Focus Area | Description |
|---|---|---|
| `0_refactor_ic_analysis.ipynb` | IC Audit | 6-step initial-condition audit: hash, structure, stats, plots, consistency, campaign summary (`workflows/diagnostics/*` IC modules, `ic_config.yaml`) |
| `1a_refactor_drift_map.ipynb` | Drift Maps | Global two-reference distance-change maps (obs & model attractor) |
| `1b_refactor_drift_region.ipynb` | Regional Drift | Lead-time drift trajectories and RMSE/spread curves by region |
| `1c_refactor_drift_regime.ipynb` | Drift Regimes | Fraction of area in each drift regime (converging vs drifting) |
| `1d_refactor_drift_skill_relationship.ipynb` | Drift–Skill | Early drift vs later skill attribution (Methods 1 & 2 with bootstrap) |
| `1e_refactor_drift_summary.ipynb` | S2D Summary | Integrated bias/drift/spread, ENSO frequency, and scorecards |
| `2a_refactor_physical_consistency.ipynb` | Physical Consistency | Flux partitioning, land coupling, ocean coupling, energy & water budget |
| `2b_refactor_unified_diagnostics.ipynb` | Unified S2D | Master orchestrator: field drift, physical consistency, model attractor, IC attribution |

## S2S Weekly Skill Suite (`jupyter/s2s_skill/`)

| Notebook | Focus Area | Description |
|---|---|---|
| `1a_atm_leadtime_acc_skill_map.ipynb` | Atmospheric Skill | Weekly (weeks 1–8) atmospheric ACC skill maps |
| `1b_lnd_leadtime_acc_skill_map.ipynb` | Land Skill | Weekly land and hydrologic ACC skill maps |
| `2a_leadtime_rmse_skill_map.ipynb` | Error Maps | Weekly RMSE and bias skill maps |
| `2b_leadtime_rmse_compare.ipynb` | Model Comparison | Multi-experiment weekly skill and RMSE comparison |
| `3a_s2s_telecon_modes.ipynb` | Teleconnections | Subseasonal NAO, PNA, and AO prediction |
| `4a_soil_moisture_memory.ipynb` | Land Memory | Soil-moisture memory and land–atmosphere coupling |

## Interactive Web Portal

All figures generated by this suite can be explored on the [Live E3SM-S2D Diagnostics Portal](https://portal.nersc.gov/cfs/e3sm/zhan391/e3sm-s2d_diag/) (see the [Diagnostics Gallery](../gallery.md) for an overview and example figures), featuring a one-page button matrix (one row per field or index, one button per figure type), teleconnection driver-mode filtering, and pop-out image modals.

## How to Run

1. Clone the repository:
   ```bash
   git clone -b e3sm-esp https://github.com/zhangshixuan1987/ESP-Lab.git
   cd ESP-Lab
   ```
2. Activate or create the conda environment (see [Installation](../how-to/install-esp-lab.md)).
3. Launch JupyterLab or JupyterHub and navigate to the suite folder under `jupyter/` (e.g. `jupyter/s2d_skill/`).
4. In `jupyter/s2d_skill/`, run notebooks in order (0 → 8). Choose the field, index, or mode in the configuration cell at the top of each notebook; data and output roots come from `esp_lab.env_paths` (override with the `ESP_LAB_*` environment variables).
5. Each notebook's run-control block (`derivation_mode = "auto" | "rebuild" | "require"` and `recompute_*` flags) decides whether caches are reused or recomputed.
6. Run `8_run_viewer_webpage.ipynb` last to rebuild the web gallery.
