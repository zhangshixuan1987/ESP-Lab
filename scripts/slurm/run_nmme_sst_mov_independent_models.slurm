#!/bin/bash
#SBATCH --job-name=esp_nmme_sst_mov
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-22
set -eo pipefail

: "${ESP_LAB_ROOT:?Set ESP_LAB_ROOT to this ESP-Lab checkout.}"
: "${E3SM_ENV_SCRIPT:?Set E3SM_ENV_SCRIPT to the site Python environment loader.}"
: "${NMME_ROOT:?Set NMME_ROOT to the NMME archive root.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a durable output directory.}"
MODELS=(CMC1-CanCM3 CMC2-CanCM4 COLA-RSMAS-CCSM3 COLA-RSMAS-CCSM4 COLA-RSMAS-CESM1 CanCM4i CanSIPS-IC3 CanSIPS-IC4 CanSIPSv2 GEM-NEMO GFDL-CM2p1 GFDL-CM2p1-aer04 GFDL-CM2p5-FLOR-A06 GFDL-CM2p5-FLOR-B01 GFDL-SPEAR IRI-ECHAM4p5-AnomalyCoupled IRI-ECHAM4p5-DirectCoupled NASA-GEOSS2S NASA-GMAO NASA-GMAO-062012 NCAR-CESM1 NCEP-CFSv1 NCEP-CFSv2)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
STAGE_ROOT="${TMPDIR:-/tmp}/${USER}/esp_nmme_sst_mov_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
FINAL_ROOT="${OUTPUT_ROOT}/nmme_models/${MODEL}"

source "${E3SM_ENV_SCRIPT}"
mkdir -p "${STAGE_ROOT}/fixed" "${FINAL_ROOT}"
cd "${ESP_LAB_ROOT}"
python scripts/run_process_modes_of_variability.py \
  --outdir "${STAGE_ROOT}" --modes PDO NPGO AMO --sources obs nmme \
  --init-months 11 --start-year 1991 --end-year 2009 --clim-start 1991 --clim-end 2009 \
  --obs-start-year 1950 --obs-end-year 2020 --monthly-nlead 12 \
  --target-dlat 2.5 --target-dlon 2.5 --dask-workers 2 --nmme-root "${NMME_ROOT}" \
  --nmme-models "${MODEL}" --nmme-fixed-dir "${STAGE_ROOT}/fixed" \
  --eof-strategy fixed_obs_projection --eof-reference-source obs \
  --eof-reference-start-year 1950 --eof-reference-end-year 2020 --force
rm -rf "${FINAL_ROOT}/NMME" "${FINAL_ROOT}/_manifests"
cp -a "${STAGE_ROOT}/NMME" "${STAGE_ROOT}/_manifests" "${FINAL_ROOT}/"
