#!/bin/bash
#SBATCH --job-name=esp_nmme_test
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=cpu
#SBATCH --qos=regular
# -----------------------------------------------------------------------------
# NMME vs E3SM mode-skill test with the merged ESP-Lab code.
#
# Submit from a login node (this submits both steps):
#   export ESP_LAB_ROOT=/path/to/ESP-Lab          # checkout to test
#   export OUTPUT_ROOT=/path/to/your/test/output  # everything is written here
#   bash scripts/slurm/nmme_cross_source_test.slurm
#
# Step 1 (array, one task per NMME model): run_process_modes_of_variability.py
#   builds each model's mode indices, with that model's own drift removed and
#   projected onto the fixed HadISST2/ERA5 EOF.
# Step 2 (after step 1): render_nmme_comparison.py scores FOSIRL, Reanalysis
#   and NMME over the NMME start years and draws the figures: NMME line = skill
#   of the equal-weight multi-model forecast, band = min-max across models.
#   It runs twice: the default cohort (NMME_YEARS, models covering every start)
#   and the robustness check (CHECK_YEARS, default 1991-2009 = all 23 models).
#
# E3SM products and the observed reference are read from DIAG_ROOT and never
# written. Optional settings (export before submitting):
#   MODES="PDO NPGO AMO"   INIT_MONTHS="5 11"
#   NMME_YEARS="1982 2010" CHECK_YEARS="1991 2009" (empty = skip the check)
#   NMME_MODELS="auto"     (or a list of model folder names)
#   SINGLE_MODEL=""        (e.g. CMC1-CanCM3: also draw that model alone)
#   NMME_ROOT, DIAG_ROOT, PYTHON (default: python on PATH),
#   E3SM_ENV_SCRIPT (sourced first when set), SBATCH_ACCOUNT (your allocation)
# -----------------------------------------------------------------------------
set -eo pipefail

: "${ESP_LAB_ROOT:?Set ESP_LAB_ROOT to the ESP-Lab checkout to test.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a test output directory.}"
NMME_ROOT="${NMME_ROOT:-/global/cfs/cdirs/e3sm/S2S2D/NMME/data_hindcast_by_member}"
DIAG_ROOT="${DIAG_ROOT:-/global/cfs/cdirs/e3sm/S2S2D/s2d_diag}"
MODES="${MODES:-PDO NPGO AMO}"
INIT_MONTHS="${INIT_MONTHS:-5 11}"
NMME_YEARS="${NMME_YEARS:-1982 2010}"
CHECK_YEARS="${CHECK_YEARS-1991 2009}"
NMME_MODELS="${NMME_MODELS:-auto}"
SINGLE_MODEL="${SINGLE_MODEL:-}"
PYTHON="${PYTHON:-python}"
export ESP_LAB_ROOT OUTPUT_ROOT NMME_ROOT DIAG_ROOT MODES INIT_MONTHS NMME_YEARS CHECK_YEARS NMME_MODELS SINGLE_MODEL PYTHON

if [[ "${NMME_MODELS}" == "auto" ]]; then
  MODEL_LIST=$(cd "${NMME_ROOT}" && for d in */sst; do echo "${d%/sst}"; done | sort | tr '\n' ' ')
else
  MODEL_LIST="${NMME_MODELS}"
fi
read -r -a MODEL_ARRAY <<< "${MODEL_LIST}"

# ---- Submit both steps when run with bash from a login node -----------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  process=$(sbatch --parsable --array=0-$((${#MODEL_ARRAY[@]} - 1)) \
    --output="${OUTPUT_ROOT}/logs/process_%A_%a.out" --export=ALL,STEP=process "${script}")
  render=$(sbatch --parsable --dependency=afterany:"${process}" --time=01:00:00 \
    --output="${OUTPUT_ROOT}/logs/render_%j.out" --export=ALL,STEP=render "${script}")
  echo "Submitted ${#MODEL_ARRAY[@]} model tasks as job ${process}; render job ${render}."
  echo "Output: ${OUTPUT_ROOT}"
  exit 0
fi

if [[ -n "${E3SM_ENV_SCRIPT:-}" ]]; then source "${E3SM_ENV_SCRIPT}"; fi
cd "${ESP_LAB_ROOT}"
export PYTHONPATH="${ESP_LAB_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HDF5_USE_FILE_LOCKING=FALSE

if [[ "${STEP}" == "process" ]]; then
  MODEL="${MODEL_ARRAY[${SLURM_ARRAY_TASK_ID}]}"
  OUT="${OUTPUT_ROOT}/nmme_models/${MODEL}"
  mkdir -p "${OUT}" "${OUTPUT_ROOT}/fixed"
  for mode in ${MODES}; do
    case "${mode}" in
      PDO|NPGO|AMO) obs="1950 2020"; field=sst ;;
      *)            obs="1979 2019"; field=prmsl ;;
    esac
    if [[ ! -d "${NMME_ROOT}/${MODEL}/${field}" ]]; then
      echo "${MODEL}: no ${field} in the archive, skipping ${mode}"; continue
    fi
    read -r obs0 obs1 <<< "${obs}"
    "${PYTHON}" scripts/run_process_modes_of_variability.py \
      --outdir "${OUT}" --modes "${mode}" --sources obs nmme \
      --init-months ${INIT_MONTHS} --start-year 1980 --end-year 2020 \
      --clim-start 1981 --clim-end 2010 \
      --obs-start-year "${obs0}" --obs-end-year "${obs1}" \
      --eof-reference-start-year "${obs0}" --eof-reference-end-year "${obs1}" \
      --monthly-nlead 12 --target-dlat 5 --target-dlon 5 --regrid-method conservative \
      --dask-workers "${SLURM_CPUS_PER_TASK:-4}" --eof-strategy fixed_obs_projection \
      --eof-bootstrap-iterations 0 \
      --nmme-root "${NMME_ROOT}" --nmme-models "${MODEL}" --nmme-fixed-dir "${OUTPUT_ROOT}/fixed"
  done
  exit 0
fi

if [[ "${STEP}" == "render" ]]; then
  roots=("${OUTPUT_ROOT}"/nmme_models/*)
  models_arg=()
  [[ "${NMME_MODELS}" != "auto" ]] && models_arg=(--models ${NMME_MODELS})
  run_render() {
    "${PYTHON}" scripts/render_nmme_comparison.py --diag-root "${DIAG_ROOT}" \
      --nmme-root "${roots[@]}" --modes ${MODES} --init-months ${INIT_MONTHS} "$@" || true
  }
  read -r y0 y1 <<< "${NMME_YEARS}"
  run_render --years "${y0}" "${y1}" "${models_arg[@]}" --outdir "${OUTPUT_ROOT}/figures/nmme_${y0}_${y1}"
  if [[ -n "${SINGLE_MODEL}" ]]; then
    run_render --years "${y0}" "${y1}" --single-model "${SINGLE_MODEL}" \
      --outdir "${OUTPUT_ROOT}/figures/nmme_${y0}_${y1}_${SINGLE_MODEL}"
  fi
  if [[ -n "${CHECK_YEARS}" ]]; then
    read -r c0 c1 <<< "${CHECK_YEARS}"
    run_render --years "${c0}" "${c1}" --outdir "${OUTPUT_ROOT}/figures/check_nmme_${c0}_${c1}"
  fi
  exit 0
fi

echo "Unknown STEP=${STEP}" >&2
exit 1
