#!/bin/bash
#SBATCH --job-name=esp_p6_render
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
set -eo pipefail

: "${ESP_LAB_ROOT:?Set ESP_LAB_ROOT to this ESP-Lab checkout.}"
: "${E3SM_ENV_SCRIPT:?Set E3SM_ENV_SCRIPT to the site Python environment loader.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the independent-model output root.}"
: "${P6_SOURCE_ROOT:?Set P6_SOURCE_ROOT to E3SM/SMYLE/HadISST2 source products.}"
source "${E3SM_ENV_SCRIPT}"
cd "${ESP_LAB_ROOT}"
python - <<'PY'
import os
from pathlib import Path
import xarray as xr
from workflows.modes_of_variability.cross_source import build_ensemble_spread_common_skill_cache, plot_ensemble_spread_native_skill_figure

source_root = Path(os.environ['P6_SOURCE_ROOT'])
output_root = Path(os.environ['OUTPUT_ROOT'])
models_root = output_root / 'nmme_models'
models = sorted(path.name for path in models_root.iterdir() if path.is_dir())
if not models:
    raise RuntimeError(f'No independent NMME model products under {models_root}')
for mode in ('PDO', 'NPGO', 'AMO'):
    lower = mode.lower(); products = {}
    for label, directory, stem in (('E3SM', 'e3sm', 'e3sm'), ('CESM-SMYLE', 'CESM-SMYLE', 'smyle')):
        with xr.open_dataset(source_root / directory / 'modes_variability/modes' / lower / 'indices' / f'{stem}_init11_{lower}.nc') as ds:
            products[label] = ds.load()
    members = []
    for model in models:
        with xr.open_dataset(models_root / model / 'NMME/modes_variability/modes' / lower / 'indices' / f'nmme_init11_{lower}.nc') as ds:
            members.append(ds.load())
    all_nmme = xr.concat(members, dim='M', join='outer', combine_attrs='drop_conflicts')
    all_nmme['valid_time'] = members[0]['valid_time']; all_nmme['target_month'] = members[0]['target_month']
    with xr.open_dataset(source_root / 'HadISST2/modes_variability/modes' / lower / 'indices' / f'hadisst2_{lower}_reference.nc') as ds:
        observation = ds['mode_index'].load()
    for token, label, nmme in (('single_nmme_cmc1', 'NMME CMC1', members[models.index('CMC1-CanCM3')]), ('all_nmme', 'NMME', all_nmme)):
        cache = build_ensemble_spread_common_skill_cache({**products, label: nmme}, observation, model_ensemble_sources=(label,))
        cache.attrs.update(nmme_input_contract='independent_per_model_fixed_hadisst2_projection', comparison_period='common_cohort_1991_2009_initializations')
        outdir = output_root / token / lower; path = outdir / 'diagnostics' / f'{lower}_init11_common_ensemble_spread_skill.nc'
        path.parent.mkdir(parents=True, exist_ok=True); cache.to_netcdf(path)
        plot_ensemble_spread_native_skill_figure(cache, outdir / 'figures' / f'{lower}_init11_common_ensemble_spread_skill.png', mode=mode)
PY
