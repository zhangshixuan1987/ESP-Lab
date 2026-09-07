"""Archive-backed preparation and cache planning for the initial-shock notebook."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from esp_lab import data_access_e3sm as e3sm_access
from esp_lab import data_access_cesm_smyle as smyle_access
from esp_lab import data_access_obs as obs_access
from esp_lab.diagnostics.initial_shock import VERSION, align_observation_months, compute_initial_shock_index
from esp_lab.paths import diagnostic_dir
from esp_lab.utils.netcdf_utils import atomic_to_netcdf, load_netcdf

ARCHIVE_VERSION = 'initial_shock_archive_v1'


def _inventory(paths):
    result = []
    for name in sorted(set(map(str, paths))):
        path = Path(name).resolve(strict=True)
        stat = path.stat()
        result.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return result


def _cache_valid(path, digest):
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path) as ds:
            required = {'std_ratio', 'model_index', 'observation_index', 'paired_sample_count',
                        'model_area_fraction', 'observation_area_fraction', 'block_start_time', 'block_end_time'}
            return ds.attrs.get('identity_sha256') == digest and required <= set(ds.variables)
    except (OSError, ValueError):
        return False


def plan_archive_run(settings, cases, variable):
    """Inventory every requested case/month before any expensive preparation.

    Incomplete E3SM ensembles/initializations raise rather than shrinking the
    cohort. Benchmark files are monthly, never seasonal or drift-corrected.
    Returns a serializable plan, including cache decisions and exact sources.
    """
    years = list(range(settings['run']['years'][0], settings['run']['years'][1] + 1))
    months = list(settings['run']['init_months'])
    if not years or not months or len(set(months)) != len(months) or any(m not in range(1, 13) for m in months):
        raise ValueError('Specify nonempty years and unique initialization months in 1..12')
    if not cases and not settings['smyle']['include']:
        raise ValueError('Select at least one model')
    mode = settings['cache']['mode']
    if mode not in {'auto', 'rebuild', 'require'}:
        raise ValueError('cache.mode must be auto, rebuild, or require')
    nlead = settings['run']['nlead']
    metric = settings['metric']
    if metric['start_lead'] + metric['window_months'] > nlead:
        raise ValueError('Metric window exceeds forecast length')
    if metric['window_months'] % metric['block_months'] or metric['window_months'] < 2 * metric['block_months']:
        raise ValueError('Metric window must contain at least two complete blocks')
    obs_path = obs_access.find_obs_file(
        obs_dir=settings['obs']['data_dir'], product=variable['obs_product'],
        field=variable['obs_variable'], filename=variable.get('obs_filename'),
    )
    tasks = []
    members = [f'EN{i:02d}' for i in range(settings['e3sm']['nens'])]
    for month in months:
        tags = e3sm_access.build_init_tags(years, month)
        for case, info in cases.items():
            loader = dict(data_dir=settings['e3sm']['data_dir'], case_prefix=info['case_prefix'],
                          members=members, init_tags=tags, field=variable['model_variable'],
                          realm='atm', grid=settings['e3sm']['grid'], freq='monthly',
                          ts_split=settings['e3sm']['ts_split'], require_all_members=True,
                          verify_field_name=True, verify_coverage=True, nlead=nlead)
            files, valid = e3sm_access.nested_file_list_by_init(**loader)
            if valid != tags:
                raise FileNotFoundError(f'{case} init {month:02d}: incomplete initializations {sorted(set(tags)-set(valid))}')
            tasks.append(dict(case=case, source=info['cache_tag'], month=month, kind='e3sm',
                              loader=loader, model_paths=[p for row in files for p in row]))
        if settings['smyle']['include']:
            loader = dict(field=variable['smyle_variable'], init_month=month,
                          benchmark_dir=settings['smyle']['benchmark_dir'],
                          nens=settings['smyle']['nens'], nlead=nlead, freq='mon')
            path = smyle_access.benchmark_path(**loader)
            with xr.open_dataset(path) as ds:
                missing = sorted(set(tags)-set(map(str, ds.Y.values)))
                if missing or ds.sizes.get('M') != loader['nens'] or ds.sizes.get('L') != nlead:
                    raise ValueError(f'CESM-SMYLE {month:02d}: incomplete years/members/leads; missing={missing}')
            tasks.append(dict(case='CESM-SMYLE', source='CESM-SMYLE', month=month,
                              kind='smyle', loader=loader, model_paths=[str(path)]))
    # Include local adapter code so changes to time normalization also invalidate caches.
    code_paths = [__file__, e3sm_access.__file__, smyle_access.__file__, obs_access.__file__]
    from esp_lab.diagnostics import initial_shock
    code_paths.append(initial_shock.__file__)
    code_identity = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code_paths}
    for task in tasks:
        task['years'] = years
        task['obs_path'] = obs_path
        task['inventory'] = _inventory(task['model_paths'] + [obs_path])
        payload = dict(task=task.copy(), settings=settings, variable=variable,
                       algorithm=VERSION, archive_algorithm=ARCHIVE_VERSION, code=code_identity)
        task['provenance_json'] = json.dumps(payload, sort_keys=True)
        identity = dict(payload)
        identity['settings'] = {k: v for k, v in settings.items() if k not in {'cache', 'dask', 'paths'}}
        task['digest'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        task['path'] = str(diagnostic_dir(task['source'], 'initial_shock', 'metrics', 'atm',
                                        variable['field'], root=settings['paths']['s2d_diag_root']) /
                           f"init{task['month']:02d}_{task['digest'][:20]}.nc")
        task['cached'] = _cache_valid(Path(task['path']), task['digest'])
        task['rebuild'] = mode == 'rebuild' or not task['cached']
    if mode == 'require' and any(t['rebuild'] for t in tasks):
        raise FileNotFoundError('Required initial-shock caches missing: ' + ', '.join(t['path'] for t in tasks if t['rebuild']))
    return tasks


def _remap(ds, field, target, settings):
    from esp_lab.utils import regrid_utils as regrid
    ready = regrid.prepare_latlon_ds(ds)
    if ready.lat.identical(target.lat) and ready.lon.identical(target.lon):
        return ready[field]
    mapper = regrid.make_regridder(ready, target, method=settings['method'], periodic=settings['periodic'])
    return mapper(ready[field])


def compute_archive_plan(plan, settings, variable):
    """Open, normalize, remap, compute and cache one case/month at a time.

    No multi-GB regridded monthly caches are created: only global block indices,
    coverage and ratios are written. Lazy source datasets are always closed.
    """
    from esp_lab.utils import regrid_utils as regrid
    target = regrid.make_latlon_grid(dlat=settings['regrid']['target_dlat'],
                                    dlon=settings['regrid']['target_dlon'])
    by_month = {}
    for task in plan:
        frozen = json.loads(task['provenance_json'])
        if frozen['settings'] != json.loads(json.dumps(settings)) or frozen['variable'] != variable:
            raise ValueError('Configuration changed after planning; rerun the inventory cell')
        if _inventory(task['model_paths'] + [task['obs_path']]) != task['inventory']:
            raise RuntimeError('Source files changed after planning; rerun the inventory cell')
        path = Path(task['path'])
        if settings['cache']['mode'] != 'rebuild' and _cache_valid(path, task['digest']):
            result = load_netcdf(path)
            print(f"Reusing {task['case']} init {task['month']:02d}: {path}")
        else:
            if settings['cache']['mode'] == 'require':
                raise FileNotFoundError(path)
            print(f"Preparing {task['case']} init {task['month']:02d}")
            with ExitStack() as stack:
                if task['kind'] == 'e3sm':
                    model_ds = stack.enter_context(e3sm_access.get_monthly_data(
                        **task['loader'], chunks=settings['e3sm']['chunks'], engine=settings['e3sm']['engine']))
                    name = variable['model_variable']
                    scale, offset = variable['model_scale'], variable['model_offset']
                else:
                    opened = stack.enter_context(smyle_access.load_benchmark(**task['loader'], chunks=settings['smyle']['chunks']))
                    tags = smyle_access.build_init_tags(task['years'], task['month'])
                    model_ds = opened.sel(Y=tags)
                    name = variable['smyle_variable']
                    scale, offset = variable['smyle_scale'], variable['smyle_offset']
                obs_ds = stack.enter_context(obs_access.get_monthly_data(
                    obs_dir=settings['obs']['data_dir'], product=variable['obs_product'],
                    field=variable['field'], field_map={variable['field']: variable['obs_variable']},
                    filename=Path(task['obs_path']).name, chunks=settings['obs']['chunks'],
                    start_year=str(task['years'][0]), end_year=str(task['years'][-1] + (settings['run']['nlead'] + 11)//12 + 1)))
                expected_members = settings['e3sm']['nens'] if task['kind'] == 'e3sm' else settings['smyle']['nens']
                if model_ds.sizes.get('M') != expected_members or model_ds.sizes.get('L') != settings['run']['nlead']:
                    raise ValueError('Loaded member/lead count differs from requested configuration')
                tags = e3sm_access.build_init_tags(task['years'], task['month'])
                if list(map(str, model_ds.Y.values)) != tags:
                    raise ValueError('Loaded initializations differ from the requested cohort')
                valid_time = model_ds.time.transpose('Y', 'L').assign_coords(Y=task['years'])
                expected = np.array(task['years'])[:, None] * 12 + task['month'] + np.arange(settings['run']['nlead'])[None, :]
                actual = np.asarray(valid_time.dt.year * 12 + valid_time.dt.month)
                if not np.array_equal(actual, expected):
                    raise ValueError('Model verification time differs from represented initialization + lead months')
                model = _remap(model_ds, name, target, settings['regrid']) * scale + offset
                model = model.assign_coords(Y=task['years'], verification_time=valid_time)
                # Both adapters return the resolved observation variable name.
                obs_name = variable['obs_variable'] if variable['obs_variable'] in obs_ds else variable['field']
                obs = _remap(obs_ds, obs_name, target, settings['regrid']) * variable['obs_scale'] + variable['obs_offset']
                model.attrs['units'] = obs.attrs['units'] = variable['units']
                aligned = align_observation_months(obs, valid_time)
                result = compute_initial_shock_index(model, aligned, area=target.area,
                                                     **settings['metric']).compute()
                if _inventory(task['model_paths'] + [task['obs_path']]) != task['inventory']:
                    raise RuntimeError('Source files changed during computation; result not cached')
                result.attrs.update(case=task['case'], field=variable['field'], init_month=task['month'],
                                    identity_sha256=task['digest'], provenance_json=task['provenance_json'])
                atomic_to_netcdf(result, path)
                print(f'Wrote {path}')
        by_month.setdefault(task['month'], []).append(result.expand_dims(case=[task['case']]))
    return {month: xr.concat(items, dim='case', join='exact', combine_attrs='drop_conflicts')
            for month, items in by_month.items()}
