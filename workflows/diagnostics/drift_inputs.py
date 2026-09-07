"""On-demand inputs for the two-reference drift analysis notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from workflows.diagnostics import two_reference_drift as workflow
from esp_lab.diagnostics.two_reference_drift import (
    area_weighted_mean, regional_subset, compute_regime_fraction,
    run_pipeline,
)

SOURCES = ('JRA55_FOSIRL', 'Reanalysis')
CASE_PREFIXES = {
    'Reanalysis': 'WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_BruteForce',
    'JRA55_FOSIRL': 'WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL',
}
REGIONS = {
    'Nino3.4': {'lon_bounds': (190, 240), 'lat_bounds': (-5, 5)},
    'North_Atlantic': {'lon_bounds': (280, 359.999), 'lat_bounds': (0, 60)},
    'Global_land': {'lon_bounds': (0, 359.999), 'lat_bounds': (-90, 90)},
}


def default_variable_config():
    ARCHIVE_ROOT = Path('/global/cfs/cdirs/e3sm/S2S2D/E3SMLE')
    OBS_ROOT = Path('/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series')
    LAND_OBS_ROOT = Path('/global/cfs/cdirs/e3sm/zhan391/data/CPC_SOM/monthly')
    VARIABLE_CONFIG = {
        'TREFHT': {
            'component': 'atm',
            'hindcast_variable': 'TREFHT',
            'observation_path': OBS_ROOT / 'ERA5/tas_197901_201912.nc',
            'observation_variable': 'tas',
            'historical_variable': 'TREFHT',
        },
        'SST': {
            'component': 'ocn',
            'hindcast_variable': 'timeMonthly_avg_activeTracers_temperature',
            'observation_path': OBS_ROOT / 'HadISST2/sst_186901_202212.nc',
            'observation_variable': 'sst',
            'historical_variable': 'timeMonthly_avg_surface_temperature',
        },
        'PSL': {
            'component': 'atm',
            'hindcast_variable': 'PSL',
            'observation_path': OBS_ROOT / 'ERA5/psl_197901_201912.nc',
            'observation_variable': 'psl',
            'historical_variable': 'PSL',
        },
        'PRECT': {
            'component': 'atm',
            'hindcast_variable': 'PRECT',
            'observation_path': OBS_ROOT / 'GPCP_v2.3/PRECT_197901_201712.nc',
            'observation_variable': 'PRECT',
            'historical_variable': 'PRECT',
            'analysis_years': (1980, 2015),
        },
        'H2OSOI': {
            'component': 'lnd',
            'hindcast_variable': 'H2OSOI',
            'observation_paths': sorted(LAND_OBS_ROOT.glob('soilw_*.nc')),
            'observation_variable': 'soilw',
            'historical_variable': 'H2OSOI',
            'include_spread': False,
            'depth_range_m': (0.0, 1.6),
        },
    }
    
    for variable, config in VARIABLE_CONFIG.items():
        component = config['component']
        archive_variable = config['historical_variable']
        relative = Path(component) / 'ts/monthly/180x360_aave/1yr'
        config['mean_paths'] = sorted((ARCHIVE_ROOT / 'ensmean/post' / relative).glob(f'{archive_variable}_*.nc'))
        config['spread_paths'] = (
            sorted((ARCHIVE_ROOT / 'ensspread/post' / relative).glob(f'{archive_variable}_*.nc'))
            if config.get('include_spread', True) else []
        )
        if 'observation_paths' not in config:
            config['observation_paths'] = [config['observation_path']]
    return VARIABLE_CONFIG


def _check_mode(mode):
    if mode not in {'auto', 'require', 'rebuild'}:
        raise ValueError('DRIFT_INPUT_MODE must be auto, require, or rebuild')


def ensure_reference(variable, init_month, source, *, mode='auto',
                     analysis_years=(1980, 2017), climatology_years=(1981, 2010),
                     variable_config=None,
                     data_root='/global/cfs/cdirs/e3sm/S2S2D/post_process',
                     members=tuple(f'EN{i:02d}' for i in range(10))):
    """Ensure a monthly hindcast and its observation/attractor reference pair."""
    _check_mode(mode)
    config = (variable_config or default_variable_config())[variable]
    job = dict(source=source, variable=variable, component=config['component'],
               init_month=int(init_month), hindcast_variable=config['hindcast_variable'], regrid=True)
    try:
        hindcast = workflow.discover_monthly_hindcast(job)
    except FileNotFoundError:
        if mode == 'require':
            raise
        hindcast = workflow.prepare_monthly_hindcast_cache(
            job, case_prefix=CASE_PREFIXES[source],
            init_years=list(range(analysis_years[0], analysis_years[1] + 1)),
            members=members, data_root=data_root, lead_count=24,
            chunks={'Y': 3, 'L': 24, 'M': 2, 'lat': 90, 'lon': 180}, force=True,
        )
    path = workflow.reference_output_path(job)
    reason = 'forced reference rebuild'
    if mode != 'rebuild':
        try:
            with workflow.load_drift_references(
                source, init_month, variable, component=config['component'], path=path,
                require_attractor_spread=config.get('include_spread', True),
            ) as existing:
                expected = {
                    'attractor_climatology_period': f'{climatology_years[0]}-{climatology_years[1]}',
                    'analysis_start_year': analysis_years[0],
                    'analysis_end_year': analysis_years[1],
                }
                if any(existing.attrs.get(k) != v for k, v in expected.items()):
                    raise ValueError('Reference analysis/climatology years changed')
                with xr.open_dataset(hindcast, chunks={}) as monthly:
                    time = workflow.select_initialization_years(monthly.time, analysis_years)
                    for name in ('Y', 'L', 'lat', 'lon'):
                        xr.testing.assert_equal(existing[name], time[name] if name in ('Y', 'L') else monthly[name])
                    np.testing.assert_array_equal(existing.valid_time.values, time.values)
                return path
        except (OSError, ValueError, KeyError, AssertionError) as error:
            reason = str(error)
    if mode == 'require':
        raise FileNotFoundError(f'Missing or stale drift reference {path}: {reason}')
    print(f'[drift inputs] Preparing {variable}/{source}/init{init_month:02d}: {reason}')
    product = workflow.prepare_reference_product(
        job, observation_path=config['observation_paths'], historical_path=config['mean_paths'],
        observation_variable=config['observation_variable'], historical_variable=config['historical_variable'],
        spread_path=config['spread_paths'] or None, spread_variable=config['historical_variable'],
        climatology_years=climatology_years, analysis_years=analysis_years,
    )
    try:
        workflow.atomic_to_netcdf(product, path)
    finally:
        product.close()
    return path


def _identity(variable, month, years, regions, baseline, tolerance):
    return json.dumps(dict(schema=1, variable=variable, month=int(month), years=list(years),
                           regions={name: REGIONS[name] for name in regions},
                           baseline=int(baseline), tolerance=float(tolerance)), sort_keys=True)


def regional_is_current(path, *, variable, month, years, regions, baseline, tolerance):
    """Accept legacy regional products only with the expected fields and metadata."""
    required = {'hindcast_mean', 'e_obs', 'e_att', 'delta_e_obs', 'delta_e_att',
                'delta_abs_e_obs', 'delta_abs_e_att', 'regime',
                'skill_acc', 'skill_rmse', *(f'fraction_regime_{i}' for i in range(1, 5))}
    try:
        with xr.open_dataset(path) as ds:
            if not required <= set(ds.data_vars) or 'region' not in ds.dims:
                return False
            if tuple(ds.region.values.astype(str)) != tuple(regions):
                return False
            if ds.attrs.get('analysis_years') != str(tuple(years)):
                return False
            identity = ds.attrs.get('drift_input_identity')
            if identity is not None:
                return identity == _identity(variable, month, years, regions, baseline, tolerance)
            # Original runner used these fixed conventions and wrote region bounds.
            return (baseline == 1 and tolerance == 1.e-6
                    and ds.attrs.get('variable') == variable
                    and ds.attrs.get('initialization_month') == month
                    and ds.attrs.get('region_definitions') == str({name: REGIONS[name] for name in regions}))
    except (OSError, ValueError, KeyError):
        return False


def _regional_mean(field, regions):
    return xr.concat([area_weighted_mean(regional_subset(field, **REGIONS[name]))
                      for name in regions], dim='region').assign_coords(region=list(regions))


def ensure_regional_products(output_root, variables, init_months, sources, regions, *,
                             mode='auto', analysis_years=(1980, 2017),
                             variable_years=None, baseline_lead=1, distance_tolerance=1.e-6):
    """Build only the compact regional diagnostics required by 5b/5c/5d."""
    _check_mode(mode)
    if set(sources) != set(SOURCES):
        raise ValueError(f'Two-reference regional processing requires {SOURCES}')
    variable_years = {'PRECT': (1980, 2015)} if variable_years is None else variable_years
    root = Path(output_root)
    rows, pending = [], []
    for variable in variables:
        years = tuple(variable_years.get(variable, analysis_years))
        for month in init_months:
            paths = {source: root / f'{source}_{variable}_{month:02d}_regional.nc' for source in sources}
            current = all(regional_is_current(
                path, variable=variable, month=month, years=years, regions=regions[variable],
                baseline=baseline_lead, tolerance=distance_tolerance,
            ) for path in paths.values())
            if mode == 'rebuild' or not current:
                pending.append((variable, month, years, paths))
            rows.extend(dict(variable=variable, init_month=month, source=source,
                             product='regional', path=str(path)) for source, path in paths.items())
    if pending and mode == 'require':
        raise FileNotFoundError(f'Missing or stale regional drift products: {[(v, m) for v, m, *_ in pending]}')
    config = default_variable_config() if pending else None
    for variable, month, years, paths in pending:
        regional, observations, attractors, spreads, times = {}, {}, {}, {}, {}
        component = config[variable]['component']
        include_spread = config[variable].get('include_spread', True)
        for source in sources:
            reference = ensure_reference(variable, month, source, mode=mode,
                                         analysis_years=years, variable_config=config)
            job = dict(variable=variable, init_month=month, source=source, component=component)
            with xr.open_dataset(workflow.discover_monthly_hindcast(job), chunks={}) as hc, workflow.load_drift_references(
                source, month, variable, component=component, path=reference,
                require_attractor_spread=include_spread,
            ) as ref:
                values = workflow.select_initialization_years(
                    workflow.apply_unit_transform(hc[variable], component, variable), years)
                values, ref = xr.align(values, ref, join='exact', exclude={'M'})
                regional[source] = _regional_mean(values, regions[variable]).load()
                observations[source] = _regional_mean(ref.X_obs, regions[variable]).load()
                attractors[source] = _regional_mean(ref.X_att, regions[variable]).load()
                times[source] = ref.valid_time.load()
                if include_spread:
                    spreads[source] = _regional_mean(ref.sigma_att, regions[variable]).load()
        pipeline = run_pipeline(regional, observations, attractors,
                                attractor_spreads=spreads if include_spread else None,
                                baseline_lead=baseline_lead, distance_tolerance=distance_tolerance)
        for source, path in paths.items():
            fields = xr.Dataset({name: pipeline[name][source] for name in (
                'hindcast_mean', 'e_obs', 'e_att', 'delta_e_obs', 'delta_e_att',
                'delta_abs_e_obs', 'delta_abs_e_att', 'regime', 'z_att') if name in pipeline})
            fractions = compute_regime_fraction(fields.regime, spatial_dims=(), sample_dims=('Y',))
            skill = pipeline['skill'][source]
            product = xr.merge([fields, fractions, skill.rename({k: f'skill_{k}' for k in skill})],
                               join='exact', compat='override').assign_coords(valid_time=times[source])
            product.attrs.update(source=source, variable=variable, initialization_month=month,
                                 analysis_years=str(years), regions=','.join(regions[variable]),
                                 region_definitions=str({name: REGIONS[name] for name in regions[variable]}),
                                 drift_input_identity=_identity(variable, month, years, regions[variable], baseline_lead, distance_tolerance))
            workflow.atomic_to_netcdf(product, path)
            print(f'[drift inputs] Saved {path}')
    return pd.DataFrame(rows)
