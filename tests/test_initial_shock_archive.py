import copy
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from workflows.diagnostics import initial_shock_archive as archive
from workflows.diagnostics import initial_shock_error_archive as error_archive


@pytest.fixture
def archive_inputs(tmp_path, monkeypatch):
    years = [1980, 1981]
    dates = xr.date_range('1980-05-01', periods=36, freq='MS', calendar='noleap', use_cftime=True)
    obs = xr.DataArray(np.broadcast_to(np.arange(36)[:, None, None], (36, 2, 2)).astype(float),
                       dims=('time', 'lat', 'lon'),
                       coords={'time': dates, 'lat': [-30., 30.], 'lon': [0., 180.]})
    tags = [f'{y}050100' for y in years]
    model = xr.concat([obs.isel(time=slice(i*12,i*12+24)).rename(time='L').assign_coords(L=np.arange(1,25))
                       for i in range(2)], dim=xr.IndexVariable('Y',tags)) * 2 + 100
    model = model.expand_dims(M=[0,1]).rename('TREFHT').to_dataset()
    model['time'] = (('Y', 'L'), np.array([dates[:24], dates[12:]], dtype=object))
    model.to_netcdf(tmp_path/'model.nc')
    obs.rename('tas').to_dataset().to_netcdf(tmp_path/'obs.nc')
    paths = [str(tmp_path/'model.nc')]
    monkeypatch.setattr(archive.e3sm_access, 'nested_file_list_by_init', lambda **kw: ([paths], tags))
    monkeypatch.setattr(archive.e3sm_access, 'get_monthly_data', lambda **kw: xr.open_dataset(paths[0]))
    monkeypatch.setattr(archive.obs_access, 'find_obs_file', lambda **kw: str(tmp_path/'obs.nc'))
    monkeypatch.setattr(archive.obs_access, 'get_monthly_data', lambda **kw: xr.open_dataset(tmp_path/'obs.nc'))
    monkeypatch.setattr(archive, '_remap', lambda ds, field, target, settings: ds[field])
    from esp_lab.utils import regrid_utils
    grid = xr.Dataset({'area': (('lat','lon'),np.ones((2,2)))},coords={'lat':obs.lat,'lon':obs.lon})
    monkeypatch.setattr(regrid_utils, 'make_latlon_grid', lambda **kw: grid)
    settings = {'paths':{'s2d_diag_root':str(tmp_path/'out')},
                'run':{'years':years,'init_months':[5],'nlead':24},
                'e3sm':{'nens':2,'data_dir':str(tmp_path),'grid':'grid','ts_split':'2yr','chunks':{},'engine':'netcdf4'},
                'smyle':{'include':False}, 'obs':{'data_dir':str(tmp_path),'chunks':{}},
                'metric':{'start_lead':0,'window_months':24,'block_months':12},
                'cache':{'mode':'auto'}, 'regrid':{'target_dlat':1.,'target_dlon':1.}}
    variable = {'field':'TREFHT','model_variable':'TREFHT','obs_variable':'tas','obs_product':'ERA5',
                'model_scale':1.,'model_offset':0.,'obs_scale':1.,'obs_offset':0.,'units':'degC'}
    cases = {'test':{'case_prefix':'example','cache_tag':'test'}}
    return settings,cases,variable


def test_archive_compute_cache_reuse_and_configuration_change(archive_inputs):
    settings,cases,variable = archive_inputs
    plan = archive.plan_archive_run(settings,cases,variable)
    assert plan[0]['rebuild']
    results = archive.compute_archive_plan(plan,settings,variable)
    np.testing.assert_allclose(results[5].std_ratio,2)
    np.testing.assert_array_equal(results[5].Y,[1980,1981])
    assert 'block_start_time' in results[5].coords
    second = archive.plan_archive_run(settings,cases,variable)
    assert not second[0]['rebuild']
    xr.testing.assert_allclose(archive.compute_archive_plan(second,settings,variable)[5],results[5])
    settings['metric']['block_months']=3
    with pytest.raises(ValueError,match='Configuration changed'):
        archive.compute_archive_plan(second,settings,variable)
    changed = archive.plan_archive_run(settings,cases,variable)[0]
    assert changed['path'] == second[0]['path']
    assert changed['rebuild']


def test_archive_cache_uses_stable_exact_period_filename(archive_inputs):
    settings, cases, variable = archive_inputs

    task = archive.plan_archive_run(settings, cases, variable)[0]

    assert Path(task['path']).name == 'init05_1980_1981.nc'


def test_plot_source_file_is_excluded_from_cache_identity(archive_inputs, monkeypatch):
    settings, cases, variable = archive_inputs
    diagnostic_path = Path(archive.initial_shock.__file__).resolve()
    read_bytes = Path.read_bytes

    def reject_full_diagnostic_hash(path):
        if path.resolve() == diagnostic_path:
            raise AssertionError('Plot and numerical source file was hashed as a whole')
        return read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', reject_full_diagnostic_hash)
    plan = archive.plan_archive_run(settings, cases, variable)
    provenance = json.loads(plan[0]['provenance_json'])
    scientific_key = f'{archive.initial_shock.__file__}::scientific-functions'
    assert scientific_key in provenance['code']


def test_incomplete_archive_stops_before_compute(archive_inputs,monkeypatch):
    settings,cases,variable=archive_inputs
    monkeypatch.setattr(archive.e3sm_access,'nested_file_list_by_init',lambda **kw: ([],[]))
    with pytest.raises(FileNotFoundError,match='incomplete initializations'):
        archive.plan_archive_run(settings,cases,variable)


@pytest.mark.parametrize(
    ('setting_path', 'value', 'message'),
    [
        (('run', 'years'), [1981, 1980], 'ordered integer endpoint years'),
        (('run', 'init_months'), [5.0], 'integer initialization months'),
        (('run', 'nlead'), 0, 'nlead must be positive'),
        (('metric', 'start_lead'), -1, 'Metric window exceeds forecast length'),
        (('metric', 'block_months'), 0, 'at least two complete blocks'),
    ],
)
def test_invalid_plan_configuration_stops_before_archive_access(
    archive_inputs, monkeypatch, setting_path, value, message,
):
    settings, cases, variable = archive_inputs
    settings[setting_path[0]][setting_path[1]] = value
    monkeypatch.setattr(
        archive.obs_access,
        'find_obs_file',
        lambda **kwargs: pytest.fail('invalid configuration reached archive access'),
    )
    with pytest.raises(ValueError, match=message):
        archive.plan_archive_run(settings, cases, variable)


def test_sources_changing_after_plan_are_rejected(archive_inputs):
    settings,cases,variable=archive_inputs
    plan=archive.plan_archive_run(settings,cases,variable)
    path=Path(plan[0]['model_paths'][0])
    path.touch()
    with pytest.raises(RuntimeError,match='Source files changed'):
        archive.compute_archive_plan(plan,settings,variable)


def test_notebook_cells_smoke_execution(archive_inputs,tmp_path):
    settings,cases,variable=archive_inputs
    _nb_matches = list((Path(__file__).parents[1] / "jupyter").rglob("6a_refactor_shock_ts.ipynb"))
    notebook = _nb_matches[0] if _nb_matches else Path(__file__).parents[1] / "jupyter/6a_refactor_shock_ts.ipynb"
    nb = json.loads(notebook.read_text())
    settings['paths']['figure_outdir']=str(tmp_path/'figures')
    settings['dask']={'enabled':False}
    ns={
        'WORKFLOW_SETTINGS': settings,
        'E3SM_CASES': cases,
        'variable': variable,
        'field': 'TREFHT',
    }
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code' and i != 3:  # replace only archive configuration
            exec(compile(''.join(cell['source']),f'notebook cell {i}','exec'),ns)
    assert list((tmp_path/'figures').glob('*_seasonal_absolute_normalized_change.png'))
    assert list((tmp_path/'figures').glob('*_seasonal_normalized_change_scatter.png'))
    assert list((tmp_path/'figures').glob('*_monthly_anomaly_member_vs_mean.png'))
    assert list((tmp_path/'figures').glob('*_summary.csv'))
    assert list((tmp_path/'figures').glob('*_seasonal_summary.csv'))


def test_require_reuses_auto_cache(archive_inputs, monkeypatch):
    settings, cases, variable = archive_inputs
    initial = archive.plan_archive_run(settings, cases, variable)
    archive.compute_archive_plan(initial, settings, variable)
    settings['cache']['mode'] = 'require'
    cached = archive.plan_archive_run(settings, cases, variable)
    assert cached[0]['path'] == initial[0]['path']
    assert not cached[0]['rebuild']
    def unexpected_load(**kwargs):
        raise AssertionError('Cache-only run opened raw model data')
    monkeypatch.setattr(archive.e3sm_access, 'get_monthly_data', unexpected_load)
    np.testing.assert_allclose(archive.compute_archive_plan(cached, settings, variable)[5].std_ratio, 2)


def test_force_compute_refreshes_same_cache(archive_inputs, monkeypatch):
    settings, cases, variable = archive_inputs
    initial = archive.plan_archive_run(settings, cases, variable)
    archive.compute_archive_plan(initial, settings, variable)
    original_loader = archive.e3sm_access.get_monthly_data
    calls = []

    def tracked_loader(**kwargs):
        calls.append(kwargs)
        return original_loader(**kwargs)

    monkeypatch.setattr(archive.e3sm_access, 'get_monthly_data', tracked_loader)
    settings['cache']['force_compute'] = True
    forced = archive.plan_archive_run(settings, cases, variable)
    assert forced[0]['rebuild']
    assert forced[0]['path'] == initial[0]['path']
    archive.compute_archive_plan(forced, settings, variable)
    assert len(calls) == 1


def test_force_compute_rejects_require_mode(archive_inputs):
    settings, cases, variable = archive_inputs
    settings['cache'].update(mode='require', force_compute=True)
    with pytest.raises(ValueError, match='force_compute'):
        archive.plan_archive_run(settings, cases, variable)


def test_rmse_mae_archive_cache_and_notebook_smoke(archive_inputs, tmp_path):
    settings, cases, variable = archive_inputs
    settings['paths']['figure_outdir'] = str(tmp_path / 'figures')
    settings['dask'] = {'enabled': False}
    settings['metric']['climatology_years'] = [1980, 1981]
    settings['metric']['monthly_error_min_samples'] = 1
    settings['metric']['seasonal_error_min_samples'] = 1
    plan = error_archive.plan_archive_run(settings, cases, variable)
    result = error_archive.compute_archive_plan(plan, settings, variable)[5]
    expected_rmse = np.sqrt((result.monthly_raw_error ** 2).mean('lead_month'))
    expected_mae = abs(result.monthly_raw_error).mean('lead_month')
    expected_nrmse = np.sqrt((result.monthly_standardized_error ** 2).mean('lead_month'))
    expected_nmae = abs(result.monthly_standardized_error).mean('lead_month')
    xr.testing.assert_allclose(result.rmse, expected_rmse)
    xr.testing.assert_allclose(result.mae, expected_mae)
    xr.testing.assert_allclose(result.normalized_rmse, expected_nrmse)
    xr.testing.assert_allclose(result.normalized_mae, expected_nmae)
    assert result.seasonal_normalized_rmse.dims == ('case', 'Y', 'lead_year', 'season')
    second = error_archive.plan_archive_run(settings, cases, variable)
    assert not second[0]['rebuild'] and not second[0]['error_rebuild']
    xr.testing.assert_allclose(error_archive.compute_archive_plan(second, settings, variable)[5], result)

    _nb_matches = list((Path(__file__).parents[1] / "jupyter").rglob("6b_refactor_shock_index.ipynb"))
    notebook = _nb_matches[0] if _nb_matches else Path(__file__).parents[1] / "jupyter/6b_refactor_shock_index.ipynb"
    nb = json.loads(notebook.read_text())
    ns = {'WORKFLOW_SETTINGS': settings, 'E3SM_CASES': cases, 'variable': variable,
          'field': 'TREFHT',
          'ERROR_HEATMAP_LEVELS': np.arange(0., 2.0 + .2, .2),
          'PLOT_SEASONAL_ERROR_HEATMAPS': True}
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and i != 3:
            exec(compile(''.join(cell['source']), f'notebook cell {i}', 'exec'), ns)
    assert list((tmp_path / 'figures').glob('*_monthly_normalized_rmse.png'))
    assert list((tmp_path / 'figures').glob('*_seasonal_normalized_rmse.png'))
    assert list((tmp_path / 'figures').glob('*_monthly_rmse_mae_summary.csv'))
    assert list((tmp_path / 'figures').glob('*_seasonal_rmse_mae_summary.csv'))


def test_rmse_mae_force_compute_refreshes_same_cache(archive_inputs):
    settings, cases, variable = archive_inputs
    initial = error_archive.plan_archive_run(settings, cases, variable)
    error_archive.compute_archive_plan(initial, settings, variable)
    settings['cache']['force_compute'] = True
    forced = error_archive.plan_archive_run(settings, cases, variable)
    assert forced[0]['error_rebuild']
    assert forced[0]['error_path'] == initial[0]['error_path']


def test_rmse_threshold_change_reuses_6a_cache_and_rebuilds_only_6b(archive_inputs):
    settings, cases, variable = archive_inputs
    initial = error_archive.plan_archive_run(settings, cases, variable)
    error_archive.compute_archive_plan(initial, settings, variable)

    settings['metric']['monthly_error_min_samples'] = 12
    changed = error_archive.plan_archive_run(settings, cases, variable)

    assert not changed[0]['rebuild']
    assert changed[0]['path'] == initial[0]['path']
    assert changed[0]['error_rebuild']
    assert changed[0]['error_path'] != initial[0]['error_path']
