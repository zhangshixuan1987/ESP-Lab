import numpy as np
import pytest
import xarray as xr

from esp_lab.diagnostics.initial_shock import (
    align_observation_months, compute_initial_shock_index, plot_std_ratio,
)
from workflows.diagnostics.initial_shock import run_initial_shock


def fields(nmonths=60):
    obs = xr.DataArray(
        np.broadcast_to(np.repeat(np.arange(nmonths // 12), 12)[None, :, None, None], (2, nmonths, 2, 2)).copy().astype(float),
        dims=("Y", "L", "lat", "lon"),
        coords={"Y": [1981, 1986], "L": np.arange(nmonths), "lat": [-30., 30.], "lon": [0., 180.]},
        attrs={"units": "degC"},
    )
    model = xr.concat([obs * 2 + 10, obs * 2 + 20], dim=xr.IndexVariable("M", [0, 1]))
    model.attrs["units"] = "degC"
    return model, obs


def test_ncl_five_annual_samples_and_offset_invariance():
    model, obs = fields()
    result = compute_initial_shock_index(model, obs)
    np.testing.assert_allclose(result.std_ratio, 2)
    np.testing.assert_allclose(result.observation_std, np.std(np.arange(5), ddof=1))
    np.testing.assert_allclose(result.model_index[0], np.arange(5) * 2 + 15)
    assert (result.paired_sample_count == 5).all()
    np.testing.assert_allclose(compute_initial_shock_index(model + 100, obs.assign_attrs(units='degC')) .std_ratio, 2)


def test_ensemble_mean_precedes_std():
    model, obs = fields()
    model.loc[dict(M=0)] = obs
    model.loc[dict(M=1)] = -obs
    assert (compute_initial_shock_index(model, obs).std_ratio == 0).all()


def test_missing_month_invalidates_whole_annual_block():
    model, obs = fields()
    model.loc[dict(L=2)] = np.nan
    result = compute_initial_shock_index(model, obs)
    assert result.std_ratio.isnull().all()
    assert (result.paired_sample_count == 4).all()
    assert result.model_index.sel(block=1).isnull().all()
    relaxed = compute_initial_shock_index(model, obs, min_samples=4)
    np.testing.assert_allclose(relaxed.std_ratio, 2)


def test_constant_observation_and_area_coverage():
    model, obs = fields()
    assert compute_initial_shock_index(model, xr.zeros_like(obs)).std_ratio.isnull().all()
    model.loc[dict(lat=-30)] = np.nan
    result = compute_initial_shock_index(model, obs, min_area_fraction=.75)
    np.testing.assert_allclose(result.model_area_fraction, .5)
    assert result.std_ratio.isnull().all()


def test_area_average_after_annual_average():
    model, obs = fields()
    model.loc[dict(L=0, lat=-30)] = np.nan
    result = compute_initial_shock_index(model, obs)
    assert (result.model_area_fraction.sel(block=1) == .5).all()
    np.testing.assert_allclose(result.std_ratio, 2)


@pytest.mark.parametrize('change', ['units', 'grid', 'lead', 'short'])
def test_bad_inputs_fail(change):
    model, obs = fields()
    if change == 'units': obs.attrs['units'] = 'K'
    if change == 'grid': obs = obs.assign_coords(lon=[1, 181])
    if change == 'lead': model = model.assign_coords(L=np.arange(60) * 2)
    if change == 'short': model, obs = model.isel(L=slice(24)), obs.isel(L=slice(24))
    with pytest.raises(ValueError): compute_initial_shock_index(model, obs)


def test_dask_matches_eager():
    model, obs = fields()
    eager = compute_initial_shock_index(model, obs)
    lazy = compute_initial_shock_index(model.chunk({'L': 7}), obs.chunk({'L': 9})).compute()
    xr.testing.assert_allclose(eager, lazy)


def test_calendar_alignment_and_missing_duplicate_months():
    obs = xr.DataArray(np.arange(24.), dims='time', coords={'time': xr.date_range('2000-01-01', periods=24, freq='MS', calendar='noleap', use_cftime=True)})
    time = xr.DataArray(np.array(xr.date_range('2000-02-01', periods=12, freq='MS'))[None,:], dims=('Y','L'), coords={'Y':[2000], 'L':np.arange(12)})
    np.testing.assert_array_equal(align_observation_months(obs, time), np.arange(1,13)[None,:])
    with pytest.raises(ValueError, match='Missing'): align_observation_months(obs.isel(time=slice(5,None)),time)
    with pytest.raises(ValueError, match='duplicate'): align_observation_months(xr.concat([obs,obs],dim='time'),time)


def test_cached_workflow_and_plot(tmp_path):
    model, obs = fields(24)
    times = np.array(xr.date_range('1981-01-01', periods=48, freq='MS')).reshape(2,24)
    model_ds = model.to_dataset(name='tas').assign(verification_time=(('Y','L'),times))
    model_ds.to_netcdf(tmp_path/'model.nc')
    obs_time = obs.rename(L='month').stack(time=('Y','month')).transpose('time','lat','lon').reset_index('time',drop=True).assign_coords(time=times.ravel())
    obs_time.to_dataset(name='tas').to_netcdf(tmp_path/'obs.nc')
    config={'output_root':str(tmp_path/'out'), 'cases':{'case1':{'path':str(tmp_path/'model.nc'),'variable':'tas','units':'degC'}},
            'observation':{'path':str(tmp_path/'obs.nc'),'variable':'tas','units':'degC'},
            'settings':{'window_months':24,'block_months':12},'figure_path':str(tmp_path/'plot.png')}
    result, paths = run_initial_shock(config)
    np.testing.assert_allclose(result.std_ratio, 2)
    assert (tmp_path/'plot.png').stat().st_size > 0
    stamp = paths[0].stat().st_mtime_ns
    config['cache_mode']='require'
    cached, same_paths = run_initial_shock(config)
    assert same_paths == paths and paths[0].stat().st_mtime_ns == stamp
    xr.testing.assert_allclose(cached, result)
    config['settings']['block_months']=6
    with pytest.raises(FileNotFoundError): run_initial_shock(config)
    config['cache_mode'] = 'auto'
    config['settings']['block_months'] = 12
    model_ds['verification_time'] = (('Y', 'L'), times[::-1])
    model_ds.to_netcdf(tmp_path / 'different_months.nc')
    config['cases']['case2'] = dict(config['cases']['case1'], path=str(tmp_path / 'different_months.nc'))
    with pytest.raises(ValueError, match='different verification months'):
        run_initial_shock(config)
