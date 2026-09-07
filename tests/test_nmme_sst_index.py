import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr

from esp_lab import data_access_nmme as nmme_access


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_process_nmme_sst_index.py"
SPEC = importlib.util.spec_from_file_location("run_process_nmme_sst_index", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_nmme_sst_mask_is_applied_before_regional_mean():
    raw = xr.DataArray(
        [[0.0, 300.0], [0.0, 300.0]],
        dims=("Y", "X"),
        coords={"Y": [-5.0, 5.0], "X": [120.0, 180.0]},
        attrs={"units": "K", "scale_min": 260.0},
        name="sst",
    )

    masked = nmme_access.mask_invalid_sst(raw, apply_land_mask=False)
    result = MODULE._regional_mean(
        masked.to_dataset(name="sst"),
        [120.0, 180.0, -5.0, 5.0],
    )

    assert bool(masked.sel(X=120.0).isnull().all())
    assert float(result) == 300.0


def test_eli_ignores_finite_zero_land_fill_values():
    sst = xr.DataArray(
        np.zeros((2, 3, 6), dtype=np.float32),
        dims=("time", "Y", "X"),
        coords={
            "time": [0, 1],
            "Y": [-5.0, 0.0, 5.0],
            "X": [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        },
        attrs={"units": "K", "scale_min": 260.0},
    )
    sst.loc[{"Y": 0.0, "X": [120.0, 180.0, 240.0]}] = [
        [280.0, 300.0, 280.0],
        [280.0, 280.0, 300.0],
    ]

    masked = nmme_access.mask_invalid_sst(sst, apply_land_mask=False)
    result = MODULE._eli_from_sst(masked, "X", "Y")

    xr.testing.assert_allclose(
        result,
        xr.DataArray(
            [180.0, 240.0],
            dims="time",
            coords={"time": [0, 1]},
            name="sst",
        ),
    )
    assert result.attrs["units"] == "degrees_east"


def test_nmme_sst_sanity_check_supports_celsius_units():
    sst = xr.DataArray(
        [-999.0, -2.0, 25.0, 100.0],
        dims="point",
        attrs={"units": "degrees_Celsius", "scale_min": -5.0},
    )

    result = nmme_access.mask_invalid_sst(sst, apply_land_mask=False)

    np.testing.assert_array_equal(
        np.isfinite(result),
        [False, True, True, False],
    )
    assert "degC" in result.attrs["valid_sst_mask"]


def test_nmme_sst_land_mask_can_be_disabled(monkeypatch):
    sst = xr.DataArray(
        [[298.0, 299.0]],
        dims=("Y", "X"),
        coords={"Y": [0.0], "X": [120.0, 180.0]},
        attrs={"units": "K"},
    )
    ocean_mask = xr.DataArray(
        [[False, True]],
        dims=("Y", "X"),
        coords={"Y": sst.Y, "X": sst.X},
    )
    monkeypatch.setattr(
        nmme_access,
        "_natural_earth_ocean_mask",
        lambda *_args, **_kwargs: ocean_mask,
    )

    masked = nmme_access.mask_invalid_sst(sst, apply_land_mask=True)
    unmasked = nmme_access.mask_invalid_sst(sst, apply_land_mask=False)

    assert bool(masked.sel(X=120.0).isnull().all())
    assert bool(unmasked.notnull().all())


def test_nmme_model_land_mask_is_reused_from_fixed_directory(
    monkeypatch, tmp_path
):
    sst = xr.DataArray(
        [[298.0, 299.0]],
        dims=("Y", "X"),
        coords={"Y": [0.0], "X": [120.0, 180.0]},
        attrs={"units": "K"},
    )
    ocean_mask = xr.DataArray(
        [[False, True]],
        dims=("Y", "X"),
        coords={"Y": sst.Y, "X": sst.X},
    )
    calls = []

    def build_mask(*_args, **_kwargs):
        calls.append(True)
        return ocean_mask

    monkeypatch.setattr(nmme_access, "_natural_earth_ocean_mask", build_mask)
    first = nmme_access.mask_invalid_sst(
        sst,
        model="test-model",
        fixed_dir=tmp_path,
        lon_name="X",
        lat_name="Y",
    )
    second = nmme_access.mask_invalid_sst(
        sst,
        model="test-model",
        fixed_dir=tmp_path,
        lon_name="X",
        lat_name="Y",
    )

    assert len(calls) == 1
    assert (tmp_path / "sftlf.NMME.test-model.nc").is_file()
    xr.testing.assert_equal(first, second)


def test_nmme_initial_anomaly_uses_configured_climatology_window():
    initialization = xr.date_range(
        "1980-01-01",
        "2020-12-01",
        freq="MS",
        calendar="360_day",
        use_cftime=True,
    )
    years = np.asarray(initialization.year)
    months = np.asarray(initialization.month)
    values = (
        20.0
        + 0.2 * (years - 1980)
        + np.sin(2.0 * np.pi * months / 12.0)
    )[:, None, None]
    regional_sst = xr.DataArray(
        values,
        dims=("S", "L", "M"),
        coords={"S": initialization, "L": [1], "M": [1]},
    )

    anomaly = MODULE._remove_init_month_climatology(
        regional_sst,
        "test-model",
        1981,
        2010,
    )
    climatology = anomaly.where(
        (anomaly["S"].dt.year >= 1981) & (anomaly["S"].dt.year <= 2010),
        drop=True,
    )

    monthly_mean = climatology.groupby("S.month").mean(("S", "M"))
    assert float(np.abs(monthly_mean).max()) < 1.0e-12


def test_nmme_observed_iod_is_west_anomaly_minus_east_anomaly():
    time = xr.date_range(
        "1980-01-01",
        "2011-12-01",
        freq="MS",
        calendar="noleap",
        use_cftime=True,
    )
    month = np.asarray(time.month)
    year = np.asarray(time.year)
    west = xr.DataArray(
        28.0 + np.sin(2.0 * np.pi * month / 12.0) + 0.1 * (year - 1980),
        dims="time",
        coords={"time": time},
    )
    east = xr.DataArray(
        24.0 + np.cos(2.0 * np.pi * month / 12.0) + 0.04 * (year - 1980),
        dims="time",
        coords={"time": time},
    )

    result = MODULE._derive_obs_indices(
        {"IOD_West": west, "IOD_East": east},
        ["IOD"],
        1981,
        2010,
    )["IOD"]
    expected = MODULE._obs_anom(west, 1981, 2010) - MODULE._obs_anom(
        east,
        1981,
        2010,
    )

    xr.testing.assert_allclose(result, expected)
    assert result.attrs["index_name"] == "DMI"
    assert result.attrs["climatology_start_year"] == 1981
    assert result.attrs["climatology_end_year"] == 2010


def test_combined_timeseries_cache_checks_model_and_mask_contract(tmp_path):
    path = tmp_path / 'nmme.nc'
    attrs = dict(region='AtlMDR', climatology_start_year=1981, climatology_end_year=2010,
                 requested_models='a,b', nmme_sst_land_mask='True',
                 nmme_sst_mask_version=nmme_access.NMME_SST_MASK_VERSION)
    ds = xr.Dataset({'sst': (('time', 'model'), [[1., 2.]])},
                    coords={'time': [0], 'model': ['a', 'b']}, attrs=attrs)
    ds.to_netcdf(path)
    kwargs = dict(region='AtlMDR', clim_start=1981, clim_end=2010,
                  models=['a', 'b'], apply_land_mask=True)
    assert MODULE.timeseries_is_current(path, **kwargs)[0]
    assert not MODULE.timeseries_is_current(path, **{**kwargs, 'models': ['a']})[0]
    assert not MODULE.timeseries_is_current(path, **{**kwargs, 'apply_land_mask': False})[0]
    ds.attrs.pop('nmme_sst_mask_version')
    ds.to_netcdf(path, mode='w')
    assert not MODULE.timeseries_is_current(path, **kwargs)[0]


def test_timeseries_writer_preserves_processing_contract(tmp_path):
    data = xr.DataArray([[[[1.0]]]], dims=('Y', 'L', 'M', 'model'),
                        coords={'Y': ['2000050100'], 'L': [1], 'M': [0], 'model': ['a']})
    time = xr.DataArray([[np.datetime64('2000-05-15')]], dims=('Y', 'L'),
                        coords={'Y': data.Y, 'L': data.L})
    processed = dict(data_start=2000, data_end=2000, region='AtlMDR',
                     climatology_start=1981, climatology_end=2010,
                     requested_models=['a'], apply_land_mask=True,
                     monthly_dd={m: data for m in MODULE.INIT_MONTHS},
                     seasonal_dd={m: data for m in MODULE.INIT_MONTHS},
                     monthly_time={m: time for m in MODULE.INIT_MONTHS},
                     seasonal_time={m: time for m in MODULE.INIT_MONTHS})
    directory = MODULE.save_timeseries_outputs(tmp_path, processed)
    for path in directory.glob('*.nc'):
        assert MODULE.timeseries_is_current(
            path, region='AtlMDR', clim_start=1981, clim_end=2010,
            models=['a'], apply_land_mask=True,
        )[0]
    assert len(list(directory.glob('*.nc'))) == 2 * len(MODULE.INIT_MONTHS)
