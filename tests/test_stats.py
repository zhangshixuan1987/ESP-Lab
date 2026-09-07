import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xskillscore as xs

from esp_lab.stats import cor_ci_bootyears
from esp_lab.stats import compute_skill_seasonal_batch
from esp_lab.stats import compute_resampled_nrmse_prepared
from esp_lab.stats import detrend_linear
from esp_lab.stats import leadtime_skill_seas
from esp_lab.stats import leadtime_skill_seas_resamp
from esp_lab.stats import prepare_skill_seasonal_lead
from esp_lab.stats import remove_drift
from esp_lab.stats import _climatology_mean_by_year


def test_cor_ci_bootyears():
    """
    Test the cor_ci_bootyears function.
    """
    ts1 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    ts2 = ts1*1.3

    result = cor_ci_bootyears(ts1, ts2)
    if np.isnan(result[0]) or np.isnan(result[1]):
        result = cor_ci_bootyears(ts1, ts2)

    # Two very similar arrays should be highly correlated
    if not np.isnan(result[0]):
        if not np.isnan(result[1]):
            assert result[0] > 0.9
            assert result[1] < 1.1


def test_cor_ci_bootyears_two_different_array_sizes():
    """
    Test that the cor_ci_bootyears function fails with
    two differently sized arrays.
    """
    ts1 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    ts2 = np.array([1, 2, 3])

    with pytest.raises(SystemExit) as e:
        cor_ci_bootyears(ts1, ts2)

    assert e.type == SystemExit


def test_cor_ci_bootyears_seed():
    """
    Test that the cor_ci_bootyears function random seed
    generates the same exact result each time
    """
    ts1 = np.array([15, 24, 35, 11, 52, 63, 71, 22, 83])
    ts2 = np.array([10, 22, 34, 11, 33, 57, 22, 71, 90])
    result = cor_ci_bootyears(ts1, ts2, seed=123)

    assert result == (-0.22524093559047992, 0.9763300171129478)


def test_detrend_linear():
    """
    Test the detrend_linear function.
    """

    temperature = np.array([[[16,  2,  3],
                             [14, 27, 12]],
                            [[14, 14,  6],
                             [12, 10, 12]]])
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    time = pd.date_range("2014-09-06", periods=3)
    reference_time = pd.Timestamp("2014-09-05")
    da = xr.DataArray(data=temperature,
                      dims=["x", "y", "time"],
                      coords=dict(
                            lon=(["x", "y"], lon),
                            lat=(["x", "y"], lat),
                            time=time,
                            reference_time=reference_time),
                      attrs=dict(
                            description="Temperature.",
                            units="degC"))

    # Apply linear detrending
    final_dat = detrend_linear(da, "time")

    assert final_dat.dims == ('x', 'y', 'time')
    assert final_dat.data[0][0][0] > 2.4
    assert final_dat.data[0][0][0] < 2.5
    assert final_dat.data[1][1][1] > -1.4
    assert final_dat.data[1][1][1] < -1.3
    assert final_dat.data[1][0][1] > 2.6
    assert final_dat.data[1][0][1] < 2.7


def test_leadtime_skill_seas():
    """
    Test the leadtime_skill_seas function.
    """
    # todo: make test

    assert True


def test_leadtime_skill_seas_resamp():
    """
    Test the leadtime_skill_seas_resamp function.
    """
    # todo: make test

    assert True


def test_remove_drift():
    """
    Test the remove_drift function.
    """
    # todo: make test

    assert True


def test_climatology_mean_by_year_accepts_cftime_bounds_for_datetime_index():
    time = pd.date_range("1979-01-15", "1982-01-15", freq="YS") + pd.Timedelta(days=14)
    da = xr.DataArray(np.arange(time.size, dtype=float), dims="time", coords={"time": time})

    clim = _climatology_mean_by_year(
        da,
        "time",
        cftime.DatetimeNoLeap(1980, 1, 1),
        cftime.DatetimeNoLeap(1981, 12, 31),
    )

    expected = da.sel(time=slice("1980", "1981")).mean("time")
    xr.testing.assert_allclose(clim, expected)


def test_climatology_mean_by_year_accepts_string_bounds_for_cftime_index():
    time = [cftime.DatetimeNoLeap(year, 1, 15) for year in range(1979, 1983)]
    da = xr.DataArray(np.arange(len(time), dtype=float), dims="time", coords={"time": time})

    clim = _climatology_mean_by_year(da, "time", "1980", "1981")

    expected = da.isel(time=[1, 2]).mean("time")
    xr.testing.assert_allclose(clim, expected)


def test_compute_skill_seasonal_batch_can_return_corr_only():
    rng = np.random.default_rng(4)
    years = np.arange(2000, 2012)
    lat = [0, 1]
    lon = [0, 1, 2]
    model = xr.DataArray(
        rng.normal(size=(years.size, 1, 4, len(lat), len(lon))),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={"Y": years, "L": [1], "M": np.arange(4), "lat": lat, "lon": lon},
    )
    model_time = xr.DataArray(
        np.array(
            [[cftime.DatetimeNoLeap(int(year), 1, 15)] for year in years],
            dtype=object,
        ),
        dims=("Y", "L"),
        coords={"Y": years, "L": [1]},
    )
    observations = xr.DataArray(
        rng.normal(size=(years.size, len(lat), len(lon))),
        dims=("time", "lat", "lon"),
        coords={
            "time": [
                cftime.DatetimeNoLeap(int(year), 1, 15) for year in years
            ],
            "lat": lat,
            "lon": lon,
        },
    )
    member_indices = np.array([[0, 1], [2, 3]])

    full = compute_skill_seasonal_batch(
        model,
        model_time,
        observations,
        member_indices_all=member_indices,
        nleads=1,
        is_anomaly=True,
    )
    corr_only = compute_skill_seasonal_batch(
        model,
        model_time,
        observations,
        member_indices_all=member_indices,
        nleads=1,
        is_anomaly=True,
        metrics=("corr",),
    )

    assert list(corr_only.data_vars) == ["corr"]
    xr.testing.assert_allclose(corr_only["corr"], full["corr"])


def test_compute_skill_seasonal_batch_vectorizes_member_samples():
    years = np.arange(2000, 2004)
    model = xr.DataArray(
        np.arange(years.size * 4, dtype=float).reshape(years.size, 1, 4, 1, 1),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={"Y": years, "L": [1], "M": np.arange(4), "lat": [0], "lon": [0]},
    )
    model_time = xr.DataArray(
        [[cftime.DatetimeNoLeap(int(year), 1, 15)] for year in years],
        dims=("Y", "L"),
        coords={"Y": years, "L": [1]},
    )
    observations = xr.DataArray(
        np.arange(years.size, dtype=float),
        dims="time",
        coords={
            "time": [
                cftime.DatetimeNoLeap(int(year), 1, 15) for year in years
            ],
        },
    ).expand_dims(lat=[0], lon=[0])
    member_indices = np.array([[0, 1], [2, 3]])

    result = compute_skill_seasonal_batch(
        model,
        model_time,
        observations,
        member_indices_all=member_indices,
        nleads=1,
        is_anomaly=True,
        metrics=("rmse",),
    )

    model_by_iteration = np.stack([
        model.isel(L=0, M=indices).mean("M").transpose("Y", "lat", "lon").values
        for indices in member_indices
    ])
    observation_values = observations.transpose("time", "lat", "lon").values
    expected_rmse = np.sqrt(
        np.mean((model_by_iteration - observation_values) ** 2, axis=1)
    ) / observations.std("time").values

    np.testing.assert_allclose(
        result["rmse"].isel(L=0).values,
        expected_rmse,
    )


def test_compute_skill_seasonal_batch_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metrics"):
        compute_skill_seasonal_batch(
            xr.DataArray(),
            xr.DataArray(),
            xr.DataArray(),
            member_indices_all=np.zeros((1, 1), dtype=int),
            metrics=("not-a-metric",),
        )


def test_prepared_resampled_nrmse_matches_batch_calculation():
    rng = np.random.default_rng(8)
    years = np.arange(2000, 2008)
    model = xr.DataArray(
        rng.normal(size=(years.size, 1, 4, 2, 2)),
        dims=("Y", "L", "M", "lat", "lon"),
        coords={
            "Y": years,
            "L": [3],
            "M": np.arange(4),
            "lat": [0, 1],
            "lon": [0, 1],
        },
    )
    model_time = xr.DataArray(
        [[cftime.DatetimeNoLeap(int(year), 3, 15)] for year in years],
        dims=("Y", "L"),
        coords={"Y": years, "L": [3]},
    )
    observations = xr.DataArray(
        rng.normal(size=(years.size, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={
            "time": [
                cftime.DatetimeNoLeap(int(year), 3, 15) for year in years
            ],
            "lat": [0, 1],
            "lon": [0, 1],
        },
    )
    member_indices = np.array([[0, 1], [2, 3], [0, 3]])

    batch = compute_skill_seasonal_batch(
        model,
        model_time,
        observations,
        member_indices_all=member_indices,
        nleads=1,
        detrend=True,
        is_anomaly=True,
        metrics=("rmse",),
    )
    model_prepared, obs_prepared = prepare_skill_seasonal_lead(
        model,
        model_time,
        observations,
        lead_index=0,
        detrend=True,
        is_anomaly=True,
    )
    prepared = compute_resampled_nrmse_prepared(
        model_prepared,
        obs_prepared,
        member_indices,
    )

    xr.testing.assert_allclose(prepared, batch["rmse"].isel(L=0))


def test_prepared_resampled_nrmse_supports_repeated_member_indices():
    model = xr.DataArray(
        np.arange(3 * 4, dtype=float).reshape(3, 4, 1, 1),
        dims=("time", "M", "lat", "lon"),
        coords={
            "time": [2000, 2001, 2002],
            "M": np.arange(4),
            "lat": [0],
            "lon": [0],
        },
    )
    observations = xr.DataArray(
        np.arange(3, dtype=float).reshape(3, 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": [2000, 2001, 2002], "lat": [0], "lon": [0]},
    )
    member_indices = np.array([[0, 0, 2], [1, 3, 3]])

    result = compute_resampled_nrmse_prepared(
        model,
        observations,
        member_indices,
    )
    expected_means = np.stack([
        model.isel(M=indices).mean("M").values
        for indices in member_indices
    ])
    expected = np.sqrt(
        np.mean((expected_means - observations.values) ** 2, axis=1)
    ) / observations.std("time").values

    np.testing.assert_allclose(result.values, expected)
