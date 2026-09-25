"""
Unit tests for esp_lab.diagnostics.s2s_core
"""

import numpy as np
import pytest
import xarray as xr

from esp_lab.diagnostics.s2s_core import (
    S2S_WEEKLY_WINDOWS,
    WeeklyWindowDef,
    aggregate_daily_to_weekly,
    build_weekly_window,
    compute_weekly_acc,
    compute_weekly_acc_significance,
    compute_weekly_anomalies,
    compute_weekly_climatology,
    compute_weekly_rmse,
    get_weekly_window,
    paired_acc_difference,
    valid_dates_for_week,
)


def test_weekly_windows():
    assert len(S2S_WEEKLY_WINDOWS) == 8
    # Week 1: 1..7
    w1 = get_weekly_window(1)
    assert w1.day_first == 1
    assert w1.day_last == 7
    assert w1.num_days == 7
    assert w1.lead_days == [1, 2, 3, 4, 5, 6, 7]

    # Week 8: 50..56
    w8 = get_weekly_window(8)
    assert w8.day_first == 50
    assert w8.day_last == 56
    assert w8.num_days == 7
    assert len(w8.lead_days) == 7


def test_valid_dates():
    # 1980-05-01 initialization
    dates_w1 = valid_dates_for_week(1980, 5, 1, init_day=1, calendar="noleap")
    assert len(dates_w1) == 7
    # Day 1 is 1980-05-02
    assert dates_w1[0] == (1980, 5, 2)
    # Day 7 is 1980-05-08
    assert dates_w1[-1] == (1980, 5, 8)


def test_aggregate_daily_to_weekly():
    # Synthetic daily array: 84 lead days, 10 lats, 20 lons
    days = np.arange(1, 85)
    data = np.ones((84, 10, 20))
    # Give week 1 value 2.0, week 2 value 4.0
    data[:7, :, :] = 2.0
    data[7:14, :, :] = 4.0

    da = xr.DataArray(
        data,
        coords={"d": days, "lat": np.linspace(-90, 90, 10), "lon": np.linspace(0, 360, 20)},
        dims=["d", "lat", "lon"],
        name="test_var",
    )

    weekly = aggregate_daily_to_weekly(da, day_dim="d", weeks=range(1, 9))
    assert "L" in weekly.dims
    assert list(weekly.L.values) == list(range(1, 9))
    assert np.allclose(weekly.sel(L=1).values, 2.0)
    assert np.allclose(weekly.sel(L=2).values, 4.0)
    assert np.allclose(weekly.sel(L=3).values, 1.0)


def test_acc_and_rmse():
    # Create synthetic anomalies over 10 years, 8 weeks, 5 lat, 5 lon
    years = np.arange(1980, 1990)
    leads = np.arange(1, 9)
    rng = np.random.RandomState(42)

    truth = rng.randn(10, 8, 5, 5)
    noise = rng.randn(10, 8, 5, 5) * 0.1
    forecast = truth + noise  # High correlation

    da_truth = xr.DataArray(
        truth,
        coords={"Y": years, "L": leads, "lat": range(5), "lon": range(5)},
        dims=["Y", "L", "lat", "lon"],
    )
    da_fcst = xr.DataArray(
        forecast,
        coords={"Y": years, "L": leads, "lat": range(5), "lon": range(5)},
        dims=["Y", "L", "lat", "lon"],
    )

    acc = compute_weekly_acc(da_fcst, da_truth, year_dim="Y", lead_dim="L")
    assert acc.shape == (8, 5, 5)
    # Since noise is small, acc should be high (> 0.9)
    assert (acc > 0.8).all()

    rmse = compute_weekly_rmse(da_fcst, da_truth, year_dim="Y", lead_dim="L")
    assert rmse.shape == (8, 5, 5)
    assert (rmse < 0.3).all()

    # Significance
    sig = compute_weekly_acc_significance(acc, n_samples=10, alpha=0.05)
    assert sig.all()


def test_paired_diff():
    leads = np.arange(1, 9)
    acc1 = xr.DataArray(np.full((8, 3, 3), 0.7), coords={"L": leads, "lat": range(3), "lon": range(3)}, dims=["L", "lat", "lon"])
    acc2 = xr.DataArray(np.full((8, 3, 3), 0.5), coords={"L": leads, "lat": range(3), "lon": range(3)}, dims=["L", "lat", "lon"])

    diff = paired_acc_difference(acc1, acc2, lead_dim="L")
    assert np.allclose(diff.values, 0.2)

