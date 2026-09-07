import cftime
import numpy as np
import pandas as pd
import xarray as xr
from esp_lab.data_access_obs import drop_feb29, preprocessor_monthly, transform_to_mid_month


def test_preprocessor_monthly_datetime64():
    time_pd = pd.date_range("1999-01-01", periods=24, freq="MS")
    ds = xr.Dataset({"obs_var": ("time", np.random.rand(24))}, coords={"time": time_pd})
    res = preprocessor_monthly(ds, field="obs_var", start_year=2000, end_year=2000, harmonize_time=True)
    # Month-start observational timestamps should stay in their represented
    # month instead of being shifted backward.
    assert len(res.time) == 12


def test_preprocessor_monthly_cftime():
    time_cf = [cftime.DatetimeNoLeap(y, m, 1) for y in [1999, 2000] for m in range(1, 13)]
    ds = xr.Dataset({"obs_var": ("time", np.random.rand(24))}, coords={"time": time_cf})
    res = preprocessor_monthly(ds, field="obs_var", start_year="2000", end_year="2000", harmonize_time=True)
    assert len(res.time) == 12


def test_transform_to_mid_month_keeps_existing_midmonth():
    time_cf = [cftime.DatetimeNoLeap(2000, m, 15) for m in range(1, 4)]
    ds = xr.Dataset({"obs_var": ("time", np.arange(3))}, coords={"time": time_cf})

    res = transform_to_mid_month(ds)

    assert list(res.time.values) == time_cf


def test_transform_to_mid_month_uses_bounds_for_following_month_start():
    ds = xr.Dataset(
        {
            "obs_var": ("time", np.arange(2)),
            "time_bnds": (
                ("time", "bnds"),
                [
                    [cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2000, 2, 1)],
                    [cftime.DatetimeNoLeap(2000, 2, 1), cftime.DatetimeNoLeap(2000, 3, 1)],
                ],
            ),
        },
        coords={
            "time": [
                cftime.DatetimeNoLeap(2000, 2, 1),
                cftime.DatetimeNoLeap(2000, 3, 1),
            ],
            "bnds": [0, 1],
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"

    res = transform_to_mid_month(ds)

    assert list(res.time.values) == [
        cftime.DatetimeNoLeap(2000, 1, 15),
        cftime.DatetimeNoLeap(2000, 2, 15),
    ]


def test_transform_to_mid_month_preserves_datetime64_calendar():
    time_pd = pd.date_range("2000-01-01", periods=2, freq="MS")
    ds = xr.Dataset({"obs_var": ("time", np.arange(2))}, coords={"time": time_pd})

    res = transform_to_mid_month(ds)

    assert np.issubdtype(res.time.dtype, np.datetime64)
    assert list(res.time.values) == [
        np.datetime64("2000-01-15T00:00:00.000000000"),
        np.datetime64("2000-02-15T00:00:00.000000000"),
    ]


def test_drop_feb29_datetime64():
    time_pd = pd.to_datetime(["2000-02-28", "2000-02-29", "2000-03-01"])
    ds = xr.Dataset({"obs_var": ("time", np.arange(3))}, coords={"time": time_pd})

    res = drop_feb29(ds)

    assert list(res.time.values) == [
        np.datetime64("2000-02-28T00:00:00.000000000"),
        np.datetime64("2000-03-01T00:00:00.000000000"),
    ]


def test_preprocessor_monthly_preserves_feb29_month_end():
    time_pd = pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31"])
    ds = xr.Dataset({"obs_var": ("time", np.arange(3))}, coords={"time": time_pd})

    res = preprocessor_monthly(ds, field="obs_var", harmonize_time=True)

    assert list(res.time.dt.month.values) == [1, 2, 3]
    assert list(res.time.dt.day.values) == [15, 15, 15]
    assert list(res.obs_var.values) == [0, 1, 2]
