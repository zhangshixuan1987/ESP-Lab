import cftime
import numpy as np
import pandas as pd
import xarray as xr

from esp_lab.data_access_e3sm import drop_feb29, time_set_midmonth


def _dataset_with_time(times):
    return xr.Dataset(
        {"TS": ("time", np.arange(len(times), dtype=float))},
        coords={"time": times},
    )


def test_time_set_midmonth_keeps_represented_month_from_filename():
    ds = _dataset_with_time(
        [
            cftime.DatetimeNoLeap(1980, 5, 1),
            cftime.DatetimeNoLeap(1980, 6, 1),
        ]
    )
    ds.encoding["source"] = "/tmp/TS_198005_198006.nc"

    res = time_set_midmonth(ds, "time")

    assert list(res.time.values) == [
        cftime.DatetimeNoLeap(1980, 5, 15),
        cftime.DatetimeNoLeap(1980, 6, 15),
    ]


def test_time_set_midmonth_shifts_following_month_from_filename():
    ds = _dataset_with_time(
        [
            cftime.DatetimeNoLeap(1980, 6, 1),
            cftime.DatetimeNoLeap(1980, 7, 1),
        ]
    )
    ds.encoding["source"] = "/tmp/TS_198005_198006.nc"

    res = time_set_midmonth(ds, "time")

    assert list(res.time.values) == [
        cftime.DatetimeNoLeap(1980, 5, 15),
        cftime.DatetimeNoLeap(1980, 6, 15),
    ]


def test_time_set_midmonth_shifts_day_one_without_filename():
    ds = _dataset_with_time(
        [
            cftime.DatetimeNoLeap(1980, 6, 1),
            cftime.DatetimeNoLeap(1980, 7, 1),
        ]
    )

    res = time_set_midmonth(ds, "time")

    assert list(res.time.values) == [
        cftime.DatetimeNoLeap(1980, 5, 15),
        cftime.DatetimeNoLeap(1980, 6, 15),
    ]


def test_time_set_midmonth_keeps_end_of_month_without_filename():
    ds = _dataset_with_time(
        [
            cftime.DatetimeNoLeap(1980, 5, 31),
            cftime.DatetimeNoLeap(1980, 6, 30),
        ]
    )

    res = time_set_midmonth(ds, "time")

    assert list(res.time.values) == [
        cftime.DatetimeNoLeap(1980, 5, 15),
        cftime.DatetimeNoLeap(1980, 6, 15),
    ]


def test_time_set_midmonth_uses_bounds_before_filename():
    ds = xr.Dataset(
        {
            "TS": ("time", np.arange(2, dtype=float)),
            "time_bnds": (
                ("time", "bnds"),
                [
                    [cftime.DatetimeNoLeap(1980, 5, 1), cftime.DatetimeNoLeap(1980, 6, 1)],
                    [cftime.DatetimeNoLeap(1980, 6, 1), cftime.DatetimeNoLeap(1980, 7, 1)],
                ],
            ),
        },
        coords={
            "time": [
                cftime.DatetimeNoLeap(1980, 6, 1),
                cftime.DatetimeNoLeap(1980, 7, 1),
            ],
            "bnds": [0, 1],
        },
    )
    ds["time"].attrs["bounds"] = "time_bnds"
    ds.encoding["source"] = "/tmp/TS_198006_198007.nc"

    res = time_set_midmonth(ds, "time")

    assert list(res.time.values) == [
        cftime.DatetimeNoLeap(1980, 5, 15),
        cftime.DatetimeNoLeap(1980, 6, 15),
    ]


def test_time_set_midmonth_preserves_datetime64_calendar():
    ds = _dataset_with_time(pd.date_range("1980-06-01", periods=2, freq="MS"))

    res = time_set_midmonth(ds, "time")

    assert np.issubdtype(res.time.dtype, np.datetime64)
    assert list(res.time.values) == [
        np.datetime64("1980-05-15T00:00:00.000000000"),
        np.datetime64("1980-06-15T00:00:00.000000000"),
    ]


def test_model_drop_feb29_datetime64():
    ds = _dataset_with_time(pd.to_datetime(["2000-02-28", "2000-02-29", "2000-03-01"]))

    res = drop_feb29(ds)

    assert list(res.time.values) == [
        np.datetime64("2000-02-28T00:00:00.000000000"),
        np.datetime64("2000-03-01T00:00:00.000000000"),
    ]


def test_time_set_midmonth_preserves_feb29_month_end():
    ds = _dataset_with_time(pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31"]))

    res = time_set_midmonth(ds, "time")

    assert list(res.time.values) == [
        np.datetime64("2000-01-15T00:00:00.000000000"),
        np.datetime64("2000-02-15T00:00:00.000000000"),
        np.datetime64("2000-03-15T00:00:00.000000000"),
    ]
