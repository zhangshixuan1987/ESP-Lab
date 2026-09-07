import cftime
import numpy as np
import pandas as pd
import xarray as xr
from esp_lab.data_access_obs import crop_time


def test_crop_time_datetime64():
    ds = xr.Dataset(
        {"data": ("time", [1, 2, 3, 4, 5])},
        coords={"time": pd.date_range("1999-01-01", periods=5, freq="YS")}
    )
    res = crop_time(ds, start_year=2000, end_year=2002)
    assert len(res.time) == 3


def test_crop_time_cftime():
    time_cftime = [cftime.DatetimeNoLeap(y, 1, 1) for y in [1999, 2000, 2001, 2002, 2003]]
    ds = xr.Dataset({"data": ("time", [1, 2, 3, 4, 5])}, coords={"time": time_cftime})
    res = crop_time(ds, start_year="2000", end_year="2002")
    assert len(res.time) == 3


def test_crop_time_integer_coords():
    ds = xr.Dataset(
        {"data": ("time", [1, 2, 3, 4, 5])},
        coords={"time": [1999, 2000, 2001, 2002, 2003]}
    )
    res = crop_time(ds, start_year=2000, end_year=2002)
    assert len(res.time) == 3
