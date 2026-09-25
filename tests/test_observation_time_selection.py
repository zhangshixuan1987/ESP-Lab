import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from esp_lab.data_access_obs import crop_time


@pytest.mark.parametrize("bounds", [(2000, 2002), ("2000", "2002")])
def test_crop_time_datetime64_accepts_integer_and_string_years(bounds):
    ds = xr.Dataset(
        {"data": ("time", [1, 2, 3, 4, 5])},
        coords={"time": pd.date_range("1999-01-01", periods=5, freq="YS")}
    )
    result = crop_time(ds, start_year=bounds[0], end_year=bounds[1])

    np.testing.assert_array_equal(result.time.dt.year, [2000, 2001, 2002])
    np.testing.assert_array_equal(result.data, [2, 3, 4])


def test_crop_time_cftime():
    time_cftime = [cftime.DatetimeNoLeap(y, 1, 1) for y in [1999, 2000, 2001, 2002, 2003]]
    ds = xr.Dataset({"data": ("time", [1, 2, 3, 4, 5])}, coords={"time": time_cftime})
    result = crop_time(ds, start_year="2000", end_year="2002")

    np.testing.assert_array_equal(result.time.dt.year, [2000, 2001, 2002])
    np.testing.assert_array_equal(result.data, [2, 3, 4])


def test_crop_time_integer_coords():
    ds = xr.Dataset(
        {"data": ("time", [1, 2, 3, 4, 5])},
        coords={"time": [1999, 2000, 2001, 2002, 2003]}
    )
    result = crop_time(ds, start_year=2000, end_year=2002)

    np.testing.assert_array_equal(result.time, [2000, 2001, 2002])
    np.testing.assert_array_equal(result.data, [2, 3, 4])


def test_crop_time_without_time_coordinate_returns_equivalent_dataset():
    source = xr.Dataset({"data": ("station", [1, 2, 3])})

    result = crop_time(source, start_year=2000, end_year=2002)

    xr.testing.assert_identical(result, source)


def test_crop_time_rejects_reversed_year_bounds():
    source = xr.Dataset(
        {"data": ("time", [1, 2])},
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="YS")},
    )

    with pytest.raises(ValueError, match="must be <="):
        crop_time(source, start_year=2002, end_year=2000)
