import cftime
import numpy as np
import xarray as xr

from esp_lab import eli


def _hindcast(years, missing=None):
    leads = [1, 2]
    values = np.arange(len(years) * 2 * 2, dtype=float).reshape(len(years), 2, 2)
    data = xr.DataArray(
        values,
        dims=("Y", "L", "M"),
        coords={"Y": years, "L": leads, "M": [0, 1]},
    )
    if missing is not None:
        data.loc[missing] = np.nan
    time = xr.DataArray(
        [[cftime.DatetimeNoLeap(year, lead, 15) for lead in leads] for year in years],
        dims=("Y", "L"),
        coords={"Y": years, "L": leads},
    )
    return data, time


def test_initialization_years_supports_numeric_and_tagged_coordinates():
    np.testing.assert_array_equal(eli.initialization_years([1980, 1981]), [1980, 1981])
    np.testing.assert_array_equal(
        eli.initialization_years(["1980050100", "1981050100"]), [1980, 1981]
    )


def test_common_target_years_are_intersected_per_lead():
    first, first_time = _hindcast([2000, 2001, 2002, 2003])
    second, second_time = _hindcast(
        [2000, 2001, 2002, 2003], missing={"Y": 2001, "L": 1}
    )
    observation = xr.DataArray(
        np.arange(6, dtype=float),
        dims="time",
        coords={
            "time": [
                cftime.DatetimeNoLeap(year, month, 15)
                for year in [2000, 2001, 2002, 2003]
                for month in [1, 2]
            ]
        },
    )

    cohorts = eli.common_target_years_by_lead(
        {"first": {5: first}, "second": {5: second}},
        {"first": {5: first_time}, "second": {5: second_time}},
        observation,
        5,
    )

    assert cohorts == {1: [2000, 2002, 2003], 2: [2000, 2001, 2002, 2003]}
