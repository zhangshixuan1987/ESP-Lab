import cftime
from functools import partial
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr

from esp_lab.data_access_smyle import time_set_midmonth
from esp_lab.data_access_smyle import file_dict
from esp_lab.data_access_smyle import get_monthly_data
from esp_lab.data_access_smyle import nested_file_list_by_year
from esp_lab.data_access_smyle import preprocessor

_TEST_DATA = Path(__file__).parent / "test_data"


def test_file_dict():
    """
    Test the file_dict function.
    """

    filetemplate = str(_TEST_DATA / 'b.e21.BSMYLE.f09_g17.????-MM.EEE.pop.h.zsatcalc.*.nc')
    filetype = '.pop.h.'
    mem = 3
    stmon = 2

    filepaths = file_dict(filetemplate, filetype, mem, stmon)

    expected = str(_TEST_DATA / 'b.e21.BSMYLE.f09_g17.1986-02.003.pop.h.zsatcalc.198602-198801.nc')
    assert filepaths[1986] == expected
    assert len(filepaths.keys()) == 3


def test_get_monthly_data():
    """
    Test the get_monthly_data function.
    """
    # filetemplate = 'tests/test_data/b.e21.BSMYLE.f09_g17.????-MM.EEE.pop.h.zsatcalc.*.nc'
    # filetype = '.pop.h.'
    # ens = 3
    # firstyear = 1986
    # lastyear = 1988
    # stmon = 2
    # nlead = 24
    # field = 'zsatcalc'
    # preproc = preprocessor

    # ds0 = get_monthly_data(filetemplate, filetype, ens, nlead, field,
    #                  firstyear, lastyear, stmon, preproc)

    # print("ds0 {}".format(ds0))

    # todo: make test

    assert True


def test_nested_file_list_by_year():
    """
    Test the nested_file_list_by_year function.
    """
    filetemplate = str(_TEST_DATA / 'b.e21.BSMYLE.f09_g17.????-MM.EEE.pop.h.zsatcalc.*.nc')
    filetype = '.pop.h.'
    ens = 3
    start_years = [1986, 1987, 1988]
    stmon = 2

    nested_files = nested_file_list_by_year(filetemplate, filetype, ens, start_years, stmon)

    assert nested_files[1][2] == 1988


def test_bad_nested_file_list_by_year():
    """
    Test the nested_file_list_by_year function for a bad value.
    """
    filetemplate = str(_TEST_DATA / 'b.e21.BSMYLE.f09_g17.????-MM.EEE.pop.h.NONEXISTANT_FIELD.*.nc')
    filetype = '.pop.h.'
    ens = 3
    start_years = [1986, 1987, 1988]
    stmon = 2

    nested_files = nested_file_list_by_year(filetemplate, filetype, ens, start_years, stmon)

    assert nested_files[0] == []


def test_preprocessor():
    """
    Test the preprocessor function.
    """
    # todo: make test

    assert True


def test_time_set_midmonth():
    """
    Test the time_set_midmonth function.
    """
    ds = xr.Dataset(
        {"foo": ("time", np.arange(2))},
        coords={
            "time": [
                cftime.DatetimeNoLeap(1980, 6, 1),
                cftime.DatetimeNoLeap(1980, 6, 30),
            ]
        },
    )

    out = time_set_midmonth(ds, "time")

    assert list(out.time.values) == [
        cftime.DatetimeNoLeap(1980, 5, 15),
        cftime.DatetimeNoLeap(1980, 6, 15),
    ]


def test_smyle_time_set_midmonth_preserves_feb29_month_end():
    ds = xr.Dataset(
        {"foo": ("time", np.arange(3))},
        coords={"time": pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31"])},
    )

    out = time_set_midmonth(ds, "time")

    assert list(out.time.values) == [
        np.datetime64("2000-01-15T00:00:00.000000000"),
        np.datetime64("2000-02-15T00:00:00.000000000"),
        np.datetime64("2000-03-15T00:00:00.000000000"),
    ]
