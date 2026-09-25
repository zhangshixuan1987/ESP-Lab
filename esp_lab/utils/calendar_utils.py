# routines for dealing with the calendar e.g., calculating timeseries of seasonal means
import xarray as xr
import pandas as pd
from math import nan
import cftime
import cf_units


def time_set_midmonth(ds, time_name, deep=False):
    """
    Return copy of ds with values of ds[time_name] replaced with mid-month
    values (day=15) rather than end-month values.
    
    Author: S. Yeager
    """

    #ds_out = ds.copy(deep)
    year = ds[time_name].dt.year
    month = ds[time_name].dt.month
    year = xr.where(month==1,year-1,year)
    month = xr.where(month==1,12,month-1)
    nmonths = len(month)
    newtime = [cftime.DatetimeNoLeap(year[i], month[i], 15) for i in range(nmonths)]
    ds[time_name] = newtime

    return ds


def mon_to_seas_dask(ds):
    """ Converts a Dask DataSet containing monthly data to one containing 
    seasonal-average data. Dask Dataset is assumed to be in initialized-prediction
    format, with dimensions (Y,L,M,...). Time should be a variable, not a coordinate.
    """
    # drop time(Y,L) variable for now
    ds_seas = ds.drop_vars('time')
    # do a simple 3-month rolling mean along L-dimension
    ds_seas = ds_seas.rolling(L=3,min_periods=3, center=True).mean()
    # add time back into dataset
    ds_seas['time'] = ds['time']
    # subselect seasons:  DJF, MAM, JJA, SON
    mon = ds_seas.isel(Y=0).time.dt.month.values
    L = ds_seas.L
    Lkeep = L.where((mon == 1) | (mon == 4) | (mon == 7) | (mon == 10)).dropna('L')
    ds_seas = ds_seas.sel(L=Lkeep)
    return ds_seas

