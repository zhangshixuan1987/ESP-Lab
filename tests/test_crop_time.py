import pandas as pd
import xarray as xr


def _crop_time(ds, start_year=None, end_year=None):
    if "time" not in ds.coords:
        return ds
    if start_year is None and end_year is None:
        return ds
    time_slice = slice(
        str(start_year) if start_year is not None else None,
        str(end_year) if end_year is not None else None
    )
    return ds.sel(time=time_slice)


def test_crop_time_single_year():
    ds = xr.Dataset({"data": ("time", [1, 2, 3])},
                    coords={"time": pd.date_range("2000-01-01", periods=3, freq="YS")})
    result = _crop_time(ds, start_year=2000, end_year=2001)
    assert len(result.time) == 2


def test_crop_time_no_time_coord():
    ds = xr.Dataset({"data": ("x", [1, 2, 3])})
    result = _crop_time(ds)
    assert "time" not in result.coords
