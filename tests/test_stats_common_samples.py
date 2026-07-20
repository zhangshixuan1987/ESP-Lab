import cftime
import numpy as np
import xarray as xr

from esp_lab import stats


def _hindcast(years, leads=(1, 2), members=2):
    leads = list(leads)
    times = np.asarray(
        [
            [cftime.DatetimeNoLeap(year + lead - 1, 1, 15) for lead in leads]
            for year in years
        ],
        dtype=object,
    )
    values = np.arange(len(years) * len(leads) * members, dtype=float).reshape(
        len(years), len(leads), members
    )
    index = xr.DataArray(
        values,
        dims=("Y", "L", "M"),
        coords={"Y": [f"{year}110100" for year in years], "L": leads, "M": range(members)},
    )
    valid_time = xr.DataArray(
        times,
        dims=("Y", "L"),
        coords={"Y": index.Y, "L": leads},
    )
    return index, valid_time


def test_common_valid_target_years_intersects_each_lead_separately():
    first, first_time = _hindcast([2000, 2001, 2002])
    second, second_time = _hindcast([2000, 2001, 2002])
    second.loc[{"Y": "2001110100", "L": 1}] = np.nan
    obs = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims="time",
        coords={
            "time": [
                cftime.DatetimeNoLeap(year, 1, 15)
                for year in range(2000, 2004)
            ]
        },
    )

    result = stats.common_valid_target_years_seasonal(
        {"first": first, "second": second},
        {"first": first_time, "second": second_time},
        obs,
        leads=[1, 2],
    )

    assert result[1] == [2000, 2002]
    assert result[2] == [2001, 2002, 2003]


def test_compute_skill_records_explicit_target_year_cohort():
    model, valid_time = _hindcast([2000, 2001, 2002, 2003], leads=(1,))
    obs = xr.DataArray(
        [0.0, 1.0, 2.0, 3.0],
        dims="time",
        coords={
            "time": [
                cftime.DatetimeNoLeap(year, 1, 15)
                for year in range(2000, 2004)
            ]
        },
    )

    result = stats.compute_skill_seasonal(
        model,
        valid_time,
        obs,
        nleads=1,
        detrend=False,
        is_anomaly=True,
        target_years_by_lead={1: [2000, 2002, 2003]},
    )

    assert int(result.sample_count.sel(L=1)) == 3
    assert int(result.target_year_start.sel(L=1)) == 2000
    assert int(result.target_year_end.sel(L=1)) == 2003
