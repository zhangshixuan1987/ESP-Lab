"""Monthly skill keeps L=1..N and leaves leads without reference months missing."""

import numpy as np
import pandas as pd
import xarray as xr

from workflows.leadtime_skill.monthly_skill import produce_monthly_skill_cache


def test_leads_without_reference_month_are_missing_not_fatal(tmp_path):
    rng = np.random.default_rng(0)
    years = np.arange(2000, 2008)
    nlead = 4  # November starts: Dec, Jan, Feb, Mar
    valid = np.array([[pd.Timestamp(y, 11, 15) + pd.DateOffset(months=l + 1) for l in range(nlead)]
                      for y in years], dtype="datetime64[ns]")
    time = xr.DataArray(valid, dims=("Y", "L"), coords={"Y": years, "L": np.arange(1, nlead + 1)})
    model = xr.DataArray(rng.normal(size=(len(years), nlead, 3, 1, 1)), dims=("Y", "L", "M", "lat", "lon"),
                         coords={"Y": years, "L": np.arange(1, nlead + 1), "lat": [0.0], "lon": [0.0]})
    obs_time = pd.date_range("1999-01-15", "2009-12-15", freq="MS") + pd.Timedelta(days=14)
    obs_time = obs_time[obs_time.month != 1]  # the reference never has January
    obs = xr.DataArray(rng.normal(size=(len(obs_time), 1, 1)), dims=("time", "lat", "lon"),
                       coords={"time": obs_time, "lat": [0.0], "lon": [0.0]})

    skill = produce_monthly_skill_cache(
        model, time, obs, climatology_years=(2000, 2007), cache_path=tmp_path / "skill.nc",
        expected_attrs={"field": "X"}, detrend=False,
    )

    assert skill.sizes["L"] == nlead
    assert np.isnan(skill["corr"].sel(L=2)).all()  # January target
    assert np.isfinite(skill["corr"].sel(L=[1, 3, 4])).all()
    assert skill.attrs["unverifiable_leads"] == "2"
