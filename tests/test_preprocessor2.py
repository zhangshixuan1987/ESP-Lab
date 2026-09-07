import numpy as np
import pandas as pd
import xarray as xr
from esp_lab.data_access_obs import preprocessor_monthly


def test_preprocessor_monthly_datetime64():
    time_pd = pd.date_range("1999-01-01", periods=24, freq="MS")
    ds = xr.Dataset({"obs_var": ("time", np.random.rand(24))}, coords={"time": time_pd})
    res = preprocessor_monthly(ds, field="obs_var", start_year=2000, end_year=2000, harmonize_time=True)
    # Month-start observational timestamps stay in their represented month.
    assert len(res.time) == 12
