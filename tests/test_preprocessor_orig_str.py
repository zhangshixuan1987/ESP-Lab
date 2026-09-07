import numpy as np
import pandas as pd
import xarray as xr
from esp_lab.data_access_obs import crop_time


def test_crop_time_string_args():
    time_pd = pd.date_range("1999-01-01", periods=24, freq="MS")
    ds = xr.Dataset({"obs_var": ("time", np.random.rand(24))}, coords={"time": time_pd})
    result = crop_time(ds, "2000", "2000")
    assert len(result.time) == 12
