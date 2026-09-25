import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr


_MODULE_PATH = Path(__file__).resolve().parents[1] / "esp_lab" / "data_access_obs.py"
_SPEC = importlib.util.spec_from_file_location("data_access_obs", _MODULE_PATH)
obs_access = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(obs_access)


def _write_monthly_obs(path, var_name="PRECT"):
    time = [pd.Timestamp(2000, month, 15) for month in range(1, 13)]
    ds = xr.Dataset(
        {
            var_name: (
                ("time", "lat", "lon"),
                np.ones((12, 2, 3), dtype=np.float32),
            )
        },
        coords={
            "time": time,
            "lat": [-45.0, 45.0],
            "lon": [0.0, 120.0, 240.0],
        },
    )
    ds.to_netcdf(path)


def test_get_monthly_data_falls_back_to_native_obs_filename(tmp_path):
    product_dir = tmp_path / "GPCP_v2.3"
    product_dir.mkdir()
    _write_monthly_obs(product_dir / "PRECT_200001_200012.nc")

    ds = obs_access.get_monthly_data(
        obs_dir=str(tmp_path),
        product="GPCP_v2.3",
        field="PRECT",
        field_map={"PRECT": "pr"},
        start_year=2000,
        end_year=2000,
        harmonize_time=False,
    )

    assert list(ds.data_vars) == ["pr"]
    assert ds["pr"].sizes == {"time": 12, "lat": 2, "lon": 3}


def test_get_monthly_data_can_preserve_native_obs_variable_name(tmp_path):
    product_dir = tmp_path / "GPCP_v2.3"
    product_dir.mkdir()
    _write_monthly_obs(product_dir / "PRECT_200001_200012.nc")

    ds = obs_access.get_monthly_data(
        obs_dir=str(tmp_path),
        product="GPCP_v2.3",
        field="PRECT",
        field_map={"PRECT": "PRECT"},
        start_year=2000,
        end_year=2000,
        harmonize_time=False,
    )

    assert list(ds.data_vars) == ["PRECT"]
    assert ds["PRECT"].sizes == {"time": 12, "lat": 2, "lon": 3}


def test_obs_region_mask_treats_zero_to_360_as_full_longitude():
    da = xr.DataArray(
        np.ones((3, 4)),
        dims=("lat", "lon"),
        coords={"lat": [-30.0, 0.0, 30.0], "lon": [0.0, 90.0, 180.0, 270.0]},
    )

    mask = obs_access.obs_region_mask(da, [0.0, 360.0, -20.0, 20.0])

    assert mask.sel(lat=0.0).all()
    assert not mask.sel(lat=-30.0).any()
    assert not mask.sel(lat=30.0).any()
