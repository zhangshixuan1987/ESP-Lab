import numpy as np
import xarray as xr

from workflows.leadtime_skill.regional_acc import regional_acc


def _skill():
    return xr.Dataset({"corr": xr.DataArray(
        [[[0.1, 0.3], [0.5, 0.7]], [[0.2, 0.4], [0.6, 0.8]]],
        dims=("L", "lat", "lon"), coords={"L": [1, 2], "lat": [-20, 40], "lon": [250, 300]},
    )})


def test_regional_acc_applies_region_and_land_masks():
    regions = {
        "Global": {"bounds": (0, 360, -90, 90), "mask": None},
        "Land": {"bounds": (0, 360, -90, 90), "mask": "land"},
        "CONUS": {"bounds": (-124.78, -66.92, 24.7, 49.4), "mask": None},
    }
    land = xr.DataArray([[True, False], [False, True]], dims=("lat", "lon"), coords={"lat": [-20, 40], "lon": [250, 300]})
    area = xr.ones_like(land, dtype=float)
    result = regional_acc(_skill(), regions, land_mask=land, area=area)
    np.testing.assert_allclose(result.sel(region="Land"), [0.4, 0.5])
    # 250E is 110W and lies in CONUS; 300E is 60W and lies east of it.
    np.testing.assert_allclose(result.sel(region="CONUS"), [0.5, 0.6])
