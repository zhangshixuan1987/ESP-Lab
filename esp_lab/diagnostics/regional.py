import xarray as xr

def build_landmask(da, spatial):
    lm = spatial.create_land_sea_mask(
        da,
        lat_key="lat",
        lon_key="lon",
    )
    return lm.astype("int8").rename("sftlf")


def compute_regional_mean(da, weights):
    return da.weighted(weights).mean(("lat", "lon"))


def compute_weights(data_access, da, regionlonlat, mask):
    return data_access.e3sm_regional_weights(
        da,
        regionlonlat,
        lat_name="lat",
        lon_name="lon",
        mask=mask,
    )
