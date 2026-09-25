import xarray as xr
import numpy as np
import cf_xarray
import xesmf as xe


def add_grid_bounds(ds):
    """
    Add lat/lon bounds for structured 1D lat-lon grids using cf_xarray.
    Clip latitude bounds to [-90, 90].
    """
    # Only add bounds if they do not exist to avoid conflicting operations
    if 'lon_bounds' not in ds.variables or 'lat_bounds' not in ds.variables:
        ds = ds.cf.add_bounds(['lon', 'lat'])

    if 'lat_bounds' in ds.variables:
        lb = ds['lat_bounds']
        if ((lb < -90) | (lb > 90)).any():
            saveattrs = ds['lat'].attrs
            lb = xr.where(lb > 90, 90, lb)
            lb = xr.where(lb < -90, -90, lb)
            ds['lat_bounds'] = lb
            ds['lat'] = ds['lat'].assign_attrs(saveattrs)

    return ds


def make_latlon_grid(dlat=5.0, dlon=5.0, lon0=0.0, lon1=360.0, lat0=-90.0, lat1=90.0,
                     add_bounds=True, add_mask=True, add_area=True):
    """
    Create a regular structured lat-lon destination grid.
    """
    lat = np.arange(lat0, lat1 + dlat, dlat)
    lon = np.arange(lon0, lon1, dlon)

    ds = xr.Dataset({
        'lat': xr.DataArray(lat, dims='lat'),
        'lon': xr.DataArray(lon, dims='lon')
    })

    ny = ds.sizes['lat']
    nx = ds.sizes['lon']

    if add_mask:
        ds['mask'] = xr.DataArray(
            np.ones((ny, nx)),
            dims=['lat', 'lon'],
            coords={'lat': ds.lat, 'lon': ds.lon}
        )

    if add_area:
        rEarth = 6378.1  # km
        ds['area'] = xr.DataArray(
            np.ones((ny, nx)),
            dims=['lat', 'lon'],
            coords={'lat': ds.lat, 'lon': ds.lon}
        ) * np.cos(np.deg2rad(ds.lat)) * rEarth**2

    ds['lat'] = ds['lat'].assign_attrs({'units': 'degrees_north', 'long_name': 'latitude'})
    ds['lon'] = ds['lon'].assign_attrs({'units': 'degrees_east', 'long_name': 'longitude'})

    if add_bounds:
        ds = add_grid_bounds(ds)

    return ds


def is_latlon_grid(ds, lat_name='lat', lon_name='lon'):
    """
    Return True if dataset has 1D structured lat/lon coordinates.
    """
    return (
        lat_name in ds.coords and lon_name in ds.coords
        and ds[lat_name].ndim == 1
        and ds[lon_name].ndim == 1
    )


def prepare_latlon_ds(ds, lon_name='lon', lat_name='lat', add_bounds_if_missing=True):
    """
    Prepare a structured lat-lon dataset for xESMF:
    - ensure 1D lat/lon
    - sort lat/lon if needed
    - convert lon to [0, 360) if needed
    - add bounds if missing
    """
    if not is_latlon_grid(ds, lat_name=lat_name, lon_name=lon_name):
        raise ValueError("Dataset is not a structured 1D lat-lon grid.")

    out = ds

    # normalize longitude to [0, 360)
    if (out[lon_name] < 0).any():
        out = out.assign_coords({lon_name: (out[lon_name] % 360)})
        out = out.sortby(lon_name)

    # ensure monotonic increasing coordinates
    if np.any(np.diff(out[lat_name].values) <= 0):
        out = out.sortby(lat_name)
    if np.any(np.diff(out[lon_name].values) <= 0):
        out = out.sortby(lon_name)

    if add_bounds_if_missing:
        if 'lat_bounds' not in out.variables or 'lon_bounds' not in out.variables:
            out = add_grid_bounds(out)

    return out


def make_regridder(src, dst, method='conservative', periodic=True,
                   reuse_weights=False, filename=None):
    """
    Build an xESMF regridder after preparing structured lat-lon source/destination grids.
    """
    src_ready = prepare_latlon_ds(src)
    dst_ready = prepare_latlon_ds(dst)

    return xe.Regridder(
        src_ready, dst_ready, method,
        periodic=periodic,
        reuse_weights=reuse_weights,
        filename=filename
    )


