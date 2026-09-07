import xarray as xr
import numpy as np
import scipy.sparse as sps
import cf_xarray
import xesmf as xe


def remap_camse(ds, dsw, varlst=None):
    """
    Remap CAM-SE / E3SM native unstructured ncol data to a structured lat-lon grid
    using a sparse weight file.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset with ncol dimension.
    dsw : xr.Dataset
        Weight dataset containing S, row, col, xc_b, yc_b, dst_grid_dims.
    varlst : list[str] or None
        Variables to remap. If None, all variables containing ncol are remapped.

    Returns
    -------
    xr.Dataset
        Remapped dataset on lat/lon grid.
    """
    if varlst is None:
        varlst = []

    if 'ncol' not in ds.dims:
        return ds.copy()

    # Drop ncol dimension cleanly
    dso = ds.drop_dims('ncol', errors='ignore').copy()

    dst_lon_dim = int(np.asarray(dsw.dst_grid_dims[0]))
    dst_lat_dim = int(np.asarray(dsw.dst_grid_dims[1]))

    lonb = np.asarray(dsw.xc_b).reshape([dst_lat_dim, dst_lon_dim])
    latb = np.asarray(dsw.yc_b).reshape([dst_lat_dim, dst_lon_dim])

    weights = sps.coo_matrix(
        (dsw.S, (dsw.row - 1, dsw.col - 1)),
        shape=[dsw.dims['n_b'], dsw.dims['n_a']]
    )

    if not varlst:
        varlst = [v for v in ds.data_vars if 'ncol' in ds[v].dims and v not in ('lon', 'lat', 'area')]

    remapped_vars = {}
    for varname in varlst:
        # Guarantee ncol is the last dimension before reshaping
        da = ds[varname].transpose(..., 'ncol')
        shape = da.shape
        invar_flat = da.values.reshape(-1, shape[-1])
        remapped_flat = weights.dot(invar_flat.T).T
        remapped_vals = remapped_flat.reshape(
            [*shape[0:-1], dst_lat_dim, dst_lon_dim]
        )

        dimlst = list(da.dims[0:-1])
        dims = list(dimlst) + ['lat', 'lon']
        coords = {dim: dso.coords[dim] for dim in dimlst if dim in dso.coords}
        coords['lat'] = latb[:, 0]
        coords['lon'] = lonb[0, :]

        remapped_vars[varname] = xr.DataArray(
            remapped_vals,
            coords=coords,
            dims=dims,
            attrs=da.attrs
        )
        
    dso = dso.assign(**remapped_vars)

    return dso


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


def has_ncol(ds):
    """
    Return True if dataset contains native unstructured ncol dimension.
    """
    return 'ncol' in ds.dims


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


def regrid_dataset(ds, dst, method='conservative', periodic=True,
                   reuse_weights=False, filename=None):
    """
    Regrid a structured lat-lon dataset to destination grid.
    """
    src_ready = prepare_latlon_ds(ds)
    regridder = make_regridder(
        src_ready, dst, method=method, periodic=periodic,
        reuse_weights=reuse_weights, filename=filename
    )
    return regridder(src_ready)


def regrid_dataarray(da, src_ds, dst, method='conservative', periodic=True,
                     reuse_weights=False, filename=None):
    """
    Regrid a DataArray using source/destination grid definitions.
    """
    src_ready = prepare_latlon_ds(src_ds)
    regridder = make_regridder(
        src_ready, dst, method=method, periodic=periodic,
        reuse_weights=reuse_weights, filename=filename
    )
    return regridder(da)


def remap_then_regrid(ds, dsw, dst, varlst=None, method='conservative', periodic=True,
                      reuse_weights=False, filename=None):
    """
    Two-step helper for native E3SM/CAM-SE:
    1) remap ncol -> structured lat/lon using sparse weights
    2) regrid structured lat/lon -> destination grid using xESMF
    """
    ds_latlon = remap_camse(ds, dsw, varlst=varlst)
    ds_latlon = prepare_latlon_ds(ds_latlon)
    ds_out = regrid_dataset(
        ds_latlon, dst, method=method, periodic=periodic,
        reuse_weights=reuse_weights, filename=filename
    )
    return ds_out


def auto_regrid(ds, dst, dsw=None, varlst=None, method='conservative', periodic=True,
                reuse_weights=False, filename=None):
    """
    Flexible entry point:
    - if ds has ncol, remap with dsw first, then regrid
    - if ds already has structured lat/lon, regrid directly

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset.
    dst : xr.Dataset
        Destination structured lat-lon grid.
    dsw : xr.Dataset or None
        Sparse weight file required when ds has ncol.
    """
    if has_ncol(ds):
        if dsw is None:
            raise ValueError("Source dataset has ncol; dsw weight file is required.")
        return remap_then_regrid(
            ds, dsw, dst, varlst=varlst, method=method, periodic=periodic,
            reuse_weights=reuse_weights, filename=filename
        )
    elif is_latlon_grid(ds):
        return regrid_dataset(
            ds, dst, method=method, periodic=periodic,
            reuse_weights=reuse_weights, filename=filename
        )
    else:
        raise ValueError("Unsupported grid: expected either ncol or structured 1D lat/lon.")