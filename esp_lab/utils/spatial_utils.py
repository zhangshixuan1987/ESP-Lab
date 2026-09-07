import warnings
import numpy as np
import xarray as xr

try:
    import regionmask
except (ImportError, ValueError):
    regionmask = None

try:
    import xesmf as xe
except ImportError:
    xe = None


def _normalize_lon_180(lon):
    """Normalize longitude to [-180, 180]."""
    return (lon + 180) % 360 - 180


def _slice_coord(coord, bounds):
    """Return a slice that works for ascending or descending coordinates."""
    if len(bounds) != 2:
        raise ValueError(f"bounds must have exactly 2 elements, got {len(bounds)}.")
    if coord.size == 0:
        raise ValueError("Coordinate has no values.")
    v0 = coord.values[0]
    v1 = coord.values[-1]
    if v0 <= v1:
        return slice(bounds[0], bounds[1])
    return slice(bounds[1], bounds[0])


def _prep_lon_bounds(lon_bnds, lon_coord):
    """
    Convert lon bounds into the convention used by lon_coord.
    Assumes the region does not cross the dateline.
    """
    lon_min, lon_max = lon_bnds
    lon_vals = lon_coord.values
    if abs(lon_max - lon_min) >= 360:
        return float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals))

    if np.nanmin(lon_vals) >= 0 and np.nanmax(lon_vals) > 180:
        lon_min = lon_min % 360
        lon_max = lon_max % 360
    else:
        lon_min = _normalize_lon_180(lon_min)
        lon_max = _normalize_lon_180(lon_max)

    if lon_min > lon_max:
        warnings.warn(
            f"lon_bnds ({lon_bnds}) appears to cross the dateline after normalization "
            f"(lon_min={lon_min}, lon_max={lon_max}). "
            "This function does not support dateline-crossing regions; "
            "the resulting selection may be incorrect."
        )

    return lon_min, lon_max


def _infer_xy_keys(obj, lat_key=None, lon_key=None):
    """Infer lat/lon coordinate names if not provided."""
    if lat_key is None:
        for k in ["lat", "latitude", "y"]:
            if k in obj.coords:
                lat_key = k
                break
    if lon_key is None:
        for k in ["lon", "longitude", "x"]:
            if k in obj.coords:
                lon_key = k
                break

    if lat_key is None or lon_key is None:
        raise ValueError("Could not infer latitude/longitude coordinate names.")

    return lat_key, lon_key


def _ensure_dataset(obj, data_var="__tmpvar__"):
    """Convert DataArray to Dataset if needed."""
    if isinstance(obj, xr.Dataset):
        return obj.copy(), None
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=data_var), data_var
    raise TypeError("obj must be an xarray Dataset or DataArray.")


def _coord_bounds_1d(coord):
    """
    Construct 1D cell bounds from 1D coordinate centers.
    Returns array of size N+1 for xesmf.
    """
    vals = np.asarray(coord.values, dtype=float)
    if vals.ndim != 1:
        raise ValueError("Bounds helper expects a 1D coordinate.")
    if vals.size == 0:
        raise ValueError("Need at least 1 coordinate value to infer bounds.")

    name = str(coord.name).lower() if coord.name else ""
    if vals.size == 1:
        half_width = 0.5
        bounds = np.array([vals[0] - half_width, vals[0] + half_width])
        if "lat" in name or name == "y":
            bounds = np.clip(bounds, -90.0, 90.0)
        return xr.DataArray(bounds, dims=(f"{coord.dims[0]}_b",))

    mid = 0.5 * (vals[1:] + vals[:-1])
    first = vals[0] - 0.5 * (vals[1] - vals[0])
    last = vals[-1] + 0.5 * (vals[-1] - vals[-2])

    bounds = np.concatenate([[first], mid, [last]])
    
    if "lat" in name or name == "y":
        bounds = np.clip(bounds, -90.0, 90.0)

    # Name the dimension something unique so it doesn't conflict
    return xr.DataArray(bounds, dims=(f"{coord.dims[0]}_b",))


def _make_grid_ds(lat, lon, lat_name="lat", lon_name="lon"):
    """Build a minimal rectilinear grid dataset for xesmf."""
    return xr.Dataset(
        coords={
            lat_name: lat,
            lon_name: lon,
            f"{lat_name}_b": _coord_bounds_1d(lat),
            f"{lon_name}_b": _coord_bounds_1d(lon),
        }
    )


def _xesmf_regrid_dataarray(da, target_grid, method="conservative", periodic=True):
    """Regrid a 2D DataArray with dims (lat, lon) onto target_grid using xesmf."""
    if xe is None:
        raise ImportError("xesmf is required for method='iterative'.")

    src_grid = _make_grid_ds(da["lat"], da["lon"], "lat", "lon")
    dst_grid = _make_grid_ds(target_grid["lat"], target_grid["lon"], "lat", "lon")

    regridder = xe.Regridder(
        src_grid,
        dst_grid,
        method=method,
        periodic=periodic,
        ignore_degenerate=True,
        reuse_weights=False,
    )
    return regridder(da)


def _build_regionmask_land_source(resolution=0.25, data_var="sftlf"):
    """
    Build a high-resolution binary land source using regionmask.
    """
    if regionmask is None:
        raise ImportError("regionmask is required to build the default land mask source.")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}.")

    lat_vals = np.arange(-90 + resolution / 2, 90, resolution)
    lon_vals = np.arange(0 + resolution / 2, 360, resolution)

    lat = xr.DataArray(
        lat_vals,
        dims=("lat",),
        coords={"lat": lat_vals},
        name="lat",
        attrs={"units": "degrees_north"},
    )
    lon = xr.DataArray(
        lon_vals,
        dims=("lon",),
        coords={"lon": lon_vals},
        name="lon",
        attrs={"units": "degrees_east"},
    )

    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = land.mask(lon, lat=lat)

    sftlf = xr.where(mask.isnull(), 0.0, 1.0).rename(data_var)
    return xr.Dataset({data_var: sftlf})


def create_land_sea_mask(
    obj,
    lon_key=None,
    lat_key=None,
    as_boolean=False,
    method="iterative",
    source=None,
    data_var="sftlf",
    threshold_1=0.2,
    threshold_2=0.3,
    regrid_method="conservative",
):
    """
    Generate a land-sea mask.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Target object whose grid is used to create the mask.
    lon_key, lat_key : str, optional
        Longitude/latitude coordinate names.
    as_boolean : bool, default False
        If True, return boolean mask.
    method : {"iterative", "regionmask"}, default "iterative"
        - "regionmask": fast polygon-based land mask
        - "iterative": regrid high-resolution land mask and iteratively refine
                       to produce a grid-consistent land/sea mask
    source : xr.Dataset, optional
        Source land-fraction dataset with variable `data_var`.
    data_var : str, default "sftlf"
        Land-fraction variable name in source.
    threshold_1, threshold_2 : float
        Iterative refinement thresholds.
    regrid_method : str, default "conservative"
        xesmf regrid method.

    Returns
    -------
    xr.DataArray
        Land-sea mask on target grid.
    """
    ds_target, _ = _ensure_dataset(obj)
    lat_key, lon_key = _infer_xy_keys(ds_target, lat_key=lat_key, lon_key=lon_key)

    if ds_target[lat_key].ndim != 1 or ds_target[lon_key].ndim != 1:
        raise ValueError("Only 1D rectilinear lat/lon grids are supported.")

    if not (0.0 <= threshold_1 <= 1.0):
        raise ValueError(f"threshold_1 must be in [0, 1], got {threshold_1}.")
    if not (0.0 <= threshold_2 <= 1.0):
        raise ValueError(f"threshold_2 must be in [0, 1], got {threshold_2}.")

    method = method.lower()
    if method == "pcmdi":
        warnings.warn(
            "'pcmdi' is deprecated, use 'iterative' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "iterative"

    target_grid = xr.Dataset(
        coords={
            "lat": ds_target[lat_key].copy(),
            "lon": ds_target[lon_key].copy(),
        }
    )

    if method == "regionmask":
        if regionmask is None:
            raise ImportError("regionmask is required for method='regionmask'.")

        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        mask = land.mask(target_grid["lon"], lat=target_grid["lat"], wrap_lon=False)

        if as_boolean:
            out = xr.where(mask.isnull(), False, True)
        else:
            out = xr.where(mask.isnull(), 0, 1)

        return out.rename("lsmask")

    if method != "iterative":
        raise ValueError("method must be 'iterative' or 'regionmask'")

    out = _generate_iterative_land_mask(
        target_grid=target_grid,
        source=source,
        data_var=data_var,
        maskname="lsmask",
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        regrid_method=regrid_method,
    )

    if as_boolean:
        out = out.astype(bool)

    return out


def get_land_ocean_mask(
    da,
    lat_name="lat",
    lon_name="lon",
    return_ocean=False,
    method="iterative",
):
    """
    Return boolean land/ocean mask.
    True means keep.
    """
    land_mask = create_land_sea_mask(
        da,
        lon_key=lon_name,
        lat_key=lat_name,
        as_boolean=True,
        method=method,
    )
    return ~land_mask if return_ocean else land_mask


def compute_regional_average(
    da,
    lat_bnds,
    lon_bnds,
    lat_name="lat",
    lon_name="lon",
    weights=None,
    mask=None,
    land_mask=False,
    ocean_mask=False,
    mask_method="iterative",
):
    """
    Compute area-weighted average over a specified region.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray on a 1D rectilinear lat/lon grid.
    lat_bnds : tuple
        (min_lat, max_lat)
    lon_bnds : tuple
        (min_lon, max_lon)
    lat_name, lon_name : str
        Coordinate names.
    weights : xr.DataArray, optional
        Area weights. If None, cos(lat) weighting is used.
    mask : xr.DataArray, optional
        External boolean mask where True means keep.
    land_mask : bool, optional
        If True, internally generate a land-only mask.
    ocean_mask : bool, optional
        If True, internally generate an ocean-only mask.
    mask_method : {"iterative", "regionmask"}
        Method used for internal land/ocean mask generation.

    Returns
    -------
    xr.DataArray
        Weighted regional mean.
    """
    if land_mask and ocean_mask:
        raise ValueError("Cannot set both land_mask=True and ocean_mask=True.")
    if len(lat_bnds) != 2:
        raise ValueError(f"lat_bnds must have exactly 2 elements, got {len(lat_bnds)}.")
    if len(lon_bnds) != 2:
        raise ValueError(f"lon_bnds must have exactly 2 elements, got {len(lon_bnds)}.")
    if lat_bnds[0] > lat_bnds[1]:
        raise ValueError(f"lat_bnds[0] ({lat_bnds[0]}) must be <= lat_bnds[1] ({lat_bnds[1]}).")

    if da[lat_name].ndim != 1 or da[lon_name].ndim != 1:
        raise ValueError("This function currently expects 1D lat/lon coordinates.")

    lat_slice = _slice_coord(da[lat_name], lat_bnds)
    lon_min, lon_max = _prep_lon_bounds(lon_bnds, da[lon_name])
    lon_slice = _slice_coord(da[lon_name], (lon_min, lon_max))

    da_reg = da.sel({lat_name: lat_slice, lon_name: lon_slice})
    if da_reg[lat_name].size == 0 or da_reg[lon_name].size == 0:
        raise ValueError(
            f"Regional selection is empty for lat_bnds={lat_bnds}, lon_bnds={lon_bnds}. "
            "Check that bounds overlap the data domain."
        )

    if land_mask:
        effective_mask_method = mask_method
        if (
            mask_method == "iterative"
            and (da_reg[lat_name].size < 4 or da_reg[lon_name].size < 4)
        ):
            effective_mask_method = "regionmask"
        mask = get_land_ocean_mask(
            da_reg,
            lat_name=lat_name,
            lon_name=lon_name,
            return_ocean=False,
            method=effective_mask_method,
        )
    elif ocean_mask:
        effective_mask_method = mask_method
        if (
            mask_method == "iterative"
            and (da_reg[lat_name].size < 4 or da_reg[lon_name].size < 4)
        ):
            effective_mask_method = "regionmask"
        mask = get_land_ocean_mask(
            da_reg,
            lat_name=lat_name,
            lon_name=lon_name,
            return_ocean=True,
            method=effective_mask_method,
        )
    elif mask is not None:
        mask = mask.sel({lat_name: lat_slice, lon_name: lon_slice}).astype(bool)

    if mask is not None:
        da_reg = da_reg.where(mask)

    if weights is None:
        weights_reg = np.cos(np.deg2rad(da_reg[lat_name]))
        if mask is not None:
            # broadcast cos(lat) weights to 2D and zero out masked points so
            # they don't inflate the weighted-mean denominator
            weights_reg, _ = xr.broadcast(weights_reg, da_reg.isel(
                {k: 0 for k in da_reg.dims if k not in (lat_name, lon_name)},
                drop=True,
            ))
            weights_reg = weights_reg.where(mask, 0)
    else:
        weights_reg = weights.sel({lat_name: lat_slice, lon_name: lon_slice})
        if mask is not None:
            weights_reg = weights_reg.where(mask, 0)

    return da_reg.weighted(weights_reg).mean(dim=[lat_name, lon_name], skipna=True)


def _generate_iterative_land_mask(
    target_grid,
    source=None,
    data_var="sftlf",
    maskname="lsmask",
    threshold_1=0.2,
    threshold_2=0.3,
    regrid_method="conservative",
    max_iter=25,
):
    """
    Iterative land-sea mask generation using xarray + xesmf only.

    Notes
    -----
    - Rectilinear 1D lat/lon only.
    - If source is None, a high-resolution binary regionmask source is built internally.
    """
    if not (0.0 <= threshold_1 <= 1.0):
        raise ValueError(f"threshold_1 must be in [0, 1], got {threshold_1}.")
    if not (0.0 <= threshold_2 <= 1.0):
        raise ValueError(f"threshold_2 must be in [0, 1], got {threshold_2}.")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")
    if xe is None:
        raise ImportError("xesmf is required for method='iterative'.")

    if source is None:
        ds_src = _build_regionmask_land_source(resolution=0.25, data_var=data_var)
    else:
        if not isinstance(source, xr.Dataset):
            raise ValueError("source must be an xarray.Dataset")
        ds_src = source.copy()

    if "lat" not in ds_src.coords or "lon" not in ds_src.coords:
        raise ValueError("source must contain 1D 'lat' and 'lon' coordinates.")
    if data_var not in ds_src:
        raise ValueError(f"source does not contain variable '{data_var}'")

    frac = _xesmf_regrid_dataarray(
        ds_src[data_var],
        target_grid,
        method=regrid_method,
        periodic=True,
    ).clip(0.0, 1.0)

    mask = xr.where(frac > 0.5, 1, 0).astype(np.int8)

    nlat = mask.sizes["lat"]
    nlon = mask.sizes["lon"]
    if nlat < 3 or nlon < 3:
        return mask.rename(maskname)

    converged = False
    for _ in range(max_iter):
        new_mask = _improve_mask_once(mask, frac, threshold_1, threshold_2)
        if new_mask.identical(mask):
            converged = True
            break
        mask = new_mask

    if not converged:
        warnings.warn(
            f"Iterative land mask did not converge after {max_iter} iterations. "
            "Consider increasing max_iter or relaxing thresholds."
        )

    return mask.rename(maskname)


def _roll2d(da, lat_shift=0, lon_shift=0):
    """
    Roll in lon cyclically, and in lat without wrap contribution at the poles.
    """
    out = da
    if lon_shift != 0:
        out = out.roll(lon=lon_shift, roll_coords=False)
    if lat_shift != 0:
        out = out.shift(lat=lat_shift)
    return out


def _neighbor_stack(da):
    """
    Return the 8-neighbor stack for a (lat, lon) DataArray.
    """
    neighbors = [
        _roll2d(da,  1, -1),  # UL
        _roll2d(da,  1,  0),  # UC
        _roll2d(da,  1,  1),  # UR
        _roll2d(da,  0, -1),  # ML
        _roll2d(da,  0,  1),  # MR
        _roll2d(da, -1, -1),  # LL
        _roll2d(da, -1,  0),  # LC
        _roll2d(da, -1,  1),  # LR
    ]
    return xr.concat(neighbors, dim="neighbor")


def _map2four(mask):
    """
    Approximate fractional coverage from a binary mask using local neighborhood smoothing.
    """
    maskf = mask.astype(float)
    nbrs = _neighbor_stack(maskf)
    approx = (maskf + nbrs.mean("neighbor", skipna=True)) / 2.0
    return approx.clip(0.0, 1.0)


def _improve_mask_once(mask, frac, threshold_1=0.2, threshold_2=0.3):
    """
    One iteration of iterative coastal-mask refinement.
    """
    approx = _map2four(mask)
    diff = frac - approx

    nbr_frac = _neighbor_stack(frac)
    nbr_max = nbr_frac.max("neighbor", skipna=True)
    nbr_min = nbr_frac.min("neighbor", skipna=True)

    to_land = (
        (mask == 0)
        & (diff > threshold_1)
        & (frac > threshold_2)
        & (frac >= nbr_max.fillna(-np.inf))
    )

    to_ocean = (
        (mask == 1)
        & (diff < -threshold_1)
        & (frac < (1.0 - threshold_2))
        & (frac <= nbr_min.fillna(np.inf))
    )

    new_mask = xr.where(to_land, 1, mask)
    new_mask = xr.where(to_ocean, 0, new_mask)

    # keep polar rows unchanged using xr.where (avoids unreliable item assignment)
    if new_mask.sizes["lat"] >= 2:
        lat_coord = new_mask["lat"]
        is_polar = (lat_coord == lat_coord[0]) | (lat_coord == lat_coord[-1])
        new_mask = new_mask.where(~is_polar, mask)

    return new_mask.astype(np.int8)
