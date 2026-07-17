import os
import glob
import argparse
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


DEFAULT_SEARCH_BASE_DIRS = (
    "/global/cfs/cdirs/e3sm/S2S2D/post_process",
    "/pscratch/sd/z/zhan391/e3sm_project/E3SMv3_S2D",
)


def _candidate_base_dirs(base_dir):
    seen = set()
    for candidate in (base_dir, *DEFAULT_SEARCH_BASE_DIRS):
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate


def _find_ts_dir(member_dir, frequency="monthly", allow_glb=True):
    grids = ("180x360_aave", "glb") if allow_glb else ("180x360_aave",)
    for grid in grids:
        ts_dir = os.path.join(member_dir, "post", "atm", grid, "ts", frequency, "2yr")
        if os.path.exists(ts_dir):
            return ts_dir, grid
    return None, None


def _simulation_has_ts_data(sim_dir, frequency="monthly"):
    member_dirs = sorted(glob.glob(os.path.join(sim_dir, "EN*"))) or [sim_dir]
    return any(
        _find_ts_dir(member_dir, frequency=frequency, allow_glb=True)[0] is not None
        for member_dir in member_dirs
    )


def _select_matching_file(files, init_tag=None):
    files = sorted(files)
    if not files:
        return None

    if init_tag:
        init_yyyymm = str(init_tag)[:6]
        tagged = [
            path for path in files
            if str(init_tag) in os.path.basename(path)
            or init_yyyymm in os.path.basename(path)
        ]
        if tagged:
            return tagged[0]
        raise FileNotFoundError(
            f"No file matched init_tag={init_tag}; candidates were: "
            f"{[os.path.basename(path) for path in files]}"
        )

    return files[0]


def _format_time_label(time_value):
    try:
        return f"{time_value.year:04d}-{time_value.month:02d}"
    except AttributeError:
        return np.datetime_as_string(np.datetime64(time_value), unit="M")


def _time_values_to_plot_axis(time_values, nlead):
    if time_values is None:
        return np.arange(1, nlead + 1), False

    values = np.asarray(time_values)[:nlead]
    if len(values) != nlead:
        return np.arange(1, nlead + 1), False

    try:
        return values.astype("datetime64[ns]"), True
    except (TypeError, ValueError):
        converted = []
        for value in values:
            if not all(hasattr(value, attr) for attr in ("year", "month")):
                return np.arange(1, nlead + 1), False
            day = getattr(value, "day", 1)
            hour = getattr(value, "hour", 0)
            minute = getattr(value, "minute", 0)
            second = getattr(value, "second", 0)
            try:
                converted.append(
                    datetime(
                        int(value.year),
                        int(value.month),
                        int(day),
                        int(hour),
                        int(minute),
                        int(second),
                    )
                )
            except ValueError:
                converted.append(
                    datetime(
                        int(value.year),
                        int(value.month),
                        min(int(day), 28),
                        int(hour),
                        int(minute),
                        int(second),
                    )
                )
        return np.asarray(converted), True


def _find_landfrac_file(ts_dir, init_tag=None):
    ts_dir = str(ts_dir)
    search_dirs = [ts_dir]
    monthly_dir = ts_dir.replace(os.path.join("ts", "6hourly", "2yr"), os.path.join("ts", "monthly", "2yr"))
    if monthly_dir != ts_dir:
        search_dirs.append(monthly_dir)

    post_marker = os.path.join(os.sep, "post", "atm", "")
    if post_marker in ts_dir:
        prefix, suffix = ts_dir.split(post_marker, 1)
        rel_parts = suffix.split(os.sep)
        if len(rel_parts) >= 2:
            grid = rel_parts[0]
            member_dir = prefix
            sim_dir = os.path.dirname(member_dir)
            if os.path.basename(member_dir).startswith("EN") and os.path.isdir(sim_dir):
                sibling_dirs = sorted(glob.glob(os.path.join(sim_dir, "EN*")))
                for sibling_dir in sibling_dirs:
                    sibling_monthly_dir = os.path.join(
                        sibling_dir,
                        "post",
                        "atm",
                        grid,
                        "ts",
                        "monthly",
                        "2yr",
                    )
                    if sibling_monthly_dir not in search_dirs:
                        search_dirs.append(sibling_monthly_dir)

    candidates = []
    for search_dir in search_dirs:
        candidates.extend(glob.glob(os.path.join(search_dir, "LANDFRAC_*.nc")))

    try:
        return _select_matching_file(candidates, init_tag=init_tag)
    except FileNotFoundError:
        return sorted(candidates)[0] if candidates else None


def _get_time_bounds(ds):
    bounds_name = ds["time"].attrs.get("bounds") if "time" in ds.coords else None
    if bounds_name not in ds:
        return None, None
    bounds = ds[bounds_name]
    return bounds, bounds.dims[-1]


def _has_zero_duration_initial_sample(ds, frequency="monthly"):
    """Detect saved initial conditions that are not interval means."""
    if frequency != "6hourly" or ds.sizes.get("time", 0) == 0:
        return False

    bounds, bounds_dim = _get_time_bounds(ds)
    if bounds is None:
        return False

    start = bounds.isel(time=0, **{bounds_dim: 0}).item()
    end = bounds.isel(time=0, **{bounds_dim: 1}).item()
    return start == end


def _time_offset_to_6hour_grid(value):
    """Return the offset from the nearest same-day 00/06/12/18 UTC grid."""
    if isinstance(value, np.datetime64):
        value_s = value.astype("datetime64[s]")
        day_s = value_s.astype("datetime64[D]").astype("datetime64[s]")
        return (value_s - day_s) % np.timedelta64(6, "h")

    return timedelta(
        hours=int(getattr(value, "hour", 0)) % 6,
        minutes=int(getattr(value, "minute", 0)),
        seconds=int(getattr(value, "second", 0)),
        microseconds=int(getattr(value, "microsecond", 0)),
    )


def _subtract_time_offset(values, offset):
    if isinstance(offset, np.timedelta64):
        return np.asarray(values) - offset
    return np.asarray([value - offset for value in values])


def _get_time_indices(ds, nlead, frequency="monthly"):
    """Return time indices to load."""
    time_size = ds.sizes.get("time", 0)
    if _has_zero_duration_initial_sample(ds, frequency=frequency):
        return np.arange(1, min(nlead + 1, time_size))
    return np.arange(min(nlead, time_size))


def _get_initial_value_mask(ds, time_indices, frequency="monthly"):
    """Return True for instantaneous initial samples that are not interval means."""
    if frequency != "6hourly":
        return np.zeros(len(time_indices), dtype=bool)

    bounds, bounds_dim = _get_time_bounds(ds)
    if bounds is None:
        return np.zeros(len(time_indices), dtype=bool)

    starts = bounds.isel(time=time_indices, **{bounds_dim: 0}).values
    ends = bounds.isel(time=time_indices, **{bounds_dim: 1}).values
    return np.asarray([start == end for start, end in zip(starts, ends)])


def _get_time_values(ds, time_indices, frequency="monthly"):
    """Return representative times for plotting and member alignment."""
    if frequency == "6hourly":
        bounds, bounds_dim = _get_time_bounds(ds)
        if bounds is not None:
            interval_starts = bounds.isel(time=time_indices, **{bounds_dim: 0}).values
            if (
                len(interval_starts) > 0
                and _has_zero_duration_initial_sample(ds, frequency=frequency)
            ):
                offset = _time_offset_to_6hour_grid(interval_starts[0])
                return _subtract_time_offset(interval_starts, offset)
            if all(
                interval_starts[i] < interval_starts[i + 1]
                for i in range(len(interval_starts) - 1)
            ):
                return interval_starts
    return ds["time"].isel(time=time_indices).values

# Variables configuration and conversions
VARIABLES_INFO = {
    'TREFHT': {
        'file_var': 'TREFHT',
        'units': '°C',
        'title': 'Reference Height Temperature (2m)',
        'conversion': lambda x: x - 273.15,
        'use_land_mask': True
    },
    'PRECT': {
        'file_var': 'PRECT',
        'units': 'mm/day',
        'title': 'Total Precipitation Rate',
        'conversion': lambda x: x * 8.64e7,
        'use_land_mask': False
    },
    'FLUT': {
        'file_var': 'FLUT',
        'units': 'W/m²',
        'title': 'Outgoing Longwave Radiation',
        'conversion': lambda x: x,
        'use_land_mask': False
    },
    'PSL': {
        'file_var': 'PSL',
        'units': 'hPa',
        'title': 'Sea Level Pressure',
        'conversion': lambda x: x / 100.0,
        'use_land_mask': False
    },
    'SWCF': {
        'file_var': 'SWCF',
        'units': 'W/m²',
        'title': 'Shortwave Cloud Forcing',
        'conversion': lambda x: x,
        'use_land_mask': False
    },
    'LWCF': {
        'file_var': 'LWCF',
        'units': 'W/m²',
        'title': 'Longwave Cloud Forcing',
        'conversion': lambda x: x,
        'use_land_mask': False
    }
}

DEFAULT_OBS_INFO = {
    'monthly': {
        'PRECT': {
            'path': '/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series/GPCP_v2.3/PRECT_197901_201712.nc',
            'file_var': 'PRECT',
            'label': 'GPCP v2.3',
            'conversion': lambda x: x,
        },
        'TREFHT': {
            'path': '/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series/ERA5/t2m_197901_201912.nc',
            'file_var': 't2m',
            'label': 'ERA5',
            'conversion': lambda x: x - 273.15,
        },
        'PSL': {
            'path': '/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series/ERA5/psl_197901_201912.nc',
            'file_var': 'psl',
            'label': 'ERA5',
            'conversion': lambda x: x / 100.0,
        },
        'FLUT': {
            'path': '/global/cfs/cdirs/e3sm/e3sm_diags/obs_for_e3sm_diags/time-series/ERA5/rlut_197901_201912.nc',
            'file_var': 'rlut',
            'label': 'ERA5',
            'conversion': lambda x: x,
        },
    },
    '6hourly': {
        'PRECT': {
            'path_template': '/global/cfs/cdirs/e3sm/zhan391/data/CVDP_RGD/ERA5.6hourly/ERA5.6hourly.en00.PRECT.{year}01-{year}12.nc',
            'file_var': 'PRECT',
            'label': 'ERA5',
            'conversion': lambda x: x * 4000.0,
        },
        'PSL': {
            'path_template': '/global/cfs/cdirs/e3sm/zhan391/data/CVDP_RGD/ERA5.6hourly/ERA5.6hourly.en00.PSL.{year}01-{year}12.nc',
            'file_var': 'PSL',
            'label': 'ERA5',
            'conversion': lambda x: x / 100.0,
        },
    },
}


def _coord_name(da, candidates):
    for name in candidates:
        if name in da.coords or name in da.dims:
            return name
    return None


def _month_key(value):
    try:
        return f"{int(value.year):04d}-{int(value.month):02d}"
    except AttributeError:
        return np.datetime_as_string(np.datetime64(value), unit="M")


def _month_keys(values):
    return np.asarray([_month_key(value) for value in np.asarray(values)])


def _time_key(value, frequency="monthly"):
    if frequency == "monthly":
        return _month_key(value)
    try:
        return (
            f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}T"
            f"{int(getattr(value, 'hour', 0)):02d}:"
            f"{int(getattr(value, 'minute', 0)):02d}"
        )
    except AttributeError:
        return np.datetime_as_string(np.datetime64(value), unit="m")


def _time_keys(values, frequency="monthly"):
    return np.asarray([_time_key(value, frequency=frequency) for value in np.asarray(values)])


def load_observation_timeseries(
    variable,
    reference_time=None,
    frequency="monthly",
    obs_info=None,
):
    """
    Load observation time series and compute a weighted spatial mean.
    """
    obs_catalog = obs_info or DEFAULT_OBS_INFO
    if frequency in obs_catalog:
        obs_cfg = obs_catalog[frequency].get(variable)
    else:
        obs_cfg = obs_catalog.get(variable)
    if obs_cfg is None:
        return None

    if 'path_template' in obs_cfg:
        if reference_time is None:
            print(f"Warning: {frequency} observations for {variable} require a reference time axis")
            return None
        ref_keys = _time_keys(reference_time, frequency=frequency)
        years = sorted({_time_key(value, frequency=frequency)[:4] for value in np.asarray(reference_time)})
        values = []
        times = []
        for year in years:
            path = obs_cfg['path_template'].format(year=year)
            if not os.path.exists(path):
                print(f"Warning: Observation file not found for {variable}: {path}")
                continue
            with xr.open_dataset(path) as ds:
                loaded = _load_observation_dataset(ds, obs_cfg, ref_keys, frequency=frequency)
                if loaded is not None:
                    values.extend(loaded['mean'])
                    times.extend(loaded['time'])
        if not values:
            print(f"Warning: No observation times matched {variable} model time axis")
            return None
        order = {key: i for i, key in enumerate(ref_keys)}
        paired = sorted(
            zip(times, values),
            key=lambda item: order.get(_time_key(item[0], frequency=frequency), len(order)),
        )
        return {
            'mean': np.asarray([value for _, value in paired]),
            'time': np.asarray([time for time, _ in paired]),
            'label': obs_cfg.get('label', 'Observations'),
            'source_file': obs_cfg.get('path_template'),
        }

    path = obs_cfg['path']
    if not os.path.exists(path):
        print(f"Warning: Observation file not found for {variable}: {path}")
        return None

    with xr.open_dataset(path) as ds:
        ref_keys = _time_keys(reference_time, frequency=frequency) if reference_time is not None else None
        loaded = _load_observation_dataset(ds, obs_cfg, ref_keys, frequency=frequency)
        if loaded is None:
            return None
        loaded['label'] = obs_cfg.get('label', 'Observations')
        loaded['source_file'] = path
        return loaded


def _load_observation_dataset(ds, obs_cfg, ref_keys=None, frequency="monthly"):
    file_var = obs_cfg['file_var']
    if file_var not in ds:
        print(f"Warning: Observation variable {file_var} not found")
        return None

    da = ds[file_var]
    if ref_keys is not None:
        obs_keys = _time_keys(ds['time'].values, frequency=frequency)
        selected_indices = [
            int(np.where(obs_keys == key)[0][0])
            for key in ref_keys
            if np.any(obs_keys == key)
        ]
        if not selected_indices:
            return None
        da = da.isel(time=selected_indices)

    lat_name = _coord_name(da, ("lat", "latitude"))
    lon_name = _coord_name(da, ("lon", "longitude"))
    if lat_name is None or lon_name is None:
        print("Warning: Could not identify observation lat/lon coordinates")
        return None

    weights = ds['area'] if 'area' in ds else np.cos(np.deg2rad(da[lat_name]))
    spatial_mean = da.weighted(weights).mean(dim=[lat_name, lon_name], skipna=True)
    spatial_mean = obs_cfg['conversion'](spatial_mean)

    return {
        'mean': np.asarray(spatial_mean.values),
        'time': np.asarray(da['time'].values),
    }


def load_simulation_timeseries(
    base_dir,
    simulations,
    variable,
    init_tag="1980050100",
    nlead=24,
    frequency="monthly",
):
    """
    Loads netcdf timeseries files for the selected variable and calculates spatial means.
    """
    if variable not in VARIABLES_INFO:
        raise ValueError(f"Variable {variable} not configured in VARIABLES_INFO")
    
    var_cfg = VARIABLES_INFO[variable]
    file_var = var_cfg['file_var']
    
    sim_data = {}
    
    for sim_name, sim_val in simulations.items():
        if isinstance(sim_val, dict):
            sim_subdir = sim_val.get('run_name')
            ens_filter = sim_val.get('ens')
        else:
            sim_subdir = sim_val
            ens_filter = None
            
        # Locate simulation directory by checking base_dir, CFS, and scratch.
        sim_dir_path = None
        for b_dir in _candidate_base_dirs(base_dir):
            test_path = os.path.join(b_dir, sim_subdir)
            if os.path.exists(test_path) and _simulation_has_ts_data(test_path, frequency=frequency):
                sim_dir_path = test_path
                break
                
        if sim_dir_path is None:
            print(f"Warning: Directory containing ts data does not exist for {sim_name}: {sim_subdir}")
            continue
            
        # Check if there are ENxx subdirectories
        all_en_dirs = sorted(glob.glob(os.path.join(sim_dir_path, "EN*")))
        
        # Decide if this is a single-member direct case or a multi-member case
        if all_en_dirs:
            # Filter ensemble members
            en_dirs = []
            if ens_filter is not None:
                if isinstance(ens_filter, int):
                    en_dirs = all_en_dirs[:ens_filter]
                elif isinstance(ens_filter, (list, tuple)):
                    for item in ens_filter:
                        if isinstance(item, int):
                            if 0 <= item < len(all_en_dirs):
                                en_dirs.append(all_en_dirs[item])
                        elif isinstance(item, str):
                            matching = [d for d in all_en_dirs if os.path.basename(d) == item]
                            if matching:
                                en_dirs.extend(matching)
                else:
                    en_dirs = all_en_dirs
            else:
                en_dirs = all_en_dirs
        else:
            # No ENxx subdirectories; treat the simulation directory itself as a single member
            en_dirs = [sim_dir_path]
            
        nmembers = len(en_dirs)
        if nmembers == 0:
            print(f"Warning: No matching ensemble members found for {sim_name}")
            continue
            
        member_timeseries = []
        source_files = []
        source_grids = []
        time_values = None
        initial_value_mask = None
        expected_length = None
        
        for en_dir in en_dirs:
            ts_dir, source_grid = _find_ts_dir(
                en_dir,
                frequency=frequency,
                allow_glb=not var_cfg['use_land_mask'],
            )
            if ts_dir is None:
                if var_cfg['use_land_mask']:
                    print(
                        f"Warning: land-masked {file_var} requires 180x360_aave "
                        f"{frequency} ts data; skipping member: {en_dir}"
                    )
                else:
                    print(f"Warning: ts directory does not exist for member: {en_dir}")
                continue
                
            # Locate file matching the variable
            pattern = os.path.join(ts_dir, f"{file_var}_*.nc")
            matching_files = glob.glob(pattern)
            if not matching_files:
                print(f"Warning: No files found matching pattern: {pattern}")
                continue
                
            try:
                file_path = _select_matching_file(matching_files, init_tag=init_tag)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
            
            try:
                with xr.open_dataset(file_path) as ds:
                    time_indices = _get_time_indices(
                        ds,
                        nlead,
                        frequency=frequency,
                    )
                    has_discarded_initial = _has_zero_duration_initial_sample(
                        ds,
                        frequency=frequency,
                    )
                    if len(time_indices) == 0 or (
                        len(time_indices) < nlead and not has_discarded_initial
                    ):
                        print(
                            f"Warning: Skipping {file_path}; expected at least "
                            f"{nlead} valid time steps, found {len(time_indices)}"
                        )
                        continue
                    if len(time_indices) < nlead:
                        print(
                            f"Warning: Using {len(time_indices)} valid time steps from "
                            f"{file_path}; discarded zero-duration initial sample"
                        )
                    da = ds[file_var].isel(time=time_indices)
                    
                    current_time_values = _get_time_values(
                        ds,
                        time_indices,
                        frequency=frequency,
                    )
                    current_initial_value_mask = _get_initial_value_mask(
                        ds,
                        time_indices,
                        frequency=frequency,
                    )

                    if (
                        time_values is not None
                        and not np.array_equal(current_time_values, time_values)
                    ):
                        print(
                            f"Warning: Skipping {file_path}; time coordinate does not "
                            "match previous valid members for this simulation"
                        )
                        continue

                    # Capture time values from first valid file for this simulation
                    if time_values is None:
                        time_values = current_time_values
                        initial_value_mask = current_initial_value_mask
                        
                    # Calculate spatial mean or use pre-computed global mean
                    if source_grid == "glb":
                        # glb files are already global means, index 0 of rgn dimension
                        # is the 'Global' region
                        if 'rgn' in da.dims:
                            spatial_mean = da.isel(rgn=0)
                        else:
                            spatial_mean = da
                    else:
                        # Compute area-weighted mean over the grid
                        lat = da['lat']
                        weights = np.cos(np.deg2rad(lat))
                        
                        if var_cfg['use_land_mask']:
                            lf_path = _find_landfrac_file(ts_dir, init_tag=init_tag)
                            if lf_path is not None:
                                with xr.open_dataset(lf_path) as ds_lf:
                                    lf = ds_lf['LANDFRAC']
                                    if "time" in lf.dims and lf.sizes.get('time', 0) != da.sizes.get('time', 0):
                                        lf = lf.isel(time=0, drop=True)
                                    elif lf.sizes.get('time', 0) > nlead:
                                        lf = lf.isel(time=slice(0, nlead))
                                    # Weight by both latitude cosine and land fraction
                                    weights = weights * lf
                            else:
                                print(
                                    f"Warning: LANDFRAC not found for {file_path}; "
                                    "using latitude-only weights"
                                )
                                    
                        weighted_da = da.weighted(weights)
                        spatial_dims = [dim for dim in ['lat', 'lon'] if dim in da.dims]
                        spatial_mean = weighted_da.mean(dim=spatial_dims, skipna=True)
                    
                    # Apply variable-specific conversion/scaling
                    spatial_mean = var_cfg['conversion'](spatial_mean)
                    values = np.asarray(spatial_mean.values)
                    if expected_length is None:
                        expected_length = values.shape[0]
                    elif values.shape[0] != expected_length:
                        print(
                            f"Warning: Skipping {file_path}; length {values.shape[0]} "
                            f"does not match previous members ({expected_length})"
                        )
                        continue
                    member_timeseries.append(values)
                    source_files.append(file_path)
                    source_grids.append(source_grid)
                    
            except Exception as e:
                print(f"Error reading/processing file {file_path}: {e}")
                continue
                
        if not member_timeseries:
            continue
            
        # Convert to numpy array of shape (nmembers, nlead)
        member_timeseries = np.array(member_timeseries)
        
        # Calculate ensemble stats
        ens_mean = np.nanmean(member_timeseries, axis=0)
        ens_std = np.nanstd(member_timeseries, axis=0)
        
        sim_data[sim_name] = {
            'members': member_timeseries,
            'mean': ens_mean,
            'std': ens_std,
            'time': time_values,
            'initial_value_mask': initial_value_mask,
            'source_files': source_files,
            'source_grids': source_grids,
            'frequency': frequency,
        }
        
    return sim_data

def plot_timeseries_comparison(
    sim_data,
    variable,
    output_dir,
    figsize=[10, 6],
    colors=None,
    fontz=14,
    frequency="monthly",
    output_tag=None,
    x_axis="time",
    obs_data=None,
):
    """
    Plots time series comparison across the simulations.
    """
    if not sim_data:
        print("No simulation data loaded to plot.")
        return
        
    var_cfg = VARIABLES_INFO[variable]
    
    if colors is None:
        colors = {
            'Reanalysis': '#10b981',       # Vibrant Emerald Green
            'JRA55_FOSIRL': '#f59e0b',     # Amber Yellow
            '4DEnVar_branch': '#3b82f6',   # Bright Blue
            '4DEnVar_hybrid': '#ef4444'    # Bright Red
        }
        
    plt.figure(figsize=figsize)
    plt.rc('font', size=fontz)
    plt.rc('axes', labelsize=fontz)
    plt.rc('xtick', labelsize=fontz - 2)
    plt.rc('ytick', labelsize=fontz - 2)
    plt.rc('legend', fontsize=fontz - 2)
    
    use_lead_axis = x_axis == "lead"
    use_date_axis = (not use_lead_axis) and all(
        _time_values_to_plot_axis(data.get('time'), len(data['mean']))[1]
        for data in sim_data.values()
    )
    
    for sim_name, data in sim_data.items():
        color = colors.get(sim_name, '#6b7280')
        nlead = len(data['mean'])
        if use_lead_axis:
            step_days = 0.25 if frequency == "6hourly" else 1.0
            x = np.arange(nlead) * step_days
        elif use_date_axis:
            x, _ = _time_values_to_plot_axis(data.get('time'), nlead)
        else:
            x = np.arange(1, nlead + 1)
            
        nmembers = data['members'].shape[0]
        
        # Plot individual members for ensemble runs
        if nmembers > 1:
            for i in range(nmembers):
                plt.plot(x, data['members'][i], color=color, alpha=0.25, linewidth=1.0)
            label = f"{sim_name} Mean ({nmembers} mem)"
        else:
            label = f"{sim_name} (1 mem)"
            
        # Plot ensemble mean
        plt.plot(x, data['mean'], color=color, linewidth=2.5, label=label)

        initial_mask = data.get('initial_value_mask')
        if initial_mask is not None and np.any(initial_mask):
            initial_mask = np.asarray(initial_mask, dtype=bool)
            plt.scatter(
                np.asarray(x)[initial_mask],
                np.asarray(data['mean'])[initial_mask],
                facecolors='white',
                edgecolors=color,
                linewidths=1.8,
                s=55,
                zorder=5,
                label="_nolegend_",
            )
        
        # Plot ensemble spread (mean +/- 1 std) for ensemble runs
        if nmembers > 1:
            plt.fill_between(x, data['mean'] - data['std'], data['mean'] + data['std'], 
                             color=color, alpha=0.1)

    if obs_data is not None:
        nlead = len(obs_data['mean'])
        if use_lead_axis:
            step_days = 0.25 if frequency == "6hourly" else 1.0
            obs_x = np.arange(nlead) * step_days
        elif use_date_axis:
            obs_x, obs_is_time = _time_values_to_plot_axis(obs_data.get('time'), nlead)
            if not obs_is_time:
                obs_x = np.arange(1, nlead + 1)
        else:
            obs_x = np.arange(1, nlead + 1)
        obs_label = obs_data.get('label', 'Observations')
        plt.plot(
            obs_x,
            obs_data['mean'],
            color='black',
            linestyle='--',
            linewidth=2.6,
            label=obs_label,
            zorder=6,
        )
            
    frequency_label = "6-hourly" if frequency == "6hourly" else "Monthly"
    plt.title(
        f"{frequency_label} {var_cfg['title']} ({variable})",
        fontsize=fontz + 2,
        fontweight='bold',
        pad=15,
    )
    if use_lead_axis:
        xlabel = "Lead Day" if frequency == "6hourly" else "Lead"
    else:
        xlabel = "Verification Date" if use_date_axis else "Lead Month"
    plt.xlabel(xlabel, fontsize=fontz, labelpad=10)
    mean_label = "Land Mean" if var_cfg['use_land_mask'] else "Global Mean"
    plt.ylabel(f"{mean_label} ({var_cfg['units']})", fontsize=fontz, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if use_lead_axis:
        max_nlead = max(len(data['mean']) for data in sim_data.values())
        step_days = 0.25 if frequency == "6hourly" else 1.0
        max_lead = (max_nlead - 1) * step_days
        if frequency == "6hourly":
            interval = 5 if max_lead <= 45 else 30
            plt.xticks(np.arange(0, max_lead + interval, interval))
        else:
            plt.xticks(np.arange(0, max_lead + 1, 2))
        plt.xlim(0, max_lead)
    elif use_date_axis:
        ax = plt.gca()
        if frequency == "6hourly":
            x_limits = ax.get_xlim()
            span_days = x_limits[1] - x_limits[0]
            if span_days <= 45:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
            else:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
    else:
        max_nlead = max(len(data['mean']) for data in sim_data.values())
        plt.xticks(np.arange(1, max_nlead + 1, 2))
        
    if not use_date_axis and not use_lead_axis:
        max_nlead = max(len(data['mean']) for data in sim_data.values())
        plt.xlim(1, max_nlead)
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='lightgray')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    if output_tag is not None:
        suffix = f"{output_tag}_{variable}"
    else:
        suffix = variable if frequency == "monthly" else f"{frequency}_{variable}"
    out_path = os.path.join(output_dir, f"e3sm_timeseries_{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Success! Saved time series comparison to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot area-weighted monthly time series comparisons across E3SM simulations.")
    parser.add_argument("--base-dir", type=str, default="/pscratch/sd/z/zhan391/e3sm_project/E3SMv3_S2D", help="Base directory of E3SM S2D runs")
    parser.add_argument("--output-dir", type=str, default="/global/cfs/cdirs/e3sm/www/zhan391/E3SMv3_S2D", help="Output directory for comparison plots")
    parser.add_argument("--variable", type=str, required=True, choices=list(VARIABLES_INFO.keys()), help="Variable to process and plot")
    parser.add_argument("--nlead", type=int, default=24, help="Number of lead months")
    parser.add_argument("--frequency", type=str, default="monthly", choices=["monthly", "6hourly"], help="Time-series frequency")
    parser.add_argument("--fontz", type=int, default=14, help="Font size scaling")
    
    args = parser.parse_args()
    
    simulations = {
        'Reanalysis': {'run_name': 'WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_BruteForce_1980050100', 'ens': 10},
        'JRA55_FOSIRL': {'run_name': 'WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_JRA55_FOSIRL_1980050100', 'ens': 10},
        '4DEnVar_branch': {'run_name': 'test_WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_4DEnVar_branch', 'ens': 1},
        '4DEnVar_hybrid': {'run_name': 'test_WCYCL20TR_ne30pg2_r05_IcoswISC30E3r5_4DEnVar_hybrid', 'ens': 1},
    }
    
    print(f"--- Processing variable: {args.variable} ---")
    data = load_simulation_timeseries(
        args.base_dir,
        simulations,
        args.variable,
        nlead=args.nlead,
        frequency=args.frequency,
    )
    plot_timeseries_comparison(
        data,
        args.variable,
        args.output_dir,
        fontz=args.fontz,
        frequency=args.frequency,
    )

if __name__ == '__main__':
    main()
