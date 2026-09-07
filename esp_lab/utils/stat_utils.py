import xarray as xr
import numpy as np
import cftime
import xskillscore as xs

def cor_ci_bootyears(ts1, ts2, seed=None, nboots=1000, conf=95):
    """ """
    ptilemin = (100.-conf)/2.
    ptilemax = conf + (100-conf)/2.

    if ts1.size != ts2.size:
        raise ValueError(f"The two arrays must have the same size, got {ts1.size} and {ts2.size}.")
    if ts1.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if seed is not None:
        np.random.seed(seed)

    samplesize = ts1.size
    ranu = np.random.uniform(0, samplesize, nboots*samplesize)
    ranu = np.floor(ranu).astype(int)

    bootdat1 = np.array(ts1[ranu])
    bootdat2 = np.array(ts2[ranu])
    bootdat1 = bootdat1.reshape([samplesize, nboots])
    bootdat2 = bootdat2.reshape([samplesize, nboots])
   
 
    bootcor = xr.corr(xr.DataArray(bootdat1), xr.DataArray(bootdat2), dim='dim_0')
    minci = np.percentile(bootcor,ptilemin)
    maxci = np.percentile(bootcor,ptilemax)

    return minci, maxci

def detrend_linear(dat, dim):
    """ linear detrend dat along the axis dim """
    params = dat.polyfit(dim=dim, deg=1, skipna=True)
    fit = xr.polyval(dat[dim], params.polyfit_coefficients)
    dat = dat-fit
    return dat

def remove_drift(da, da_time, y1, y2):
    """
    Function to convert raw DP DataArray into anomaly DP DataArray with leadtime-dependent climatology removed.
    --Inputs--
        da:  Raw DP DataArray with dimensions (Y,L,M,...)
        da_time:  Verification time of DP DataArray (Y,L)
        y1:  Start year of climatology
        y2:  End year of climatology
        
    --Outputs--
        da_anom:  De-drifted DP DataArray
        da_climo:  Leadtime-dependent climatology
    
    Author: E. Maroon (modified by S. Yeager)
    """
    if y1 >= y2:
        raise ValueError(f"y1 ({y1}) must be strictly less than y2 ({y2}).")
    d1 = cftime.DatetimeNoLeap(y1,1,1,0,0,0)
    d2 = cftime.DatetimeNoLeap(y2,12,31,23,59,59)
    masked_period = da.where((da_time>d1) & (da_time<d2))
    time_mask = (da_time>d1) & (da_time<d2)
    if int(time_mask.sum()) == 0:
        raise ValueError(
            f"Climatology period {y1}-{y2} has no overlap with da_time. "
            "Check that y1/y2 fall within the data time range."
        )
    da_climo = masked_period.mean('M').mean('Y')
    da_anom = da - da_climo
    return da_anom, da_climo

def leadtime_skill_seas(mod_da,mod_time,obs_da,detrend=False):
    """ 
    Computes a suite of deterministic skill metrics given two DataArrays corresponding to model and observations, which 
    must share the same lat/lon coordinates (if any). Assumes time coordinates are compatible
    (can be aligned). Both DataArrays should represent 3-month seasonal averages (DJF, MAM, JJA, SON).
    
        Inputs
        mod_da: a seasonally-averaged hindcast DataArray dimensioned (Y,L,M,...)
        mod_time: a hindcast time DataArray dimensioned (Y,L). NOTE: assumes mod_time.dt.month
            returns the mid-month of a 3-month seasonal average (e.g., mon=1 ==> "DJF").
        obs_da: an OBS DataArray dimensioned (season,year,...)
    """
    seasons = {1:'DJF',4:'MAM',7:'JJA',10:'SON'}
    if mod_da.L.size == 0:
        raise ValueError("mod_da has an empty L dimension.")
    corr_list = []; pval_list = []; rmse_list = []; msss_list = []; rpc_list = []
    # convert L to leadtime values:
    leadtime = mod_da.L - 2
    for i in mod_da.L.values:
        ens_ts = mod_da.sel(L=i).rename({'Y':'time'})
        time_data = mod_time.sel(L=i)
        if time_data.size == 0:
            raise ValueError(f"mod_time has no years for L={i}.")
        ens_time_year = time_data.dt.year.data
        ens_time_month = int(time_data.dt.month.data[0])
        if ens_time_month not in seasons:
            raise ValueError(
                f"Unexpected month {ens_time_month} for L={i}. "
                f"Expected one of {list(seasons.keys())} (mid-months of DJF, MAM, JJA, SON)."
            )
        obs_ts = obs_da.sel(season=seasons[ens_time_month]).rename({'year':'time'})
        ens_ts = ens_ts.assign_coords(time=("time",ens_time_year))
        a,b = xr.align(ens_ts,obs_ts)
        if a.time.size == 0:
            raise ValueError(
                f"No overlapping years between model and observations "
                f"at L={i}, season={seasons[ens_time_month]}."
            )
        if detrend:
            a = detrend_linear(a,'time')
            b = detrend_linear(b,'time')
        amean = a.mean('M')
        sigobs = b.std('time')
        sigsig = amean.std('time')
        sigtot = a.std('time').mean('M')
        _skipna = bool(amean.isnull().any() or b.isnull().any())
        r = xs.pearson_r(amean,b,dim='time',skipna=_skipna)
        rpc = (r/(sigsig/sigtot)).where(sigsig > 0)
        bvar = b.var('time')
        corr_list.append(r)
        rpc_list.append(rpc.where(r>0))
        rmse_list.append((xs.rmse(amean,b,dim='time',skipna=_skipna)/sigobs).where(sigobs > 0))
        msss_list.append((1-(xs.mse(amean,b,dim='time',skipna=_skipna)/bvar)).where(bvar > 0))
        pval_list.append(xs.pearson_r_eff_p_value(amean,b,dim='time',skipna=_skipna))
    corr = xr.concat(corr_list,leadtime)
    pval = xr.concat(pval_list,leadtime)
    rmse = xr.concat(rmse_list,leadtime)
    msss = xr.concat(msss_list,leadtime)
    rpc = xr.concat(rpc_list,leadtime)
    return xr.Dataset({'corr':corr,'pval':pval,'nrmse':rmse,'msss':msss,'rpc':rpc})

def leadtime_skill_seas_resamp(mod_da,mod_time,obs_da,sampsize,N,detrend=False):
    """ 
    Same as leadtime_skill_seas(), but this version resamples the mod_da member dimension (M) to generate
    a distribution of skill scores using a smaller ensemble size (N, where N<M). Returns the 
    mean of the resampled skill score distribution.
    """
    dslist = []
    seasons = {1:'DJF',4:'MAM',7:'JJA',10:'SON'}
    if mod_da.L.size == 0:
        raise ValueError("mod_da has an empty L dimension.")
    # convert L to leadtime values:
    leadtime = mod_da.L - 2
    # Perform resampling
    if (not N<mod_da.M.size):
        raise ValueError('ERROR: expecting resampled ensemble size to be less than original')
    mod_da_r = xs.resample_iterations(mod_da.chunk(), sampsize, 'M', dim_max=N)
    for l in mod_da_r.iteration.values:
        corr_list = []; pval_list = []; rmse_list = []; msss_list = []; rpc_list = []
        for i in mod_da.L.values:
            ens_ts = mod_da_r.sel(iteration=l).sel(L=i).rename({'Y':'time'})
            time_data = mod_time.sel(L=i)
            if time_data.size == 0:
                raise ValueError(f"mod_time has no years for L={i}.")
            ens_time_year = time_data.dt.year.data
            ens_time_month = int(time_data.dt.month.data[0])
            if ens_time_month not in seasons:
                raise ValueError(
                    f"Unexpected month {ens_time_month} for L={i}. "
                    f"Expected one of {list(seasons.keys())} (mid-months of DJF, MAM, JJA, SON)."
                )
            obs_ts = obs_da.sel(season=seasons[ens_time_month]).rename({'year':'time'})
            ens_ts = ens_ts.assign_coords(time=("time",ens_time_year))
            a,b = xr.align(ens_ts,obs_ts)
            if a.time.size == 0:
                raise ValueError(
                    f"No overlapping years between model and observations "
                    f"at L={i}, season={seasons[ens_time_month]}."
                )
            if detrend:
                a = detrend_linear(a,'time')
                b = detrend_linear(b,'time')
            amean = a.mean('M')
            sigobs = b.std('time')
            sigsig = amean.std('time')
            sigtot = a.std('time').mean('M')
            _skipna = bool(amean.isnull().any() or b.isnull().any())
            r = xs.pearson_r(amean,b,dim='time',skipna=_skipna)
            rpc = (r/(sigsig/sigtot)).where(sigsig > 0)
            bvar = b.var('time')
            corr_list.append(r)
            rpc_list.append(rpc.where(r>0))
            rmse_list.append((xs.rmse(amean,b,dim='time',skipna=_skipna)/sigobs).where(sigobs > 0))
            msss_list.append((1-(xs.mse(amean,b,dim='time',skipna=_skipna)/bvar)).where(bvar > 0))
            pval_list.append(xs.pearson_r_eff_p_value(amean,b,dim='time',skipna=_skipna))
        corr = xr.concat(corr_list,leadtime)
        pval = xr.concat(pval_list,leadtime)
        rmse = xr.concat(rmse_list,leadtime)
        msss = xr.concat(msss_list,leadtime)
        rpc = xr.concat(rpc_list,leadtime)
        dslist.append(xr.Dataset({'corr':corr,'pval':pval,'nrmse':rmse,'msss':msss,'rpc':rpc}))
    dsout = xr.concat(dslist,dim='iteration').mean('iteration').compute()
    return dsout


