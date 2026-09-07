"""Windowed variability index from Compute_Std_Index_Initial_Shock_share.ncl.

Use monthly physical fields, before lead-dependent drift removal or detrending.
The NCL statistic is temporal std(model ensemble-mean global index) / std(obs),
not ensemble spread and not a direct measure of a discontinuity at initialization.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

VERSION = "initial_shock_std_v1"


def align_observation_months(observation, verification_time, *, time_dim="time"):
    """Select observations by represented year/month, allowing different calendars.

    verification_time must contain normalized represented months, not uncorrected
    E3SM end-of-interval timestamps. Duplicate or unavailable months are errors.
    """
    if verification_time.dims != ("Y", "L"):
        raise ValueError("verification_time must have dimensions (Y, L)")
    stamps = observation[time_dim]
    keys = np.asarray(stamps.dt.year * 12 + stamps.dt.month)
    if len(set(keys)) != len(keys):
        raise ValueError("Observations contain duplicate year/month entries")
    target = np.asarray(verification_time.dt.year * 12 + verification_time.dt.month)
    if target.shape[1] < 1 or not np.all(np.diff(target, axis=1) == 1):
        raise ValueError("Verification months must be consecutive along L")
    lookup = {int(k): i for i, k in enumerate(keys)}
    missing = sorted(set(target.ravel()) - set(keys))
    if missing:
        labels = [f"{(int(k)-1)//12:04d}-{(int(k)-1)%12+1:02d}" for k in missing]
        raise ValueError(f"Missing observation months: {labels}")
    indices = xr.DataArray(
        np.array([lookup[int(k)] for k in target.ravel()]).reshape(target.shape),
        dims=("Y", "L"), coords={d: verification_time[d] for d in ("Y", "L")},
    )
    selected = observation.isel({time_dim: indices})
    return selected.drop_vars(time_dim).assign_coords(verification_time=verification_time)


def _weights(field, area):
    for dim in ("lat", "lon"):
        if dim not in field.dims or dim not in field.coords:
            raise ValueError("Fields must have labeled lat/lon dimensions; regrid upstream")
    if area is None:
        for dim in ("lat", "lon"):
            values = np.asarray(field[dim])
            if values.ndim != 1 or len(values) < 2:
                raise ValueError("Supply cell areas for non-regular or single-cell grids")
            steps = np.diff(values)
            if not (np.all(steps > 0) or np.all(steps < 0)) or not np.allclose(steps, steps[0]):
                raise ValueError("Cosine weights require a regular lat/lon grid; supply areas")
        if np.any(np.abs(field.lat.values) > 90):
            raise ValueError("Latitude outside [-90, 90]")
        area = np.cos(np.deg2rad(field.lat)).broadcast_like(field.isel(Y=0, L=0, drop=True))
    if set(area.dims) != {"lat", "lon"}:
        raise ValueError("area must have only lat/lon dimensions")
    _, area = xr.align(field, area, join="exact")
    if not bool((np.isfinite(area) & (area >= 0)).all().compute()) or float(area.sum()) <= 0:
        raise ValueError("Cell areas must be finite, non-negative and have a positive sum")
    return area


def compute_initial_shock_index(
    model: xr.DataArray,
    observation: xr.DataArray,
    *,
    area: xr.DataArray | None = None,
    block_months: int = 12,
    window_months: int = 60,
    start_lead: int = 0,
    min_samples: int | None = None,
    min_area_fraction: float = 0.0,
) -> xr.Dataset:
    """Compute one variability ratio for every initialization (Y).

    Inputs: model(Y,L,M,lat,lon) or ensemble mean(Y,L,lat,lon), and aligned
    observation(Y,L,lat,lon), in compatible physical units on the same grid.
    L must be consecutive numeric monthly leads. start_lead is a zero-based
    POSITION. Non-overlapping blocks start at that position; they are calendar
    years only when the first represented month is January and block_months=12.

    Ensemble averaging and block averaging require every member/month to be
    finite. Spatial means independently omit missing cells as in NCL. Statistics
    use paired valid blocks, ddof=1, no detrending, and no lead climatology removal.
    By default all requested blocks must be valid. Missing/constant observation
    series produce NaN ratios, accompanied by coverage and validity variables.
    """
    required = {"Y", "L", "lat", "lon"}
    if set(model.dims) not in (required, required | {"M"}) or set(observation.dims) != required:
        raise ValueError("Expected model(Y,L,[M,]lat,lon) and observation(Y,L,lat,lon)")
    if any(d not in model.coords or d not in observation.coords for d in required):
        raise ValueError("All dimensions must have explicit coordinates")
    for data in (model, observation):
        if any(data.sizes[d] == 0 or not data.get_index(d).is_unique for d in data.dims):
            raise ValueError("Coordinates must be nonempty and unique")
    leads = np.asarray(model.L)
    if not np.issubdtype(leads.dtype, np.number) or not np.all(np.diff(leads) == 1):
        raise ValueError("L must contain consecutive numeric monthly leads")
    if any(not isinstance(v, (int, np.integer)) for v in (start_lead, block_months, window_months)):
        raise ValueError("Window settings must be integers")
    if block_months < 1 or window_months < 2 * block_months or window_months % block_months:
        raise ValueError("window_months must contain at least two complete blocks")
    if start_lead < 0 or start_lead + window_months > model.sizes["L"]:
        raise ValueError("Requested window exceeds available monthly leads")
    nblocks = window_months // block_months
    min_samples = nblocks if min_samples is None else min_samples
    if not isinstance(min_samples, (int, np.integer)) or not 2 <= min_samples <= nblocks:
        raise ValueError("min_samples must be between 2 and the number of blocks")
    if not 0 <= min_area_fraction <= 1:
        raise ValueError("min_area_fraction must be between 0 and 1")
    if not model.attrs.get("units") or model.attrs.get("units") != observation.attrs.get("units"):
        raise ValueError("Convert model and observation to identical explicit units first")
    model, observation = xr.align(model, observation, join="exact")
    units = model.attrs["units"]
    nens = model.sizes.get("M", 1)
    model = model.where(np.isfinite(model))
    observation = observation.where(np.isfinite(observation))
    if "M" in model.dims:
        model = model.mean("M", skipna=False)
    weights = _weights(model, area)
    selection = slice(start_lead, start_lead + window_months)
    def block_index(data):
        if "verification_time" in data.coords:
            data = data.drop_vars("verification_time")
        blocks = data.isel(L=selection).coarsen(L=block_months, boundary="exact").mean(skipna=False)
        coverage = weights.where(blocks.notnull(), 0).sum(("lat", "lon")) / weights.sum()
        index = blocks.weighted(weights).mean(("lat", "lon"), skipna=True)
        index = index.where((coverage > 0) & (coverage >= min_area_fraction))
        return index, coverage
    model_index, model_area = block_index(model)
    obs_index, obs_area = block_index(observation)
    paired = model_index.notnull() & obs_index.notnull()
    n = paired.sum("L")
    m = model_index.where(paired)
    o = obs_index.where(paired)
    # Explicit sample variance avoids ddof warnings for insufficient samples.
    def sample_std(data):
        return np.sqrt(((data - data.mean("L")) ** 2).sum("L") / (n - 1).where(n > 1))
    mstd, ostd = sample_std(m), sample_std(o)
    valid = (n >= min_samples) & (ostd > 0)
    result = xr.Dataset({
        "std_ratio": (mstd / ostd.where(ostd > 0)).where(valid),
        "model_std": mstd.where(n >= min_samples),
        "observation_std": ostd.where(n >= min_samples),
        "paired_sample_count": n,
        "valid_ratio": valid.astype("int8"),
        "model_index": model_index,
        "observation_index": obs_index,
        "model_area_fraction": model_area,
        "observation_area_fraction": obs_area,
    }).rename({"L": "block"}).assign_coords(block=np.arange(1, nblocks + 1))
    result = result.assign_coords(
        block_start_lead=("block", leads[start_lead:start_lead + window_months:block_months]),
        block_end_lead=("block", leads[start_lead + block_months - 1:start_lead + window_months:block_months]),
    )
    if "verification_time" in observation.coords:
        time = observation.verification_time
        for name, offset in (("block_start_time", 0), ("block_end_time", block_months - 1)):
            values = time.isel(L=slice(start_lead + offset, start_lead + window_months, block_months))
            result = result.assign_coords({name: (("Y", "block"), values.values)})
    for name in ("model_std", "observation_std", "model_index", "observation_index"):
        result[name].attrs["units"] = units
    result.std_ratio.attrs.update(units="1", long_name="Model / observation temporal standard deviation")
    result.attrs.update(
        diagnostic_version=VERSION, block_months=int(block_months), window_months=int(window_months),
        start_lead_position=int(start_lead), min_samples=int(min_samples), ddof=1,
        ensemble_members=nens, min_area_fraction=float(min_area_fraction),
        detrended="false", spatial_mask="independent finite cells per block",
        averaging_order="ensemble mean; unweighted monthly blocks; area mean; temporal sample std",
        interpretation="Variability amplitude ratio; does not uniquely identify initialization shock",
    )
    return result


def plot_std_ratio(result: xr.Dataset):
    """Return a case-by-initialization heatmap; NaN denotes an invalid ratio."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ratio = result.std_ratio
    if "case" not in ratio.dims:
        ratio = ratio.expand_dims(case=[result.attrs.get("case", "model")])
    ratio = ratio.transpose("Y", "case")
    colors = ["gray", "white", "white", "orange", "orangered", "red", "darkred"]
    cmap = ListedColormap(colors)
    cmap.set_bad("lightgray")
    bounds = [0, 0.4, 1.0, 1.6, 2.2, 2.8, 3.4, 10]
    fig, ax = plt.subplots(figsize=(max(5, ratio.sizes["case"] * 1.3), max(3, ratio.sizes["Y"] * .25)))
    mesh = ax.imshow(ratio.values, aspect="auto", cmap=cmap, norm=BoundaryNorm(bounds, cmap.N, clip=True))
    ax.set_xticks(np.arange(ratio.sizes["case"]), labels=ratio.case.values, rotation=45, ha="right")
    ax.set_yticks(np.arange(ratio.sizes["Y"]), labels=[str(v) for v in ratio.Y.values])
    ax.set_ylabel("Initialization")
    ax.set_title(f"Variability ratio: {result.attrs.get('window_months', '?')} months, "
                 f"{result.attrs.get('block_months', '?')}-month means")
    fig.colorbar(mesh, ax=ax, ticks=bounds[1:-1], label="Model std / observed std (1 = equal)", extend="max")
    fig.tight_layout()
    return fig
