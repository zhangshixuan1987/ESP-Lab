"""S2D Initialization Consistency and Drift Summary Diagnostic Utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import cftime
import numpy as np
import xarray as xr


# -----------------------------------------------------------------------------
# Filename & Token Sanitization
# -----------------------------------------------------------------------------

def figure_filename(*parts, ext: str = "png") -> str:
    """Build standardized, readable figure filenames: fig_<diagnostic>_<model>_<reference>_<period>_<other>.ext."""
    clean_parts = ["fig"]
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if not text:
            continue
        text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
        if text:
            clean_parts.append(text)
    return "_".join(clean_parts) + f".{ext.lstrip('.').lower()}"


def safe_token(value: object) -> str:
    """Sanitize arbitrary strings/values for filenames and keys."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


# -----------------------------------------------------------------------------
# Unit Conversion Utilities
# -----------------------------------------------------------------------------

def convert_kelvin_to_celsius(da: xr.DataArray) -> xr.DataArray:
    out = da - 273.15
    out.attrs.update(da.attrs)
    out.attrs["units"] = "degC"
    return out


def _temperature_units_kind(da: xr.DataArray) -> str:
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin", "degrees_k", "degree_k"} or "kelvin" in units:
        return "kelvin"
    if any(token in units for token in ("degc", "degree_c", "degrees_c", "celsius", "c")):
        return "celsius"
    return "unknown"


def convert_kelvin_to_celsius_if_needed(da: xr.DataArray) -> xr.DataArray:
    kind = _temperature_units_kind(da)
    if kind == "kelvin":
        return convert_kelvin_to_celsius(da)
    if kind == "celsius":
        return da

    # Fallback for regional/cache series with missing units.
    try:
        indexers = {dim: 0 for dim in da.dims if da.sizes.get(dim, 0) > 0}
        sample = float(da.isel(indexers).mean(skipna=True).load())
    except Exception:
        return da
    if np.isfinite(sample) and sample > 100.0:
        return convert_kelvin_to_celsius(da)
    return da


def normalize_regional_temperature_units(da: xr.DataArray) -> xr.DataArray:
    kind = _temperature_units_kind(da)
    if kind == "kelvin":
        return convert_kelvin_to_celsius(da)
    if kind == "celsius":
        return da
    try:
        mean_value = float(da.mean(skipna=True).load())
    except Exception:
        return da
    if np.isfinite(mean_value) and mean_value > 100.0:
        return convert_kelvin_to_celsius(da)
    return da


def convert_precip_mps_to_mmday(da: xr.DataArray) -> xr.DataArray:
    out = da * 1000.0 * 86400.0
    out.attrs.update(da.attrs)
    out.attrs["units"] = "mm/day"
    return out


def convert_pa_to_hpa(da: xr.DataArray) -> xr.DataArray:
    out = da * 1.0e-2
    out.attrs.update(da.attrs)
    out.attrs["units"] = "hPa"
    return out


def no_conversion(da: xr.DataArray) -> xr.DataArray:
    return da


# -----------------------------------------------------------------------------
# Calendar & Lead-Time Coordinate Math
# -----------------------------------------------------------------------------

def lead_to_cal_month(init_month: int, lead_1based: int) -> int:
    """Compute verification calendar month (1-12) from 1-based lead time."""
    return ((init_month - 1 + int(lead_1based) - 1) % 12) + 1


def lead_to_valid_time(init_year: int, init_month: int, lead_1based: int):
    """Compute mid-month cftime verification time."""
    month_index = init_month - 1 + int(lead_1based) - 1
    year = init_year + month_index // 12
    month = month_index % 12 + 1
    return cftime.DatetimeNoLeap(year, month, 15)


def valid_times_for_leads(init_year: int, init_month: int, leads: Sequence[int]):
    """Construct sequence of valid cftime mid-month points for given leads."""
    return [lead_to_valid_time(init_year, init_month, L) for L in leads]


def lead_axis_ticks(leads: Sequence[int], init_month: int):
    """Compute major tick locations and labels for lead-time axes."""
    lead_months = np.asarray(leads) - 1
    month_abbrs = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    tick_labels = [
        f"L{int(L)-1}\n{month_abbrs[lead_to_cal_month(init_month, L) - 1]}"
        for L in leads
    ]
    return lead_months, tick_labels


def format_lead_time_axis(ax, leads: Sequence[int], init_month: int,
                          show_labels: bool = True, label_every: int = 3,
                          grid_alpha: float = 0.16, tick_labelsize: int | None = None):
    """Standardize lead-time x-axis with dual L-index and calendar month labels."""
    lead_months = np.asarray(leads) - 1
    if len(lead_months) == 0:
        return lead_months
    month_abbrs = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    tick_idx = list(range(0, len(lead_months), label_every))
    if tick_idx[-1] != len(lead_months) - 1:
        tick_idx.append(len(lead_months) - 1)
    major_ticks = lead_months[tick_idx]
    major_labels = [
        f"L{int(lead_months[i])}\n{month_abbrs[lead_to_cal_month(init_month, leads[i]) - 1]}"
        for i in tick_idx
    ]
    ax.set_xlim(float(np.nanmin(lead_months)) - 0.75, float(np.nanmax(lead_months)) + 0.75)
    ax.set_xticks(major_ticks)
    ax.set_xticks(lead_months, minor=True)
    if show_labels:
        ax.set_xticklabels(major_labels, fontsize=tick_labelsize)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.grid(which="minor", axis="x", alpha=grid_alpha)
    return lead_months


def apply_axis_style(ax, grid_alpha: float = 0.25, tick_labelsize: int | None = None):
    """Apply consistent grid and tick styling to a matplotlib axis."""
    ax.grid(True, alpha=grid_alpha)
    if tick_labelsize is not None:
        ax.tick_params(labelsize=tick_labelsize)


# -----------------------------------------------------------------------------
# Observational Alignment & Statistics Helpers
# -----------------------------------------------------------------------------

def select_init_year(da: xr.DataArray, init_year: int | str) -> xr.DataArray:
    """Extract a specific initialization year across varying coordinate types."""
    yvals = da.Y.values
    if init_year in yvals:
        return da.sel(Y=init_year)
    if str(init_year) in yvals:
        return da.sel(Y=str(init_year))
    ystr = np.array([str(y) for y in yvals])
    matches = [y for y, ys in zip(yvals, ystr) if ys.startswith(str(init_year))]
    if len(matches) == 1:
        return da.sel(Y=matches[0])
    raise KeyError(f"Cannot match init_year={init_year} in Y: {yvals}")


def obs_clim_by_lead(obs_monthly_clim: xr.DataArray, init_month: int, leads: Sequence[int]) -> xr.DataArray:
    """Extract monthly observational climatology aligned to forecast lead calendar months."""
    months = [lead_to_cal_month(init_month, L) for L in leads]
    return xr.DataArray(
        [obs_monthly_clim.sel(month=m).item() for m in months],
        dims=("L",),
        coords={"L": list(leads)},
        name="obs_clim_by_lead",
    )


def obs_values_by_valid_time(obs_series: xr.DataArray, init_year: int, init_month: int, leads: Sequence[int]) -> xr.DataArray:
    """Sample time-series observations at verification valid times for a single initialization start."""
    times = valid_times_for_leads(init_year, init_month, leads)
    values = obs_series.sel(time=times, method="nearest").load()
    return xr.DataArray(values.values, dims=("L",), coords={"L": list(leads)}, name="obs_by_lead")


def obs_matrix_by_valid_times(obs_series: xr.DataArray, init_years: Sequence[int], init_month: int, leads: Sequence[int]) -> xr.DataArray:
    """Construct (Y, L) matrix of observations at verification valid times across all initialization years."""
    rows = []
    for year in init_years:
        times = [lead_to_valid_time(year, init_month, L) for L in leads]
        obs_at_lead = obs_series.sel(time=times, method="nearest").load()
        rows.append(obs_at_lead.values)
    return xr.DataArray(
        np.asarray(rows),
        dims=("Y", "L"),
        coords={"Y": list(init_years), "L": list(leads)},
        name="obs_valid_by_year_lead",
    )


def obs_mean_by_valid_times(obs_series: xr.DataArray, init_years: Sequence[int], init_month: int, leads: Sequence[int]) -> xr.DataArray:
    """Mean of observations at valid times across initialization years."""
    return obs_matrix_by_valid_times(obs_series, init_years, init_month, leads).mean("Y", skipna=True).rename("obs_valid_lead_mean")


def infer_model_init_years(da: xr.DataArray, fallback_years: Sequence[int]) -> list[int]:
    """Infer integer initialization years from a model regional-index Y coordinate."""
    n_years = da.sizes.get("Y")
    inferred = []
    for yval in da.Y.values:
        if hasattr(yval, "year"):
            inferred.append(int(yval.year))
            continue
        text = str(yval)
        match = re.search(r"(\d{4})", text)
        if match:
            inferred.append(int(match.group(1)))
    if len(inferred) == n_years:
        return inferred

    fallback = list(fallback_years)
    if len(fallback) >= n_years:
        return fallback[:n_years]
    raise ValueError(
        "Cannot infer enough model initialization years from da.Y and fallback_years "
        f"({len(fallback)} fallback years for {n_years} model years)."
    )


def compute_rmse_and_acc(model: xr.DataArray, obs_by_year_lead: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate RMSE and anomaly correlation coefficient (ACC) between hindcast ensemble mean and observations."""
    model_ens_mean = model.mean("M", skipna=True)
    if obs_by_year_lead.sizes.get("Y") != model_ens_mean.sizes.get("Y"):
        raise ValueError(
            "obs_by_year_lead and model must have the same number of initialization years "
            f"({obs_by_year_lead.sizes.get('Y')} vs {model_ens_mean.sizes.get('Y')})."
        )
    obs_aligned = xr.DataArray(
        obs_by_year_lead.values,
        dims=("Y", "L"),
        coords={"Y": model_ens_mean.Y.values, "L": model_ens_mean.L.values},
        name="obs_valid_by_year_lead",
    )
    error = model_ens_mean - obs_aligned
    rmse = np.sqrt((error ** 2).mean("Y", skipna=True)).rename("rmse")

    model_anom = model_ens_mean - model_ens_mean.mean("Y", skipna=True)
    obs_anom = obs_aligned - obs_aligned.mean("Y", skipna=True)
    acc = xr.corr(model_anom, obs_anom, dim="Y").rename("acc")
    return rmse.load(), acc.load()


def safe_sel_mean(da: xr.DataArray, lead_slice: slice) -> float:
    """Compute mean over a lead slice returning float."""
    return float(da.sel(L=lead_slice).mean("L", skipna=True))


def shock_index_from_bias(bias: xr.DataArray, lead_slice: slice) -> float:
    """Calculate RMS initial-shock index over an early lead window."""
    return float(np.sqrt((bias.sel(L=lead_slice) ** 2).mean("L", skipna=True)))


def season1_drift_from_bias(bias: xr.DataArray, lead_slice: slice = slice(1, 6)) -> float:
    """Calculate early drift rate (bias change per month) over season 1."""
    early = bias.sel(L=lead_slice)
    if early.sizes.get("L", 0) < 2:
        return np.nan
    return float((early.isel(L=-1) - early.isel(L=0)) / (early.L.isel(L=-1) - early.L.isel(L=0)))
