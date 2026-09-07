"""Utilities for PSL index references and reference-sensitive skill metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from . import stats


PSL_INDEX_FILES: Dict[str, str] = {
    "IOD": "iod.HadISST.PSL.csv",
    "Nino12": "nino12.HadISST.PSL.csv",
    "Nino3": "nino3.HadISST.PSL.csv",
    "Nino3.4": "nino34.HadISST.PSL.csv",
    "TNA": "tna.HadISST.PSL.csv",
    "TSA": "tsa.HadISST.PSL.csv",
    "TNI": "tni.HadISST.PSL.csv",
    "ONI": "oni.ERSSTv5.PSL.csv",
    "RONI": "roni.ERSSTv5.PSL.csv",
    "PACWRAMPOOL": "pacwarmpool.ERSSTv5.PSL.csv",
}

DEFAULT_REFERENCE_ORDER: Tuple[str, str] = ("HadISST2", "PSL")


def default_external_dir() -> Path:
    """Return the repository-packaged external index directory."""
    return Path(__file__).resolve().parents[1] / "external"


def _midmonth_noleap_timestamps(values) -> list[cftime.DatetimeNoLeap]:
    dates = pd.to_datetime(values)
    return [
        cftime.DatetimeNoLeap(int(date.year), int(date.month), 15)
        for date in dates
    ]


def _clean_missing_values(values: xr.DataArray, missing_threshold: float = -90.0) -> xr.DataArray:
    return values.where(values > missing_threshold)


def load_psl_index(
    index_name: str,
    external_dir: str | Path | None = None,
    *,
    missing_threshold: float = -90.0,
) -> xr.DataArray:
    """Load a packaged NOAA PSL monthly index CSV as a ``time`` DataArray.

    The CSVs in ``external/`` have two columns: a date column and one value
    column whose name includes source metadata. PSL missing values are encoded
    as large negative sentinels (for example ``-99.99`` or ``-9999``), so values
    less than or equal to ``missing_threshold`` are converted to NaN.
    """
    key = index_name.strip()
    if key not in PSL_INDEX_FILES:
        valid = ", ".join(sorted(PSL_INDEX_FILES))
        raise KeyError(f"Unknown PSL index '{index_name}'. Valid indices: {valid}")

    base = default_external_dir() if external_dir is None else Path(external_dir)
    path = base / PSL_INDEX_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"PSL index file not found: {path}")

    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in PSL index file: {path}")

    value_column = frame.columns[1]
    time = _midmonth_noleap_timestamps(frame.iloc[:, 0])
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(dtype=float)

    da = xr.DataArray(
        values,
        dims="time",
        coords={"time": time},
        name=f"{key.lower().replace('.', '')}_psl",
        attrs={
            "index": key,
            "source": "NOAA PSL",
            "source_file": str(path),
            "source_column": str(value_column),
            "units": "degC",
            "is_anomaly": "true",
        },
    )
    return _clean_missing_values(da, missing_threshold=missing_threshold)


def monthly_climatology_anomaly(
    da: xr.DataArray,
    climy0: int,
    climy1: int,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Remove a monthly climatology from a raw monthly index."""
    years = da[time_dim].dt.year
    clim = da.where((years >= climy0) & (years <= climy1), drop=True)
    clim = clim.groupby(f"{time_dim}.month").mean(time_dim)
    out = da.groupby(f"{time_dim}.month") - clim
    out.attrs.update(da.attrs)
    out.attrs["climatology"] = f"{climy0}-{climy1}"
    out.attrs["is_anomaly"] = "true"
    return out


def seasonal_centered_mean(
    da: xr.DataArray,
    *,
    window: int = 3,
    season_center_months: Tuple[int, ...] = (1, 4, 7, 10),
    min_periods: int | None = None,
) -> xr.DataArray:
    """Return centered seasonal means at DJF/MAM/JJA/SON center months."""
    if "time" not in da.dims:
        raise ValueError("Seasonal averaging requires a 'time' dimension.")
    if not bool(da.notnull().any()):
        source = da.encoding.get("source", da.name or "input")
        raise ValueError(
            f"Cannot compute seasonal means: {source} contains no finite values. "
            "Regenerate the upstream SST-index product."
        )
    if min_periods is None:
        min_periods = window

    out = da.rolling(time=window, center=True, min_periods=min_periods).mean()
    out = out.dropna("time", how="all")
    mask = xr.DataArray(
        np.isin(out["time"].dt.month, list(season_center_months)),
        dims="time",
        coords={"time": out["time"]},
    )
    out = out.where(mask, drop=True)
    out.attrs.update(da.attrs)
    out.attrs["temporal_average"] = f"{window}-month centered mean"
    return out


def retain_complete_seasonal_leads(
    data: xr.DataArray,
    valid_time: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, list[int]]:
    """Exclude seasonal lead coordinates containing no finite hindcast data.

    Centered three-month means require a month on either side of the target
    month. A 24-month hindcast therefore has seven complete seasonal centers;
    an upstream ``L=24`` coordinate may be retained as an all-missing
    placeholder. This helper preserves every partially or fully populated
    lead and removes only leads that are entirely missing across all other
    dimensions.
    """
    if "L" not in data.dims or "L" not in valid_time.dims:
        raise ValueError("data and valid_time must both contain an L dimension")
    if not data["L"].identical(valid_time["L"]):
        raise ValueError("data and valid_time must have identical L coordinates")

    finite_by_lead = data.notnull()
    for dim in tuple(dim for dim in finite_by_lead.dims if dim != "L"):
        finite_by_lead = finite_by_lead.any(dim)
    if finite_by_lead.chunks is not None:
        # Boolean indexing with ``drop=True`` requires a realized mask.
        finite_by_lead = finite_by_lead.compute()

    complete = data["L"].where(finite_by_lead, drop=True)
    dropped = data["L"].where(~finite_by_lead, drop=True)
    if complete.size == 0:
        raise ValueError("no complete seasonal leads are available")

    return (
        data.sel(L=complete),
        valid_time.sel(L=complete),
        [int(value) for value in dropped.values],
    )


def subset_hindcast_initialization_years(
    data: xr.DataArray,
    valid_time: xr.DataArray,
    year0: int,
    year1: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Select a complete, inclusive initialization-year cohort.

    The cohort is derived from the first verification lead.  Selection is
    applied through the shared ``Y`` coordinate so later-lead verification
    dates remain unchanged.
    """
    if year1 < year0:
        raise ValueError(f"year1 ({year1}) must be greater than or equal to year0 ({year0})")
    if "Y" not in data.dims or "Y" not in valid_time.dims:
        raise ValueError(f"Expected a Y dimension; data={data.dims}, time={valid_time.dims}")
    if not data["Y"].identical(valid_time["Y"]):
        raise ValueError("data and valid_time must have identical Y coordinates")

    first_lead_time = valid_time.isel(L=0) if "L" in valid_time.dims else valid_time
    init_year = first_lead_time.dt.year
    mask = np.asarray((init_year >= year0) & (init_year <= year1), dtype=bool)
    selected_y = np.asarray(valid_time["Y"].values)[mask]
    expected_years = np.arange(year0, year1 + 1)
    selected_years = np.asarray(init_year.values)[mask].astype(int)
    if not np.array_equal(selected_years, expected_years):
        raise ValueError(
            f"Expected every initialization year from {year0} through {year1}; "
            f"found {selected_years.tolist()}."
        )
    return data.sel(Y=selected_y), valid_time.sel(Y=selected_y)


def psl_reference(
    index_name: str,
    external_dir: str | Path | None = None,
    *,
    seasonal: bool = False,
) -> xr.DataArray:
    """Load a PSL monthly anomaly index, optionally converted to seasons."""
    da = load_psl_index(index_name, external_dir=external_dir)
    if seasonal:
        da = seasonal_centered_mean(da)
    return da


def observation_agreement(
    local_obs: xr.DataArray,
    psl_obs: xr.DataArray,
    climy0: int,
    climy1: int,
    *,
    local_is_anomaly: bool = False,
    seasonal: bool = False,
) -> xr.Dataset:
    """Compare a local observational index against a PSL anomaly index."""
    local = local_obs if local_is_anomaly else monthly_climatology_anomaly(local_obs, climy0, climy1)
    psl = psl_obs
    if seasonal:
        local = seasonal_centered_mean(local)
        psl = seasonal_centered_mean(psl)

    local, psl = xr.align(local, psl, join="inner")
    valid = np.isfinite(local) & np.isfinite(psl)
    local = local.where(valid, drop=True)
    psl = psl.where(valid, drop=True)

    diff = local - psl
    local_std = local.std("time")
    psl_std = psl.std("time")
    return xr.Dataset(
        {
            "corr": xr.corr(local, psl, dim="time"),
            "rmse": np.sqrt((diff ** 2).mean("time")),
            "bias": diff.mean("time"),
            "local_std": local_std,
            "psl_std": psl_std,
            "std_ratio": local_std / psl_std,
            "n": xr.DataArray(local.sizes.get("time", 0)),
        }
    )


def compute_reference_skill(
    model_anom: xr.DataArray,
    model_time: xr.DataArray,
    references: Mapping[str, tuple[xr.DataArray, bool]],
    climy0: int,
    climy1: int,
    *,
    nleadavg: int = 1,
    nleads: int | None = None,
    resamp: int = 0,
    detrend: bool = True,
    monthly: bool = False,
) -> xr.Dataset:
    """Compute model skill against multiple observational references.

    Parameters
    ----------
    model_anom
        Drift-corrected model index with dimensions ``Y``, ``L`` and
        optionally ``M``.
    model_time
        Verification timestamps matching ``model_anom`` on ``Y`` and ``L``.
    references
        Mapping of reference name to ``(obs_data, is_anomaly)``. Raw references
        receive climatology removal inside :func:`esp_lab.stats.compute_skill_seasonal`;
        anomaly references such as PSL DMI should use ``is_anomaly=True``.
    climy0, climy1
        Climatology years used for raw references.
    """
    if nleads is None:
        nleads = int(model_anom.sizes["L"])

    names = []
    datasets = []
    for name, (obs, is_anomaly) in references.items():
        skill = stats.compute_skill_seasonal(
            model_anom,
            model_time,
            obs,
            climy0,
            climy1,
            nleadavg=nleadavg,
            nleads=nleads,
            resamp=resamp,
            detrend=detrend,
            monthly=monthly,
            is_anomaly=is_anomaly,
        )
        names.append(name)
        datasets.append(skill)

    reference = xr.DataArray(names, dims="reference", name="reference")
    out = xr.concat(datasets, dim=reference)
    out.attrs.update(
        {
            "climatology": f"{climy0}-{climy1}",
            "detrend": str(bool(detrend)).lower(),
            "monthly": str(bool(monthly)).lower(),
        }
    )
    return out


def add_reference_sensitivity(
    skill: xr.Dataset,
    *,
    primary: str = "HadISST2",
    secondary: str = "PSL",
) -> xr.Dataset:
    """Add PSL-minus-primary deltas and conservative robust metrics."""
    if "reference" not in skill.dims:
        raise ValueError("skill must have a 'reference' dimension")

    refs = set(str(value) for value in skill["reference"].values)
    missing = {primary, secondary} - refs
    if missing:
        raise KeyError(f"Missing reference(s): {sorted(missing)}")

    out = skill.copy()
    primary_skill = skill.sel(reference=primary)
    secondary_skill = skill.sel(reference=secondary)

    if "corr" in skill:
        out["dacc_ref"] = secondary_skill["corr"] - primary_skill["corr"]
        out["corr_mean"] = skill["corr"].mean("reference")
        out["corr_robust"] = xr.concat(
            [primary_skill["corr"], secondary_skill["corr"]],
            dim="reference_pair",
        ).min("reference_pair")

    if "rmse" in skill:
        out["dnrmse_ref"] = secondary_skill["rmse"] - primary_skill["rmse"]
        out["rmse_mean"] = skill["rmse"].mean("reference")
        out["rmse_robust"] = xr.concat(
            [primary_skill["rmse"], secondary_skill["rmse"]],
            dim="reference_pair",
        ).max("reference_pair")

    out.attrs["primary_reference"] = primary
    out.attrs["secondary_reference"] = secondary
    return out
