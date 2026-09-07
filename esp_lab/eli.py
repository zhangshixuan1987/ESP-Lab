"""Reusable loading and skill helpers for Equatorial Longitude Index diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr


@dataclass(frozen=True)
class ELIModelSpec:
    """Description of one monthly hindcast ELI product family."""

    key: str
    label: str
    root: Path
    filename_template: str
    ensemble_size: int
    color: str
    marker: str

    def path(self, init_month: int, nlead: int) -> Path:
        return Path(self.root) / self.filename_template.format(
            init_month=int(init_month),
            nens=int(self.ensemble_size),
            nlead=int(nlead),
        )


def initialization_years(values) -> np.ndarray:
    """Return integer initialization years from numeric or tagged Y values."""

    years = []
    for value in np.asarray(values):
        text = str(value)
        if len(text) < 4 or not text[:4].isdigit():
            raise ValueError(f"Cannot determine initialization year from Y={value!r}")
        years.append(int(text[:4]))
    return np.asarray(years, dtype=int)


def load_model_hindcasts(
    specs: Mapping[str, ELIModelSpec],
    init_months: Sequence[int],
    *,
    nlead: int,
    start_year: int,
    end_year: int,
) -> tuple[dict, dict, dict]:
    """Load and validate every requested model/month ELI cache.

    Returns nested dictionaries containing ELI data, valid times, and source
    provenance.  All products are normalized to integer initialization-year
    coordinates so E3SM and CESM-SMYLE can be compared directly.
    """

    expected_years = np.arange(start_year, end_year + 1, dtype=int)
    data = {key: {} for key in specs}
    valid_time = {key: {} for key in specs}
    provenance = {key: {} for key in specs}
    missing = []

    for key, spec in specs.items():
        for init_month in init_months:
            path = spec.path(init_month, nlead)
            if not path.is_file():
                missing.append(path)
                continue

            with xr.open_dataset(path) as source:
                required = {"eli", "time"}
                absent = required - set(source.variables)
                if absent:
                    raise ValueError(f"{path} is missing variables {sorted(absent)}")
                eli = source["eli"].load()
                time = source["time"].load()
                attrs = dict(source.attrs)

            if eli.dims != ("Y", "L", "M"):
                raise ValueError(f"{path}: expected eli dims ('Y', 'L', 'M'), got {eli.dims}")
            if time.dims != ("Y", "L"):
                raise ValueError(f"{path}: expected time dims ('Y', 'L'), got {time.dims}")
            if eli.sizes["M"] != spec.ensemble_size:
                raise ValueError(
                    f"{path}: expected {spec.ensemble_size} members, got {eli.sizes['M']}"
                )
            if eli.sizes["L"] != nlead:
                raise ValueError(f"{path}: expected {nlead} leads, got {eli.sizes['L']}")

            years = initialization_years(eli.Y.values)
            if len(np.unique(years)) != len(years):
                raise ValueError(f"{path}: duplicate initialization years")
            eli = eli.assign_coords(Y=years).sel(Y=expected_years)
            time = time.assign_coords(Y=years).sel(Y=expected_years)
            if not np.array_equal(eli.Y.values, expected_years):
                raise ValueError(
                    f"{path}: expected initialization years {start_year}-{end_year}, "
                    f"got {eli.Y.values.tolist()}"
                )
            if not bool(eli.notnull().any()):
                raise ValueError(f"{path}: ELI data are entirely missing")
            if not eli.L.identical(time.L):
                raise ValueError(f"{path}: ELI and valid-time lead coordinates differ")

            data[key][int(init_month)] = eli
            valid_time[key][int(init_month)] = time
            provenance[key][int(init_month)] = {"path": str(path), **attrs}

    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing required ELI model caches:\n{formatted}")
    return data, valid_time, provenance


def remove_lead_drift(
    data: Mapping[str, Mapping[int, xr.DataArray]],
    *,
    clim_start: int,
    clim_end: int,
) -> tuple[dict, dict]:
    """Remove each model's lead-dependent ensemble climatology."""

    anomaly = {model: {} for model in data}
    drift = {model: {} for model in data}
    for model, by_month in data.items():
        for init_month, values in by_month.items():
            climatology = values.sel(Y=slice(clim_start, clim_end)).mean(("Y", "M"))
            if not bool(climatology.notnull().all()):
                raise ValueError(
                    f"{model} init {init_month:02d}: missing lead climatology values"
                )
            drift[model][init_month] = climatology
            anomaly[model][init_month] = values - climatology
    return anomaly, drift


def common_target_years_by_lead(
    anomaly: Mapping[str, Mapping[int, xr.DataArray]],
    valid_time: Mapping[str, Mapping[int, xr.DataArray]],
    observation: xr.DataArray,
    init_month: int,
) -> dict[int, list[int]]:
    """Find one identical valid target-year cohort for all compared models."""

    obs_year = np.asarray(observation.time.dt.year.values, dtype=int)
    obs_month = np.asarray(observation.time.dt.month.values, dtype=int)
    obs_valid = np.asarray(observation.notnull().values, dtype=bool)
    common_leads = sorted(
        set.intersection(
            *(set(map(int, anomaly[model][init_month].L.values)) for model in anomaly)
        )
    )
    cohorts = {}
    for lead in common_leads:
        target_months = set()
        common = None
        for model in anomaly:
            values = anomaly[model][init_month].sel(L=lead)
            times = valid_time[model][init_month].sel(L=lead)
            years = np.asarray(times.dt.year.values, dtype=int)
            months = np.asarray(times.dt.month.values, dtype=int)
            target_months.update(np.unique(months).tolist())
            finite = np.asarray(values.notnull().any("M").values, dtype=bool)
            available = set(years[finite].tolist())
            common = available if common is None else common & available
        if len(target_months) != 1:
            raise ValueError(
                f"Init {init_month:02d}, lead {lead}: inconsistent target months "
                f"{sorted(target_months)}"
            )
        target_month = target_months.pop()
        observed = set(obs_year[(obs_month == target_month) & obs_valid].tolist())
        cohorts[lead] = sorted((common or set()) & observed)
        if len(cohorts[lead]) < 3:
            raise ValueError(
                f"Init {init_month:02d}, lead {lead}: only "
                f"{len(cohorts[lead])} common samples"
            )
    return cohorts


def observations_for_times(observation: xr.DataArray, times) -> np.ndarray:
    """Return observation values matched exactly by target year and month."""

    def year_month(value):
        if hasattr(value, "year") and hasattr(value, "month"):
            return int(value.year), int(value.month)
        timestamp = pd.Timestamp(value)
        return int(timestamp.year), int(timestamp.month)

    lookup = {
        year_month(t): float(value)
        for t, value in zip(observation.time.values, observation.values)
        if np.isfinite(value)
    }
    return np.asarray(
        [lookup.get(year_month(t), np.nan) for t in times],
        dtype=float,
    )


def compute_multimodel_skill(
    anomaly: Mapping[str, Mapping[int, xr.DataArray]],
    valid_time: Mapping[str, Mapping[int, xr.DataArray]],
    observation: xr.DataArray,
    init_months: Sequence[int],
) -> tuple[dict, dict]:
    """Compute ACC/RMSE on common model-observation cohorts for every lead."""

    skill = {model: {} for model in anomaly}
    cohorts = {}
    for init_month in init_months:
        cohort = common_target_years_by_lead(
            anomaly, valid_time, observation, int(init_month)
        )
        cohorts[int(init_month)] = cohort
        for model in anomaly:
            leads = np.asarray(sorted(cohort), dtype=int)
            records = {name: [] for name in (
                "acc", "pval", "rmse", "nrmse", "sample_count",
                "target_year_start", "target_year_end",
            )}
            for lead in leads:
                values = anomaly[model][init_month].sel(L=lead).mean("M")
                times = valid_time[model][init_month].sel(L=lead)
                target_year = np.asarray(times.dt.year.values, dtype=int)
                selected = np.isin(target_year, cohort[int(lead)])
                model_values = np.asarray(values.values, dtype=float)[selected]
                selected_times = np.asarray(times.values, dtype=object)[selected]
                obs_values = observations_for_times(observation, selected_times)
                finite = np.isfinite(model_values) & np.isfinite(obs_values)
                model_values = model_values[finite]
                obs_values = obs_values[finite]
                years = target_year[selected][finite]
                if model_values.size != len(cohort[int(lead)]):
                    raise ValueError(
                        f"{model} init {init_month:02d} lead {lead}: cohort changed "
                        "during exact observation alignment"
                    )
                corr, pvalue = pearsonr(model_values, obs_values)
                error = model_values - obs_values
                rmse = float(np.sqrt(np.mean(error**2)))
                records["acc"].append(float(corr))
                records["pval"].append(float(pvalue))
                records["rmse"].append(rmse)
                records["nrmse"].append(rmse / float(np.std(obs_values)))
                records["sample_count"].append(int(model_values.size))
                records["target_year_start"].append(int(years.min()))
                records["target_year_end"].append(int(years.max()))
            skill[model][int(init_month)] = xr.Dataset(
                {
                    name: xr.DataArray(values, dims="L", coords={"L": leads})
                    for name, values in records.items()
                },
                attrs={
                    "model": model,
                    "init_month": int(init_month),
                    "sample_alignment": "common valid target years across all compared models and observations, independently by lead",
                },
            )
    return skill, cohorts
