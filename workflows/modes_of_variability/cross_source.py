"""Validation and alignment helpers for shared MOV source comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs

from esp_lab import stats
from esp_lab.diagnostics.sst_index import BASIC_REGIONS, HELPER_REGIONS


REFERENCE_ATTRIBUTES = (
    "eof_strategy",
    "common_basis_reference",
    "eof_reference_period",
)


def _atomic_to_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    """Write ``dataset`` to a unique temporary file, then rename it over ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{uuid.uuid4().hex}")
    try:
        dataset.to_netcdf(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _shared_coordinate(products: Mapping[str, xr.Dataset], name: str) -> np.ndarray:
    shared: set[object] | None = None
    for source, product in products.items():
        if name not in product.coords:
            raise ValueError(f"{source} product lacks coordinate {name!r}.")
        values = set(product[name].values.tolist())
        shared = values if shared is None else shared & values
    if not shared:
        raise ValueError(f"No common {name} values across sources.")
    return np.asarray(sorted(shared))


def _initialization_years(product: xr.Dataset, source: str) -> np.ndarray:
    """Return integer initialization years from source-native ``Y`` labels."""
    if "Y" not in product.coords:
        raise ValueError(f"{source} product lacks coordinate 'Y'.")
    values = np.asarray(product["Y"].values)
    try:
        if np.issubdtype(values.dtype, np.datetime64):
            years = np.asarray(product["Y"].dt.year.values, dtype=int)
        elif np.issubdtype(values.dtype, np.number):
            years = np.asarray([
                int(str(abs(int(value)))[:4])
                if abs(int(value)) >= 10_000 else int(value)
                for value in values
            ])
        else:
            years = []
            for value in values:
                match = re.search(r"(?<!\d)(\d{4})", str(value))
                years.append(int(match.group(1)) if match else int(value.year))
            years = np.asarray(years)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"Cannot derive initialization years from {source} Y coordinates."
        ) from error
    if np.unique(years).size != years.size:
        raise ValueError(f"{source} product has duplicate initialization years.")
    return years


def _target_year_month(valid_time: xr.DataArray) -> np.ndarray:
    """Canonical target dates across numpy, Gregorian, and CF calendars."""
    return np.stack(
        (valid_time.dt.year.values.astype(int), valid_time.dt.month.values.astype(int)),
        axis=-1,
    )


def _validate_grid(products: Mapping[str, xr.Dataset]) -> None:
    first_source, first = next(iter(products.items()))
    for name in ("lat", "lon"):
        if name not in first.coords:
            raise ValueError(f"{first_source} product lacks {name!r} grid coordinates.")
        for source, product in products.items():
            if name not in product.coords or not np.array_equal(product[name], first[name]):
                raise ValueError(
                    f"Incompatible target grid: {source} {name} differs from {first_source}."
                )


def _validate_reference(products: Mapping[str, xr.Dataset]) -> None:
    first_source, first = next(iter(products.items()))
    for attribute in REFERENCE_ATTRIBUTES:
        value = first.attrs.get(attribute)
        if value is None:
            raise ValueError(f"{first_source} product lacks reference attribute {attribute!r}.")
        for source, product in products.items():
            if product.attrs.get(attribute) != value:
                raise ValueError(
                    f"Incompatible fixed reference: {source} {attribute} differs from {first_source}."
                )


def _validate_populated_leads(product: xr.Dataset, source: str, leads: Sequence[object]) -> None:
    if "mode_index" not in product:
        raise ValueError(f"{source} product lacks 'mode_index'.")
    for lead in leads:
        values = np.asarray(product["mode_index"].sel(L=lead).values)
        if not np.isfinite(values).any():
            raise ValueError(f"{source} lead {lead} is empty; padded leads are not allowed.")


def align_common_mode_products(
    products: Mapping[str, xr.Dataset], *, required_leads: Sequence[int] | None = None
) -> dict[str, xr.Dataset]:
    """Return products restricted to scientifically identical source cohorts.

    The selector deliberately rejects, rather than fills, absent NMME leads.
    It is therefore safe for joint first-year metrics while source-only views
    may retain longer E3SM or CESM-SMYLE lead axes elsewhere.
    """
    if len(products) < 2:
        raise ValueError("At least two source products are required for comparison.")
    _validate_grid(products)
    _validate_reference(products)
    common_leads = _shared_coordinate(products, "L")
    leads = np.asarray(required_leads if required_leads is not None else common_leads)
    missing = {
        source: sorted(set(leads.tolist()) - set(product.L.values.tolist()))
        for source, product in products.items()
    }
    missing = {source: values for source, values in missing.items() if values}
    if missing:
        raise ValueError(f"Requested leads are unavailable and will not be padded: {missing}.")

    source_years = {
        source: _initialization_years(product, source)
        for source, product in products.items()
    }
    shared_years = set.intersection(
        *(set(years.tolist()) for years in source_years.values())
    )
    if not shared_years:
        raise ValueError("No common initialization years across sources.")
    years = np.asarray(sorted(shared_years), dtype=int)
    aligned = {}
    for source, product in products.items():
        positions = [
            int(np.flatnonzero(source_years[source] == year)[0]) for year in years
        ]
        aligned[source] = (
            product.isel(Y=positions).sel(L=leads).assign_coords(Y=years)
        )
    first_source, first = next(iter(aligned.items()))
    if "valid_time" not in first:
        raise ValueError(f"{first_source} product lacks 'valid_time'.")
    for source, product in aligned.items():
        if "valid_time" not in product or not np.array_equal(
            _target_year_month(product["valid_time"]),
            _target_year_month(first["valid_time"]),
        ):
            raise ValueError(f"Target-time mismatch between {source} and {first_source}.")
        _validate_populated_leads(product, source, leads)
        if "target_month" in first and (
            "target_month" not in product
            or not np.array_equal(product["target_month"], first["target_month"])
        ):
            raise ValueError(f"Target-month mismatch between {source} and {first_source}.")
    return aligned


def _observations_at_target_times(observation: xr.DataArray, times: xr.DataArray) -> xr.DataArray:
    if observation.dims != ("time",):
        raise ValueError("Observed mode index must have exactly the ('time',) dimension.")
    months = np.unique(times.dt.month.values)
    if months.size != 1:
        raise ValueError("A compared lead must map to one target month.")
    years = np.asarray(times.dt.year.values, dtype=int)
    monthly = observation.where(observation.time.dt.month == int(months[0]), drop=True)
    monthly = monthly.assign_coords(time=("time", monthly.time.dt.year.values))
    if np.unique(monthly.time.values).size != monthly.time.size:
        raise ValueError("Observed mode index has duplicate target-month years.")
    return monthly.reindex(time=years)


def build_common_skill_cache(
    products: Mapping[str, xr.Dataset], observation: xr.DataArray
) -> xr.Dataset:
    """Compute comparable projected-index skill on one shared source cohort.

    The returned cache is deliberately limited to leads shared by every source.
    It records paired sample coverage so plots cannot imply that longer
    source-only horizons contributed to a joint metric.
    """
    aligned = align_common_mode_products(products)
    sources = list(aligned)
    first = aligned[sources[0]]
    records: dict[str, list[xr.DataArray]] = {
        name: []
        for name in ("corr", "pval", "rmse", "msss", "rpc", "sample_count", "coverage")
    }
    for lead in first.L.values:
        times = first["valid_time"].sel(L=lead)
        observed = _observations_at_target_times(observation, times)
        per_source: dict[str, list[float]] = {name: [] for name in records}
        for source, product in aligned.items():
            forecast = product["mode_index"].sel(L=lead).rename(Y="time")
            forecast = forecast.assign_coords(time=("time", times.dt.year.values))
            ensemble_mean = forecast.mean("M")
            paired = ensemble_mean.notnull() & observed.notnull()
            sample_count = int(paired.sum())
            coverage = sample_count / int(times.size)
            values = ensemble_mean.where(paired, drop=True)
            observed_values = observed.where(paired, drop=True)
            if sample_count < 3:
                metrics = {name: np.nan for name in ("corr", "pval", "rmse", "msss", "rpc")}
            else:
                corr = float(xs.pearson_r(values, observed_values, dim="time"))
                sig_obs = float(observed_values.std("time"))
                sig_sig = float(values.std("time"))
                sig_tot = float(forecast.where(paired, drop=True).std("time").mean("M"))
                mse = float(xs.mse(values, observed_values, dim="time"))
                metrics = {
                    "corr": corr,
                    "pval": float(xs.pearson_r_eff_p_value(values, observed_values, dim="time")),
                    "rmse": float(xs.rmse(values, observed_values, dim="time")),
                    "msss": 1.0 - mse / sig_obs**2 if sig_obs > 0 else np.nan,
                    "rpc": corr / (sig_sig / sig_tot) if corr > 0 and sig_sig > 0 and sig_tot > 0 else np.nan,
                }
            for name, value in metrics.items():
                per_source[name].append(value)
            per_source["sample_count"].append(sample_count)
            per_source["coverage"].append(coverage)
        for name, values in per_source.items():
            records[name].append(xr.DataArray(values, dims="source", coords={"source": sources}).expand_dims(L=[lead]))

    result = xr.Dataset({name: xr.concat(values, dim="L") for name, values in records.items()})
    result["target_month"] = first["target_month"]
    result["target_year_start"] = xr.DataArray(
        [int(first.valid_time.sel(L=lead).dt.year.min()) for lead in first.L.values],
        dims="L", coords={"L": first.L},
    )
    result["target_year_end"] = xr.DataArray(
        [int(first.valid_time.sel(L=lead).dt.year.max()) for lead in first.L.values],
        dims="L", coords={"L": first.L},
    )
    result.attrs.update(
        sample_alignment="identical source-intersection target times by lead",
        source_only_longer_leads="excluded from this comparison cache",
        eof_strategy=str(first.attrs["eof_strategy"]),
        common_basis_reference=str(first.attrs["common_basis_reference"]),
        eof_reference_period=str(first.attrs["eof_reference_period"]),
    )
    return result




def write_common_skill_cache(
    products: Mapping[str, xr.Dataset], observation: xr.DataArray, path: str | Path
) -> Path:
    """Build and atomically save a shared-cohort comparison cache."""
    return _atomic_to_netcdf(build_common_skill_cache(products, observation), path)


def _build_native_source_skill(
    product: xr.Dataset, observation: xr.DataArray, *, source: str
) -> xr.Dataset:
    """Compute projected-index skill for one source without cross-source year filtering."""
    required = {"mode_index", "valid_time", "target_month"}
    missing = required - set(product.data_vars)
    if missing:
        raise ValueError(f"{source} product lacks {sorted(missing)}.")
    forecast = product["mode_index"].transpose("Y", "L", "M")
    valid_time = product["valid_time"].transpose("Y", "L")
    records: dict[str, list[xr.DataArray]] = {
        name: [] for name in (
            "corr", "pval", "rmse", "msss", "rpc", "sample_count", "coverage",
            "target_year_start", "target_year_end", "target_month",
        )
    }
    for lead in forecast.L.values:
        times = valid_time.sel(L=lead)
        observed = _observations_at_target_times(observation, times)
        ensemble_mean = forecast.sel(L=lead).mean("M").rename(Y="time")
        ensemble_mean = ensemble_mean.assign_coords(time=("time", times.dt.year.values))
        paired = ensemble_mean.notnull() & observed.notnull()
        sample_count = int(paired.sum())
        values = ensemble_mean.where(paired, drop=True)
        observed_values = observed.where(paired, drop=True)
        if sample_count < 3:
            metrics = {name: np.nan for name in ("corr", "pval", "rmse", "msss", "rpc")}
        else:
            corr = float(xs.pearson_r(values, observed_values, dim="time"))
            sig_obs = float(observed_values.std("time"))
            sig_sig = float(values.std("time"))
            forecast_members = forecast.sel(L=lead).rename(Y="time")
            forecast_members = forecast_members.assign_coords(time=("time", times.dt.year.values))
            sig_tot = float(forecast_members.where(paired, drop=True).std("time").mean("M"))
            mse = float(xs.mse(values, observed_values, dim="time"))
            metrics = {
                "corr": corr,
                "pval": float(xs.pearson_r_eff_p_value(values, observed_values, dim="time")),
                "rmse": float(xs.rmse(values, observed_values, dim="time")),
                "msss": 1.0 - mse / sig_obs**2 if sig_obs > 0 else np.nan,
                "rpc": corr / (sig_sig / sig_tot) if corr > 0 and sig_sig > 0 and sig_tot > 0 else np.nan,
            }
        lead_coord = {"L": [lead]}
        for name, value in metrics.items():
            records[name].append(xr.DataArray([value], dims="L", coords=lead_coord))
        records["sample_count"].append(xr.DataArray([sample_count], dims="L", coords=lead_coord))
        records["coverage"].append(
            xr.DataArray([sample_count / int(times.size)], dims="L", coords=lead_coord)
        )
        records["target_year_start"].append(
            xr.DataArray([int(times.dt.year.min())], dims="L", coords=lead_coord)
        )
        records["target_year_end"].append(
            xr.DataArray([int(times.dt.year.max())], dims="L", coords=lead_coord)
        )
        records["target_month"].append(
            xr.DataArray(
                [int(product["target_month"].sel(L=lead))], dims="L", coords=lead_coord
            )
        )
    return xr.Dataset({name: xr.concat(values, dim="L") for name, values in records.items()})


def build_native_skill_cache(
    products: Mapping[str, xr.Dataset], observation: xr.DataArray
) -> xr.Dataset:
    """Build full-period, source-native projected-index skill.

    Sources retain their own valid years and native lead horizon. The cache is
    intended for descriptive side-by-side panels, not matched-period rankings.
    """
    if len(products) < 2:
        raise ValueError("A side-by-side comparison requires at least two sources.")
    if observation.dims != ("time",):
        raise ValueError("Observed mode index must have exactly the ('time',) dimension.")
    _validate_reference(products)
    source_caches = [
        _build_native_source_skill(product, observation, source=source).expand_dims(source=[source])
        for source, product in products.items()
    ]
    result = xr.concat(source_caches, dim="source", join="outer")
    result.attrs.update(
        period_alignment="source-native full valid period; years are not intersected",
        comparison_kind="descriptive side-by-side system estimates",
        difference_panels="not computed",
        native_leads="each source retains only its populated native leads",
        eof_strategy=str(next(iter(products.values())).attrs["eof_strategy"]),
        common_basis_reference=str(next(iter(products.values())).attrs["common_basis_reference"]),
        eof_reference_period=str(next(iter(products.values())).attrs["eof_reference_period"]),
    )
    return result


def write_native_skill_cache(
    products: Mapping[str, xr.Dataset], observation: xr.DataArray, path: str | Path
) -> Path:
    """Build and atomically save a full-period source-native skill cache."""
    return _atomic_to_netcdf(build_native_skill_cache(products, observation), path)


def plot_native_skill_figure(cache: xr.Dataset, path: str | Path, *, mode: str) -> Path:
    """Render ACC and RMSE as source columns with common row scales."""
    import matplotlib.pyplot as plt

    required = {"corr", "pval", "rmse", "sample_count", "target_year_start", "target_year_end", "source", "L"}
    missing = required - (set(cache.data_vars) | set(cache.coords))
    if missing:
        raise ValueError(f"Native-period cache lacks {sorted(missing)}.")
    sources = [str(value) for value in cache.source.values]
    figure, axes = plt.subplots(
        2, len(sources), figsize=(4.0 * len(sources), 7.0),
        sharey="row", squeeze=False, layout="constrained",
    )
    finite_rmse = np.asarray(cache["rmse"].values, dtype=float)
    rmse_max = float(np.nanmax(finite_rmse)) if np.isfinite(finite_rmse).any() else 1.0
    metric_specs = (("corr", "ACC", 0.0, (-1.0, 1.0)), ("rmse", "RMSE", None, (0.0, 1.05 * rmse_max)))
    colors = {"e3sm": "#176d8f", "smyle": "#c46b27", "nmme": "#3a8a5b"}
    for column, source in enumerate(sources):
        selected = cache.sel(source=source)
        leads = np.asarray(selected.L.values)
        color = colors.get(source.lower(), "#4c6a87")
        valid_leads = leads[np.isfinite(selected["sample_count"].values)]
        periods = selected[["target_year_start", "target_year_end", "sample_count"]].sel(
            L=valid_leads
        )
        if valid_leads.size:
            years = f"{int(periods.target_year_start.min())}-{int(periods.target_year_end.max())}"
            counts = np.asarray(periods.sample_count.values, dtype=int)
            sample_label = str(int(counts[0])) if np.all(counts == counts[0]) else f"{counts.min()}-{counts.max()}"
        else:
            years, sample_label = "no valid samples", "0"
        for row, (metric, label, baseline, ylim) in enumerate(metric_specs):
            axis = axes[row, column]
            values = selected[metric]
            if baseline is not None:
                axis.axhline(baseline, color="0.45", linewidth=0.8, zorder=0)
            finite = np.isfinite(values.values)
            axis.plot(leads[finite], values.values[finite], color=color, linewidth=2.2, marker="o")
            if metric == "corr":
                significant = finite & np.isfinite(selected["pval"].values) & (selected["pval"].values < 0.1)
                axis.scatter(leads[significant], values.values[significant], color=color, s=36, zorder=3)
            axis.set_ylim(*ylim)
            axis.set_xticks(valid_leads)
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel(label)
            if row == 0:
                axis.set_title(f"{source}\n{years}; n={sample_label}")
        axes[-1, column].set_xlabel("Native lead (months)")
    figure.suptitle(f"{mode.upper()} full-period projected-index skill by source")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination

def _skill_stack_by_member(
    product: xr.Dataset, observation: xr.DataArray, *, source: str
) -> xr.Dataset:
    """Return one skill curve per source member."""
    members = [str(value) for value in product["M"].values]
    if len(members) < 2:
        raise ValueError(f"{source} requires at least two members for a spread.")
    return xr.concat(
        [
            _build_native_source_skill(product.sel(M=[member]), observation, source=source)
            .expand_dims(spread_unit=[member])
            for member in members
        ],
        dim="spread_unit",
        join="outer",
    )


def _skill_stack_by_model_mean(
    product: xr.Dataset, observation: xr.DataArray, *, source: str
) -> xr.Dataset:
    """Return one skill curve per NMME model after reducing its own members."""
    groups: dict[str, list[str]] = {}
    for member in product["M"].values.astype(str):
        if ":" not in member:
            raise ValueError(
                f"{source} member {member!r} lacks MODEL:MEMBER identity required for model spread."
            )
        model, _ = member.split(":", 1)
        groups.setdefault(model, []).append(member)
    if len(groups) < 2:
        raise ValueError(f"{source} requires at least two models for a model-spread band.")
    return xr.concat(
        [
            _build_native_source_skill(product.sel(M=labels), observation, source=source)
            .expand_dims(spread_unit=[model])
            for model, labels in sorted(groups.items())
        ],
        dim="spread_unit",
        join="outer",
    )


def build_ensemble_spread_native_skill_cache(
    products: Mapping[str, xr.Dataset],
    observation: xr.DataArray,
    *,
    model_ensemble_sources: Sequence[str] = ("NMME",),
) -> xr.Dataset:
    """Build full-period source skill with member or model-mean standard deviations.

    E3SM and SMYLE retain their own ensemble-mean skill as the center curve and
    use the standard deviation of member skill curves for shading. A configured
    NMME source containing one model uses the same member-level definition.
    For a multi-model NMME source, each model is first reduced to its
    member-ensemble mean; the center and standard deviation are then computed
    across models.
    """
    if len(products) < 2:
        raise ValueError("An ensemble-spread comparison requires at least two sources.")
    if observation.dims != ("time",):
        raise ValueError("Observed mode index must have exactly the ('time',) dimension.")
    _validate_reference(products)
    metric_names = ("corr", "rmse", "msss", "rpc")
    source_caches = []
    for source, product in products.items():
        if source in model_ensemble_sources:
            members = product["M"].values.astype(str)
            if any(":" not in member for member in members):
                raise ValueError(
                    f"{source} requires MODEL:MEMBER identities for its NMME spread."
                )
            models = {member.split(":", 1)[0] for member in members}
            if len(models) == 1:
                center = _build_native_source_skill(product, observation, source=source)
                stack = _skill_stack_by_member(product, observation, source=source)
                spread_kind = "standard deviation across member skill (single NMME model)"
            else:
                stack = _skill_stack_by_model_mean(product, observation, source=source)
                center = stack.mean("spread_unit", skipna=True)
                spread_kind = "standard deviation across per-model member-ensemble-mean skill"
        else:
            center = _build_native_source_skill(product, observation, source=source)
            stack = _skill_stack_by_member(product, observation, source=source)
            spread_kind = "standard deviation across member skill"
        for metric in metric_names:
            center[f"{metric}_spread"] = stack[metric].std("spread_unit", skipna=True)
        center["spread_unit_count"] = stack["corr"].count("spread_unit").astype("int64")
        center["spread_definition"] = xr.DataArray(spread_kind)
        source_caches.append(center.expand_dims(source=[source]))
    result = xr.concat(source_caches, dim="source", join="outer")
    result.attrs.update(
        period_alignment="source-native full valid period; years are not intersected",
        comparison_kind="three-source line comparison with within-source spread",
        difference_panels="not computed",
        model_ensemble_sources=",".join(model_ensemble_sources),
        native_leads="each source retains only its populated native leads",
    )
    return result


def write_ensemble_spread_native_skill_cache(
    products: Mapping[str, xr.Dataset],
    observation: xr.DataArray,
    path: str | Path,
    *,
    model_ensemble_sources: Sequence[str] = ("NMME",),
) -> Path:
    """Build and atomically save a native ensemble-spread skill cache."""
    cache = build_ensemble_spread_native_skill_cache(
        products, observation, model_ensemble_sources=model_ensemble_sources
    )
    return _atomic_to_netcdf(cache, path)


def plot_ensemble_spread_native_skill_figure(
    cache: xr.Dataset, path: str | Path, *, mode: str
) -> Path:
    """Render ACC/RMSE with E3SM, SMYLE, and NMME on common axes and spread bands."""
    import matplotlib.pyplot as plt

    required = {
        "corr", "rmse", "pval", "corr_spread", "rmse_spread",
        "sample_count", "target_year_start", "target_year_end", "source", "L",
    }
    missing = required - (set(cache.data_vars) | set(cache.coords))
    if missing:
        raise ValueError(f"Ensemble-spread cache lacks {sorted(missing)}.")
    colors = {"e3sm": "#176d8f", "cesm-smyle": "#c46b27", "nmme": "#3a8a5b"}
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=False, layout="constrained")
    finite_rmse = np.asarray(cache["rmse"].values, dtype=float)
    max_rmse = float(np.nanmax(finite_rmse)) if np.isfinite(finite_rmse).any() else 1.0
    for axis, metric, label, baseline, ylim in (
        (axes[0], "corr", "ACC", 0.0, (-1.0, 1.0)),
        (axes[1], "rmse", "RMSE", 0.0, (0.0, 1.05 * max_rmse)),
    ):
        axis.axhline(baseline, color="0.45", linewidth=0.8, zorder=0)
        for number, source in enumerate(cache.source.values.astype(str)):
            selected = cache.sel(source=source)
            finite = np.isfinite(selected[metric].values)
            if not finite.any():
                continue
            leads = np.asarray(selected.L.values)[finite]
            values = np.asarray(selected[metric].values, dtype=float)[finite]
            spread = np.asarray(selected[f"{metric}_spread"].values, dtype=float)[finite]
            color = colors.get(source.lower(), ("#4c6a87", "#8b4b7a")[number % 2])
            years = (
                f"{int(selected.target_year_start.where(selected.sample_count.notnull(), drop=True).min())}-"
                f"{int(selected.target_year_end.where(selected.sample_count.notnull(), drop=True).max())}"
            )
            count = np.asarray(selected.sample_count.where(selected.sample_count.notnull(), drop=True).values, dtype=int)
            n_label = str(count[0]) if np.all(count == count[0]) else f"{count.min()}-{count.max()}"
            label_text = f"{source} ({years}; n={n_label})"
            axis.plot(leads, values, color=color, linewidth=2.2, marker="o", label=label_text)
            axis.fill_between(leads, values - spread, values + spread, color=color, alpha=0.18)
            if metric == "corr":
                significant = finite & np.isfinite(selected.pval.values) & (selected.pval.values < 0.1)
                axis.scatter(np.asarray(selected.L.values)[significant], selected[metric].values[significant], color=color, s=30, zorder=3)
        axis.set_ylabel(label)
        axis.set_xlabel("Native lead (months)")
        axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle(f"{mode.upper()} full-period projected-index skill (shading: +/- 1 SD)")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination

def plot_common_skill_figure(
    cache: xr.Dataset,
    path: str | Path,
    *,
    mode: str,
    source_only_skill: Mapping[str, xr.Dataset] | None = None,
) -> Path:
    """Plot common-cohort ACC/RMSE and explicitly marked longer source-only leads."""
    import matplotlib.pyplot as plt

    required = {"corr", "rmse", "source", "L"}
    missing = required - (set(cache.data_vars) | set(cache.coords))
    if missing:
        raise ValueError(f"Comparison cache lacks {sorted(missing)}.")
    source_only_skill = source_only_skill or {}
    colors = {"e3sm": "#176d8f", "smyle": "#c46b27", "nmme": "#3a8a5b"}
    metrics = (("corr", "ACC", 0.0), ("rmse", "RMSE", None))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, layout="constrained")
    common_last_lead = int(cache.L.max())
    fallback_colors = ("#3a8a5b", "#8b4b7a", "#8f6d1f", "#4c6a87")
    for axis, (metric, label, reference) in zip(axes, metrics):
        if reference is not None:
            axis.axhline(reference, color="0.4", linewidth=0.8, zorder=0)
        for source_number, source in enumerate(cache.source.values.astype(str)):
            values = cache[metric].sel(source=source)
            color = colors.get(source.lower(), fallback_colors[source_number % len(fallback_colors)])
            lower_name, upper_name = f"model_{metric}_min", f"model_{metric}_max"
            if lower_name in cache and upper_name in cache and np.isfinite(cache[lower_name].sel(source=source)).any():
                axis.fill_between(cache.L, cache[lower_name].sel(source=source), cache[upper_name].sel(source=source), color=color, alpha=0.18, label=f"{source.upper()} model range")
            axis.plot(cache.L, values, marker="o", color=color, linewidth=2.2, label=f"{source.upper()} common cohort")
            extended = source_only_skill.get(source)
            if extended is None or metric not in extended:
                continue
            extension = extended[metric].sel(L=extended.L > common_last_lead)
            if not extension.L.size:
                continue
            anchor = values.sel(L=common_last_lead)
            axis.plot(
                np.concatenate(([common_last_lead], extension.L.values)),
                np.concatenate(([float(anchor)], extension.values)),
                marker="o", color=color, linewidth=2.0, linestyle="--",
                label=f"{source.upper()} source-only extension",
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title("Projected-index correlation")
    axes[1].set_title("Projected-index error")
    axes[0].legend(fontsize=8, frameon=False)
    figure.supxlabel("Seasonal lead (months)")
    figure.suptitle(f"{mode.upper()} first-year common-cohort skill")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def build_projected_pc_timeseries(product: xr.Dataset, reference: xr.Dataset | xr.DataArray) -> xr.Dataset:
    """Build the notebook-style projected-PC time-series cache for native leads.

    Each lead retains its own target time and observed reference values.  This
    deliberately avoids manufacturing a continuous NMME horizon from missing
    monthly or second-year leads.
    """
    required = {"mode_index", "valid_time", "target_month"}
    missing = required - set(product.data_vars)
    if missing:
        raise ValueError(f"Projected PC time series requires {sorted(required)}; missing {sorted(missing)}.")
    observation = reference["mode_index"] if isinstance(reference, xr.Dataset) else reference
    if observation.dims != ("time",):
        raise ValueError("Observed mode index must have exactly the ('time',) dimension.")

    forecast = product["mode_index"].transpose("Y", "L", "M")
    valid_time = product["valid_time"].transpose("Y", "L")
    observed_by_lead = []
    for lead in forecast.L.values:
        times = valid_time.sel(L=lead)
        observed = _observations_at_target_times(observation, times)
        observed_by_lead.append(
            observed.rename(time="Y").assign_coords(Y=forecast.Y).expand_dims(L=[lead])
        )
    observed = xr.concat(observed_by_lead, dim="L", coords="minimal", compat="override").transpose("Y", "L")
    result = xr.Dataset(
        {
            "forecast_mean": forecast.mean("M"),
            "forecast_std": forecast.std("M"),
            "observation": observed,
            "sample_available": forecast.mean("M").notnull() & observed.notnull(),
            "member_count": forecast.count("M"),
            "valid_time": valid_time,
            "target_month": product["target_month"],
        }
    )
    result.attrs.update(
        mode=str(product.attrs.get("mode", "unknown")),
        source=str(product.attrs.get("source", "unknown")),
        time_alignment="observations selected by exact valid-time target month and year",
        spread_definition="one ensemble standard deviation across members",
        native_leads="only populated product leads are retained",
    )
    return result


def _matplotlib_time_axis(values: np.ndarray) -> np.ndarray:
    """Return Matplotlib-compatible dates without changing cache calendar values."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values
    if values.size and hasattr(values.flat[0], "strftime"):
        return np.asarray([np.datetime64(value.strftime("%Y-%m-%dT%H:%M:%S")) for value in values])
    return values


def plot_projected_pc_timeseries(
    cache: xr.Dataset, path: str | Path, *, mode: str, init_month: int, lead: int
) -> Path:
    """Render the ESP-Lab notebook's observed-PC and hindcast-spread layout."""
    import matplotlib.pyplot as plt

    required = {"forecast_mean", "forecast_std", "observation", "valid_time", "L"}
    missing = required - (set(cache.data_vars) | set(cache.coords))
    if missing:
        raise ValueError(f"Projected PC cache lacks {sorted(missing)}.")
    if lead not in cache.L.values:
        raise ValueError(f"Lead {lead} is unavailable; native leads are {cache.L.values.tolist()}.")
    selected = cache.sel(L=lead)
    times = _matplotlib_time_axis(selected["valid_time"].values)
    mean = selected["forecast_mean"].values
    spread = selected["forecast_std"].values
    observed = selected["observation"].values
    source = str(cache.attrs.get("source", "model"))
    color = {"e3sm": "#176d8f", "smyle": "#c46b27", "nmme": "#3a8a5b"}.get(source.lower(), "#3a8a5b")

    figure, axis = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
    axis.axhline(0.0, color="0.45", linewidth=0.8, zorder=0)
    axis.plot(times, observed, color="black", linewidth=1.8, label="Observed PC")
    axis.plot(times, mean, color=color, linewidth=2.0, label=f"{source.upper()} ensemble mean")
    axis.fill_between(times, mean - spread, mean + spread, color=color, alpha=0.22, label="+/- 1 ensemble std")
    target_month = int(selected["target_month"])
    axis.set_title(
        f"{mode.upper()} projected EOF PC: init {init_month:02d}, lead {lead} "
        f"(target month {target_month:02d})"
    )
    axis.set_ylabel("Projected PC")
    axis.set_xlabel("Valid time")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_mode_domain_regression_figure(
    product: xr.Dataset,
    reference: xr.Dataset,
    path: str | Path,
    *,
    mode: str,
    init_month: int,
    lead: int,
    stipple_stride: int = 1,
) -> Path:
    """Plot fixed-reference and model regression patterns for one native lead."""
    required_product = {
        "mode_regression_pattern",
        "mode_regression_significant",
        "target_month",
        "mode_pattern_rmse_reference",
        "mode_pattern_pcc_reference",
    }
    missing_product = required_product - set(product.data_vars)
    if missing_product:
        raise ValueError(f"Mode-domain figure missing product fields {sorted(missing_product)}.")
    if "mode_pattern" not in reference:
        raise ValueError("Mode-domain figure requires reference 'mode_pattern'.")
    if lead not in product.L.values:
        raise ValueError(f"Lead {lead} is unavailable; native leads are {product.L.values.tolist()}.")
    if stipple_stride < 1:
        raise ValueError("stipple_stride must be at least one.")

    target_month = int(product["target_month"].sel(L=lead))
    model_pattern = product["mode_regression_pattern"].sel(L=lead)
    model_significant = product["mode_regression_significant"].sel(L=lead).astype(bool)
    reference_pattern = reference["mode_pattern"].sel(target_month=target_month)
    if not np.array_equal(model_pattern.lat, reference_pattern.lat) or not np.array_equal(model_pattern.lon, reference_pattern.lon):
        raise ValueError("Model and reference mode-pattern grids differ.")
    maximum = float(np.nanmax(np.abs(np.concatenate((model_pattern.values.ravel(), reference_pattern.values.ravel())))))
    if not np.isfinite(maximum) or maximum == 0:
        maximum = 1.0
    levels = np.linspace(-maximum, maximum, 17)
    rmse = float(product["mode_pattern_rmse_reference"].sel(L=lead))
    pcc = float(product["mode_pattern_pcc_reference"].sel(L=lead))
    source = str(product.attrs.get("source", "model"))

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharex=True, sharey=True, layout="constrained")
    images = []
    for axis, pattern, title in (
        (axes[0], reference_pattern, "ERA5 fixed-EOF reference"),
        (axes[1], model_pattern, f"{source.upper()} regression pattern"),
    ):
        image = axis.contourf(pattern.lon, pattern.lat, pattern, levels=levels, cmap="RdBu_r", extend="both")
        images.append(image)
        axis.contour(pattern.lon, pattern.lat, pattern, levels=levels[::2], colors="0.25", linewidths=0.35, alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Latitude")
    stipple = model_significant.isel(lat=slice(None, None, stipple_stride), lon=slice(None, None, stipple_stride))
    latitude, longitude = np.meshgrid(stipple.lat.values, stipple.lon.values, indexing="ij")
    axes[1].scatter(longitude[stipple.values], latitude[stipple.values], s=7, c="black", alpha=0.45, linewidths=0)
    colorbar = figure.colorbar(images[-1], ax=axes, shrink=0.9, pad=0.02)
    colorbar.set_label(str(model_pattern.attrs.get("units", "regression amplitude")))
    axes[1].text(0.02, 0.02, f"RMSE = {rmse:.3f}\nPCC = {pcc:.3f}", transform=axes[1].transAxes, va="bottom", ha="left", fontsize=9, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    figure.suptitle(f"{mode.upper()} mode-domain patterns: init {init_month:02d}, lead {lead}, target month {target_month:02d}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_cross_source_teleconnection_figure(
    products: Mapping[str, xr.Dataset],
    path: str | Path,
    *,
    mode: str,
    lead: int,
    target_month: int,
    stipple_stride: int = 3,
) -> Path:
    """Render fixed-reference and source global regression maps on one scale."""
    if len(products) < 2:
        raise ValueError("A cross-source teleconnection figure requires at least two products.")
    if stipple_stride < 1:
        raise ValueError("stipple_stride must be at least one.")
    selected = {}
    for source, product in products.items():
        required = {
            "mode_global_regression_pattern",
            "mode_global_regression_significant",
        }
        missing = required - set(product.data_vars)
        if missing:
            raise ValueError(f"{source} teleconnection lacks {sorted(missing)}.")
        if "L" in product["mode_global_regression_pattern"].dims:
            if lead not in product.L.values:
                raise ValueError(f"{source} teleconnection lacks lead {lead}.")
            selected[source] = product.sel(L=lead)
        elif "target_month" in product["mode_global_regression_pattern"].dims:
            if target_month not in product.target_month.values:
                raise ValueError(
                    f"{source} teleconnection lacks target month {target_month}."
                )
            selected[source] = product.sel(target_month=target_month)
        else:
            raise ValueError(f"{source} teleconnection has no L or target_month axis.")
    first_source, first = next(iter(selected.items()))
    for source, product in selected.items():
        if not np.array_equal(product.lat, first.lat) or not np.array_equal(
            product.lon, first.lon
        ):
            raise ValueError(
                f"Teleconnection grid mismatch between {source} and {first_source}."
            )
    maximum = max(
        float(np.nanmax(np.abs(product["mode_global_regression_pattern"].values)))
        for product in selected.values()
    )
    if not np.isfinite(maximum) or maximum == 0:
        maximum = 1.0
    levels = np.linspace(-maximum, maximum, 17)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(selected), 1, figsize=(11.5, 3.2 * len(selected)),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    image = None
    for axis, (source, product) in zip(axes[:, 0], selected.items()):
        pattern = product["mode_global_regression_pattern"]
        significant = product["mode_global_regression_significant"].astype(bool)
        image = axis.contourf(
            pattern.lon, pattern.lat, pattern, levels=levels, cmap="RdBu_r",
            extend="both",
        )
        axis.contour(
            pattern.lon, pattern.lat, pattern, levels=levels[::2],
            colors="0.25", linewidths=0.3, alpha=0.45,
        )
        stipple = significant.isel(
            lat=slice(None, None, stipple_stride),
            lon=slice(None, None, stipple_stride),
        )
        latitude, longitude = np.meshgrid(
            stipple.lat.values, stipple.lon.values, indexing="ij"
        )
        axis.scatter(
            longitude[stipple.values], latitude[stipple.values],
            s=5, c="black", alpha=0.4, linewidths=0,
        )
        axis.set_title(str(source))
        axis.set_ylabel("Latitude")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Longitude")
    colorbar = figure.colorbar(image, ax=axes[:, 0], shrink=0.9, pad=0.02)
    colorbar.set_label(
        str(first["mode_global_regression_pattern"].attrs.get(
            "units", "regression amplitude"
        ))
    )
    figure.suptitle(
        f"{mode.upper()} global teleconnection: lead {lead}, "
        f"target month {target_month:02d}"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def build_projected_mode_skill_cache(
    product: xr.Dataset, reference: xr.Dataset | xr.DataArray, *, detrend: bool = True
) -> xr.Dataset:
    """Compute notebook-equivalent projected-index skill for one source.

    Prefer the product's ``*_skill`` axis when supplied.  SST MOV products use
    it to retain their native monthly cache while evaluating the notebook's
    quarterly verification leads.
    """
    observation = reference["mode_index"] if isinstance(reference, xr.Dataset) else reference
    if observation.dims != ("time",):
        raise ValueError("Observed mode index must have exactly the ('time',) dimension.")
    if {"mode_index_skill", "valid_time_skill"} <= set(product.data_vars):
        forecast = product["mode_index_skill"].rename(skill_L="L").transpose("Y", "L", "M")
        valid_time = product["valid_time_skill"].rename(skill_L="L").transpose("Y", "L")
        target_month = product.get("target_month_skill")
        if target_month is not None:
            target_month = target_month.rename(skill_L="L")
    else:
        required = {"mode_index", "valid_time", "target_month"}
        missing = required - set(product.data_vars)
        if missing:
            raise ValueError(f"Projected-mode skill requires {sorted(required)}; missing {sorted(missing)}.")
        forecast = product["mode_index"].transpose("Y", "L", "M")
        valid_time = product["valid_time"].transpose("Y", "L")
        target_month = product["target_month"]
    if target_month is None:
        target_month = xr.DataArray(
            [int(valid_time.sel(L=lead).dt.month.values[0]) for lead in forecast.L.values],
            dims="L", coords={"L": forecast.L}, name="target_month",
        )
    result = stats.compute_skill_seasonal(
        forecast,
        valid_time,
        observation,
        nleadavg=1,
        nleads=forecast.sizes["L"],
        detrend=detrend,
        is_anomaly=True,
    )
    result["target_month"] = target_month
    result.attrs.update(
        mode=str(product.attrs.get("mode", "unknown")),
        source=str(product.attrs.get("source", "unknown")),
        skill_axis="product mode_index_skill when available; otherwise native mode_index",
        detrend=str(bool(detrend)).lower(),
    )
    return result


def plot_projected_mode_skill_figure(cache: xr.Dataset, path: str | Path, *, mode: str, init_month: int) -> Path:
    """Render ACC and normalized-RMSE curves for one source's projected mode."""
    import matplotlib.pyplot as plt

    required = {"corr", "rmse", "pval", "L", "target_month"}
    missing = required - (set(cache.data_vars) | set(cache.coords))
    if missing:
        raise ValueError(f"Projected-mode skill cache lacks {sorted(missing)}.")
    source = str(cache.attrs.get("source", "model"))
    color = {"e3sm": "#176d8f", "smyle": "#c46b27", "nmme": "#3a8a5b"}.get(source.lower(), "#3a8a5b")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, layout="constrained")
    for axis, variable, label, baseline in (
        (axes[0], "corr", "ACC", 0.0),
        (axes[1], "rmse", "Normalized RMSE", 1.0),
    ):
        values = cache[variable].values
        axis.axhline(baseline, color="0.45", linewidth=0.8, zorder=0)
        axis.plot(cache.L, values, color=color, linewidth=2.2, marker="o")
        if variable == "corr":
            significant = np.isfinite(cache.pval.values) & (cache.pval.values < 0.1)
            axis.scatter(cache.L.values[significant], values[significant], color=color, s=38, zorder=3, label="p < 0.1")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_xticks(cache.L.values)
    axes[0].set_xticklabels([f"{int(lead)} ({int(month):02d})" for lead, month in zip(cache.L.values, cache.target_month.values)])
    figure.supxlabel("Lead month (target month)")
    figure.suptitle(f"{mode.upper()} projected-mode skill: {source.upper()} init {init_month:02d}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def build_nao_station_eof_skill_cache(
    product: xr.Dataset, reference: xr.Dataset, *, detrend: bool = True
) -> xr.Dataset:
    """Compute notebook-equivalent NAO EOF and station-index skill together."""
    required_product = {"mode_index", "nao_station", "valid_time", "target_month"}
    required_reference = {"mode_index", "nao_station"}
    missing_product = required_product - set(product.data_vars)
    missing_reference = required_reference - set(reference.data_vars)
    if missing_product or missing_reference:
        raise ValueError(
            f"NAO station skill requires product={sorted(required_product)} and "
            f"reference={sorted(required_reference)}; missing product={sorted(missing_product)}, "
            f"reference={sorted(missing_reference)}."
        )
    results = []
    for method, model_variable, reference_variable in (
        ("eof", "mode_index", "mode_index"),
        ("station", "nao_station", "nao_station"),
    ):
        forecast = product[model_variable].transpose("Y", "L", "M")
        skill = stats.compute_skill_seasonal(
            forecast,
            product["valid_time"],
            reference[reference_variable],
            nleadavg=1,
            nleads=forecast.sizes["L"],
            detrend=detrend,
            is_anomaly=True,
        ).rename(rmse="nrmse")
        results.append(skill.expand_dims(method=[method]))
    result = xr.concat(results, dim="method")
    result["target_month"] = product["target_month"]
    result.attrs.update(
        mode="NAO",
        source=str(product.attrs.get("source", "unknown")),
        skill_definition="ESP-Lab seasonal ensemble-mean skill; station and observed-EOF methods",
        detrend=str(bool(detrend)).lower(),
    )
    return result


def skill_summary_dataframe(cache: xr.Dataset, *, init_month: int) -> pd.DataFrame:
    """Return notebook-style machine-readable rows from a method/lead skill cache."""
    required = {"method", "L", "target_month", "corr", "pval", "nrmse", "sample_count"}
    missing = required - (set(cache.data_vars) | set(cache.coords))
    if missing:
        raise ValueError(f"Skill cache lacks {sorted(missing)}.")
    rows = []
    for method in cache.method.values.astype(str):
        for lead in cache.L.values:
            row = cache.sel({"method": method, "L": lead})
            rows.append({
                "mode": str(cache.attrs.get("mode", "unknown")),
                "source": str(cache.attrs.get("source", "unknown")),
                "method": method,
                "init_month": int(init_month),
                "lead": int(lead),
                "target_month": int(row["target_month"]),
                "target_year_start": int(row["target_year_start"]),
                "target_year_end": int(row["target_year_end"]),
                "sample_count": int(row["sample_count"]),
                "acc": float(row["corr"]),
                "pval": float(row["pval"]),
                "nrmse": float(row["nrmse"]),
                "msss": float(row["msss"]),
                "rpc": float(row["rpc"]),
            })
    return pd.DataFrame(rows)


def write_skill_summary_table(cache: xr.Dataset, directory: str | Path, *, init_month: int) -> tuple[Path, Path]:
    """Write notebook-style CSV and text summaries for one MOV skill cache."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    mode = str(cache.attrs.get("mode", "mode")).lower()
    source = str(cache.attrs.get("source", "source")).lower()
    stem = f"{source}_{mode}_init{init_month:02d}_skill_summary"
    dataframe = skill_summary_dataframe(cache, init_month=init_month)
    csv_path, text_path = output / f"{stem}.csv", output / f"{stem}.txt"
    dataframe.to_csv(csv_path, index=False, float_format="%.4f")
    text_path.write_text(dataframe.to_string(index=False, float_format=lambda value: f"{value:.4f}") + "\n")
    return csv_path, text_path


def build_model_mean_common_skill_cache(
    products: Mapping[str, xr.Dataset], observation: xr.DataArray, *, source: str = "nmme"
) -> xr.Dataset:
    """Build a common-skill cache with optional equal-weight NMME model means."""
    if source not in products:
        raise ValueError(f"Requested grouped source {source!r} is absent.")
    members = [str(value) for value in products[source]["M"].values]
    groups: dict[str, list[str]] = {}
    for member in members:
        if ":" not in member:
            return build_common_skill_cache(products, observation)
        model, _ = member.split(":", 1)
        groups.setdefault(model, []).append(member)
    if len(groups) <= 1:
        return build_common_skill_cache(products, observation)
    expanded = {name: product for name, product in products.items() if name != source}
    model_sources = []
    for model, labels in sorted(groups.items()):
        name = f"{source}:{model}"
        expanded[name] = products[source].sel(M=labels)
        model_sources.append(name)
    raw = build_common_skill_cache(expanded, observation)
    source_order = list(products)
    variables = {}
    for variable in raw.data_vars:
        if "source" not in raw[variable].dims:
            variables[variable] = raw[variable]
            continue
        pieces = []
        for name in source_order:
            value = raw[variable].sel(source=model_sources).mean("source") if name == source else raw[variable].sel(source=name)
            pieces.append(value.expand_dims(source=[name]))
        variables[variable] = xr.concat(pieces, dim="source")
    result = xr.Dataset(variables, attrs=raw.attrs)
    for metric in ("corr", "rmse"):
        lower, upper = [], []
        for name in source_order:
            if name == source:
                values = raw[metric].sel(source=model_sources)
                lower.append(values.min("source").expand_dims(source=[name]))
                upper.append(values.max("source").expand_dims(source=[name]))
            else:
                template = xr.full_like(raw[metric].sel(source=name), np.nan)
                lower.append(template.expand_dims(source=[name]))
                upper.append(template.expand_dims(source=[name]))
        result[f"model_{metric}_min"] = xr.concat(lower, dim="source")
        result[f"model_{metric}_max"] = xr.concat(upper, dim="source")
    result["model_count"] = xr.DataArray(
        [len(groups) if name == source else 1 for name in source_order],
        dims="source", coords={"source": source_order},
    )
    result.attrs["model_mean_source"] = source
    result.attrs["model_mean_method"] = "equal-weight model ensemble means; min-max model skill range"
    return result


def write_model_mean_common_skill_cache(
    products: Mapping[str, xr.Dataset], observation: xr.DataArray, path: str | Path, *, source: str = "nmme"
) -> Path:
    """Atomically write an optional grouped-model common-skill cache."""
    return _atomic_to_netcdf(
        build_model_mean_common_skill_cache(products, observation, source=source), path
    )


def build_ensemble_spread_common_skill_cache(
    products: Mapping[str, xr.Dataset], observation: xr.DataArray, *,
    model_ensemble_sources: Sequence[str] = ("NMME",),
) -> xr.Dataset:
    """Build Phase-7-style ensemble-spread skill on one exact common cohort."""
    aligned = align_common_mode_products(products)
    result = build_ensemble_spread_native_skill_cache(
        aligned, observation, model_ensemble_sources=model_ensemble_sources
    )
    result.attrs.update(
        period_alignment="identical source-intersection target times by lead",
        comparison_kind="common-cohort three-source line comparison with within-source spread",
        source_only_longer_leads="excluded from this comparison cache",
    )
    return result

# ---------------------------------------------------------------------------
# Regional SST indices from source-native SST fields
# ---------------------------------------------------------------------------
# Supported indices: every BASIC_REGIONS region, IOD (West minus East), ONI
# (centered three-month mean of Nino3.4), and the derived TNI and RONI.
# Forecast products apply running means along lead ``L``; observations along
# ``time``. NMME anomalies remove each model's own member-and-start-year mean
# at every lead; other sources use their ``SST_anom`` field as provided.

DERIVED_SST_INDICES = ("TNI", "RONI")
_DERIVED_SST_COMPONENTS = {"TNI": ("Nino12", "Nino4"), "RONI": ("Nino3.4", "TropicalMean")}
_SOURCE_SST_ANOMALY_METHOD = "source SST_anom field"


def _is_nmme(field_cache: xr.Dataset) -> bool:
    return str(field_cache.attrs.get("source", "")).upper() == "NMME"


def _base_sst_index(index: str) -> str:
    """ONI is the smoothed Nino3.4 anomaly; every other index is its own base."""
    return "Nino3.4" if index == "ONI" else index


def _regional_sst_anomaly_mean(anomaly: xr.DataArray, *, index: str) -> xr.DataArray:
    """Return the cosine-weighted mean SST anomaly over one standard region."""
    definitions = {**BASIC_REGIONS, **HELPER_REGIONS}
    if index not in definitions:
        raise ValueError(f"{index!r} is not a supported regional SST definition.")
    if not {"lat", "lon"}.issubset(anomaly.coords):
        raise ValueError("SST anomaly requires lat/lon coordinates.")
    lon_min, lon_max, lat_min, lat_max = definitions[index]["lonlat"]
    # 0-360 longitudes and ascending latitudes, so the slices below select the
    # region whatever the orientation of the source grid. (Latitudes are only
    # re-sorted when descending, which leaves ascending grids bit-for-bit unchanged.)
    normalized = anomaly.assign_coords(lon=(anomaly.lon % 360.0)).sortby("lon")
    if normalized.lat.size > 1 and bool(normalized.lat[0] > normalized.lat[-1]):
        normalized = normalized.sortby("lat")
    regional = normalized.sel(lat=slice(lat_min, lat_max))
    if lon_min <= lon_max:
        regional = regional.sel(lon=slice(lon_min, lon_max))
    else:
        regional = xr.concat(
            [regional.sel(lon=slice(lon_min, 360.0)), regional.sel(lon=slice(0.0, lon_max))],
            dim="lon",
        )
    if not regional.sizes.get("lat") or not regional.sizes.get("lon"):
        raise ValueError(f"{index} bounds select no SST grid cells.")
    weights = xr.DataArray(
        np.cos(np.deg2rad(regional.lat.values)), dims=("lat",), coords={"lat": regional.lat}
    )
    return regional.weighted(weights).mean(("lat", "lon"), skipna=True).rename("mode_index").assign_attrs(
        long_name=str(definitions[index]["long_name"]) + " anomaly",
        units=str(anomaly.attrs.get("units", "degC")),
        index=index,
        index_definition=(
            f"cosine-weighted SST anomaly mean over "
            f"{lon_min:g}-{lon_max:g}E, {lat_min:g}-{lat_max:g}N"
        ),
    )


def _sst_index_from_field(field: xr.DataArray, *, index: str) -> xr.DataArray:
    """Return a regional mean, or the IOD as West minus East."""
    if index == "IOD":
        west = _regional_sst_anomaly_mean(field, index="IOD_West")
        east = _regional_sst_anomaly_mean(field, index="IOD_East")
        return (west - east).rename("mode_index").assign_attrs(
            long_name="Dipole Mode Index SST anomaly",
            units=str(field.attrs.get("units", "degC")),
            index="IOD",
            index_definition="IOD West minus IOD East regional SST anomaly",
        )
    return _regional_sst_anomaly_mean(field, index=index)


def _apply_sst_index_transform(anomaly: xr.DataArray, *, index: str, rolling_dim: str) -> xr.DataArray:
    """Apply the post-anomaly transform of ``index`` (ONI smoothing; others unchanged)."""
    if index == "ONI":
        return anomaly.rolling({rolling_dim: 3}, center=True, min_periods=1).mean().rename(
            "mode_index"
        ).assign_attrs(
            long_name="Oceanic Nino Index",
            units=str(anomaly.attrs.get("units", "degC")),
            index="ONI",
            index_definition="centered three-month running mean of Nino3.4 SST anomalies",
        )
    return anomaly


def _require_per_model_nmme_anomaly(field_cache: xr.Dataset) -> str:
    """Return the NMME anomaly method, rejecting products with a pooled climatology."""
    labels = field_cache["SST_anom"].M.values.astype(str)
    if any(":" not in label for label in labels):
        raise ValueError("NMME regional SST requires MODEL:MEMBER labels.")
    method = str(
        field_cache["SST_anom"].attrs.get(
            "nmme_anomaly_method", field_cache.attrs.get("nmme_anomaly_method", "")
        )
    )
    if not method.startswith("per-model"):
        raise ValueError(
            "NMME SST_anom must have each model's drift removed separately "
            "(attribute nmme_anomaly_method starting with 'per-model'); rebuild the "
            "NMME field product with scripts/run_process_modes_of_variability.py."
        )
    return method


def _forecast_sst_index(field_cache: xr.Dataset, *, index: str) -> xr.DataArray:
    """Non-derived index (regional, IOD, ONI) for one forecast source."""
    anomaly = _sst_index_from_field(field_cache["SST_anom"], index=_base_sst_index(index))
    if _is_nmme(field_cache):
        anomaly = anomaly.assign_attrs(regional_anomaly_method=_require_per_model_nmme_anomaly(field_cache))
    return _apply_sst_index_transform(anomaly, index=index, rolling_dim="L")


def _derived_components(field_cache: xr.Dataset, index: str, *, forecast: bool) -> dict[str, xr.DataArray]:
    """Regional anomalies that TNI or RONI are built from."""
    names = _DERIVED_SST_COMPONENTS[index]
    if forecast:
        return {name: _forecast_sst_index(field_cache, index=name) for name in names}
    return {name: _sst_index_from_field(field_cache["SST_anom"], index=name) for name in names}


def _derived_value(
    parts: Mapping[str, xr.DataArray], index: str, rolling_dim: str, sample_dims: Sequence[str]
) -> xr.DataArray:
    """TNI (standardized Nino12 minus Nino4, 5-month mean) or RONI (scaled relative Nino3.4)."""
    if index == "TNI":
        def standardized(data: xr.DataArray) -> xr.DataArray:
            return data / data.std(sample_dims, skipna=True).where(lambda value: value > 0, 1.0)

        value = (standardized(parts["Nino12"]) - standardized(parts["Nino4"])).rolling(
            {rolling_dim: 5}, center=True, min_periods=1
        ).mean()
        return value.where(parts["Nino12"].notnull() & parts["Nino4"].notnull()).rename(
            "mode_index"
        ).assign_attrs(
            index="TNI",
            units="1",
            long_name="Trans-Niño Index",
            index_definition="centered five-month mean of standardized Nino12 minus Nino4 anomalies",
        )
    difference = (parts["Nino3.4"] - parts["TropicalMean"]).rolling(
        {rolling_dim: 3}, center=True, min_periods=1
    ).mean()
    nino34 = parts["Nino3.4"].rolling({rolling_dim: 3}, center=True, min_periods=1).mean()
    denominator = difference.std(sample_dims, skipna=True)
    scale = (nino34.std(sample_dims, skipna=True) / denominator).fillna(1.0).where(denominator > 0, 1.0)
    return (difference * scale).where(
        parts["Nino3.4"].notnull() & parts["TropicalMean"].notnull()
    ).rename("mode_index").assign_attrs(
        index="RONI",
        units=str(nino34.attrs.get("units", "degC")),
        long_name="Relative Oceanic Niño Index",
        index_definition="variance-scaled centered three-month Nino3.4 minus tropical-mean anomaly",
    )


def build_regional_sst_index_product(field_cache: xr.Dataset, *, index: str) -> xr.Dataset:
    """Adapt source-native forecast SST fields to a regional, IOD, ONI, TNI, or RONI product."""
    if not {"SST_anom", "valid_time"}.issubset(field_cache.data_vars):
        raise ValueError("SST field cache requires SST_anom and valid_time.")
    if index in DERIVED_SST_INDICES:
        parts = _derived_components(field_cache, index, forecast=True)
        anomaly = _derived_value(parts, index, "L", ("Y", "M"))
        method = next(iter(parts.values())).attrs.get("regional_anomaly_method", _SOURCE_SST_ANOMALY_METHOD)
    else:
        anomaly = _forecast_sst_index(field_cache, index=index)
        method = anomaly.attrs.get("regional_anomaly_method", _SOURCE_SST_ANOMALY_METHOD)
    mode_index = anomaly.transpose("Y", "L", "M")
    valid_time = field_cache["valid_time"].transpose("Y", "L")
    result = xr.Dataset(
        {
            "mode_index": mode_index,
            "valid_time": valid_time,
            "target_month": valid_time.isel(Y=0).dt.month.rename("target_month"),
        }
    )
    result.attrs.update({key: str(field_cache.attrs[key]) for key in REFERENCE_ATTRIBUTES})
    result.attrs.update(
        index=index,
        index_definition=str(mode_index.attrs["index_definition"]),
        source_native_sst_anomaly="true",
        regional_anomaly_method=str(method),
    )
    return result


def build_regional_sst_index_observation(field_cache: xr.Dataset, *, index: str) -> xr.DataArray:
    """Adapt observed SST anomalies to the matching regional, IOD, ONI, TNI, or RONI index."""
    if "SST_anom" not in field_cache:
        raise ValueError("Observed SST field cache lacks SST_anom.")
    if index in DERIVED_SST_INDICES:
        return _derived_value(_derived_components(field_cache, index, forecast=False), index, "time", ("time",))
    anomaly = _sst_index_from_field(field_cache["SST_anom"], index=_base_sst_index(index))
    return _apply_sst_index_transform(anomaly, index=index, rolling_dim="time")


# ---------------------------------------------------------------------------
# NMME multi-model skill: one cohort, one line, one min-max model band.
# ---------------------------------------------------------------------------

# Default cohort: the NMME hindcast period. 20 of the 23 archived models cover
# every May and November start in 1982-2010; (1991, 2009) keeps all 23.
NMME_DEFAULT_YEARS = (1982, 2010)
NMME_MIN_MEMBERS = 4
NMME_LINE_METHOD = (
    "skill of the equal-weight multi-model forecast: each model's ensemble mean, "
    "averaged over models with one weight per model"
)
NMME_BAND_METHOD = "min-max of the individual models' skill"


def nmme_model_members(index: xr.DataArray) -> dict[str, list[str]]:
    """Group ``MODEL:MEMBER`` labels by model, keeping archive order."""
    groups: dict[str, list[str]] = {}
    for label in index.M.values.astype(str):
        if ":" not in label:
            raise ValueError("NMME products require MODEL:MEMBER member labels.")
        groups.setdefault(label.split(":", 1)[0], []).append(label)
    return groups


def select_nmme_cohort(
    index: xr.DataArray,
    years: Sequence[int] = NMME_DEFAULT_YEARS,
    *,
    models: str | Sequence[str] = "auto",
    min_members: int = NMME_MIN_MEMBERS,
) -> tuple[xr.DataArray, dict[str, str]]:
    """Restrict ``index`` to start years ``years`` and to models that cover them.

    A model qualifies when every start year in ``years`` has at least
    ``min_members`` finite members at the first lead. ``models="auto"`` keeps all
    qualifying models; an explicit list must qualify in full. Returns the subset
    and ``{model: reason}`` for each model left out.
    """
    year0, year1 = (int(value) for value in years)
    if year0 > year1:
        raise ValueError(f"NMME years must be increasing, got {years!r}.")
    wanted = np.arange(year0, year1 + 1)
    present = np.asarray(index.Y.values).astype(int)
    groups = nmme_model_members(index)
    if isinstance(models, str):
        if models != "auto":
            models = [models]
    requested = None if isinstance(models, str) else [str(model) for model in models]
    if requested is not None:
        unknown = sorted(set(requested) - set(groups))
        if unknown:
            raise ValueError(f"NMME models not in the product: {unknown}")
    first_lead = index.isel(L=0)
    while set(first_lead.dims) - {"Y", "M"}:
        first_lead = first_lead.isel({next(iter(set(first_lead.dims) - {"Y", "M"})): 0})
    dropped: dict[str, str] = {}
    kept: list[str] = []
    for model in requested or list(groups):
        missing = [int(year) for year in wanted if year not in present]
        if not missing:
            counts = first_lead.sel(Y=wanted, M=groups[model]).notnull().sum("M")
            missing = [int(year) for year, count in zip(wanted, counts.values) if count < min_members]
        if missing:
            dropped[model] = (
                f"fewer than {min_members} members for {len(missing)} start year(s) "
                f"({missing[0]}-{missing[-1]})"
            )
        else:
            kept.append(model)
    if requested is not None and dropped:
        raise ValueError(
            f"Requested NMME models do not cover {year0}-{year1}: "
            + "; ".join(f"{model}: {reason}" for model, reason in dropped.items())
        )
    if not kept:
        raise ValueError(f"No NMME model covers every start in {year0}-{year1}.")
    members = [label for model in kept for label in groups[model]]
    subset = index.sel(Y=wanted, M=members).assign_attrs(
        index.attrs,
        nmme_models=",".join(kept),
        nmme_years=f"{year0}-{year1}",
        nmme_min_members=str(min_members),
    )
    return subset, dropped


def nmme_multimodel_mean(index: xr.DataArray) -> xr.DataArray:
    """Equal-weight multi-model forecast: model ensemble means, then their mean.

    Models with a shorter forecast drop out at the leads they lack, so later
    leads average fewer models; ``nmme_model_count`` records how many.
    """
    groups = nmme_model_members(index)
    model_means = xr.concat(
        [index.sel(M=labels).mean("M", skipna=True) for labels in groups.values()],
        dim=xr.DataArray(list(groups), dims="model", name="model"),
    )
    combined = model_means.mean("model", skipna=True).expand_dims(M=["NMME"]).transpose(*index.dims)
    return combined.assign_attrs(index.attrs, nmme_line_method=NMME_LINE_METHOD)


def nmme_skill(
    index: xr.DataArray,
    skill_function,
    *,
    model: str | None = None,
    keep_units: bool = False,
) -> xr.Dataset:
    """Score NMME with the caller's skill function and add a min-max band.

    ``skill_function(index)`` scores one (Y, L, M) forecast and returns a Dataset
    whose metrics have an ``L`` dimension; it is the same function used for the
    other sources, so every line is scored identically. With ``model=None`` the
    line is the equal-weight multi-model forecast and the band spans the
    individual models. With one ``model`` the line is that model's ensemble mean
    and the band spans its members. ``keep_units=True`` also returns each
    model's (or member's) skill as ``{metric}_by_model`` (``_by_member``), so a
    caller can form the band of an average, e.g. over start months.
    """
    groups = nmme_model_members(index)
    if model is None:
        line = skill_function(nmme_multimodel_mean(index))
        units = {name: index.sel(M=labels) for name, labels in groups.items()}
        unit_name, band_method = "model", NMME_BAND_METHOD
    else:
        if model not in groups:
            raise ValueError(f"NMME model {model!r} is not in the product.")
        line = skill_function(index.sel(M=groups[model]))
        units = {label: index.sel(M=[label]) for label in groups[model]}
        unit_name, band_method = "member", f"min-max of {model}'s individual members' skill"
    unit_skill = xr.concat(
        [skill_function(member_index) for member_index in units.values()],
        dim=xr.DataArray(list(units), dims=unit_name, name=unit_name),
    )
    result = line.copy()
    for name, variable in unit_skill.data_vars.items():
        if unit_name in variable.dims and "L" in variable.dims and np.issubdtype(variable.dtype, np.number):
            result[f"{name}_min"] = variable.min(unit_name, skipna=True)
            result[f"{name}_max"] = variable.max(unit_name, skipna=True)
            if keep_units:
                result[f"{name}_by_{unit_name}"] = variable
    reference = next(
        (name for name in ("corr", "acc", "rmse") if name in unit_skill and unit_name in unit_skill[name].dims),
        None,
    )
    if reference is not None:
        counted = unit_skill[reference]
        result[f"{unit_name}_count"] = counted.notnull().sum(unit_name).where(
            counted.notnull().any(unit_name)
        )
    result.attrs.update(
        nmme_line_method=(NMME_LINE_METHOD if model is None else f"skill of {model}'s ensemble mean"),
        nmme_band_method=band_method,
        nmme_models=",".join(groups if model is None else [model]),
        nmme_unit_count=str(len(units)),
    )
    for key in ("nmme_years", "nmme_min_members"):
        if key in index.attrs:
            result.attrs[key] = index.attrs[key]
    return result


# ---------------------------------------------------------------------------
# NMME regional SST indices and ELI, one model at a time.
# ---------------------------------------------------------------------------

NMME_SST_INDICES = (
    "Nino3.4", "Nino12", "Nino3", "Nino4", "TNA", "TSA", "PACWRAMPOOL",
    "AtlNino", "AtlMDR", "IOD", "TNI", "ONI", "RONI",
)
NMME_SST_TROPICAL_BAND = (-40.0, 40.0)


def nmme_index_variable(index: str) -> str:
    """NetCDF variable name of an index, e.g. 'Nino3.4' -> 'Nino34'."""
    return index.replace(".", "")


def build_nmme_model_sst_indices(
    sst: xr.DataArray,
    valid_time: xr.DataArray,
    *,
    model: str,
    climatology_years: Sequence[int],
    indices: Sequence[str] = NMME_SST_INDICES,
) -> xr.Dataset:
    """Regional SST indices and ELI for one NMME model, drift removed per model.

    ``sst`` is the model's total SST (Y, L, M, lat, lon) with ``MODEL:MEMBER``
    labels. The lead-dependent drift is removed with this model's own
    climatology (forecasts verifying in ``climatology_years``); regional indices
    then follow ``build_regional_sst_index_product`` and ELI follows the shared
    ``compute_eli_latlon_sst`` on total SST before its own drift removal.
    """
    from esp_lab.diagnostics.sst_index import compute_eli_latlon_sst

    year0, year1 = (int(value) for value in climatology_years)
    labels = sst.M.values.astype(str)
    if any(not label.startswith(model + ":") for label in labels):
        raise ValueError(f"Expected only {model}:MEMBER labels.")
    anomaly, _ = stats.remove_drift(sst, valid_time, year0, year1)
    method = f"per-model lead-dependent drift removed over {year0}-{year1}"
    field_cache = xr.Dataset(
        {"SST_anom": anomaly.assign_attrs(units="degC", nmme_anomaly_method=method), "valid_time": valid_time},
        attrs={"source": "NMME", **{key: "not_applicable" for key in REFERENCE_ATTRIBUTES}},
    )
    variables = {}
    for index in indices:
        product = build_regional_sst_index_product(field_cache, index=index)
        variables[nmme_index_variable(index)] = product["mode_index"].astype("float32").assign_attrs(
            index=index, index_definition=product.attrs["index_definition"],
        )
    eli = compute_eli_latlon_sst(sst, lat_name="lat", lon_name="lon", output_name="ELI")
    eli_anomaly, _ = stats.remove_drift(eli, valid_time, year0, year1)
    variables["ELI"] = eli_anomaly.astype("float32").transpose("Y", "L", "M").assign_attrs(
        eli.attrs, index="ELI", long_name="Equatorial Longitude Index anomaly (drift removed)",
    )
    variables["ELI_total"] = eli.astype("float32").transpose("Y", "L", "M")
    # Seasonal ELI from centred three-month-mean SST, as the E3SM and observed
    # seasonal ELI are made (ELI is nonlinear, so it is not a mean of monthly ELI).
    seasonal_sst = sst.rolling(L=3, center=True, min_periods=3).mean()
    eli_seas = compute_eli_latlon_sst(seasonal_sst, lat_name="lat", lon_name="lon", output_name="ELI_seas")
    eli_seas_anomaly, _ = stats.remove_drift(eli_seas, valid_time, year0, year1)
    variables["ELI_seas"] = eli_seas_anomaly.astype("float32").transpose("Y", "L", "M").assign_attrs(
        eli_seas.attrs, index="ELI",
        long_name="ELI of centred three-month-mean SST, anomaly (drift removed); L is the centre month",
    )
    result = xr.Dataset(variables)
    result["valid_time"] = valid_time
    result.attrs.update(
        source="NMME", model=model, nmme_anomaly_method=method,
        climatology=f"{year0}-{year1}", indices=",".join(indices) + ",ELI",
        processing="workflows.modes_of_variability.cross_source.build_nmme_model_sst_indices",
    )
    return result


def load_nmme_sst_indices(paths: Sequence[str | Path], *, index: str) -> tuple[xr.DataArray, xr.DataArray]:
    """Concatenate one index over per-model files; returns (index, valid_time)."""
    name = nmme_index_variable(index)
    indices, times = [], []
    for path in paths:
        with xr.open_dataset(path) as dataset:
            indices.append(dataset[name].load())
            times.append(dataset["valid_time"].load())
    if not indices:
        raise FileNotFoundError(f"No NMME SST-index files for {index}.")
    combined = xr.concat(indices, dim="M", join="outer", combine_attrs="drop_conflicts")
    time = times[0]
    for other in times[1:]:
        time = time.combine_first(other)
    return combined, time.sel(Y=combined.Y)
