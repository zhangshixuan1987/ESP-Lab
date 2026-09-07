"""ENSO event-frequency diagnostics for initialized ensemble forecasts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr


EVENT_CATEGORIES = ("el_nino", "la_nina", "enso")


def _persistent_mask_1d(condition: np.ndarray, min_run: int) -> np.ndarray:
    """Mark every element belonging to a True run of at least ``min_run``."""
    values = np.asarray(condition, dtype=bool)
    result = np.zeros(values.shape, dtype=bool)
    start = None
    for index, value in enumerate(np.r_[values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_run:
                result[start:index] = True
            start = None
    return result


def persistent_event_mask(
    condition: xr.DataArray,
    *,
    min_run: int = 5,
    lead_dim: str = "L",
) -> xr.DataArray:
    """Return a mask for threshold runs that satisfy an ONI-style persistence rule."""
    if min_run < 1:
        raise ValueError("min_run must be at least one")
    if lead_dim not in condition.dims:
        raise ValueError(f"condition lacks lead dimension {lead_dim!r}")
    return xr.apply_ufunc(
        _persistent_mask_1d,
        condition.fillna(False),
        input_core_dims=[[lead_dim]],
        output_core_dims=[[lead_dim]],
        kwargs={"min_run": int(min_run)},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).transpose(*condition.dims)


def compute_enso_frequency_product(
    model_monthly: xr.DataArray,
    observation_monthly_anomaly: xr.DataArray,
    *,
    initialization_years: np.ndarray,
    climatology_years: tuple[int, int] = (1981, 2010),
    threshold: float = 0.5,
    min_persistence: int = 5,
    start_dim: str = "Y",
    lead_dim: str = "L",
    member_dim: str = "M",
) -> xr.Dataset:
    """Build deterministic and member-probability ENSO classifications.

    The model climatology is calculated independently at every lead from the
    requested initialization-year climatology cohort. Both model and observed
    anomalies are converted to centered three-month means before applying the
    +/- threshold and persistence rule.
    """
    required = {start_dim, lead_dim, member_dim}
    if not required <= set(model_monthly.dims):
        raise ValueError(f"model_monthly must contain {sorted(required)}")
    if not {start_dim, lead_dim} <= set(observation_monthly_anomaly.dims):
        raise ValueError("observation_monthly_anomaly must contain Y and L")
    years = np.asarray(initialization_years, dtype=int)
    if years.size != model_monthly.sizes[start_dim]:
        raise ValueError("initialization_years does not match the model Y dimension")
    if observation_monthly_anomaly.sizes[start_dim] != years.size:
        raise ValueError("observation_monthly_anomaly does not match the model Y dimension")
    # Observation matrices are commonly indexed by integer initialization year,
    # while model Y coordinates contain case labels. They represent the same
    # ordered starts, so give observations the model coordinate before aligning.
    observation_monthly_anomaly = observation_monthly_anomaly.assign_coords(
        {start_dim: model_monthly[start_dim]}
    )

    clim_start, clim_end = map(int, climatology_years)
    climatology_positions = np.flatnonzero((years >= clim_start) & (years <= clim_end))
    if climatology_positions.size == 0:
        raise ValueError("No model initialization years fall inside climatology_years")

    model_climatology = model_monthly.isel(
        {start_dim: climatology_positions}
    ).mean((start_dim, member_dim), skipna=True)
    model_anomaly = model_monthly - model_climatology
    model_seasonal = model_anomaly.rolling(
        {lead_dim: 3}, center=True, min_periods=3
    ).mean().isel({lead_dim: slice(1, -1)})
    obs_seasonal = observation_monthly_anomaly.rolling(
        {lead_dim: 3}, center=True, min_periods=3
    ).mean().isel({lead_dim: slice(1, -1)})
    model_seasonal, obs_seasonal = xr.align(model_seasonal, obs_seasonal, join="inner")

    forecast_index = model_seasonal.mean(member_dim, skipna=True)
    member_el_nino = persistent_event_mask(
        model_seasonal >= threshold, min_run=min_persistence, lead_dim=lead_dim
    )
    member_la_nina = persistent_event_mask(
        model_seasonal <= -threshold, min_run=min_persistence, lead_dim=lead_dim
    )
    forecast_el_nino = persistent_event_mask(
        forecast_index >= threshold, min_run=min_persistence, lead_dim=lead_dim
    )
    forecast_la_nina = persistent_event_mask(
        forecast_index <= -threshold, min_run=min_persistence, lead_dim=lead_dim
    )
    obs_el_nino = persistent_event_mask(
        obs_seasonal >= threshold, min_run=min_persistence, lead_dim=lead_dim
    )
    obs_la_nina = persistent_event_mask(
        obs_seasonal <= -threshold, min_run=min_persistence, lead_dim=lead_dim
    )

    valid = forecast_index.notnull() & obs_seasonal.notnull()
    product = xr.Dataset(
        {
            "forecast_index": forecast_index,
            "observation_index": obs_seasonal,
            "valid": valid,
            "forecast_el_nino": forecast_el_nino,
            "forecast_la_nina": forecast_la_nina,
            "forecast_enso": forecast_el_nino | forecast_la_nina,
            "observation_el_nino": obs_el_nino,
            "observation_la_nina": obs_la_nina,
            "observation_enso": obs_el_nino | obs_la_nina,
            "probability_el_nino": member_el_nino.mean(member_dim),
            "probability_la_nina": member_la_nina.mean(member_dim),
            "probability_enso": (member_el_nino | member_la_nina).mean(member_dim),
        }
    )
    product = product.assign_coords(
        initialization_year=(start_dim, years),
    )
    product.attrs.update(
        threshold_degC=float(threshold),
        persistence_seasons=int(min_persistence),
        climatology=f"{clim_start}-{clim_end}",
        seasonal_definition="centered three-month mean",
        frequency_note="event-season rates retain each initialization/lead sample",
    )
    return product


def _bootstrap_interval(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(n_bootstrap), values.size))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def startwise_frequency(
    product: xr.Dataset,
    category: str,
    *,
    basis: str = "event_season",
    start_dim: str = "Y",
    lead_dim: str = "L",
) -> xr.DataArray:
    """Return one deterministic forecast-frequency value per initialization year."""
    if category not in EVENT_CATEGORIES:
        raise ValueError(f"Unknown category {category!r}")
    flag = product[f"forecast_{category}"].where(product.valid)
    if basis == "event_season":
        result = flag.mean(lead_dim, skipna=True)
    elif basis == "start_with_event":
        result = flag.fillna(False).any(lead_dim).astype(float).where(product.valid.any(lead_dim))
    else:
        raise ValueError("basis must be 'event_season' or 'start_with_event'")
    # Pair cross-experiment contrasts by the physical initialization year, not
    # by experiment-specific case labels stored in the model Y coordinate.
    if "initialization_year" in product.coords:
        result = result.assign_coords({start_dim: product.initialization_year.values})
    return result


def summarize_enso_frequency(
    product: xr.Dataset,
    *,
    experiment: str,
    init_month: int,
    n_bootstrap: int = 2000,
    seed: int = 42,
    start_dim: str = "Y",
    lead_dim: str = "L",
) -> pd.DataFrame:
    """Summarize event frequency and categorical verification for one forecast set."""
    rows = []
    valid = product.valid
    for offset, category in enumerate(EVENT_CATEGORIES):
        forecast = product[f"forecast_{category}"].where(valid)
        observed = product[f"observation_{category}"].where(valid)
        probability = product[f"probability_{category}"].where(valid)
        forecast_by_start = forecast.mean(lead_dim, skipna=True)
        observed_by_start = observed.mean(lead_dim, skipna=True)
        bias_by_start = forecast_by_start - observed_by_start
        has_valid_lead = valid.any(lead_dim)
        forecast_start = forecast.fillna(False).any(lead_dim).astype(float).where(has_valid_lead)
        observed_start = observed.fillna(False).any(lead_dim).astype(float).where(has_valid_lead)
        start_bias = forecast_start - observed_start
        # ``where`` promotes boolean flags to floating point because NaN is
        # introduced, so cast back before categorical logical operations.
        forecast_bool = forecast.fillna(False).astype(bool)
        observed_bool = observed.fillna(False).astype(bool)
        hits = float((forecast_bool & observed_bool & valid).sum())
        forecast_yes = float((forecast_bool & valid).sum())
        observed_yes = float((observed_bool & valid).sum())
        low, high = _bootstrap_interval(
            np.asarray(forecast_by_start), n_bootstrap=n_bootstrap, seed=seed + offset
        )
        bias_low, bias_high = _bootstrap_interval(
            np.asarray(bias_by_start), n_bootstrap=n_bootstrap, seed=seed + 10 + offset
        )
        start_low, start_high = _bootstrap_interval(
            np.asarray(forecast_start), n_bootstrap=n_bootstrap, seed=seed + 20 + offset
        )
        start_bias_low, start_bias_high = _bootstrap_interval(
            np.asarray(start_bias), n_bootstrap=n_bootstrap, seed=seed + 30 + offset
        )
        rows.append(
            {
                "experiment": experiment,
                "init_month": int(init_month),
                "category": category,
                "event_season_frequency": 100.0 * float(forecast.mean(skipna=True)),
                "event_season_ci_low": 100.0 * low,
                "event_season_ci_high": 100.0 * high,
                "observed_event_season_frequency": 100.0 * float(observed.mean(skipna=True)),
                "event_season_frequency_bias": 100.0 * float(bias_by_start.mean(skipna=True)),
                "event_season_bias_ci_low": 100.0 * bias_low,
                "event_season_bias_ci_high": 100.0 * bias_high,
                "starts_with_event_frequency": 100.0 * float(forecast_start.mean(skipna=True)),
                "starts_with_event_ci_low": 100.0 * start_low,
                "starts_with_event_ci_high": 100.0 * start_high,
                "observed_starts_with_event_frequency": 100.0 * float(observed_start.mean(skipna=True)),
                "starts_with_event_frequency_bias": 100.0 * float(start_bias.mean(skipna=True)),
                "starts_with_event_bias_ci_low": 100.0 * start_bias_low,
                "starts_with_event_bias_ci_high": 100.0 * start_bias_high,
                "probability_of_detection": hits / observed_yes if observed_yes else np.nan,
                "false_alarm_ratio": (forecast_yes - hits) / forecast_yes if forecast_yes else np.nan,
                "brier_score": float(((probability - observed.astype(float)) ** 2).mean(skipna=True)),
                "valid_event_seasons": int(valid.sum()),
            }
        )
    return pd.DataFrame(rows)


def paired_frequency_contrasts(
    products: Mapping[tuple[str, int], xr.Dataset],
    *,
    experiments: tuple[str, str],
    init_months: tuple[int, int],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return method, initialization, and method-by-initialization contrasts."""
    exp_a, exp_b = experiments
    init_a, init_b = map(int, init_months)
    rows = []
    contrast_specs = [
        (f"{exp_a} minus {exp_b}", [(exp_a, init_a, 1), (exp_b, init_a, -1)]),
        (f"{exp_a} minus {exp_b}", [(exp_a, init_b, 1), (exp_b, init_b, -1)]),
        (f"init {init_b:02d} minus init {init_a:02d}", [(exp_a, init_b, 1), (exp_a, init_a, -1)]),
        (f"init {init_b:02d} minus init {init_a:02d}", [(exp_b, init_b, 1), (exp_b, init_a, -1)]),
        (
            "method-by-initialization interaction",
            [(exp_a, init_b, 1), (exp_a, init_a, -1), (exp_b, init_b, -1), (exp_b, init_a, 1)],
        ),
    ]
    for category_index, category in enumerate(EVENT_CATEGORIES):
        for basis_index, basis in enumerate(("event_season", "start_with_event")):
            series = {
                key: startwise_frequency(product, category, basis=basis)
                for key, product in products.items()
            }
            for contrast_index, (name, terms) in enumerate(contrast_specs):
                arrays = xr.align(*(series[(exp, month)] for exp, month, _ in terms), join="inner")
                combined = sum(coefficient * array for array, (_, _, coefficient) in zip(arrays, terms))
                low, high = _bootstrap_interval(
                    np.asarray(combined),
                    n_bootstrap=n_bootstrap,
                    seed=seed + 100 * category_index + 10 * basis_index + contrast_index,
                )
                rows.append(
                    {
                        "category": category,
                        "basis": basis,
                        "contrast": name,
                        "context": ", ".join(f"{exp}:init{month:02d}:{coefficient:+d}" for exp, month, coefficient in terms),
                        "difference_percentage_points": 100.0 * float(combined.mean(skipna=True)),
                        "ci_low": 100.0 * low,
                        "ci_high": 100.0 * high,
                    }
                )
    return pd.DataFrame(rows)


__all__ = [
    "EVENT_CATEGORIES",
    "compute_enso_frequency_product",
    "paired_frequency_contrasts",
    "persistent_event_mask",
    "startwise_frequency",
    "summarize_enso_frequency",
]
