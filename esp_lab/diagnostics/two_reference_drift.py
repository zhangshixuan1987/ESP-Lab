"""Start-specific, two-reference lead-time drift diagnostics.

This module contains the reusable xarray calculations for the monthly S2D
workflow.  It intentionally performs no file discovery or regridding: callers
must put hindcasts, observations, and the model-attractor climatology on the
same grid and in the same physical units before calling :func:`compute_diagnostics`.

The core contract is ``Y x L x spatial dimensions``.  ``Y`` is never averaged
inside the primary calculation, which leaves start-specific products available
for later drift--skill analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr
from scipy.stats import rankdata, spearmanr


REGIME_DEFINITIONS = {
    0: "neutral/no appreciable change",
    1: "adjustment toward observations and model climatology",
    2: "adjustment toward observations and away from model climatology",
    3: "canonical relaxation toward biased model climatology",
    4: "non-attractor error growth",
}


def compute_ensemble_mean(values: xr.DataArray, member_dim: str = "M") -> xr.DataArray:
    """Average ensemble members while retaining initialization year and lead."""
    required = {"Y", "L"}
    if not required <= set(values.dims):
        raise ValueError(f"Hindcast must contain dimensions {sorted(required)}; got {values.dims}")
    result = values.mean(member_dim, skipna=True) if member_dim in values.dims else values.copy()
    if "Y" not in result.dims:
        raise AssertionError("Initialization-year dimension Y was unexpectedly removed")
    result.name = "hindcast_mean"
    result.attrs.update(values.attrs)
    result.attrs["ensemble_processing"] = (
        f"arithmetic mean over {member_dim}, skipna=True"
        if member_dim in values.dims
        else f"no {member_dim} dimension present; values retained"
    )
    return result


def _year_month(value: object) -> tuple[int, int]:
    """Return calendar year and month for NumPy, pandas, or cftime values."""
    if hasattr(value, "year") and hasattr(value, "month"):
        return int(value.year), int(value.month)
    text = np.datetime_as_string(np.asarray(value, dtype="datetime64[M]"), unit="M")
    year, month = text.split("-")[:2]
    return int(year), int(month)


def _validate_valid_time(valid_time: xr.DataArray) -> None:
    if tuple(valid_time.dims) != ("Y", "L"):
        raise ValueError(f"valid_time must have dimensions ('Y', 'L'); got {valid_time.dims}")
    if valid_time.size == 0:
        raise ValueError("valid_time is empty")


def match_observation_to_valid_time(
    observation: xr.DataArray,
    valid_time: xr.DataArray,
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Match one monthly observation slice to every exact hindcast valid month.

    Duplicate or missing observation months are rejected rather than silently
    choosing a record or returning a partial reference.
    """
    _validate_valid_time(valid_time)
    if time_dim not in observation.dims:
        raise ValueError(f"Observation must contain time dimension {time_dim!r}")

    lookup: dict[tuple[int, int], int] = {}
    for index, value in enumerate(observation[time_dim].values):
        key = _year_month(value)
        if key in lookup:
            raise ValueError(
                f"Observation contains duplicate monthly record {key[0]:04d}-{key[1]:02d}"
            )
        lookup[key] = index

    flat_times = valid_time.values.reshape(-1)
    missing = sorted(
        {_year_month(value) for value in flat_times if _year_month(value) not in lookup}
    )
    if missing:
        labels = ", ".join(f"{year:04d}-{month:02d}" for year, month in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"Observation is missing {len(missing)} valid month(s): {labels}{suffix}")

    indices = xr.DataArray(
        np.asarray([lookup[_year_month(value)] for value in flat_times]).reshape(valid_time.shape),
        dims=("Y", "L"),
        coords={"Y": valid_time["Y"], "L": valid_time["L"]},
    )
    matched = observation.isel({time_dim: indices}).drop_vars(time_dim, errors="ignore")
    matched = matched.assign_coords(valid_time=valid_time)
    matched.name = "obs_ref"
    matched.attrs.update(observation.attrs)
    matched.attrs["reference_matching"] = "exact valid calendar year and month"
    return matched


def build_reference_lookup(
    reference: xr.DataArray,
    valid_time: xr.DataArray,
    *,
    matching: str = "exact",
    time_dim: str = "time",
    calendar_dim: str | None = None,
) -> xr.DataArray:
    """Map a reference to the exact hindcast ``Y,L`` valid-time structure.

    ``matching="exact"`` selects observations by calendar year and month.
    ``matching="month"`` expands a monthly climatology, and
    ``matching="dayofyear"`` expands a daily climatology.  The latter keeps
    the API ready for daily diagnostics without changing the monthly path.
    """
    if matching == "exact":
        return match_observation_to_valid_time(reference, valid_time, time_dim=time_dim)
    if matching == "month":
        return match_model_climatology_to_valid_time(
            reference, valid_time, month_dim=calendar_dim or "month"
        )
    if matching != "dayofyear":
        raise ValueError("matching must be 'exact', 'month', or 'dayofyear'")

    _validate_valid_time(valid_time)
    day_dim = calendar_dim or "dayofyear"
    if day_dim not in reference.dims:
        raise ValueError(f"Daily climatology must contain dimension {day_dim!r}")
    try:
        day_values = valid_time.dt.dayofyear
    except AttributeError as exc:
        raise ValueError("valid_time must contain datetime-like values") from exc
    available = set(np.asarray(reference[day_dim].values, dtype=int).tolist())
    required = set(np.asarray(day_values.values, dtype=int).reshape(-1).tolist())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Daily climatology is missing day-of-year values: {missing[:12]}")
    matched = reference.sel({day_dim: day_values}).drop_vars(day_dim, errors="ignore")
    matched = matched.assign_coords(valid_time=valid_time)
    matched.attrs.update(reference.attrs)
    matched.attrs["reference_matching"] = "valid calendar day-of-year"
    return matched


def build_obs_lookup(
    observation: xr.DataArray, valid_time: xr.DataArray, *, time_dim: str = "time"
) -> xr.DataArray:
    """Observation-specific wrapper around :func:`build_reference_lookup`."""
    result = build_reference_lookup(
        observation, valid_time, matching="exact", time_dim=time_dim
    )
    return result.rename("obs_ref")


def build_attractor_lookup(
    climatology: xr.DataArray,
    valid_time: xr.DataArray,
    *,
    frequency: str = "monthly",
) -> xr.DataArray:
    """Attractor-specific wrapper supporting monthly and daily climatologies."""
    matching = {"monthly": "month", "daily": "dayofyear"}.get(frequency)
    if matching is None:
        raise ValueError("frequency must be 'monthly' or 'daily'")
    result = build_reference_lookup(climatology, valid_time, matching=matching)
    return result.rename("att_ref")


def match_model_climatology_to_valid_time(
    climatology: xr.DataArray,
    valid_time: xr.DataArray,
    *,
    month_dim: str = "month",
) -> xr.DataArray:
    """Expand a 12-month free-running E3SM climatology onto ``Y x L``."""
    _validate_valid_time(valid_time)
    if month_dim not in climatology.dims:
        raise ValueError(f"Model climatology must contain dimension {month_dim!r}")
    months = np.asarray(climatology[month_dim].values, dtype=int)
    if set(months.tolist()) != set(range(1, 13)) or months.size != 12:
        raise ValueError(f"Model climatology requires exactly months 1..12; got {months.tolist()}")
    valid_month = xr.DataArray(
        np.asarray([_year_month(value)[1] for value in valid_time.values.reshape(-1)]).reshape(
            valid_time.shape
        ),
        dims=("Y", "L"),
        coords={"Y": valid_time["Y"], "L": valid_time["L"]},
    )
    matched = climatology.sel({month_dim: valid_month}).drop_vars(month_dim, errors="ignore")
    matched = matched.assign_coords(valid_time=valid_time)
    matched.name = "att_ref"
    matched.attrs.update(climatology.attrs)
    matched.attrs["reference_matching"] = "valid calendar month"
    return matched


def _normalized_units(data: xr.DataArray) -> str:
    return str(data.attrs.get("units", "")).strip().lower().replace("°", "deg")


def validate_compatible_fields(*fields: xr.DataArray) -> None:
    """Require exact coordinates, dimensions, grid, and declared equal units."""
    if len(fields) < 2:
        return
    reference = fields[0]
    for candidate in fields[1:]:
        xr.align(reference, candidate, join="exact", copy=False)
        if set(reference.dims) != set(candidate.dims):
            raise ValueError(f"Field dimensions differ: {reference.dims} versus {candidate.dims}")
    units = [_normalized_units(field) for field in fields]
    if any(not unit for unit in units):
        raise ValueError("All fields must declare units before reference subtraction")
    if len(set(units)) != 1:
        raise ValueError(f"Fields have inconsistent units: {units}")


def compute_reference_departure(
    hindcast_mean: xr.DataArray, reference: xr.DataArray
) -> xr.DataArray:
    """Return signed grid-cell departure ``hindcast_mean - reference``."""
    validate_compatible_fields(hindcast_mean, reference)
    departure = hindcast_mean - reference
    departure.attrs.update(hindcast_mean.attrs)
    departure.attrs["diagnostic"] = "signed hindcast departure from reference"
    return departure


def compute_initialization_adjustment(
    departure: xr.DataArray,
    *,
    baseline_lead: int = 1,
    lead_dim: str = "L",
) -> xr.DataArray:
    """Return departure change relative to the initialization-month mean."""
    if baseline_lead not in departure[lead_dim]:
        raise ValueError(f"Baseline lead {baseline_lead} is absent")
    result = departure - departure.sel({lead_dim: baseline_lead}, drop=True)
    result.attrs.update(departure.attrs)
    result.attrs.update(
        baseline_lead=baseline_lead,
        baseline_definition="initialization-month mean (not instantaneous day 0)",
    )
    return result


def compute_distance_change(
    departure: xr.DataArray,
    *,
    baseline_lead: int = 1,
    lead_dim: str = "L",
) -> xr.DataArray:
    """Return grid-cell change in absolute distance to a reference."""
    if baseline_lead not in departure[lead_dim]:
        raise ValueError(f"Baseline lead {baseline_lead} is absent")
    result = abs(departure) - abs(departure.sel({lead_dim: baseline_lead}, drop=True))
    result.attrs.update(departure.attrs)
    result.attrs.update(
        diagnostic="change in absolute grid-cell distance from reference",
        baseline_lead=baseline_lead,
        baseline_definition="initialization-month mean (not instantaneous day 0)",
    )
    return result


def classify_drift_regime(
    delta_abs_e_obs: xr.DataArray,
    delta_abs_e_att: xr.DataArray,
    *,
    tolerance: float = 0.0,
) -> xr.DataArray:
    """Classify grid cells into neutral (0) or the four drift regimes (1--4)."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    obs, att = xr.align(delta_abs_e_obs, delta_abs_e_att, join="exact", copy=False)
    valid = obs.notnull() & att.notnull()
    obs_toward, obs_away = obs < -tolerance, obs > tolerance
    att_toward, att_away = att < -tolerance, att > tolerance
    regime = xr.zeros_like(obs, dtype=np.int8)
    regime = xr.where(obs_toward & att_toward, 1, regime)
    regime = xr.where(obs_toward & att_away, 2, regime)
    regime = xr.where(obs_away & att_toward, 3, regime)
    regime = xr.where(obs_away & att_away, 4, regime)
    regime = regime.where(valid)
    regime.name = "regime"
    regime.attrs.update(
        long_name="two-reference drift regime",
        regime_definitions="; ".join(f"{key}={value}" for key, value in REGIME_DEFINITIONS.items()),
        distance_tolerance=float(tolerance),
        missing_value_policy="NaN where either distance change is missing",
    )
    return regime


def compute_diagnostics(
    values: xr.DataArray,
    obs_ref: xr.DataArray,
    att_ref: xr.DataArray,
    *,
    baseline_lead: int = 1,
    distance_tolerance: float = 0.0,
) -> xr.Dataset:
    """Compute the complete start-specific two-reference diagnostic product."""
    hindcast_mean = compute_ensemble_mean(values)
    validate_compatible_fields(hindcast_mean, obs_ref, att_ref)
    e_obs = compute_reference_departure(hindcast_mean, obs_ref).rename("e_obs")
    e_att = compute_reference_departure(hindcast_mean, att_ref).rename("e_att")
    delta_e_obs = compute_initialization_adjustment(e_obs, baseline_lead=baseline_lead).rename(
        "delta_e_obs"
    )
    delta_e_att = compute_initialization_adjustment(e_att, baseline_lead=baseline_lead).rename(
        "delta_e_att"
    )
    delta_abs_e_obs = compute_distance_change(e_obs, baseline_lead=baseline_lead).rename(
        "delta_abs_e_obs"
    )
    delta_abs_e_att = compute_distance_change(e_att, baseline_lead=baseline_lead).rename(
        "delta_abs_e_att"
    )
    regime = classify_drift_regime(
        delta_abs_e_obs, delta_abs_e_att, tolerance=distance_tolerance
    )
    product = xr.Dataset(
        {
            "hindcast_mean": hindcast_mean,
            "e_obs": e_obs,
            "e_att": e_att,
            "delta_e_obs": delta_e_obs,
            "delta_e_att": delta_e_att,
            "delta_abs_e_obs": delta_abs_e_obs,
            "delta_abs_e_att": delta_abs_e_att,
            "regime": regime,
        }
    )
    if "Y" not in product.dims:
        raise AssertionError("Diagnostic product must retain initialization-year dimension Y")
    product.attrs.update(
        baseline_lead=baseline_lead,
        baseline_definition="L=1 initialization-month mean; not instantaneous t=0",
        distance_tolerance=float(distance_tolerance),
    )
    validate_diagnostics(product)
    return product


def compute_prediction_skill(
    values: xr.DataArray,
    obs_ref: xr.DataArray,
    *,
    member_dim: str = "M",
    start_dim: str = "Y",
) -> xr.Dataset:
    """Compute RMSE, ACC, ensemble spread, and spread/RMSE across starts."""
    hindcast_mean = compute_ensemble_mean(values, member_dim=member_dim)
    validate_compatible_fields(hindcast_mean, obs_ref)
    error = hindcast_mean - obs_ref
    rmse = np.sqrt((error**2).mean(start_dim, skipna=True)).rename("rmse")
    acc = xr.corr(hindcast_mean, obs_ref, dim=start_dim).rename("acc")
    if member_dim in values.dims:
        spread = values.std(member_dim, skipna=True).mean(start_dim, skipna=True)
    else:
        spread = xr.full_like(rmse, np.nan)
    spread = spread.rename("ensemble_spread")
    ratio = (spread / rmse.where(rmse != 0)).rename("spread_rmse_ratio")
    rmse.attrs.update(hindcast_mean.attrs)
    rmse.attrs["diagnostic"] = "root-mean-square error across initialization years"
    spread.attrs.update(hindcast_mean.attrs)
    spread.attrs["diagnostic"] = "mean ensemble standard deviation across initialization years"
    for dimensionless, diagnostic in (
        (acc, "anomaly correlation across initialization years"),
        (ratio, "ensemble spread divided by RMSE"),
    ):
        dimensionless.attrs = {
            "units": "1",
            "diagnostic": diagnostic,
        }
    return xr.Dataset(
        {"rmse": rmse, "acc": acc, "ensemble_spread": spread, "spread_rmse_ratio": ratio}
    )


def run_pipeline(
    hindcasts: Mapping[str, xr.DataArray],
    obs_references: Mapping[str, xr.DataArray],
    attractor_references: Mapping[str, xr.DataArray],
    *,
    attractor_spreads: Mapping[str, xr.DataArray] | None = None,
    baseline_lead: int = 1,
    distance_tolerance: float = 0.0,
) -> dict[str, object]:
    """Run the explicit two-reference calculation for one initialization.

    Inputs are keyed by experiment and references are already matched to each
    experiment's exact ``Y,L`` valid times by the reference-preparation stage.
    """
    labels = list(hindcasts)
    if set(labels) != set(obs_references) or set(labels) != set(attractor_references):
        raise KeyError("Hindcasts, observation references, and attractor references must share keys")
    common_years = sorted(
        set.intersection(*(set(np.asarray(hindcasts[label].Y.values).tolist()) for label in labels))
    )
    common_leads = sorted(
        set.intersection(*(set(np.asarray(hindcasts[label].L.values).tolist()) for label in labels))
    )
    if not common_years or not common_leads:
        raise ValueError("Experiments have no common Y/L sample")
    if baseline_lead not in common_leads:
        raise ValueError(f"Baseline lead {baseline_lead} is not in the common lead sample")
    aligned_hindcasts = {
        label: hindcasts[label].sel(Y=common_years, L=common_leads) for label in labels
    }
    aligned_obs = {
        label: obs_references[label].sel(Y=common_years, L=common_leads) for label in labels
    }
    aligned_att = {
        label: attractor_references[label].sel(Y=common_years, L=common_leads)
        for label in labels
    }
    products = {
        label: compute_diagnostics(
            aligned_hindcasts[label], aligned_obs[label], aligned_att[label],
            baseline_lead=baseline_lead, distance_tolerance=distance_tolerance,
        )
        for label in labels
    }
    result: dict[str, object] = {
        name: {label: products[label][name] for label in labels}
        for name in (
            "hindcast_mean", "e_obs", "e_att", "delta_e_obs", "delta_e_att",
            "delta_abs_e_obs", "delta_abs_e_att", "regime",
        )
    }
    if attractor_spreads is not None:
        if set(labels) != set(attractor_spreads):
            raise KeyError("Attractor spreads must share the hindcast experiment keys")
        result["z_att"] = {}
        for label in labels:
            spread = attractor_spreads[label].sel(Y=common_years, L=common_leads)
            validate_compatible_fields(products[label]["e_att"], spread)
            z_att = (products[label]["e_att"] / spread.where(spread > 0)).rename("z_att")
            z_att.attrs = {
                "units": "1",
                "diagnostic": "departure from model attractor normalized by historical ensemble spread",
            }
            result["z_att"][label] = z_att
    result["skill"] = {
        label: compute_prediction_skill(aligned_hindcasts[label], aligned_obs[label])
        for label in labels
    }
    result["paired"] = {}
    if {"JRA55_FOSIRL", "Reanalysis"} <= set(labels):
        paired = compare_initializations(
            products["JRA55_FOSIRL"], products["Reanalysis"]
        )
        result["paired"] = paired.rename(
            {name: f"delta_{name}" for name in paired.data_vars}
        )
    return result


def compute_lead_window_mean(
    data: xr.DataArray,
    leads: Sequence[int],
    *,
    lead_dim: str = "L",
) -> xr.DataArray:
    """Average a continuous field over an explicitly enumerated lead window."""
    requested = [int(lead) for lead in leads]
    available = set(np.asarray(data[lead_dim].values, dtype=int).tolist())
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"Lead window contains unavailable leads: {missing}")
    result = data.sel({lead_dim: requested}).mean(lead_dim, skipna=True)
    result.attrs.update(data.attrs)
    result.attrs["lead_window"] = ",".join(map(str, requested))
    return result


def compute_spatial_drift_summary(
    product: xr.Dataset,
    lead_windows: Mapping[str, Sequence[int]],
    *,
    start_dim: str = "Y",
) -> xr.Dataset:
    """Create compact climatological maps after start-specific drift is computed."""
    continuous = (
        "e_obs", "e_att", "delta_e_obs", "delta_e_att",
        "delta_abs_e_obs", "delta_abs_e_att",
    )
    fields: dict[str, xr.DataArray] = {}
    for window, leads in lead_windows.items():
        for name in continuous:
            if name in product:
                fields[f"{name}_{window}"] = compute_lead_window_mean(
                    product[name], leads
                ).mean(start_dim, skipna=True)
        if "z_att" in product:
            fields[f"z_att_{window}"] = compute_lead_window_mean(
                product.z_att, leads
            ).mean(start_dim, skipna=True)
        if "regime" in product:
            fractions = compute_regime_fraction(
                product.regime, leads=leads, spatial_dims=(), sample_dims=(start_dim,)
            )
            for name, field in fractions.data_vars.items():
                fields[f"{name}_{window}"] = field
    result = xr.Dataset(fields)
    result.attrs.update(
        summary="climatological spatial maps computed after Y-specific diagnostics",
        categorical_handling="regime frequencies; integer regime codes were not averaged",
    )
    return result


def infer_area_weights(data: xr.DataArray, area: xr.DataArray | None = None) -> xr.DataArray:
    """Use supplied cell area, or cosine-latitude weights on a rectilinear grid."""
    if area is not None:
        weights = area.astype(float)
        weights.attrs.setdefault("weight_source", "provided grid-cell area")
        return weights
    if "lat" not in data.coords:
        raise ValueError("Supply area weights when data have no latitude coordinate")
    if data["lat"].ndim != 1:
        raise ValueError("Cosine-latitude fallback requires a one-dimensional lat coordinate")
    weights = np.cos(np.deg2rad(data["lat"])).clip(min=0.0)
    weights.name = "area_weight"
    weights.attrs["weight_source"] = "cosine latitude"
    return weights


def _weighted_mean(
    data: xr.DataArray, weights: xr.DataArray, spatial_dims: Sequence[str]
) -> xr.DataArray:
    dims = [dim for dim in spatial_dims if dim in data.dims]
    if not dims:
        raise ValueError(f"No requested spatial dimensions {tuple(spatial_dims)} occur in data")
    valid_weights = weights.broadcast_like(data).where(data.notnull())
    denominator = valid_weights.sum(dims, skipna=True)
    return (data * valid_weights).sum(dims, skipna=True) / denominator.where(denominator > 0)


def area_weighted_mean(
    data: xr.DataArray,
    area: xr.DataArray | None = None,
    *,
    spatial_dims: Sequence[str] = ("lat", "lon"),
) -> xr.DataArray:
    """Finite-cell-aware area-weighted spatial mean."""
    result = _weighted_mean(data, infer_area_weights(data, area), spatial_dims)
    result.attrs.update(data.attrs)
    result.attrs["spatial_reduction"] = "finite-cell-aware area-weighted mean"
    return result


def area_weighted_rmse(
    departure: xr.DataArray,
    area: xr.DataArray | None = None,
    *,
    spatial_dims: Sequence[str] = ("lat", "lon"),
) -> xr.DataArray:
    """Area-weighted RMS norm of an already-computed reference departure."""
    result = np.sqrt(
        _weighted_mean(departure**2, infer_area_weights(departure, area), spatial_dims)
    )
    result.attrs.update(departure.attrs)
    result.attrs["diagnostic"] = "area-weighted RMS reference departure"
    return result


def regional_subset(
    data: xr.DataArray | xr.Dataset,
    *,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
) -> xr.DataArray | xr.Dataset:
    """Subset a rectilinear grid, including longitude ranges crossing the dateline."""
    if "lat" not in data.coords or "lon" not in data.coords:
        raise ValueError("regional_subset requires lat and lon coordinates")
    lat0, lat1 = sorted(map(float, lat_bounds))
    lat_mask = (data.lat >= lat0) & (data.lat <= lat1)
    lon = data.lon % 360
    lon0, lon1 = (float(value) % 360 for value in lon_bounds)
    lon_mask = (lon >= lon0) & (lon <= lon1) if lon0 <= lon1 else (lon >= lon0) | (lon <= lon1)
    result = data.where(lat_mask & lon_mask, drop=True)
    if result.sizes.get("lat", 0) == 0 or result.sizes.get("lon", 0) == 0:
        raise ValueError("Requested regional bounds select no grid cells")
    return result


def compute_regime_fraction(
    regime: xr.DataArray,
    area: xr.DataArray | None = None,
    *,
    spatial_dims: Sequence[str] = ("lat", "lon"),
    sample_dims: Sequence[str] = (),
    lead_dim: str = "L",
    leads: Sequence[int] | None = None,
) -> xr.Dataset:
    """Return area/time fraction in regimes 0--4; never average category codes."""
    selected = regime
    reduction_dims = list(
        dict.fromkeys(dim for dim in (*spatial_dims, *sample_dims) if dim in selected.dims)
    )
    if leads is not None:
        requested = [int(lead) for lead in leads]
        selected = selected.sel({lead_dim: requested})
        if lead_dim not in reduction_dims:
            reduction_dims.append(lead_dim)
    if any(dim in selected.dims for dim in spatial_dims):
        base_weights = infer_area_weights(selected, area)
    else:
        base_weights = xr.ones_like(selected, dtype=float)
    if not reduction_dims:
        raise ValueError("No dimensions selected for regime-frequency reduction")
    weights = base_weights.broadcast_like(selected).where(selected.notnull())
    denominator = weights.sum(reduction_dims, skipna=True)
    return xr.Dataset(
        {
            f"fraction_regime_{code}": weights.where(selected == code, 0.0).sum(
                reduction_dims, skipna=True
            )
            / denominator.where(denominator > 0)
            for code in range(0, 5)
        }
    )


def compute_early_drift_late_error(
    delta_abs_e_obs: xr.DataArray,
    forecast_error: xr.DataArray,
    *,
    early_leads: Sequence[int] = (1, 2, 3),
    late_leads: Sequence[int] = (7, 8, 9),
    lead_dim: str = "L",
    start_dim: str = "Y",
) -> xr.Dataset:
    """Return start-specific early drift, later error, and correlation over ``Y``.

    Any remaining dimensions (for example latitude and longitude) are retained,
    so this works for regional series and spatial fields alike.
    """
    drift, error = xr.align(delta_abs_e_obs, forecast_error, join="inner", copy=False)
    if start_dim not in drift.dims or lead_dim not in drift.dims:
        raise ValueError(f"Inputs must retain {start_dim!r} and {lead_dim!r}")
    early = compute_lead_window_mean(drift, early_leads, lead_dim=lead_dim).rename("early_drift")
    late_mse = compute_lead_window_mean(error**2, late_leads, lead_dim=lead_dim)
    late = np.sqrt(late_mse).rename("late_error")
    late.attrs.update(error.attrs)
    late.attrs["diagnostic"] = "start-specific RMSE over later lead window"
    early, late = xr.align(early, late, join="inner", copy=False)
    correlation = xr.corr(early, late, dim=start_dim).rename("drift_skill_correlation")
    correlation.attrs = {
        "units": "1",
        "diagnostic": "Pearson correlation across initialization years",
    }
    return xr.Dataset({"early_drift": early, "late_error": late, "correlation": correlation})


def compute_early_error_growth_late_error(
    forecast_error: xr.DataArray,
    *,
    early_leads: Sequence[int] = (1, 2, 3),
    late_leads: Sequence[int] = (7, 8, 9),
    lead_dim: str = "L",
) -> xr.Dataset:
    """Return baseline error, early error-growth slope, and later RMS error.

    The growth metric is the ordinary least-squares slope of absolute forecast
    error against lead over ``early_leads``.  Unlike a mean change relative to
    the first lead, it does not include a structurally zero baseline term.
    """
    requested = [int(lead) for lead in early_leads]
    if len(requested) < 2 or len(set(requested)) < 2:
        raise ValueError("early_leads must contain at least two distinct leads")
    if lead_dim not in forecast_error.dims:
        raise ValueError(f"Forecast error must contain lead dimension {lead_dim!r}")
    missing = sorted(set(requested) - set(forecast_error[lead_dim].values.tolist()))
    if missing:
        raise ValueError(f"Forecast error is missing early leads {missing}")

    absolute_error = abs(forecast_error.sel({lead_dim: requested}))
    lead = xr.DataArray(
        np.asarray(requested, dtype=float),
        dims=(lead_dim,),
        coords={lead_dim: requested},
    )
    valid = absolute_error.notnull()
    lead_mean = lead.where(valid).mean(lead_dim, skipna=True)
    error_mean = absolute_error.mean(lead_dim, skipna=True)
    numerator = (
        (lead - lead_mean) * (absolute_error - error_mean)
    ).sum(lead_dim, skipna=True)
    denominator = ((lead - lead_mean) ** 2).where(valid).sum(lead_dim, skipna=True)
    count = valid.sum(lead_dim)
    growth = (numerator / denominator.where(denominator > 0)).where(count >= 2)
    growth = growth.rename("early_error_growth")
    growth.attrs.update(forecast_error.attrs)
    growth.attrs.update(
        diagnostic="OLS slope of absolute observation error against lead",
        early_leads=",".join(str(lead_value) for lead_value in requested),
    )

    baseline = absolute_error.sel({lead_dim: requested[0]}, drop=True).rename(
        "baseline_error"
    )
    baseline.attrs.update(forecast_error.attrs)
    baseline.attrs.update(
        diagnostic="absolute observation error at the first early lead",
        baseline_lead=requested[0],
    )
    late_mse = compute_lead_window_mean(forecast_error**2, late_leads, lead_dim=lead_dim)
    late = np.sqrt(late_mse).rename("late_error")
    late.attrs.update(forecast_error.attrs)
    late.attrs["diagnostic"] = "start-specific RMS error over later lead window"
    baseline, growth, late = xr.align(baseline, growth, late, join="inner", copy=False)
    return xr.Dataset(
        {
            "baseline_error": baseline,
            "early_error_growth": growth,
            "late_error": late,
        }
    )


def _circular_block_indices(
    rng: np.random.Generator, n_samples: int, block_length: int
) -> np.ndarray:
    """Draw one circular moving-block bootstrap sample."""
    starts = rng.integers(0, n_samples, size=int(np.ceil(n_samples / block_length)))
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n_samples).reshape(-1)[:n_samples]


def _standardized_conditional_slope(
    predictor: np.ndarray, baseline: np.ndarray, outcome: np.ndarray
) -> float:
    values = np.column_stack((predictor, baseline, outcome)).astype(float, copy=False)
    scales = values.std(axis=0, ddof=1)
    if values.shape[0] < 4 or np.any(~np.isfinite(scales)) or np.any(scales == 0):
        return np.nan
    standardized = (values - values.mean(axis=0)) / scales
    design = np.column_stack((np.ones(values.shape[0]), standardized[:, :2]))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return np.nan
    coefficients = np.linalg.lstsq(design, standardized[:, 2], rcond=None)[0]
    return float(coefficients[1])


def _row_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return Pearson correlations between corresponding matrix rows."""
    first_anomaly = first - first.mean(axis=1, keepdims=True)
    second_anomaly = second - second.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(first_anomaly**2, axis=1) * np.sum(second_anomaly**2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(first_anomaly * second_anomaly, axis=1) / denominator


def _row_standardized_conditional_slope(
    predictor: np.ndarray, baseline: np.ndarray, outcome: np.ndarray
) -> np.ndarray:
    """Return standardized partial-regression slopes for matrix rows."""
    predictor_outcome = _row_correlation(predictor, outcome)
    predictor_baseline = _row_correlation(predictor, baseline)
    baseline_outcome = _row_correlation(baseline, outcome)
    denominator = 1.0 - predictor_baseline**2
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (
            predictor_outcome - predictor_baseline * baseline_outcome
        ) / denominator
    slopes[np.isclose(denominator, 0.0)] = np.nan
    return slopes


def _leave_one_out_r2(
    predictor: np.ndarray, baseline: np.ndarray, outcome: np.ndarray
) -> float:
    predictions = np.full(outcome.shape, np.nan, dtype=float)
    for held_out in range(outcome.size):
        training = np.arange(outcome.size) != held_out
        design = np.column_stack(
            (np.ones(training.sum()), predictor[training], baseline[training])
        )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        coefficients = np.linalg.lstsq(design, outcome[training], rcond=None)[0]
        predictions[held_out] = np.dot(
            [1.0, predictor[held_out], baseline[held_out]], coefficients
        )
    valid = np.isfinite(predictions)
    if valid.sum() < 2:
        return np.nan
    denominator = np.sum((outcome[valid] - outcome[valid].mean()) ** 2)
    if denominator == 0:
        return np.nan
    return float(1.0 - np.sum((outcome[valid] - predictions[valid]) ** 2) / denominator)


def characterize_conditional_relationship(
    predictor: xr.DataArray,
    baseline: xr.DataArray,
    outcome: xr.DataArray,
    *,
    start_dim: str = "Y",
    log_baseline: bool = True,
    log_outcome: bool = True,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    block_length: int = 4,
    seed: int = 42,
) -> xr.Dataset:
    """Characterize an association after controlling for baseline magnitude.

    The primary estimate is the standardized coefficient of ``predictor`` in
    a two-predictor OLS model containing ``baseline``.  Percentile confidence
    intervals use a circular moving-block bootstrap over ordered starts.
    Spearman correlation is included as an outlier-resistant unadjusted
    sensitivity statistic, and predictive value is measured by leave-one-out
    cross-validated R-squared.
    """
    predictor, baseline, outcome = xr.align(
        predictor, baseline, outcome, join="exact", copy=False
    )
    for name, field in (
        ("predictor", predictor), ("baseline", baseline), ("outcome", outcome)
    ):
        if field.dims != (start_dim,):
            raise ValueError(f"{name} must have only dimension {start_dim!r}; got {field.dims}")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    if block_length < 1:
        raise ValueError("block_length must be positive")

    x = np.asarray(predictor.values, dtype=float)
    z = np.asarray(baseline.values, dtype=float)
    y = np.asarray(outcome.values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(z) & np.isfinite(y)
    if log_baseline:
        valid &= z > 0
    if log_outcome:
        valid &= y > 0
    x, z, y = x[valid], z[valid], y[valid]
    if log_baseline:
        z = np.log(z)
    if log_outcome:
        y = np.log(y)
    if x.size < 4:
        raise ValueError("Conditional relationship requires at least four valid starts")
    block_length = min(int(block_length), x.size)

    slope = _standardized_conditional_slope(x, z, y)
    rho = float(spearmanr(x, y).statistic)
    cv_r2 = _leave_one_out_r2(x, z, y)
    rng = np.random.default_rng(seed)
    indices = np.stack(
        [
            _circular_block_indices(rng, x.size, block_length)
            for _ in range(n_bootstrap)
        ]
    )
    sampled_x, sampled_z, sampled_y = x[indices], z[indices], y[indices]
    bootstrap_slopes = _row_standardized_conditional_slope(
        sampled_x, sampled_z, sampled_y
    )
    bootstrap_rhos = _row_correlation(
        rankdata(sampled_x, axis=1), rankdata(sampled_y, axis=1)
    )
    alpha = (1.0 - confidence) / 2.0
    quantiles = (alpha, 1.0 - alpha)
    slope_ci = np.nanquantile(bootstrap_slopes, quantiles)
    rho_ci = np.nanquantile(bootstrap_rhos, quantiles)
    result = xr.Dataset(
        {
            "standardized_slope": xr.DataArray(slope),
            "standardized_slope_ci_lower": xr.DataArray(slope_ci[0]),
            "standardized_slope_ci_upper": xr.DataArray(slope_ci[1]),
            "spearman_correlation": xr.DataArray(rho),
            "spearman_ci_lower": xr.DataArray(rho_ci[0]),
            "spearman_ci_upper": xr.DataArray(rho_ci[1]),
            "cross_validated_r2": xr.DataArray(cv_r2),
            "n_starts": xr.DataArray(x.size),
        }
    )
    result.attrs.update(
        model="standardized outcome ~ standardized predictor + standardized baseline",
        outcome_transform="log" if log_outcome else "none",
        baseline_transform="log" if log_baseline else "none",
        uncertainty="circular moving-block bootstrap over ordered initialization years",
        block_length=block_length,
        n_bootstrap=int(n_bootstrap),
        confidence=float(confidence),
        random_seed=int(seed),
    )
    return result


def bootstrap_block_mean_ci(
    values: xr.DataArray,
    *,
    start_dim: str = "Y",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    block_length: int = 4,
    seed: int = 42,
) -> xr.Dataset:
    """Return a circular moving-block bootstrap interval for a one-dimensional mean."""
    if values.dims != (start_dim,):
        raise ValueError(f"values must have only dimension {start_dim!r}; got {values.dims}")
    finite = np.asarray(values.values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        raise ValueError("Block-bootstrap mean requires at least two valid starts")
    if n_bootstrap < 1 or block_length < 1:
        raise ValueError("n_bootstrap and block_length must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    block_length = min(int(block_length), finite.size)
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [
            finite[_circular_block_indices(rng, finite.size, block_length)].mean()
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, (alpha, 1.0 - alpha))
    result = xr.Dataset(
        {
            "estimate": xr.DataArray(finite.mean()),
            "ci_lower": xr.DataArray(lower),
            "ci_upper": xr.DataArray(upper),
        }
    )
    result.attrs.update(
        uncertainty="circular moving-block bootstrap over ordered initialization years",
        block_length=block_length,
        n_bootstrap=int(n_bootstrap),
        confidence=float(confidence),
        random_seed=int(seed),
    )
    return result


def compare_drift_skill_relationship(
    fosirl: xr.Dataset, reanalysis: xr.Dataset, *, start_dim: str = "Y"
) -> xr.Dataset:
    """Relate paired FOSIRL-minus-Reanalysis drift and error across starts."""
    for name in ("early_drift", "late_error"):
        if name not in fosirl or name not in reanalysis:
            raise KeyError(f"Both inputs must contain {name!r}")
    delta_drift = compare_initializations(
        fosirl["early_drift"], reanalysis["early_drift"]
    ).rename("delta_drift")
    delta_error = compare_initializations(
        fosirl["late_error"], reanalysis["late_error"]
    ).rename("delta_error")
    delta_drift, delta_error = xr.align(delta_drift, delta_error, join="exact")
    correlation = xr.corr(delta_drift, delta_error, dim=start_dim).rename(
        "paired_drift_skill_correlation"
    )
    correlation.attrs = {
        "units": "1",
        "diagnostic": "Pearson correlation of paired experiment differences across starts",
    }
    valid = delta_drift.notnull() & delta_error.notnull()
    concordance = (
        ((delta_drift * delta_error) > 0).where(valid).mean(start_dim, skipna=True)
    ).rename("concordance_fraction")
    concordance.attrs = {
        "units": "1",
        "diagnostic": (
            "fraction of starts for which paired early-drift and later-error "
            "differences have the same sign"
        ),
        "tie_handling": "zero differences are counted as non-concordant",
    }
    return xr.Dataset(
        {
            "delta_drift": delta_drift,
            "delta_error": delta_error,
            "correlation": correlation,
            "concordance_fraction": concordance,
        }
    )


def bootstrap_paired_correlation_ci(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    start_dim: str = "Y",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> xr.Dataset:
    """Bootstrap a paired Pearson correlation by resampling starts together."""
    first, second = xr.align(first, second, join="exact", copy=False)
    if start_dim not in first.dims or start_dim not in second.dims:
        raise ValueError(f"Both inputs must contain {start_dim!r}")
    n_starts = first.sizes[start_dim]
    if n_starts < 3:
        raise ValueError("Paired correlation bootstrap requires at least three starts")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")

    rng = np.random.default_rng(seed)
    indices = xr.DataArray(
        rng.integers(0, n_starts, size=(n_bootstrap, n_starts)),
        dims=("bootstrap", "sample"),
    )
    sampled_first = first.isel({start_dim: indices})
    sampled_second = second.isel({start_dim: indices})
    sampled = xr.corr(sampled_first, sampled_second, dim="sample")
    alpha = (1.0 - confidence) / 2.0
    estimate = xr.corr(first, second, dim=start_dim).rename("estimate")
    lower = sampled.quantile(alpha, dim="bootstrap", skipna=True).drop_vars(
        "quantile"
    ).rename("ci_lower")
    upper = sampled.quantile(1.0 - alpha, dim="bootstrap", skipna=True).drop_vars(
        "quantile"
    ).rename("ci_upper")
    result = xr.Dataset({"estimate": estimate, "ci_lower": lower, "ci_upper": upper})
    result.attrs.update(
        bootstrap_dimension=start_dim,
        n_bootstrap=int(n_bootstrap),
        confidence=float(confidence),
        random_seed=int(seed),
        resampling="paired initialization years with replacement",
    )
    return result


def bootstrap_paired_mean_ci(
    difference: xr.DataArray,
    *,
    start_dim: str = "Y",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> xr.Dataset:
    """Bootstrap a paired mean difference by resampling initialization years."""
    if start_dim not in difference.dims:
        raise ValueError(f"Paired difference must retain {start_dim!r}")
    n_starts = difference.sizes[start_dim]
    if n_starts < 2:
        raise ValueError("Paired bootstrap requires at least two initialization years")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie between zero and one")
    rng = np.random.default_rng(seed)
    indices = xr.DataArray(
        rng.integers(0, n_starts, size=(n_bootstrap, n_starts)),
        dims=("bootstrap", "sample"),
    )
    sampled = difference.isel({start_dim: indices}).mean("sample", skipna=True)
    alpha = (1.0 - confidence) / 2.0
    estimate = difference.mean(start_dim, skipna=True).rename("estimate")
    lower = sampled.quantile(alpha, dim="bootstrap").drop_vars("quantile").rename("ci_lower")
    upper = sampled.quantile(1.0 - alpha, dim="bootstrap").drop_vars("quantile").rename("ci_upper")
    for field in (estimate, lower, upper):
        field.attrs.update(difference.attrs)
    result = xr.Dataset({"estimate": estimate, "ci_lower": lower, "ci_upper": upper})
    result.attrs.update(
        bootstrap_dimension=start_dim,
        n_bootstrap=int(n_bootstrap),
        confidence=float(confidence),
        random_seed=int(seed),
        resampling="paired initialization years with replacement",
    )
    return result


def compare_initializations(
    fosirl: xr.Dataset | xr.DataArray,
    reanalysis: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """Return paired ``JRA55_FOSIRL - Reanalysis`` for continuous quantities.

    For Dataset inputs the categorical ``regime`` variable is deliberately
    omitted; regimes must be compared through their area fractions.
    """
    left, right = xr.align(fosirl, reanalysis, join="exact", copy=False)
    if isinstance(left, xr.Dataset):
        shared = sorted((set(left.data_vars) & set(right.data_vars)) - {"regime"})
        if not shared:
            raise ValueError("Datasets contain no shared continuous variables")
        left, right = left[shared], right[shared]
    result = left - right
    result.attrs.update(
        comparison="JRA55_FOSIRL minus Reanalysis",
        sign_interpretation_delta_abs_e_obs=(
            "negative means FOSIRL is closer to observations; positive means Reanalysis is closer"
        ),
    )
    return result


def validate_diagnostics(product: xr.Dataset, *, atol: float = 1.0e-12) -> None:
    """Validate formulas, dimensions, masks, baseline, and regime signs."""
    required = {
        "hindcast_mean",
        "e_obs",
        "e_att",
        "delta_e_obs",
        "delta_e_att",
        "delta_abs_e_obs",
        "delta_abs_e_att",
        "regime",
    }
    missing = required - set(product.data_vars)
    if missing:
        raise ValueError(f"Diagnostic product is missing variables: {sorted(missing)}")
    if "Y" not in product.dims or "L" not in product.dims:
        raise ValueError("Diagnostic product must retain Y and L")
    baseline = int(product.attrs.get("baseline_lead", 1))
    for name in ("delta_e_obs", "delta_e_att", "delta_abs_e_obs", "delta_abs_e_att"):
        baseline_values = product[name].sel(L=baseline)
        finite = baseline_values.where(baseline_values.notnull(), 0.0)
        if not bool((abs(finite) <= atol).all()):
            raise ValueError(f"{name} is not zero at baseline lead {baseline}")
    expected = classify_drift_regime(
        product["delta_abs_e_obs"],
        product["delta_abs_e_att"],
        tolerance=float(product.attrs.get("distance_tolerance", 0.0)),
    )
    if not product["regime"].equals(expected):
        raise ValueError("Regime classification is inconsistent with distance-change signs")


__all__ = [
    "REGIME_DEFINITIONS",
    "area_weighted_mean",
    "area_weighted_rmse",
    "bootstrap_block_mean_ci",
    "build_attractor_lookup",
    "bootstrap_paired_correlation_ci",
    "bootstrap_paired_mean_ci",
    "build_obs_lookup",
    "build_reference_lookup",
    "classify_drift_regime",
    "characterize_conditional_relationship",
    "compare_initializations",
    "compare_drift_skill_relationship",
    "compute_diagnostics",
    "compute_distance_change",
    "compute_ensemble_mean",
    "compute_early_drift_late_error",
    "compute_early_error_growth_late_error",
    "compute_initialization_adjustment",
    "compute_lead_window_mean",
    "compute_reference_departure",
    "compute_regime_fraction",
    "compute_prediction_skill",
    "compute_spatial_drift_summary",
    "infer_area_weights",
    "match_model_climatology_to_valid_time",
    "match_observation_to_valid_time",
    "regional_subset",
    "run_pipeline",
    "validate_compatible_fields",
    "validate_diagnostics",
]
