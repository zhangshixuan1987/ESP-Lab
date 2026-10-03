"""Seasonal lead-year adjustment and legacy variability diagnostics.

The primary 24-month metrics compare matching three-month seasons in forecast
years two and one and normalize by season-specific observed interannual
variability. Annual changes and the original NCL temporal-std ratio are retained
for continuity; neither is ensemble spread or a direct initialization discontinuity.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

VERSION = "initial_shock_std_v4"


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
    reference_observation: xr.DataArray | None = None,
    area: xr.DataArray | None = None,
    block_months: int = 12,
    window_months: int = 60,
    start_lead: int = 0,
    min_samples: int | None = None,
    min_area_fraction: float = 0.0,
    climatology_years: tuple[int, int] | list[int] | None = None,
) -> xr.Dataset:
    """Compute seasonal/annual normalized changes and the legacy variability ratio.

    Inputs: model(Y,L,M,lat,lon) or ensemble mean(Y,L,lat,lon), and aligned
    observation(Y,L,lat,lon), in compatible physical units on the same grid.
    reference_observation(time,lat,lon), when supplied, defines one shared
    calendar-month climatology for all initialization cohorts and models.
    L must be consecutive numeric monthly leads. start_lead is a zero-based
    POSITION. Non-overlapping blocks start at that position; they are calendar
    years only when the first represented month is January and block_months=12.

    Ensemble averaging and block averaging require every member/month to be
    finite. Spatial means independently omit missing cells as in NCL. Statistics
    use paired valid blocks, ddof=1, no detrending, and no lead climatology removal.
    Seasonal changes pair each three-month mean with the same season twelve
    months later and use season-specific observed variability across Y. Annual
    normalized change uses the observed first-block means across Y. Both
    climatologies can be restricted by climatology_years. By default all requested
    legacy-ratio blocks must be valid. Missing/constant observation series produce
    NaN metrics, accompanied by coverage and validity variables.
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
    if window_months < 24:
        raise ValueError("window_months must include 24 months for same-season lead-year changes")
    if start_lead < 0 or start_lead + window_months > model.sizes["L"]:
        raise ValueError("Requested window exceeds available monthly leads")
    nblocks = window_months // block_months
    min_samples = nblocks if min_samples is None else min_samples
    if not isinstance(min_samples, (int, np.integer)) or not 2 <= min_samples <= nblocks:
        raise ValueError("min_samples must be between 2 and the number of blocks")
    if not 0 <= min_area_fraction <= 1:
        raise ValueError("min_area_fraction must be between 0 and 1")
    if climatology_years is not None:
        if (
            len(climatology_years) != 2
            or any(not isinstance(value, (int, np.integer)) for value in climatology_years)
            or climatology_years[0] > climatology_years[1]
        ):
            raise ValueError("climatology_years must be two ordered integer years")
    if not model.attrs.get("units") or model.attrs.get("units") != observation.attrs.get("units"):
        raise ValueError("Convert model and observation to identical explicit units first")
    if reference_observation is not None:
        if set(reference_observation.dims) != {"time", "lat", "lon"}:
            raise ValueError("reference_observation must have dimensions (time, lat, lon)")
        if any(dim not in reference_observation.coords for dim in ("time", "lat", "lon")):
            raise ValueError("reference_observation dimensions must have explicit coordinates")
        if not reference_observation.get_index("time").is_unique:
            raise ValueError("reference_observation time coordinates must be unique")
        if reference_observation.attrs.get("units") != model.attrs["units"]:
            raise ValueError("reference_observation must use the same explicit units as model")
    model, observation = xr.align(model, observation, join="exact")
    if reference_observation is not None:
        _, reference_observation = xr.align(
            model.isel(Y=0, L=0, drop=True), reference_observation, join="exact"
        )
    units = model.attrs["units"]
    nens = model.sizes.get("M", 1)
    model = model.where(np.isfinite(model))
    observation = observation.where(np.isfinite(observation))
    if reference_observation is not None:
        reference_observation = reference_observation.where(np.isfinite(reference_observation))
    first_ensemble_member = str(model.M.values[0]) if "M" in model.dims else "deterministic"
    first_member_model = model.isel(M=0, drop=True) if "M" in model.dims else model
    if "M" in model.dims:
        model = model.mean("M", skipna=False)
    weights = _weights(model, area)
    def block_index(data, *, months, length):
        if "verification_time" in data.coords:
            data = data.drop_vars("verification_time")
        blocks = data.isel(L=slice(start_lead, start_lead + length)).coarsen(
            L=months, boundary="exact"
        ).mean(skipna=False)
        coverage = weights.where(blocks.notnull(), 0).sum(("lat", "lon")) / weights.sum()
        index = blocks.weighted(weights).mean(("lat", "lon"), skipna=True)
        index = index.where((coverage > 0) & (coverage >= min_area_fraction))
        return index, coverage
    model_index, model_area = block_index(model, months=block_months, length=window_months)
    obs_index, obs_area = block_index(observation, months=block_months, length=window_months)
    paired = model_index.notnull() & obs_index.notnull()
    n = paired.sum("L")
    m = model_index.where(paired)
    o = obs_index.where(paired)
    # Explicit sample variance avoids ddof warnings for insufficient samples.
    def sample_std(data):
        return np.sqrt(((data - data.mean("L")) ** 2).sum("L") / (n - 1).where(n > 1))
    mstd, ostd = sample_std(m), sample_std(o)
    valid = (n >= min_samples) & (ostd > 0)

    # With a 24-month window the legacy std ratio has only two temporal
    # samples. Normalize the first-to-second block change by the much more
    # stable spread of first-block observed annual means across initialization
    # years. For May (November) starts these are unique May-April
    # (November-October) annual means.
    climatology = obs_index.isel(L=0)
    if climatology_years is not None:
        year_values = np.asarray(climatology.Y.values)
        if not np.issubdtype(year_values.dtype, np.number):
            raise ValueError("climatology_years requires numeric initialization-year coordinates")
        climatology = climatology.where(
            (climatology.Y >= climatology_years[0])
            & (climatology.Y <= climatology_years[1])
        )
    climatology_count = climatology.notnull().sum("Y")
    climatology_mean = climatology.mean("Y", skipna=True)
    climatology_std = np.sqrt(
        ((climatology - climatology_mean) ** 2).sum("Y", skipna=True)
        / (climatology_count - 1).where(climatology_count > 1)
    )
    model_change = model_index.isel(L=1) - model_index.isel(L=0)
    observation_change = obs_index.isel(L=1) - obs_index.isel(L=0)
    valid_change = model_change.notnull() & (climatology_count > 1) & (climatology_std > 0)
    signed_change = (model_change / climatology_std.where(climatology_std > 0)).where(valid_change)
    absolute_change = np.abs(signed_change)
    excess_change = (
        (model_change - observation_change) / climatology_std.where(climatology_std > 0)
    ).where(valid_change & observation_change.notnull())

    # Compare matching three-month seasons in lead years two and one. Pairing
    # seasons twelve months apart avoids treating the climatological seasonal
    # cycle as forecast adjustment.
    seasons_per_year = 4
    seasonal_model, seasonal_model_area = block_index(model, months=3, length=24)
    seasonal_obs, seasonal_obs_area = block_index(observation, months=3, length=24)

    def split_lead_years(data):
        year1 = data.isel(L=slice(0, seasons_per_year)).rename(L="season")
        year2 = data.isel(L=slice(seasons_per_year, 2 * seasons_per_year)).rename(L="season")
        season_coord = np.arange(1, seasons_per_year + 1)
        return (
            year1.assign_coords(season=season_coord),
            year2.assign_coords(season=season_coord),
        )

    seasonal_model_year1, seasonal_model_year2 = split_lead_years(seasonal_model)
    seasonal_obs_year1, seasonal_obs_year2 = split_lead_years(seasonal_obs)
    seasonal_model_area_year1, seasonal_model_area_year2 = split_lead_years(seasonal_model_area)
    seasonal_obs_area_year1, seasonal_obs_area_year2 = split_lead_years(seasonal_obs_area)
    seasonal_model_change = seasonal_model_year2 - seasonal_model_year1
    seasonal_observation_change = seasonal_obs_year2 - seasonal_obs_year1

    seasonal_climatology = seasonal_obs_year1
    if climatology_years is not None:
        seasonal_climatology = seasonal_climatology.where(
            (seasonal_climatology.Y >= climatology_years[0])
            & (seasonal_climatology.Y <= climatology_years[1])
        )
    seasonal_climatology_count = seasonal_climatology.notnull().sum("Y")
    seasonal_climatology_mean = seasonal_climatology.mean("Y", skipna=True)
    seasonal_climatology_std = np.sqrt(
        ((seasonal_climatology - seasonal_climatology_mean) ** 2).sum("Y", skipna=True)
        / (seasonal_climatology_count - 1).where(seasonal_climatology_count > 1)
    )
    valid_seasonal_change = (
        seasonal_model_change.notnull()
        & (seasonal_climatology_count > 1)
        & (seasonal_climatology_std > 0)
    )
    seasonal_signed_change = (
        seasonal_model_change / seasonal_climatology_std.where(seasonal_climatology_std > 0)
    ).where(valid_seasonal_change)
    seasonal_absolute_change = np.abs(seasonal_signed_change)
    seasonal_excess_change = (
        (seasonal_model_change - seasonal_observation_change)
        / seasonal_climatology_std.where(seasonal_climatology_std > 0)
    ).where(valid_seasonal_change & seasonal_observation_change.notnull())

    seasonal_model_index = xr.concat(
        [seasonal_model_year1, seasonal_model_year2],
        dim=xr.IndexVariable("lead_year", [1, 2]),
    ).transpose("Y", "lead_year", "season")
    seasonal_observation_index = xr.concat(
        [seasonal_obs_year1, seasonal_obs_year2],
        dim=xr.IndexVariable("lead_year", [1, 2]),
    ).transpose("Y", "lead_year", "season")
    seasonal_model_coverage = xr.concat(
        [seasonal_model_area_year1, seasonal_model_area_year2],
        dim=xr.IndexVariable("lead_year", [1, 2]),
    ).transpose("Y", "lead_year", "season")
    seasonal_observation_coverage = xr.concat(
        [seasonal_obs_area_year1, seasonal_obs_area_year2],
        dim=xr.IndexVariable("lead_year", [1, 2]),
    ).transpose("Y", "lead_year", "season")

    # Preserve monthly timing for the evolution diagnostic. Prefer a single
    # calendar-month climatology from the independent reference time series.
    monthly_model_index, monthly_model_coverage = block_index(model, months=1, length=24)
    monthly_first_member_index, monthly_first_member_coverage = block_index(
        first_member_model, months=1, length=24
    )
    monthly_observation_index, monthly_observation_coverage = block_index(
        observation, months=1, length=24
    )
    monthly_leads = np.arange(1, 25)
    monthly_model_index = monthly_model_index.rename(L="lead_month").assign_coords(
        lead_month=monthly_leads
    )
    monthly_first_member_index = monthly_first_member_index.rename(
        L="lead_month"
    ).assign_coords(lead_month=monthly_leads)
    monthly_observation_index = monthly_observation_index.rename(L="lead_month").assign_coords(
        lead_month=monthly_leads
    )
    monthly_model_coverage = monthly_model_coverage.rename(L="lead_month").assign_coords(
        lead_month=monthly_leads
    )
    monthly_first_member_coverage = monthly_first_member_coverage.rename(
        L="lead_month"
    ).assign_coords(lead_month=monthly_leads)
    monthly_observation_coverage = monthly_observation_coverage.rename(
        L="lead_month"
    ).assign_coords(lead_month=monthly_leads)

    if reference_observation is not None:
        reference_coverage = (
            weights.where(reference_observation.notnull(), 0).sum(("lat", "lon"))
            / weights.sum()
        )
        reference_index = reference_observation.weighted(weights).mean(
            ("lat", "lon"), skipna=True
        ).where((reference_coverage > 0) & (reference_coverage >= min_area_fraction))
        if climatology_years is not None:
            reference_index = reference_index.where(
                (reference_index.time.dt.year >= climatology_years[0])
                & (reference_index.time.dt.year <= climatology_years[1]),
                drop=True,
            )
        monthly_groups = reference_index.groupby("time.month")
        monthly_climatology_count = monthly_groups.count("time").rename(
            month="month_phase"
        )
        monthly_climatology_mean = monthly_groups.mean("time", skipna=True).rename(
            month="month_phase"
        )
        monthly_climatology_std = monthly_groups.std(
            "time", skipna=True, ddof=1
        ).rename(month="month_phase")
        for name, values in (
            ("count", monthly_climatology_count),
            ("mean", monthly_climatology_mean),
            ("std", monthly_climatology_std),
        ):
            if not np.array_equal(np.asarray(values.month_phase), np.arange(1, 13)):
                raise ValueError(
                    f"Shared reference climatology lacks calendar months for {name}"
                )
        if "verification_time" not in observation.coords:
            raise ValueError(
                "verification_time is required with a shared reference_observation"
            )
        calendar_month = np.asarray(
            observation.verification_time.isel(Y=0, L=slice(start_lead, start_lead + 24)).dt.month,
            dtype=int,
        )

        def map_calendar_months(data):
            return xr.concat(
                [data.sel(month_phase=int(month)) for month in calendar_month],
                dim=xr.IndexVariable("lead_month", monthly_leads),
            )

        monthly_mean_by_lead = map_calendar_months(monthly_climatology_mean)
        monthly_std_by_lead = map_calendar_months(monthly_climatology_std)
        monthly_count_by_lead = map_calendar_months(monthly_climatology_count)
        monthly_climatology_source = "shared reference observation by calendar month"
    else:
        monthly_climatology = monthly_observation_index.isel(
            lead_month=slice(0, 12)
        ).rename(lead_month="month_phase").assign_coords(month_phase=np.arange(1, 13))
        if climatology_years is not None:
            monthly_climatology = monthly_climatology.where(
                (monthly_climatology.Y >= climatology_years[0])
                & (monthly_climatology.Y <= climatology_years[1])
            )
        monthly_climatology_count = monthly_climatology.notnull().sum("Y")
        monthly_climatology_mean = monthly_climatology.mean("Y", skipna=True)
        monthly_climatology_std = np.sqrt(
            ((monthly_climatology - monthly_climatology_mean) ** 2).sum("Y", skipna=True)
            / (monthly_climatology_count - 1).where(monthly_climatology_count > 1)
        )

        def repeat_monthly_climatology(data):
            first_year = data.rename(month_phase="lead_month")
            second_year = first_year.assign_coords(lead_month=np.arange(13, 25))
            return xr.concat([first_year, second_year], dim="lead_month")

        monthly_mean_by_lead = repeat_monthly_climatology(monthly_climatology_mean)
        monthly_std_by_lead = repeat_monthly_climatology(monthly_climatology_std)
        monthly_count_by_lead = repeat_monthly_climatology(monthly_climatology_count)
        monthly_climatology_source = "first lead year of aligned observation cohort"
    valid_monthly_climatology = (monthly_count_by_lead > 1) & (monthly_std_by_lead > 0)
    monthly_model_anomaly = (
        (monthly_model_index - monthly_mean_by_lead)
        / monthly_std_by_lead.where(monthly_std_by_lead > 0)
    ).where(monthly_model_index.notnull() & valid_monthly_climatology)
    monthly_first_member_anomaly = (
        (monthly_first_member_index - monthly_mean_by_lead)
        / monthly_std_by_lead.where(monthly_std_by_lead > 0)
    ).where(monthly_first_member_index.notnull() & valid_monthly_climatology)
    monthly_observation_anomaly = (
        (monthly_observation_index - monthly_mean_by_lead)
        / monthly_std_by_lead.where(monthly_std_by_lead > 0)
    ).where(monthly_observation_index.notnull() & valid_monthly_climatology)
    monthly_standardized_error = (
        (monthly_model_index - monthly_observation_index)
        / monthly_std_by_lead.where(monthly_std_by_lead > 0)
    ).where(
        monthly_model_index.notnull()
        & monthly_observation_index.notnull()
        & valid_monthly_climatology
    )

    result = xr.Dataset({
        "std_ratio": (mstd / ostd.where(ostd > 0)).where(valid),
        "model_std": mstd.where(n >= min_samples),
        "observation_std": ostd.where(n >= min_samples),
        "model_lead_year_change": model_change,
        "observation_lead_year_change": observation_change,
        "observation_climatology_std": climatology_std,
        "observation_climatology_sample_count": climatology_count,
        "signed_normalized_change": signed_change,
        "absolute_normalized_change": absolute_change,
        "excess_normalized_change": excess_change,
        "valid_normalized_change": valid_change.astype("int8"),
        "seasonal_model_year_change": seasonal_model_change,
        "seasonal_observation_year_change": seasonal_observation_change,
        "seasonal_observation_climatology_std": seasonal_climatology_std,
        "seasonal_observation_climatology_sample_count": seasonal_climatology_count,
        "seasonal_signed_normalized_change": seasonal_signed_change,
        "seasonal_absolute_normalized_change": seasonal_absolute_change,
        "seasonal_excess_normalized_change": seasonal_excess_change,
        "valid_seasonal_normalized_change": valid_seasonal_change.astype("int8"),
        "seasonal_model_index": seasonal_model_index,
        "seasonal_observation_index": seasonal_observation_index,
        "seasonal_model_area_fraction": seasonal_model_coverage,
        "seasonal_observation_area_fraction": seasonal_observation_coverage,
        "monthly_model_index": monthly_model_index,
        "monthly_first_member_index": monthly_first_member_index,
        "monthly_observation_index": monthly_observation_index,
        "monthly_observation_climatology_mean": monthly_climatology_mean,
        "monthly_observation_climatology_std": monthly_climatology_std,
        "monthly_observation_climatology_sample_count": monthly_climatology_count,
        "monthly_model_normalized_anomaly": monthly_model_anomaly,
        "monthly_first_member_normalized_anomaly": monthly_first_member_anomaly,
        "monthly_observation_normalized_anomaly": monthly_observation_anomaly,
        "monthly_standardized_error": monthly_standardized_error,
        "monthly_model_area_fraction": monthly_model_coverage,
        "monthly_first_member_area_fraction": monthly_first_member_coverage,
        "monthly_observation_area_fraction": monthly_observation_coverage,
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
    seasonal_start_leads = leads[start_lead:start_lead + 24:3].reshape(2, seasons_per_year)
    seasonal_end_leads = leads[start_lead + 2:start_lead + 24:3].reshape(2, seasons_per_year)
    result = result.assign_coords(
        seasonal_start_lead=(("lead_year", "season"), seasonal_start_leads),
        seasonal_end_lead=(("lead_year", "season"), seasonal_end_leads),
    )
    if "verification_time" in observation.coords:
        time = observation.verification_time
        for name, offset in (("block_start_time", 0), ("block_end_time", block_months - 1)):
            values = time.isel(L=slice(start_lead + offset, start_lead + window_months, block_months))
            result = result.assign_coords({name: (("Y", "block"), values.values)})
        seasonal_start_time = time.isel(L=slice(start_lead, start_lead + 24, 3))
        seasonal_end_time = time.isel(L=slice(start_lead + 2, start_lead + 24, 3))
        result = result.assign_coords(
            seasonal_start_time=(
                ("Y", "lead_year", "season"),
                np.asarray(seasonal_start_time).reshape(model.sizes["Y"], 2, seasons_per_year),
            ),
            seasonal_end_time=(
                ("Y", "lead_year", "season"),
                np.asarray(seasonal_end_time).reshape(model.sizes["Y"], 2, seasons_per_year),
            ),
        )
        month_names = np.array(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        first_start = np.asarray(seasonal_start_time.dt.month)[0, :seasons_per_year]
        first_end = np.asarray(seasonal_end_time.dt.month)[0, :seasons_per_year]
        result = result.assign_coords(season_label=(
            "season",
            [f"{month_names[start - 1]}-{month_names[end - 1]}"
             for start, end in zip(first_start, first_end)],
        ))
        monthly_time = time.isel(L=slice(start_lead, start_lead + 24))
        calendar_month = np.asarray(monthly_time.dt.month)[0]
        result = result.assign_coords(
            monthly_verification_time=(("Y", "lead_month"), np.asarray(monthly_time)),
            calendar_month=("lead_month", calendar_month),
            calendar_month_label=(
                "lead_month", [month_names[value - 1] for value in calendar_month]
            ),
        )
    for name in ("model_std", "observation_std", "model_index", "observation_index"):
        result[name].attrs["units"] = units
    for name in (
        "model_lead_year_change", "observation_lead_year_change",
        "observation_climatology_std", "seasonal_model_year_change",
        "seasonal_observation_year_change", "seasonal_observation_climatology_std",
        "seasonal_model_index", "seasonal_observation_index",
        "monthly_model_index", "monthly_first_member_index", "monthly_observation_index",
        "monthly_observation_climatology_mean", "monthly_observation_climatology_std",
    ):
        result[name].attrs["units"] = units
    result.std_ratio.attrs.update(units="1", long_name="Model / observation temporal standard deviation")
    result.signed_normalized_change.attrs.update(
        units="1", long_name="Signed model block-2 minus block-1 change normalized by observed climatological std",
    )
    result.absolute_normalized_change.attrs.update(
        units="1", long_name="Absolute model block-2 minus block-1 change normalized by observed climatological std",
    )
    result.excess_normalized_change.attrs.update(
        units="1", long_name="Model minus observed block change normalized by observed climatological std",
    )
    result.observation_climatology_sample_count.attrs.update(
        units="1", long_name="Number of observed annual means in climatological standard deviation",
    )
    result.valid_normalized_change.attrs.update(
        units="1", long_name="Normalized lead-year change validity flag",
    )
    result.seasonal_signed_normalized_change.attrs.update(
        units="1",
        long_name="Same-season model year-2 minus year-1 change normalized by observed seasonal std",
    )
    result.seasonal_absolute_normalized_change.attrs.update(
        units="1",
        long_name="Absolute same-season model year-2 minus year-1 change normalized by observed seasonal std",
    )
    result.seasonal_excess_normalized_change.attrs.update(
        units="1",
        long_name="Same-season model minus observed year-over-year change normalized by observed seasonal std",
    )
    result.seasonal_observation_climatology_sample_count.attrs.update(
        units="1", long_name="Observed samples in each season-specific climatological standard deviation",
    )
    result.valid_seasonal_normalized_change.attrs.update(
        units="1", long_name="Same-season normalized lead-year change validity flag",
    )
    result.monthly_model_normalized_anomaly.attrs.update(
        units="1", long_name="Model monthly anomaly normalized by observed calendar-month std",
    )
    result.monthly_first_member_normalized_anomaly.attrs.update(
        units="1",
        long_name="First ensemble member monthly anomaly normalized by observed calendar-month std",
    )
    result.monthly_observation_normalized_anomaly.attrs.update(
        units="1", long_name="Observed monthly anomaly normalized by observed calendar-month std",
    )
    result.monthly_standardized_error.attrs.update(
        units="1", long_name="Model minus observed monthly mean normalized by observed calendar-month std",
    )
    result.monthly_observation_climatology_sample_count.attrs.update(
        units="1", long_name="Observed samples in each calendar-month climatology",
    )
    result.attrs.update(
        diagnostic_version=VERSION, block_months=int(block_months), window_months=int(window_months),
        start_lead_position=int(start_lead), min_samples=int(min_samples), ddof=1,
        ensemble_members=nens, min_area_fraction=float(min_area_fraction),
        detrended="false", spatial_mask="independent finite cells per block",
        averaging_order="ensemble mean; unweighted monthly blocks; area mean; temporal sample std",
        interpretation=(
            "Primary metrics compare matching three-month seasons in lead years two and one; "
            "they do not uniquely isolate initialization shock from observed climate evolution"
        ),
        normalized_change_climatology=(
            "all initialization years" if climatology_years is None
            else f"{climatology_years[0]}-{climatology_years[1]}"
        ),
        normalized_change_denominator="sample std of observed first-block means across initialization years",
        seasonal_change_definition="matching three-month season in lead year 2 minus lead year 1",
        seasonal_change_denominator="season-specific sample std of observed lead-year-1 means across initialization years",
        monthly_anomaly_climatology=monthly_climatology_source,
        monthly_standardized_error_definition="(model monthly mean - observed monthly mean) / observed calendar-month interannual std",
        first_ensemble_member=first_ensemble_member,
    )
    return result


def plot_std_ratio(
    result: xr.Dataset,
    *,
    ax=None,
    add_colorbar: bool = True,
    add_invalid_legend: bool = True,
    title: str | None = None,
):
    """Plot a case-by-initialization heatmap; NaN denotes an invalid ratio.

    Supply ``ax`` and disable the per-axis colorbar/legend when composing
    multiple initialization months into one figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    ratio = result.std_ratio
    if "case" not in ratio.dims:
        ratio = ratio.expand_dims(case=[result.attrs.get("case", "model")])
    ratio = ratio.transpose("Y", "case")
    # A ratio of one is the neutral point.  Use cool colors below one and warm
    # colors above one so that equally variable, under-variable, and
    # over-variable cases cannot share the same color.
    bounds = np.round(np.arange(0.4, 3.0 + 0.2, 0.2), 1)
    centers = (bounds[:-1] + bounds[1:]) / 2
    diverging_norm = TwoSlopeNorm(vmin=bounds[0], vcenter=1.0, vmax=bounds[-1])
    colors = plt.colormaps["RdBu_r"](diverging_norm(centers))
    cmap = ListedColormap(colors)
    invalid_color = "#bdbdbd"
    cmap.set_bad(invalid_color)
    # Make out-of-range values visibly different from the adjacent endpoint
    # bins; these colors also fill the colorbar's extension triangles.
    cmap.set_under("#021a35")
    cmap.set_over("#3b0010")
    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(
            figsize=(max(6, ratio.sizes["case"] * 1.5), max(3, ratio.sizes["Y"] * .25))
        )
    else:
        fig = ax.figure
    mesh = ax.imshow(ratio.values, aspect="auto", cmap=cmap, norm=BoundaryNorm(bounds, cmap.N))
    ax.set_xticks(np.arange(ratio.sizes["case"]), labels=ratio.case.values, rotation=30, ha="right")
    ax.set_yticks(np.arange(ratio.sizes["Y"]), labels=[str(v) for v in ratio.Y.values])
    ax.set_xticks(np.arange(-.5, ratio.sizes["case"], 1), minor=True)
    ax.set_yticks(np.arange(-.5, ratio.sizes["Y"], 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel("Initialization")
    ax.set_title(
        title if title is not None else
        f"Variability ratio: {result.attrs.get('window_months', '?')} months, "
        f"{result.attrs.get('block_months', '?')}-month means"
    )
    if add_colorbar:
        colorbar = fig.colorbar(
            mesh, ax=ax, ticks=bounds,
            label="Model std / observed std (1 = equal)", extend="both",
        )
        colorbar.ax.set_yticklabels([f"{value:.1f}" for value in bounds])
    if add_invalid_legend:
        ax.legend(
            handles=[Patch(facecolor=invalid_color, edgecolor="none", label="Invalid / missing")],
            loc="upper left", bbox_to_anchor=(1.01, 0), frameon=False, fontsize="small",
        )
    if owns_figure:
        fig.tight_layout()
    return fig


def plot_normalized_change(
    result: xr.Dataset,
    *,
    variable: str = "signed_normalized_change",
    levels=None,
    ax=None,
    add_colorbar: bool = True,
    add_invalid_legend: bool = True,
    title: str | None = None,
):
    """Plot signed, absolute, or observation-adjusted normalized change.

    Parameters
    ----------
    levels : array-like, optional
        Explicit, strictly increasing color-bin boundaries. Variable-specific
        defaults are used when omitted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    choices = {
        "signed_normalized_change": {
            "bounds": np.arange(-3.0, 3.01, 0.5),
            "cmap": "RdBu_r", "extend": "both",
            "label": r"$(M_2-M_1) / \sigma_{obs,clim}$",
        },
        "absolute_normalized_change": {
            "bounds": np.arange(0.0, 3.01, 0.25),
            "cmap": "YlOrRd", "extend": "max",
            "label": r"$|M_2-M_1| / \sigma_{obs,clim}$",
        },
        "excess_normalized_change": {
            "bounds": np.arange(-3.0, 3.01, 0.5),
            "cmap": "RdBu_r", "extend": "both",
            "label": r"$[(M_2-M_1)-(O_2-O_1)] / \sigma_{obs,clim}$",
        },
        "seasonal_signed_normalized_change": {
            "bounds": np.arange(-3.0, 3.01, 0.5),
            "cmap": "RdBu_r", "extend": "both",
            "label": r"$(M_{2,s}-M_{1,s}) / \sigma_{obs,s}$",
        },
        "seasonal_absolute_normalized_change": {
            "bounds": np.arange(0.0, 3.01, 0.25),
            "cmap": "YlOrRd", "extend": "max",
            "label": r"$|M_{2,s}-M_{1,s}| / \sigma_{obs,s}$",
        },
        "seasonal_excess_normalized_change": {
            "bounds": np.arange(-3.0, 3.01, 0.5),
            "cmap": "RdBu_r", "extend": "both",
            "label": r"$[(M_{2,s}-M_{1,s})-(O_{2,s}-O_{1,s})] / \sigma_{obs,s}$",
        },
    }
    if variable not in choices:
        raise ValueError(f"variable must be one of {tuple(choices)}")
    if variable not in result:
        raise ValueError(f"Result does not contain {variable!r}")

    settings = choices[variable]
    values = result[variable]
    if "case" not in values.dims:
        values = values.expand_dims(case=[result.attrs.get("case", "model")])
    values = values.transpose("Y", "case")
    bounds = np.asarray(settings["bounds"] if levels is None else levels, dtype=float)
    if bounds.ndim != 1 or bounds.size < 2:
        raise ValueError("levels must contain at least two one-dimensional boundaries")
    if not np.isfinite(bounds).all() or not np.all(np.diff(bounds) > 0):
        raise ValueError("levels must be finite and strictly increasing")
    centers = (bounds[:-1] + bounds[1:]) / 2
    if bounds[0] < 0 < bounds[-1]:
        scale = TwoSlopeNorm(vmin=bounds[0], vcenter=0.0, vmax=bounds[-1])
        colors = plt.colormaps[settings["cmap"]](scale(centers))
    else:
        colors = plt.colormaps[settings["cmap"]](
            (centers - bounds[0]) / (bounds[-1] - bounds[0])
        )
    cmap = ListedColormap(colors)
    invalid_color = "#bdbdbd"
    cmap.set_bad(invalid_color)
    if bounds[0] < 0 < bounds[-1]:
        cmap.set_under("#021a35")
        cmap.set_over("#3b0010")
    else:
        cmap.set_over("#4a0000")

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(
            figsize=(max(6, values.sizes["case"] * 1.5), max(3, values.sizes["Y"] * .25))
        )
    else:
        fig = ax.figure
    mesh = ax.imshow(
        values.values, aspect="auto", cmap=cmap,
        norm=BoundaryNorm(bounds, cmap.N),
    )
    ax.set_xticks(
        np.arange(values.sizes["case"]), labels=values.case.values,
        rotation=30, ha="right",
    )
    ax.set_yticks(np.arange(values.sizes["Y"]), labels=[str(value) for value in values.Y.values])
    ax.set_xticks(np.arange(-.5, values.sizes["case"], 1), minor=True)
    ax.set_yticks(np.arange(-.5, values.sizes["Y"], 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel("Initialization")
    ax.set_title(title if title is not None else values.attrs.get("long_name", variable))
    if add_colorbar:
        colorbar = fig.colorbar(
            mesh, ax=ax, ticks=bounds, label=settings["label"],
            extend=settings["extend"],
        )
        colorbar.ax.set_yticklabels([f"{value:g}" for value in bounds])
    if add_invalid_legend:
        ax.legend(
            handles=[Patch(facecolor=invalid_color, edgecolor="none", label="Invalid / missing")],
            loc="upper left", bbox_to_anchor=(1.01, 0), frameon=False, fontsize="small",
        )
    if owns_figure:
        fig.tight_layout()
    return fig
