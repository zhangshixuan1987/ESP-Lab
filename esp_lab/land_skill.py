"""Lead-time ACC skill utilities for E3SM land variables.

This module extracts the reusable scientific workflow from
``jupyter/1a_refactor_leadtime_acc_skill_map.ipynb`` for land-model fields.
It deliberately leaves the observational product configurable: snow water
equivalent, total water storage, and soil moisture generally require different
reference products and masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

from . import data_access_e3sm, stats
from .utils import calendar_utils


@dataclass(frozen=True)
class LandVariableSpec:
    """Metadata and dimensional requirements for a supported land field."""

    field: str
    long_name: str
    plot_name: str
    units: str
    vertical_dim: str | None = None


LAND_VARIABLES: Mapping[str, LandVariableSpec] = {
    "H2OSNO": LandVariableSpec(
        "H2OSNO", "snow water equivalent", "Snow water equivalent", "mm"
    ),
    "TWS": LandVariableSpec(
        "TWS", "total water storage", "Total water storage", "mm"
    ),
    "H2OSOI": LandVariableSpec(
        "H2OSOI",
        "volumetric soil water",
        "Soil moisture",
        "mm3/mm3",
        vertical_dim="levgrnd",
    ),
}


def elm_soil_layer_bounds(depths: xr.DataArray) -> xr.DataArray:
    """Return ELM layer interfaces after verifying native ``levgrnd`` centers.

    ELM uses an exponentially stretched vertical soil grid. The archived
    H2OSOI files contain layer centers but not bounds, so bounds may only be
    reconstructed safely when the coordinates match that documented grid.
    Unknown grids are rejected instead of being treated as ELM implicitly.
    """
    if depths.ndim != 1:
        raise ValueError("soil depth coordinates must be one-dimensional")
    if depths.size == 0:
        raise ValueError("soil depth coordinates must not be empty")
    values = np.asarray(depths, dtype=float)
    layer_number = np.arange(1, values.size + 1, dtype=float)
    expected_centers = 0.025 * (np.exp(0.5 * (layer_number - 0.5)) - 1.0)
    if not np.allclose(values, expected_centers, rtol=2.0e-5, atol=1.0e-7):
        raise ValueError(
            "levgrnd does not match the native ELM soil grid; provide explicit "
            "soil_layer_bounds_m"
        )
    interfaces = 0.025 * (np.exp(0.5 * np.arange(values.size + 1)) - 1.0)
    return xr.DataArray(
        np.column_stack([interfaces[:-1], interfaces[1:]]),
        dims=(depths.dims[0], "bounds"),
        coords={depths.dims[0]: depths, "bounds": [0, 1]},
        attrs={"units": "m", "source": "reconstructed native ELM grid"},
    )


def _standardize_layer_bounds(
    bounds: xr.DataArray | np.ndarray | Sequence[float],
    depths: xr.DataArray,
    vertical_dim: str,
) -> xr.DataArray:
    values = np.asarray(bounds, dtype=float)
    nlevels = depths.size
    if values.shape == (nlevels + 1,):
        values = np.column_stack([values[:-1], values[1:]])
    if values.shape != (nlevels, 2):
        raise ValueError(
            f"layer bounds must have shape ({nlevels}, 2) or ({nlevels + 1},), "
            f"got {values.shape}"
        )
    if not np.all(np.isfinite(values)) or np.any(values[:, 1] <= values[:, 0]):
        raise ValueError("layer bounds must be finite and have bottom > top")
    if np.any(values[1:, 0] < values[:-1, 1] - 1.0e-10):
        raise ValueError("soil layers must not overlap")
    return xr.DataArray(
        values,
        dims=(vertical_dim, "bounds"),
        coords={vertical_dim: depths, "bounds": [0, 1]},
        attrs={"units": "m"},
    )


def depth_weighted_soil_moisture(
    da: xr.DataArray,
    depth_range_m: tuple[float, float],
    *,
    vertical_dim: str = "levgrnd",
    layer_bounds_m: xr.DataArray | np.ndarray | Sequence[float] | None = None,
    min_coverage_fraction: float = 0.999,
) -> xr.DataArray:
    """Average volumetric soil moisture over a physical depth interval.

    Each layer is weighted by its geometric overlap with ``depth_range_m``.
    Missing layers are excluded locally and output is masked wherever their
    combined thickness covers less than ``min_coverage_fraction`` of the target
    interval.
    """
    if vertical_dim not in da.dims:
        raise ValueError(f"soil moisture must contain {vertical_dim!r}")
    if len(depth_range_m) != 2:
        raise ValueError("depth_range_m must contain (top_m, bottom_m)")
    top, bottom = map(float, depth_range_m)
    if not np.isfinite([top, bottom]).all() or top < 0 or bottom <= top:
        raise ValueError("depth_range_m must satisfy 0 <= top < bottom")
    if not 0 < min_coverage_fraction <= 1:
        raise ValueError("min_coverage_fraction must be in (0, 1]")

    depths = da[vertical_dim]
    if layer_bounds_m is None:
        if vertical_dim != "levgrnd":
            raise ValueError("non-ELM vertical grids require explicit layer_bounds_m")
        bounds = elm_soil_layer_bounds(depths)
    else:
        bounds = _standardize_layer_bounds(layer_bounds_m, depths, vertical_dim)

    overlap = np.maximum(
        0.0,
        np.minimum(bounds.isel(bounds=1), bottom)
        - np.maximum(bounds.isel(bounds=0), top),
    )
    requested_thickness = bottom - top
    geometric_coverage = float(overlap.sum()) / requested_thickness
    if geometric_coverage < min_coverage_fraction:
        raise ValueError(
            f"soil layers cover only {geometric_coverage:.3f} of requested "
            f"{top:g}-{bottom:g} m interval"
        )

    valid_overlap = overlap.where(da.notnull())
    covered_thickness = valid_overlap.sum(vertical_dim)
    safe_covered_thickness = covered_thickness.where(covered_thickness > 0)
    out = (da * overlap).sum(vertical_dim, skipna=True) / safe_covered_thickness
    coverage_fraction = covered_thickness / requested_thickness
    out = out.where(coverage_fraction >= min_coverage_fraction)
    out.attrs = dict(da.attrs)
    out.attrs.update(
        {
            "depth_top_m": top,
            "depth_bottom_m": bottom,
            "vertical_aggregation": "layer-overlap thickness-weighted mean",
            "minimum_vertical_coverage_fraction": min_coverage_fraction,
        }
    )
    out.name = da.name
    return out


def depth_integrated_soil_water_mm(
    da: xr.DataArray,
    depth_range_m: tuple[float, float],
    *,
    vertical_dim: str = "levgrnd",
    layer_bounds_m: xr.DataArray | np.ndarray | Sequence[float] | None = None,
    min_coverage_fraction: float = 0.999,
) -> xr.DataArray:
    """Integrate volumetric soil moisture into water-height equivalent (mm)."""
    mean = depth_weighted_soil_moisture(
        da,
        depth_range_m,
        vertical_dim=vertical_dim,
        layer_bounds_m=layer_bounds_m,
        min_coverage_fraction=min_coverage_fraction,
    )
    thickness_m = float(depth_range_m[1]) - float(depth_range_m[0])
    out = mean * thickness_m * 1000.0
    out.attrs.update(mean.attrs)
    out.attrs["units"] = "mm"
    out.attrs["vertical_aggregation"] = "layer-overlap integrated water equivalent"
    out.name = da.name
    return out


def get_land_variable_spec(field: str) -> LandVariableSpec:
    """Return metadata for a supported field, with a useful error if unknown."""
    try:
        return LAND_VARIABLES[field.upper()]
    except (AttributeError, KeyError) as exc:
        valid = ", ".join(LAND_VARIABLES)
        raise ValueError(f"Unsupported land field {field!r}. Available fields: {valid}") from exc


def prepare_land_field(
    data: xr.Dataset | xr.DataArray,
    field: str,
    *,
    soil_layer: int | None = None,
    soil_depth_m: float | None = None,
    soil_depth_range_m: tuple[float, float] | None = None,
    soil_layer_bounds_m: xr.DataArray | np.ndarray | Sequence[float] | None = None,
    min_soil_coverage_fraction: float = 0.999,
    soil_output: str = "volumetric_mean",
) -> xr.DataArray:
    """Select and standardize one land field for map-based verification.

    ``H2OSNO`` and ``TWS`` are already two-dimensional map fields. ``H2OSOI``
    has a ``levgrnd`` dimension, so its vertical treatment must be explicit.
    Production comparisons should pass ``soil_depth_range_m`` to form an
    overlap/thickness-weighted mean matching the observation's depth support.
    Layer-index and nearest-depth selection remain available for exploration.

    Unweighted averaging across soil layers is never performed.
    """
    spec = get_land_variable_spec(field)
    if isinstance(data, xr.Dataset):
        if spec.field not in data:
            raise KeyError(f"Dataset does not contain {spec.field!r}")
        da = data[spec.field]
    elif isinstance(data, xr.DataArray):
        da = data
    else:
        raise TypeError("data must be an xarray Dataset or DataArray")

    if spec.vertical_dim is None:
        if any(value is not None for value in (soil_layer, soil_depth_m, soil_depth_range_m)):
            raise ValueError(f"soil vertical options only apply to H2OSOI, not {spec.field}")
    else:
        vdim = spec.vertical_dim
        if vdim not in da.dims:
            raise ValueError(f"{spec.field} must contain vertical dimension {vdim!r}")
        selections = [
            soil_layer is not None,
            soil_depth_m is not None,
            soil_depth_range_m is not None,
        ]
        if sum(selections) != 1:
            raise ValueError(
                "H2OSOI requires exactly one of soil_layer, soil_depth_m, or "
                "soil_depth_range_m"
            )

        if soil_depth_range_m is not None:
            soil_processors = {
                "volumetric_mean": depth_weighted_soil_moisture,
                "water_equivalent_mm": depth_integrated_soil_water_mm,
            }
            if soil_output not in soil_processors:
                raise ValueError(
                    f"soil_output must be one of {sorted(soil_processors)}, got {soil_output!r}"
                )
            da = soil_processors[soil_output](
                da,
                soil_depth_range_m,
                vertical_dim=vdim,
                layer_bounds_m=soil_layer_bounds_m,
                min_coverage_fraction=min_soil_coverage_fraction,
            )
        elif soil_depth_m is not None:
            if not np.isfinite(soil_depth_m) or soil_depth_m < 0:
                raise ValueError("soil_depth_m must be a finite, non-negative depth")
            da = da.sel({vdim: float(soil_depth_m)}, method="nearest")
        else:
            layer = 0 if soil_layer is None else soil_layer
            if not isinstance(layer, (int, np.integer)):
                raise TypeError("soil_layer must be an integer index")
            if not -da.sizes[vdim] <= int(layer) < da.sizes[vdim]:
                raise IndexError(
                    f"soil_layer={layer} is outside {vdim} with size {da.sizes[vdim]}"
                )
            da = da.isel({vdim: int(layer)})

    out = da.rename(spec.field)
    out.attrs = dict(da.attrs)
    out.attrs.setdefault("long_name", spec.long_name)
    out.attrs.setdefault("units", spec.units)
    if spec.vertical_dim and spec.vertical_dim in out.coords:
        depth = float(out[spec.vertical_dim])
        out.attrs["selected_soil_depth_m"] = depth
        out.attrs["soil_layer_selection"] = "nearest_depth" if soil_depth_m is not None else "index"
    return out


def load_e3sm_land_monthly(
    *,
    data_dir: str,
    case_prefix: str,
    members: Sequence[str],
    init_tags: Sequence[str],
    field: str,
    nlead: int = 24,
    chunks: dict[str, int] | None = None,
    grid: str = "180x360_aave",
    freq: str = "monthly",
    ts_split: str = "2yr",
    engine: str = "netcdf4",
    require_all_members: bool = True,
    verify_coverage: bool = True,
) -> xr.Dataset:
    """Load a native E3SM land hindcast as ``(Y, L, M, ...)``.

    This is a thin, land-safe wrapper around :func:`get_monthly_data`; in
    particular it fixes ``realm='lnd'`` so an atmospheric directory cannot be
    selected accidentally.
    """
    spec = get_land_variable_spec(field)
    return data_access_e3sm.get_monthly_data(
        data_dir=data_dir,
        case_prefix=case_prefix,
        members=list(members),
        init_tags=list(init_tags),
        field=spec.field,
        nlead=nlead,
        chunks=chunks,
        realm="lnd",
        grid=grid,
        freq=freq,
        ts_split=ts_split,
        engine=engine,
        require_all_members=require_all_members,
        verify_field_name=True,
        verify_coverage=verify_coverage,
    )


def seasonal_land_hindcast(
    monthly: xr.Dataset | xr.DataArray,
    field: str,
    *,
    soil_layer: int | None = None,
    soil_depth_m: float | None = None,
    soil_depth_range_m: tuple[float, float] | None = None,
    soil_layer_bounds_m: xr.DataArray | np.ndarray | Sequence[float] | None = None,
    min_soil_coverage_fraction: float = 0.999,
    soil_output: str = "volumetric_mean",
) -> xr.DataArray:
    """Select a land map field and form centered DJF/MAM/JJA/SON means."""
    return seasonal_land_hindcast_dataset(
        monthly,
        field,
        soil_layer=soil_layer,
        soil_depth_m=soil_depth_m,
        soil_depth_range_m=soil_depth_range_m,
        soil_layer_bounds_m=soil_layer_bounds_m,
        min_soil_coverage_fraction=min_soil_coverage_fraction,
        soil_output=soil_output,
    )[get_land_variable_spec(field).field]


def seasonal_land_hindcast_dataset(
    monthly: xr.Dataset | xr.DataArray,
    field: str,
    *,
    soil_layer: int | None = None,
    soil_depth_m: float | None = None,
    soil_depth_range_m: tuple[float, float] | None = None,
    soil_layer_bounds_m: xr.DataArray | np.ndarray | Sequence[float] | None = None,
    min_soil_coverage_fraction: float = 0.999,
    soil_output: str = "volumetric_mean",
) -> xr.Dataset:
    """Return a seasonal land field together with its ``(Y, L)`` valid time."""
    da = prepare_land_field(
        monthly,
        field,
        soil_layer=soil_layer,
        soil_depth_m=soil_depth_m,
        soil_depth_range_m=soil_depth_range_m,
        soil_layer_bounds_m=soil_layer_bounds_m,
        min_soil_coverage_fraction=min_soil_coverage_fraction,
        soil_output=soil_output,
    )
    work = da.to_dataset(name=da.name)
    # The E3SM hindcast loader stores verification time as a (Y, L) data
    # variable rather than a coordinate, so DataArray selection does not carry
    # it along. Preserve it explicitly for the calendar utility.
    if "time" not in work and isinstance(monthly, xr.Dataset) and "time" in monthly:
        work["time"] = monthly["time"]
    if "time" not in work:
        raise ValueError("monthly input must provide verification time")
    seasonal = calendar_utils.mon_to_seas_dask(work)
    seasonal[da.name].attrs.update(da.attrs)
    seasonal[da.name].attrs["temporal_average"] = "centered 3-month seasonal mean"
    seasonal_field, seasonal_time, dropped = retain_valid_seasonal_leads(
        seasonal[da.name], seasonal["time"]
    )
    seasonal = seasonal_field.to_dataset(name=da.name)
    seasonal["time"] = seasonal_time
    seasonal[da.name].attrs["dropped_all_missing_leads"] = ",".join(map(str, dropped))
    return seasonal


def retain_valid_seasonal_leads(
    data: xr.DataArray,
    valid_time: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, list[int]]:
    """Drop only leads with no finite model values anywhere."""
    if "L" not in data.dims or "L" not in valid_time.dims:
        raise ValueError("data and valid_time must both contain an L dimension")
    if not np.array_equal(np.asarray(data.L), np.asarray(valid_time.L)):
        raise ValueError("data and valid_time must have identical L coordinates")

    finite = data.notnull()
    for dim in tuple(dim for dim in finite.dims if dim != "L"):
        finite = finite.any(dim)
    if finite.chunks is not None:
        finite = finite.compute()
    keep = data.L.where(finite, drop=True)
    dropped = data.L.where(~finite, drop=True)
    if keep.size == 0:
        raise ValueError("no finite seasonal leads are available")
    return (
        data.sel(L=keep),
        valid_time.sel(L=keep),
        [int(value) for value in dropped.values],
    )


def validate_hindcast_evaluation_setup(
    forecast: xr.DataArray,
    valid_time: xr.DataArray,
    *,
    init_month: int,
    initialization_years: tuple[int, int],
    expected_members: Sequence[str],
    climatology_years: tuple[int, int],
    require_complete_member_grid: bool = True,
) -> dict[int, list[int]]:
    """Validate the evaluation contract used by the refactored ACC workflow.

    The checks are intentionally stricter than dimensional validation: the
    requested initialization years and member labels must be exact, every lead
    must have one verification month and a complete climatology window, and
    represented grid cells may not silently average fewer members.

    Returns
    -------
    dict
        Target years keyed by seasonal lead.
    """
    required = {"Y", "L", "M"}
    missing = required - set(forecast.dims)
    if missing:
        raise ValueError(f"forecast is missing required dimensions: {sorted(missing)}")
    if set(valid_time.dims) != {"Y", "L"}:
        raise ValueError("valid_time must have exactly Y and L dimensions")
    if not np.array_equal(forecast.Y, valid_time.Y):
        raise ValueError("forecast and valid_time Y coordinates differ")
    if not np.array_equal(forecast.L, valid_time.L):
        raise ValueError("forecast and valid_time L coordinates differ")

    y0, y1 = map(int, initialization_years)
    expected_years = list(range(y0, y1 + 1))
    labels = [str(value) for value in forecast.Y.values]
    try:
        actual_years = [int(value[:4]) for value in labels]
        actual_months = [int(value[4:6]) for value in labels]
    except (TypeError, ValueError) as exc:
        raise ValueError("Y labels must begin with YYYYMM initialization tags") from exc
    if actual_years != expected_years:
        raise ValueError(
            f"initialization years differ: expected {y0}-{y1}, got "
            f"{actual_years[0] if actual_years else 'empty'}-"
            f"{actual_years[-1] if actual_years else 'empty'}"
        )
    if set(actual_months) != {int(init_month)}:
        raise ValueError(
            f"initialization month differs: expected {init_month}, got "
            f"{sorted(set(actual_months))}"
        )

    actual_members = [str(value) for value in forecast.M.values]
    expected_members = [str(value) for value in expected_members]
    if actual_members != expected_members:
        raise ValueError(
            f"ensemble members differ: expected {expected_members}, got {actual_members}"
        )

    target_years = {}
    climy0, climy1 = map(int, climatology_years)
    for lead in map(int, forecast.L.values):
        times = valid_time.sel(L=lead)
        years = np.asarray(times.dt.year.values, dtype=int)
        months = np.asarray(times.dt.month.values, dtype=int)
        if len(np.unique(months)) != 1:
            raise ValueError(
                f"lead {lead} has multiple verification months: "
                f"{sorted(np.unique(months).tolist())}"
            )
        if len(np.unique(years)) != len(expected_years):
            raise ValueError(f"lead {lead} does not have one unique target year per initialization")
        clim_count = int(((years >= climy0) & (years <= climy1)).sum())
        available_clim_years = [
            year for year in years.tolist() if climy0 <= year <= climy1
        ]
        if clim_count != len(available_clim_years) or clim_count < 3:
            raise ValueError(
                f"lead {lead} has an invalid model climatology cohort in the "
                f"{climy0}-{climy1} verification-time window"
            )
        target_years[lead] = years.tolist()

    if require_complete_member_grid:
        member_count = forecast.notnull().sum("M")
        partial = ((member_count > 0) & (member_count < len(expected_members))).any()
        if partial.chunks is not None:
            partial = partial.compute()
        if bool(partial.item()):
            raise ValueError(
                "at least one represented year/lead/grid cell contains fewer "
                "than the requested ensemble members"
            )

    return target_years


def validate_reference_time_coverage(
    reference: xr.DataArray,
    target_years_by_lead: Mapping[int, Sequence[int]],
    valid_time: xr.DataArray,
    climatology_years: tuple[int, int],
) -> None:
    """Require unique reference seasons for all evaluation and climo years."""
    if "time" not in reference.dims:
        raise ValueError("reference must contain time")
    ref_years = np.asarray(reference.time.dt.year.values, dtype=int)
    ref_months = np.asarray(reference.time.dt.month.values, dtype=int)
    pairs = list(zip(ref_years.tolist(), ref_months.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("reference contains duplicate year/month timestamps")

    finite = reference.notnull()
    spatial_dims = tuple(dim for dim in finite.dims if dim != "time")
    if spatial_dims:
        finite = finite.any(spatial_dims)
    if finite.chunks is not None:
        finite = finite.compute()
    available = {
        (year, month)
        for year, month, valid in zip(ref_years, ref_months, finite.values)
        if bool(valid)
    }

    climy0, climy1 = map(int, climatology_years)
    for lead, years in target_years_by_lead.items():
        months = np.unique(valid_time.sel(L=int(lead)).dt.month.values)
        if len(months) != 1:
            raise ValueError(f"lead {lead} does not have one verification month")
        month = int(months[0])
        required = {(int(year), month) for year in years}
        required.update((year, month) for year in range(climy0, climy1 + 1))
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"reference lacks {len(missing)} required seasons for lead {lead}: "
                f"{missing[:5]}"
            )


def validate_land_reference_compatibility(
    forecast: xr.DataArray,
    reference: xr.DataArray,
) -> None:
    """Reject incompatible grids or soil-depth definitions before scoring."""
    for coord in ("lat", "lon"):
        if coord in forecast.coords or coord in reference.coords:
            if coord not in forecast.coords or coord not in reference.coords:
                raise ValueError(f"forecast and reference must both contain {coord}")
            forecast_coord = np.asarray(forecast[coord])
            reference_coord = np.asarray(reference[coord])
            if (
                forecast_coord.shape != reference_coord.shape
                or not np.array_equal(forecast_coord, reference_coord)
            ):
                raise ValueError(f"forecast and reference {coord} coordinates are not identical")

    if forecast.name == "H2OSOI" or reference.name == "H2OSOI":
        depth_attrs = ("depth_top_m", "depth_bottom_m")
        for name, data in (("forecast", forecast), ("reference", reference)):
            missing = [attr for attr in depth_attrs if attr not in data.attrs]
            if missing:
                raise ValueError(
                    f"{name} H2OSOI is missing depth metadata {missing}; "
                    "vertically harmonize it before computing skill"
                )
        for attr in depth_attrs:
            if not np.isclose(
                float(forecast.attrs[attr]),
                float(reference.attrs[attr]),
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError(
                    "forecast and reference H2OSOI depth intervals differ: "
                    f"forecast={forecast.attrs['depth_top_m']}-"
                    f"{forecast.attrs['depth_bottom_m']} m, reference="
                    f"{reference.attrs['depth_top_m']}-"
                    f"{reference.attrs['depth_bottom_m']} m"
                )
        forecast_units = forecast.attrs.get("units")
        reference_units = reference.attrs.get("units")
        if forecast_units is None or reference_units is None:
            raise ValueError("forecast and reference H2OSOI must declare volumetric units")
        if str(forecast_units).lower() != str(reference_units).lower():
            raise ValueError(
                "forecast and reference H2OSOI units differ: "
                f"{forecast_units!r} versus {reference_units!r}"
            )


def compute_land_acc_skill(
    forecast: xr.DataArray,
    valid_time: xr.DataArray,
    reference: xr.DataArray,
    climy0: int,
    climy1: int,
    *,
    nleads: int | None = None,
    detrend: bool = True,
    reference_is_anomaly: bool = False,
    target_years_by_lead=None,
) -> xr.Dataset:
    """Remove lead-dependent model drift and compute seasonal map skill.

    Inputs must already be on the same horizontal grid and use seasonal means,
    matching the analysis stage of the reference notebook. The returned
    Dataset includes ACC as ``corr`` plus p-values and the other deterministic
    metrics produced by :func:`esp_lab.stats.compute_skill_seasonal`.
    """
    required_forecast = {"Y", "L", "M"}
    missing = required_forecast - set(forecast.dims)
    if missing:
        raise ValueError(f"forecast is missing required dimensions: {sorted(missing)}")
    if not {"Y", "L"}.issubset(valid_time.dims):
        raise ValueError("valid_time must contain Y and L dimensions")
    if "time" not in reference.dims:
        raise ValueError("reference must contain a time dimension")
    if "levgrnd" in forecast.dims:
        raise ValueError("forecast still has levgrnd; select an H2OSOI layer first")
    validate_land_reference_compatibility(forecast, reference)
    if nleads is None:
        nleads = int(forecast.sizes["L"])
    if not 1 <= nleads <= forecast.sizes["L"]:
        raise ValueError(f"nleads must be between 1 and {forecast.sizes['L']}")

    model_anom, _ = stats.remove_drift(forecast, valid_time, climy0, climy1)
    skill = stats.compute_skill_seasonal(
        model_anom,
        valid_time,
        reference,
        climy0,
        climy1,
        nleadavg=1,
        nleads=nleads,
        resamp=0,
        detrend=detrend,
        is_anomaly=reference_is_anomaly,
        target_years_by_lead=target_years_by_lead,
    )
    skill.attrs.update(
        {
            "climatology": f"{climy0}-{climy1}",
            "detrend": str(bool(detrend)).lower(),
            "metric": "seasonal lead-time ACC",
        }
    )
    if forecast.name:
        skill.attrs["field"] = forecast.name
    return skill


def plot_land_acc_maps(
    skill: xr.Dataset,
    *,
    leads: Sequence[int] | None = None,
    significance_level: float | None = None,
    ncols: int = 2,
    cmap: str = "RdBu_r",
):
    """Plot ACC maps for selected leads and return ``(figure, axes)``.

    Cartopy and matplotlib are imported lazily so calculation-only workflows do
    not need to initialize a plotting stack.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    if "corr" not in skill or "L" not in skill["corr"].dims:
        raise ValueError("skill must contain corr with an L dimension")
    if not {"lat", "lon"}.issubset(skill["corr"].dims):
        raise ValueError("corr must contain lat and lon dimensions")
    selected = list(skill.L.values if leads is None else leads)
    if not selected:
        raise ValueError("at least one lead must be selected")
    unknown = [lead for lead in selected if lead not in skill.L.values]
    if unknown:
        raise KeyError(f"lead values not present in skill: {unknown}")
    if significance_level is not None and not 0 < significance_level < 1:
        raise ValueError("significance_level must be between 0 and 1")

    ncols = max(1, min(int(ncols), len(selected)))
    nrows = int(np.ceil(len(selected) / ncols))
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 2.9 * nrows),
        subplot_kw={"projection": projection},
        squeeze=False,
    )
    mappable = None
    for ax, lead in zip(axes.ravel(), selected):
        corr = skill["corr"].sel(L=lead)
        if significance_level is not None:
            if "pval" not in skill:
                raise ValueError("significance masking requires skill['pval']")
            corr = corr.where(skill["pval"].sel(L=lead) < significance_level)
        mappable = corr.plot.pcolormesh(
            ax=ax,
            transform=projection,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            add_colorbar=False,
        )
        ax.coastlines(linewidth=0.6)
        ax.set_title(f"Lead {lead}")
    for ax in axes.ravel()[len(selected) :]:
        ax.set_visible(False)
    fig.colorbar(mappable, ax=list(axes.ravel()[: len(selected)]), label="ACC", shrink=0.85)
    return fig, axes


__all__ = [
    "LAND_VARIABLES",
    "LandVariableSpec",
    "compute_land_acc_skill",
    "depth_integrated_soil_water_mm",
    "depth_weighted_soil_moisture",
    "elm_soil_layer_bounds",
    "get_land_variable_spec",
    "load_e3sm_land_monthly",
    "plot_land_acc_maps",
    "prepare_land_field",
    "seasonal_land_hindcast",
    "seasonal_land_hindcast_dataset",
    "retain_valid_seasonal_leads",
    "validate_land_reference_compatibility",
]
