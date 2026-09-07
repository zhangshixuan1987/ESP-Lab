"""Preparation and cache contracts for land lead-time ACC inputs.

The functions deliberately prepare one reference or one model
case/initialization month at a time so an interrupted workflow can restart
without reopening unrelated raw archives.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Mapping, Sequence
import uuid

import numpy as np
import xarray as xr

from . import data_access_e3sm, land_skill
from .leadtime_skill_cache import (
    canonical_json,
    file_inventory_digest,
    provenance_digest,
    source_fingerprint,
)
from .prepared_skill import cache_status
from .utils import regrid_utils as regrid
from .utils.netcdf_utils import atomic_to_netcdf


_LAND_PREPARED_CACHE_SCHEMA_VERSION = "1"
_PROCESSING_STAGE = "analysis_ready_land_acc_input_v2"
_SEASONAL_CONVENTION = "centered_3month_DJF_MAM_JJA_SON"


def target_grid(dlat: float = 1.0, dlon: float = 1.0) -> xr.Dataset:
    """Return a cell-centred regular latitude/longitude target grid."""
    if (
        dlat <= 0
        or dlon <= 0
        or not np.isclose(180.0 / dlat, round(180.0 / dlat))
        or not np.isclose(360.0 / dlon, round(360.0 / dlon))
    ):
        raise ValueError("dlat and dlon must divide 180 and 360, respectively")
    return xr.Dataset(
        coords={
            "lat": np.arange(-90.0 + dlat / 2.0, 90.0, dlat),
            "lon": np.arange(dlon / 2.0, 360.0, dlon),
        }
    )


def target_grid_tag(dlat: float = 1.0, dlon: float = 1.0) -> str:
    """Return the readable grid tag used by staged land-input paths."""
    return f"{dlat:g}x{dlon:g}deg_cell_centered"


def is_target_grid(data: xr.Dataset | xr.DataArray, grid: xr.Dataset) -> bool:
    """Return whether latitude and longitude exactly match ``grid``."""
    return (
        "lat" in data.coords
        and "lon" in data.coords
        and np.array_equal(data.lat.values, grid.lat.values)
        and np.array_equal(data.lon.values, grid.lon.values)
    )


def resolve_reference_files(path_pattern: str) -> list[Path]:
    """Resolve an explicit path or glob into a stable, non-empty inventory."""
    text = str(Path(path_pattern).expanduser()) if path_pattern else ""
    paths = sorted(Path(path) for path in glob.glob(text)) if any(
        char in text for char in "*?[]"
    ) else ([Path(text)] if text and Path(text).is_file() else [])
    if not paths:
        raise FileNotFoundError(f"No raw reference files match {path_pattern!r}")
    return paths


def resolve_model_files(
    *,
    data_dir: str | Path,
    case_prefix: str,
    members: Sequence[str],
    years: Sequence[int],
    init_month: int,
    field: str,
    nlead: int,
) -> list[Path]:
    """Resolve the exact E3SM files required by one case/month preparation."""
    init_tags = data_access_e3sm.build_init_tags(years, init_month)
    nested, valid = data_access_e3sm.nested_file_list_by_init(
        data_dir=str(data_dir), case_prefix=case_prefix,
        members=list(members), init_tags=init_tags, field=field,
        realm="lnd", grid="180x360_aave", freq="monthly", ts_split="2yr",
        require_all_members=True, verify_field_name=True,
        verify_coverage=True, nlead=nlead,
    )
    if valid != init_tags:
        missing = sorted(set(init_tags) - set(valid))
        raise ValueError(
            f"Incomplete raw E3SM inventory for {case_prefix}, init={init_month}: "
            f"missing {missing}"
        )
    return [Path(path) for files in nested for path in files]


def raw_source_identity(
    *,
    paths: Sequence[str | Path],
    logical_identity: Mapping,
    source_revision: str,
    mode: str,
    inventory_root: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
) -> str:
    """Fingerprint inputs by inventory, saved inventory snapshot, or revision.

    Inventory mode preserves established land-cache identities and can record
    the resolved identity for an archive-free restart. Snapshot mode only
    accepts a record matching the exact logical request and revision token.
    """
    if mode not in {"inventory", "snapshot", "revision"}:
        raise ValueError(
            "source_identity_mode must be 'inventory', 'snapshot', or 'revision'"
        )
    revision = str(source_revision)
    source_key = source_fingerprint(logical_identity, source_revision=revision)
    snapshot_path = (
        Path(snapshot_dir) / f"{source_key.split(':')[-1]}.json"
        if snapshot_dir is not None else None
    )
    if mode == "snapshot":
        if snapshot_path is None:
            raise ValueError("snapshot_dir is required in snapshot mode")
        try:
            record = json.loads(snapshot_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"No readable land source inventory snapshot at {snapshot_path}; "
                "run once in inventory mode with archive access."
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid land source inventory snapshot: {snapshot_path}")
        identity = record.get("source_data_identity", "")
        if not isinstance(identity, str) or not identity.startswith("sha256:"):
            raise ValueError(f"Invalid land source inventory snapshot: {snapshot_path}")
        checksum = source_fingerprint(
            {"source_key": source_key}, source_revision=identity
        )
        if (record.get("source_key") != source_key or
                record.get("checksum") != checksum):
            raise ValueError(f"Invalid land source inventory snapshot: {snapshot_path}")
        return identity
    if mode == "inventory":
        revision += "|" + file_inventory_digest(list(paths), root=inventory_root)
    identity = source_fingerprint(logical_identity, source_revision=revision)
    if mode == "inventory" and snapshot_path is not None:
        record = {
            "source_key": source_key,
            "source_data_identity": identity,
            "checksum": source_fingerprint(
                {"source_key": source_key}, source_revision=identity
            ),
        }
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot_path.with_name(
            f".{snapshot_path.name}.tmp.{uuid.uuid4().hex}"
        )
        try:
            temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
            os.replace(temporary, snapshot_path)
        finally:
            temporary.unlink(missing_ok=True)
    return identity


def expected_attrs(
    *,
    field: str,
    source_kind: str,
    source_name: str,
    source_data_identity: str,
    initialization_years: tuple[int, int],
    climatology_years: tuple[int, int],
    ensemble_members: Sequence[str],
    monthly_nlead: int,
    target_grid_name: str,
    regridding_method: str,
    output_units: str,
    reference_is_anomaly: bool,
    init_month: int | None = None,
    case_prefix: str = "",
    depth_range_m: tuple[float, float] | None = None,
    reference_month_policy: str = "all seasonal center months",
) -> dict[str, str | int | float]:
    """Build the complete compatibility contract for one prepared input."""
    field = field.upper()
    y0, y1 = map(int, initialization_years)
    c0, c1 = map(int, climatology_years)
    attrs: dict[str, str | int | float] = {
        "land_prepared_cache_version": _LAND_PREPARED_CACHE_SCHEMA_VERSION,
        "processing_stage": _PROCESSING_STAGE,
        "field": field,
        "source_kind": str(source_kind),
        "source_name": str(source_name),
        "source_data_identity": str(source_data_identity),
        "case_prefix": str(case_prefix),
        "horizontal_grid": str(target_grid_name),
        "regridding_method": str(regridding_method),
        "seasonal_convention": _SEASONAL_CONVENTION,
        "reference_is_anomaly": str(bool(reference_is_anomaly)).lower(),
        "initialization_year_start": y0,
        "initialization_year_end": y1,
        "initialization_year_count": y1 - y0 + 1,
        "climatology_year_start": c0,
        "climatology_year_end": c1,
        "ensemble_members": canonical_json(list(map(str, ensemble_members))),
        "ensemble_member_count": len(ensemble_members),
        "monthly_lead_count": int(monthly_nlead),
        "output_units": str(output_units),
        "unit_conversion": "configured_scale_offset_v1",
        "reference_month_policy": str(reference_month_policy),
    }
    if source_kind == "model":
        if init_month is None or not 1 <= int(init_month) <= 12:
            raise ValueError("model inputs require init_month in 1..12")
        attrs["initialization_month"] = int(init_month)
    elif source_kind != "reference":
        raise ValueError("source_kind must be 'reference' or 'model'")
    if field == "H2OSOI":
        if depth_range_m is None:
            raise ValueError("H2OSOI requires depth_range_m")
        top, bottom = map(float, depth_range_m)
        attrs.update({
            "depth_top_m": top,
            "depth_bottom_m": bottom,
            "vertical_aggregation": f"{top:g}-{bottom:g}m water-height equivalent",
        })
    return attrs


def prepared_data_identity(attrs: Mapping) -> str:
    """Return a stable identity for validated prepared data provenance."""
    return f"sha256:{provenance_digest(dict(attrs), length=32)}"


def validate_dataset(
    dataset: xr.Dataset,
    expected: Mapping[str, str | int | float],
    *,
    grid: xr.Dataset,
) -> None:
    """Validate variables, metadata, dimensions, and target coordinates."""
    field = str(expected["field"])
    source_kind = str(expected["source_kind"])
    required = [field] + (["time"] if source_kind == "model" else [])
    from .prepared_skill import validate_cache_attrs

    validate_cache_attrs(dataset, expected, required_variables=required)
    da = dataset[field]
    if not {"lat", "lon"}.issubset(da.dims) or not is_target_grid(da, grid):
        raise ValueError("Prepared input does not match the configured target grid")
    if source_kind == "reference" and "time" not in da.dims:
        raise ValueError("Prepared reference must contain a time dimension")
    if source_kind == "model":
        if not {"Y", "L", "M"}.issubset(da.dims):
            raise ValueError("Prepared model field must contain Y, L, and M dimensions")
        if set(dataset.time.dims) != {"Y", "L"}:
            raise ValueError("Prepared model time must have exactly Y and L dimensions")
        if da.sizes["Y"] != int(expected["initialization_year_count"]):
            raise ValueError("Prepared model initialization-year count is incompatible")
        if da.sizes["M"] != int(expected["ensemble_member_count"]):
            raise ValueError("Prepared model ensemble-member count is incompatible")


def prepared_cache_status(path, expected, *, grid: xr.Dataset) -> tuple[bool, str]:
    """Return whether one prepared land cache satisfies its exact contract."""
    ok, reason = cache_status(
        path, expected_attrs=expected,
        required_variables=[str(expected["field"])]
        + (["time"] if expected["source_kind"] == "model" else []),
    )
    if not ok:
        return ok, reason
    try:
        with xr.open_dataset(path) as dataset:
            validate_dataset(dataset, expected, grid=grid)
    except Exception as exc:
        return False, str(exc)
    return True, "compatible"


def standardize_reference_coordinates(da: xr.DataArray) -> xr.DataArray:
    """Normalize reference coordinates to ascending ``lat`` and 0..360 ``lon``."""
    rename = {
        source: target for source, target in (
            ("latitude", "lat"), ("longitude", "lon")
        ) if source in da.dims or source in da.coords
    }
    da = da.rename(rename)
    missing = {"time", "lat", "lon"} - (set(da.dims) | set(da.coords))
    if missing:
        raise ValueError(f"Reference is missing coordinates: {sorted(missing)}")
    return da.assign_coords(lon=da.lon % 360).sortby("lon").sortby("lat")


def _load_reference(paths: Sequence[Path], cfg: Mapping, field: str,
                    depth_range_m: tuple[float, float] | None):
    dataset = (
        xr.open_mfdataset([str(path) for path in paths], combine="by_coords")
        if len(paths) > 1 else xr.open_dataset(paths[0])
    )
    if cfg["variable"] not in dataset:
        dataset.close()
        raise KeyError(f"{cfg['variable']!r} not found in raw reference")
    da = standardize_reference_coordinates(dataset[cfg["variable"]]).chunk(
        {"time": 24, "lat": 90, "lon": 180}
    )
    da = (da * float(cfg.get("scale", 1.0)) + float(cfg.get("offset", 0.0))).rename(field)
    if cfg.get("mask_negative_categorical_flags", False):
        da = land_skill.mask_c3s_swe_flags(da)
    da.attrs["units"] = str(cfg["output_units"])
    if field == "H2OSOI":
        vertical_dim = cfg.get("vertical_dim")
        if vertical_dim:
            bounds_name = cfg.get("layer_bounds_variable")
            if not bounds_name or bounds_name not in dataset:
                dataset.close()
                raise ValueError("Layered reference requires layer_bounds_variable")
            da = land_skill.depth_integrated_soil_water_mm(
                da, depth_range_m, vertical_dim=vertical_dim,
                layer_bounds_m=dataset[bounds_name],
            )
        elif not np.allclose(
            cfg.get("represented_depth_range_m"), depth_range_m,
            rtol=0.0, atol=1e-6,
        ):
            dataset.close()
            raise ValueError("Reference depth differs from configured model depth")
        else:
            da.attrs["depth_top_m"], da.attrs["depth_bottom_m"] = depth_range_m
    original_attrs = dict(da.attrs)
    if not cfg.get("already_seasonal", False):
        if cfg.get("require_complete_calendar_months", False):
            da = land_skill.complete_calendar_seasonal_mean(
                da, retain_missing=cfg.get("retain_missing_seasons", False)
            )
        else:
            da = da.rolling(time=3, center=True, min_periods=3).mean()
            da = da.where(da.time.dt.month.isin([1, 4, 7, 10]), drop=True)
            da = da.dropna("time", how="all")
        da.attrs.update(original_attrs)
    return dataset, da


def prepare_reference_cache(
    *,
    paths: Sequence[Path],
    cfg: Mapping,
    field: str,
    depth_range_m: tuple[float, float] | None,
    grid: xr.Dataset,
    expected: Mapping,
    output_path: str | Path,
    write_options: Mapping | None = None,
) -> Path:
    """Prepare, validate, and atomically write one reference cache."""
    source, native = _load_reference(paths, cfg, field, depth_range_m)
    regridder = None
    try:
        if is_target_grid(native, grid):
            ready = native.copy()
        else:
            regridder = regrid.make_regridder(
                native.to_dataset(name=field), grid,
                method=str(expected["regridding_method"]), periodic=True,
            )
            ready = regridder(native).rename(field)
        ready.attrs.update(native.attrs)
        ready.attrs.update(expected)
        ready.attrs["documentation"] = str(cfg.get("documentation", ""))
        dataset = ready.to_dataset(name=field)
        dataset.attrs.update(expected)
        validate_dataset(dataset, expected, grid=grid)
        return atomic_to_netcdf(dataset, output_path, **dict(write_options or {}))
    finally:
        source.close()


def prepare_model_cache(
    *,
    data_dir: str | Path,
    case_prefix: str,
    members: Sequence[str],
    years: Sequence[int],
    init_month: int,
    field: str,
    monthly_nlead: int,
    monthly_chunks: Mapping[str, int],
    depth_range_m: tuple[float, float] | None,
    reference: xr.DataArray,
    restrict_to_reference_months: bool,
    climatology_years: tuple[int, int],
    strict_member_completeness: bool,
    grid: xr.Dataset,
    expected: Mapping,
    output_path: str | Path,
    write_options: Mapping | None = None,
) -> Path:
    """Prepare, validate, and atomically write one E3SM case/month cache."""
    monthly = land_skill.load_e3sm_land_monthly(
        data_dir=str(data_dir), case_prefix=case_prefix,
        members=list(members),
        init_tags=data_access_e3sm.build_init_tags(years, init_month),
        field=field, nlead=monthly_nlead, chunks=dict(monthly_chunks),
    )
    try:
        selection = ({
            "soil_depth_range_m": depth_range_m,
            "soil_output": "water_equivalent_mm",
        } if field == "H2OSOI" else {})
        seasonal = land_skill.seasonal_land_hindcast_dataset(
            monthly, field, **selection
        )
        if restrict_to_reference_months:
            supported, valid_time, dropped = land_skill.retain_reference_supported_leads(
                seasonal[field], seasonal.time, reference
            )
            seasonal = supported.to_dataset(name=field)
            seasonal["time"] = valid_time
            seasonal[field].attrs["dropped_reference_unsupported_leads"] = ",".join(
                map(str, dropped)
            )
        land_skill.validate_hindcast_evaluation_setup(
            seasonal[field], seasonal.time, init_month=init_month,
            initialization_years=(int(years[0]), int(years[-1])),
            expected_members=members, climatology_years=climatology_years,
            require_complete_member_grid=strict_member_completeness,
        )
        if is_target_grid(seasonal[field], grid):
            ready_field = seasonal[field].copy()
        else:
            regridder = regrid.make_regridder(
                seasonal, grid, method=str(expected["regridding_method"]), periodic=True
            )
            ready_field = regridder(seasonal[field]).rename(field)
        ready_field.attrs.update(seasonal[field].attrs)
        ready_field.attrs.update(expected)
        dataset = ready_field.to_dataset(name=field)
        dataset["time"] = seasonal.time
        dataset.attrs.update(expected)
        validate_dataset(dataset, expected, grid=grid)
        return atomic_to_netcdf(dataset, output_path, **dict(write_options or {}))
    finally:
        monthly.close()


__all__ = [
    "expected_attrs", "is_target_grid", "prepare_model_cache",
    "prepare_reference_cache", "prepared_cache_status", "prepared_data_identity",
    "raw_source_identity", "resolve_model_files", "resolve_reference_files",
    "standardize_reference_coordinates", "target_grid", "target_grid_tag",
]
