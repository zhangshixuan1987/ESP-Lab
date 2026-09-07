"""Versioned cache contracts for lead-time skill workflows."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Iterable, Mapping

import xarray as xr

from esp_lab.paths import leadtime_acc_dir


_PREPARED_CACHE_SCHEMA_VERSION = "2"
REQUIRED_VARIABLES = {"anomaly", "climatology", "time"}
REQUIRED_ATTRS = {
    "prepared_skill_version",
    "source",
    "source_data_identity",
    "case_prefix",
    "component",
    "variable",
    "initialization_month",
    "climatology_start_year",
    "climatology_end_year",
    "requested_years",
    "requested_year_count",
    "target_grid",
    "regridding_method",
    "ensemble_member_count",
    "lead_count",
    "unit_conversion_version",
    "temporal_resolution",
}


def _token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _years_value(years: Iterable[int]) -> str:
    values = [int(year) for year in years]
    if not values:
        raise ValueError("requested_years must not be empty")
    return json.dumps(values, separators=(",", ":"))


def _provenance_digest(attrs: Mapping[str, str | int]) -> str:
    payload = json.dumps(dict(sorted(attrs.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def expected_prepared_skill_attrs(
    *,
    source: str,
    source_data_identity: str,
    case_prefix: str,
    component: str,
    variable: str,
    init_month: int,
    climatology_years: tuple[int, int],
    requested_years: Iterable[int],
    target_grid: str,
    regridding_method: str,
    ensemble_member_count: int,
    lead_count: int,
    unit_conversion_version: str,
    temporal_resolution: str = "seasonal",
) -> dict[str, str | int]:
    """Return metadata that uniquely identifies a compatible prepared cache."""
    climy0, climy1 = climatology_years
    years_value = _years_value(requested_years)
    return {
        "prepared_skill_version": _PREPARED_CACHE_SCHEMA_VERSION,
        "source": str(source),
        "source_data_identity": str(source_data_identity),
        "case_prefix": str(case_prefix),
        "component": str(component),
        "variable": str(variable),
        "initialization_month": int(init_month),
        "climatology_start_year": int(climy0),
        "climatology_end_year": int(climy1),
        "requested_years": years_value,
        "requested_year_count": len(json.loads(years_value)),
        "target_grid": str(target_grid),
        "regridding_method": str(regridding_method),
        "ensemble_member_count": int(ensemble_member_count),
        "lead_count": int(lead_count),
        "unit_conversion_version": str(unit_conversion_version),
        "temporal_resolution": str(temporal_resolution),
    }


def prepared_skill_path(
    source: str,
    component: str,
    variable: str,
    init_month: int,
    climatology_years: tuple[int, int],
    *,
    requested_years: Iterable[int],
    target_grid: str,
    regridding_method: str,
    ensemble_member_count: int,
    lead_count: int,
    unit_conversion_version: str,
    case_prefix: str,
    source_data_identity: str,
    root: str | Path,
) -> Path:
    """Return a readable, provenance-hashed prepared-cache path."""
    years = [int(year) for year in requested_years]
    attrs = expected_prepared_skill_attrs(
        source=source,
        source_data_identity=source_data_identity,
        case_prefix=case_prefix,
        component=component,
        variable=variable,
        init_month=init_month,
        climatology_years=climatology_years,
        requested_years=years,
        target_grid=target_grid,
        regridding_method=regridding_method,
        ensemble_member_count=ensemble_member_count,
        lead_count=lead_count,
        unit_conversion_version=unit_conversion_version,
    )
    climy0, climy1 = climatology_years
    source_token = _token(source)
    filename = (
        f"{source_token}_{init_month:02d}_{variable}_seasonal_anomaly_"
        f"y{years[0]}-{years[-1]}_ny{len(years)}_clim_{climy0}_{climy1}_"
        f"m{ensemble_member_count}_l{lead_count}_"
        f"{_provenance_digest(attrs)}.nc"
    )
    return leadtime_acc_dir(
        source, "prepared_skill", component, variable, filename, root=root
    )


def build_prepared_skill_dataset(
    anomaly: xr.DataArray,
    climatology: xr.DataArray,
    time: xr.DataArray,
    *,
    source: str,
    source_data_identity: str,
    case_prefix: str,
    component: str,
    variable: str,
    init_month: int,
    climatology_years: tuple[int, int],
    requested_years: Iterable[int],
    target_grid: str,
    regridding_method: str,
    ensemble_member_count: int,
    lead_count: int,
    unit_conversion_version: str,
    temporal_resolution: str = "seasonal",
) -> xr.Dataset:
    """Package model anomalies and their complete provenance contract."""
    if not {"Y", "L"} <= set(anomaly.dims):
        raise ValueError("anomaly must contain Y and L dimensions")
    if not {"Y", "L"} <= set(time.dims):
        raise ValueError("time must contain Y and L dimensions")
    if "L" not in climatology.dims:
        raise ValueError("climatology must contain an L dimension")

    anomaly, time = xr.align(anomaly, time, join="exact")
    anomaly, climatology = xr.align(anomaly, climatology, join="exact", exclude={"Y", "M"})
    attrs = expected_prepared_skill_attrs(
        source=source,
        source_data_identity=source_data_identity,
        case_prefix=case_prefix,
        component=component,
        variable=variable,
        init_month=init_month,
        climatology_years=climatology_years,
        requested_years=requested_years,
        target_grid=target_grid,
        regridding_method=regridding_method,
        ensemble_member_count=ensemble_member_count,
        lead_count=lead_count,
        unit_conversion_version=unit_conversion_version,
        temporal_resolution=temporal_resolution,
    )
    dataset = xr.Dataset(
        {
            "anomaly": anomaly.rename("anomaly"),
            "climatology": climatology.rename("climatology"),
            "time": time.rename("time"),
        },
        attrs=attrs,
    )
    dataset["anomaly"].attrs.update(anomaly.attrs)
    dataset["climatology"].attrs.update(climatology.attrs)
    return dataset


def validate_cache_attrs(
    dataset: xr.Dataset,
    expected_attrs: Mapping[str, str | int],
    *,
    required_variables: Iterable[str] = (),
) -> None:
    """Reject a generic NetCDF cache with missing data or stale provenance."""
    missing_variables = set(required_variables) - set(dataset.data_vars)
    if missing_variables:
        raise ValueError(f"Cache is missing variables: {sorted(missing_variables)}")
    missing_attrs = set(expected_attrs) - set(dataset.attrs)
    if missing_attrs:
        raise ValueError(f"Cache is missing attributes: {sorted(missing_attrs)}")
    mismatches = {
        key: (dataset.attrs.get(key), value)
        for key, value in expected_attrs.items()
        if dataset.attrs.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Cache metadata mismatch: {details}")


def validate_prepared_skill_dataset(
    dataset: xr.Dataset,
    expected_attrs: Mapping[str, str | int],
) -> None:
    """Reject incomplete or provenance-incompatible prepared inputs."""
    validate_cache_attrs(dataset, expected_attrs, required_variables=REQUIRED_VARIABLES)
    if not {"Y", "L"} <= set(dataset["anomaly"].dims):
        raise ValueError("Prepared anomaly must contain Y and L dimensions")
    if not {"Y", "L"} <= set(dataset["time"].dims):
        raise ValueError("Prepared time must contain Y and L dimensions")
    if "L" not in dataset["climatology"].dims:
        raise ValueError("Prepared climatology must contain an L dimension")
    missing_attrs = REQUIRED_ATTRS - set(dataset.attrs)
    if missing_attrs:
        raise ValueError(f"Prepared skill cache is missing attributes: {sorted(missing_attrs)}")
    if dataset["anomaly"].sizes["L"] != int(dataset.attrs["lead_count"]):
        raise ValueError("Prepared anomaly lead count does not match metadata")
    actual_members = dataset["anomaly"].sizes.get("M", 1)
    if actual_members != int(dataset.attrs["ensemble_member_count"]):
        raise ValueError("Prepared anomaly ensemble-member count does not match metadata")


def cache_status(
    path: str | Path,
    *,
    expected_attrs: Mapping[str, str | int],
    required_variables: Iterable[str] = (),
) -> tuple[bool, str]:
    """Return whether a generic NetCDF cache satisfies its contract."""
    path = Path(path)
    if not path.exists():
        return False, "file does not exist"
    try:
        with xr.open_dataset(path) as dataset:
            validate_cache_attrs(dataset, expected_attrs, required_variables=required_variables)
    except Exception as exc:
        return False, str(exc)
    return True, "compatible"


def prepared_skill_cache_status(
    path: str | Path,
    *,
    expected_attrs: Mapping[str, str | int],
) -> tuple[bool, str]:
    """Return whether a prepared cache exists and satisfies its contract."""
    path = Path(path)
    if not path.exists():
        return False, "file does not exist"
    try:
        with xr.open_dataset(path) as dataset:
            validate_prepared_skill_dataset(dataset, expected_attrs)
    except Exception as exc:
        return False, str(exc)
    return True, "compatible"


def write_prepared_skill_dataset(dataset: xr.Dataset, path: str | Path) -> Path:
    """Atomically write a validated prepared-skill dataset."""
    missing_attrs = REQUIRED_ATTRS - set(dataset.attrs)
    if missing_attrs:
        raise ValueError(f"Prepared skill cache is missing attributes: {sorted(missing_attrs)}")
    expected = expected_prepared_skill_attrs(
        source=str(dataset.attrs["source"]),
        source_data_identity=str(dataset.attrs["source_data_identity"]),
        case_prefix=str(dataset.attrs["case_prefix"]),
        component=str(dataset.attrs["component"]),
        variable=str(dataset.attrs["variable"]),
        init_month=int(dataset.attrs["initialization_month"]),
        climatology_years=(
            int(dataset.attrs["climatology_start_year"]),
            int(dataset.attrs["climatology_end_year"]),
        ),
        requested_years=json.loads(str(dataset.attrs["requested_years"])),
        target_grid=str(dataset.attrs["target_grid"]),
        regridding_method=str(dataset.attrs["regridding_method"]),
        ensemble_member_count=int(dataset.attrs["ensemble_member_count"]),
        lead_count=int(dataset.attrs["lead_count"]),
        unit_conversion_version=str(dataset.attrs["unit_conversion_version"]),
        temporal_resolution=str(dataset.attrs["temporal_resolution"]),
    )
    validate_prepared_skill_dataset(dataset, expected)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    encoding = {
        name: {"zlib": True, "complevel": 1}
        for name in dataset.data_vars
        if dataset[name].ndim > 0
    }
    try:
        dataset.to_netcdf(temporary, encoding=encoding)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def open_prepared_skill_dataset(
    path: str | Path,
    *,
    expected_attrs: Mapping[str, str | int],
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open a prepared input lazily and validate its complete contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    dataset = xr.open_dataset(path, chunks=dict(chunks or {}))
    try:
        validate_prepared_skill_dataset(dataset, expected_attrs)
    except Exception:
        dataset.close()
        raise
    return dataset


__all__ = [
    "build_prepared_skill_dataset",
    "cache_status",
    "expected_prepared_skill_attrs",
    "open_prepared_skill_dataset",
    "prepared_skill_cache_status",
    "prepared_skill_path",
    "validate_cache_attrs",
    "validate_prepared_skill_dataset",
    "write_prepared_skill_dataset",
]
