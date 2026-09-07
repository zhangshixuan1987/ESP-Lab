"""Reusable global teleconnection products for modes-of-variability analyses."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr
from scipy import stats


TELECONNECTION_SCHEMA = 1
INDEX_VARIABLES = ("mode_index_projected", "mode_index", "pc", "PC", "index")
FIELD_VARIABLES = (
    "SST_anom", "PSL_anom", "sst_anom", "psl_anom", "anomaly",
    "anomaly_field", "sst", "SST", "psl", "PSL", "field",
)


def _first_variable(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.data_vars:
            return name
    raise KeyError(
        f"None of {list(candidates)} found; available={list(dataset.data_vars)}"
    )


def output_path(product: str, index_path: Path, field_path: Path) -> Path:
    """Return the source-local ancillary product path."""
    for candidate in (field_path, index_path):
        if "modes_variability" in candidate.parts:
            position = candidate.parts.index("modes_variability")
            root = Path(*candidate.parts[: position + 1])
            return (
                root / "regression_patterns"
                / f"{product.replace(':', '_')}_global_teleconnection.nc"
            )
    raise ValueError(
        "Cannot infer modes_variability root from "
        f"index={index_path} and field={field_path}"
    )


def _input_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def configuration_signature(
    product: str,
    index_path: Path,
    field_path: Path,
    *,
    alpha: float,
    standardize_index: bool,
    fdr: bool,
) -> str:
    payload = {
        "schema": TELECONNECTION_SCHEMA,
        "product": product,
        "index": _input_record(index_path),
        "field": _input_record(field_path),
        "alpha": alpha,
        "standardize_index": standardize_index,
        "fdr": fdr,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def output_issues(path: Path, expected_signature: str) -> list[str]:
    if not path.is_file():
        return [f"missing {path}"]
    try:
        with xr.open_dataset(path) as dataset:
            required = {
                "mode_global_regression_pattern",
                "mode_global_regression_sample_size",
            }
            absent = required - set(dataset.data_vars)
            if absent:
                return [f"{path} lacks {', '.join(sorted(absent))}"]
            if dataset.attrs.get("teleconnection_configuration") != expected_signature:
                return [f"incompatible {path}"]
            has_pattern = bool(
                dataset["mode_global_regression_pattern"].notnull().any().compute()
            )
            has_samples = bool(
                (dataset["mode_global_regression_sample_size"] >= 3).any().compute()
            )
            if not has_pattern or not has_samples:
                return [f"invalid {path}"]
    except Exception as error:
        return [f"unreadable {path}: {error}"]
    return []


def _fdr_mask(p_values: xr.DataArray, alpha: float) -> xr.DataArray:
    values = np.asarray(p_values.values, dtype=float)
    flat = values.ravel()
    finite = np.isfinite(flat)
    result = np.zeros(flat.shape, dtype=bool)
    if finite.any():
        probabilities = flat[finite]
        ranked = np.sort(probabilities)
        thresholds = alpha * np.arange(1, ranked.size + 1) / ranked.size
        passing = ranked <= thresholds
        if passing.any():
            result[finite] = probabilities <= ranked[np.flatnonzero(passing)[-1]]
    return xr.DataArray(
        result.reshape(values.shape), coords=p_values.coords, dims=p_values.dims
    )


def regression_map(
    field: xr.DataArray,
    index: xr.DataArray,
    *,
    standardize_index: bool,
    alpha: float,
    fdr: bool,
) -> xr.Dataset:
    """Regress a field onto an index across their common non-spatial axes."""
    field, index = xr.align(field, index, join="inner")
    spatial_dims = {"lat", "latitude", "LAT", "lon", "longitude", "LON"}
    sample_dims = [
        dim for dim in index.dims if dim in field.dims and dim not in spatial_dims
    ]
    if not sample_dims:
        raise ValueError(
            f"No common sample dimensions: field={field.dims}, index={index.dims}"
        )
    field_samples = field.stack(sample=sample_dims).transpose("sample", ...)
    index_samples = index.stack(sample=sample_dims).transpose("sample")
    finite_index = np.isfinite(index_samples)
    field_samples = field_samples.where(finite_index)
    index_samples = index_samples.where(finite_index)
    if standardize_index:
        index_samples = (
            (index_samples - index_samples.mean("sample"))
            / index_samples.std("sample")
        )
    else:
        index_samples = index_samples - index_samples.mean("sample")
    field_anomaly = field_samples - field_samples.mean("sample")
    index_anomaly = index_samples - index_samples.mean("sample")
    pair = field_anomaly * index_anomaly
    sample_size = np.isfinite(pair).sum("sample")
    denominator = xr.where(sample_size > 1, sample_size - 1, np.nan)
    covariance = pair.sum("sample") / denominator
    index_variance = (index_anomaly ** 2).sum("sample") / denominator
    field_variance = (field_anomaly ** 2).sum("sample") / denominator
    beta = covariance / index_variance
    correlation = covariance / np.sqrt(index_variance * field_variance)
    t_value = correlation * np.sqrt(
        (sample_size - 2)
        / xr.where(1 - correlation ** 2 > 0, 1 - correlation ** 2, np.nan)
    )
    p_value = xr.apply_ufunc(
        lambda values: 2.0 * stats.norm.sf(np.abs(values)),
        t_value,
        dask="parallelized",
        output_dtypes=[float],
    )
    significant = _fdr_mask(p_value, alpha) if fdr else p_value < alpha
    return xr.Dataset(
        {
            "regression_beta": beta,
            "regression_r": correlation,
            "regression_pvalue": p_value,
            "regression_significant": significant,
            "regression_sample_size": sample_size,
        }
    )


def _compute_dataset(
    index_path: Path,
    field_path: Path,
    *,
    standardize_index: bool,
    alpha: float,
    fdr: bool,
) -> tuple[xr.Dataset, str, str, str]:
    with xr.open_dataset(index_path) as index_dataset, xr.open_dataset(field_path) as field_dataset:
        index_name = _first_variable(index_dataset, INDEX_VARIABLES)
        field_name = _first_variable(field_dataset, FIELD_VARIABLES)
        index = index_dataset[index_name]
        field = field_dataset[field_name]
        grouping_dim = "target_month" if "target_month" in index_dataset.dims else "L"
        results = []
        if grouping_dim == "target_month":
            if "time" not in index.dims or "time" not in field.dims:
                raise ValueError("Reference index and field must share a time dimension")
            for value in range(1, 13):
                result = regression_map(
                    field.where(field.time.dt.month == value, drop=True),
                    index.where(index.time.dt.month == value, drop=True),
                    standardize_index=standardize_index, alpha=alpha, fdr=fdr,
                )
                results.append(result.expand_dims(target_month=[value]))
        else:
            if "L" not in index.dims or "L" not in field.dims:
                raise ValueError("Hindcast index and field must share an L dimension")
            for value in index.L.values:
                result = regression_map(
                    field.sel(L=value), index.sel(L=value),
                    standardize_index=standardize_index, alpha=alpha, fdr=fdr,
                )
                results.append(result.expand_dims(L=[value]))
        combined = xr.concat(results, grouping_dim).rename(
            {
                "regression_beta": "mode_global_regression_pattern",
                "regression_r": "mode_global_regression_r",
                "regression_pvalue": "mode_global_regression_pvalue",
                "regression_significant": "mode_global_regression_significant",
                "regression_sample_size": "mode_global_regression_sample_size",
            }
        ).load()
        combined["mode_global_regression_significant"] = combined[
            "mode_global_regression_significant"
        ].astype("int8")
        units = str(field.attrs.get("units", "")).strip()
        if units:
            combined["mode_global_regression_pattern"].attrs["units"] = units
        return combined, index_name, field_name, grouping_dim


def ensure_products(
    products: Mapping[str, Mapping[str, str]],
    *,
    ensure_mode: str,
    alpha: float = 0.05,
    standardize_index: bool = True,
    fdr: bool = True,
) -> dict[str, str]:
    """Validate or materialize teleconnections for the supplied products."""
    statuses: dict[str, str] = {}
    issues: list[str] = []
    for product, details in sorted(products.items()):
        index_path = Path(details["index"])
        field_path = Path(details["field"])
        missing_inputs = [path for path in (index_path, field_path) if not path.is_file()]
        if missing_inputs:
            issues.extend(f"missing {path}" for path in missing_inputs)
            continue
        destination = output_path(product, index_path, field_path)
        signature = configuration_signature(
            product, index_path, field_path, alpha=alpha,
            standardize_index=standardize_index, fdr=fdr,
        )
        current_issues = output_issues(destination, signature)
        if ensure_mode == "require":
            issues.extend(current_issues)
            statuses[product] = "valid" if not current_issues else "invalid"
            continue
        if not current_issues and ensure_mode != "rebuild":
            statuses[product] = "reused"
            continue
        dataset, index_name, field_name, grouping_dim = _compute_dataset(
            index_path, field_path, standardize_index=standardize_index,
            alpha=alpha, fdr=fdr,
        )
        if not bool(dataset["mode_global_regression_pattern"].notnull().any()):
            raise ValueError(f"Global teleconnection for {product} is entirely NaN")
        dataset.attrs.update(
            product=product,
            index_file=str(index_path), field_file=str(field_path),
            index_variable=index_name, field_variable=field_name,
            grouping_dimension=grouping_dim,
            method="global regression of field anomalies onto projected mode index",
            teleconnection_configuration=signature,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp.{uuid.uuid4().hex}")
        try:
            dataset.to_netcdf(temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        remaining = output_issues(destination, signature)
        if remaining:
            issues.extend(remaining)
        statuses[product] = "written"
    if issues:
        raise RuntimeError("Global teleconnection products are unavailable:\n  - " + "\n  - ".join(issues))
    return statuses
