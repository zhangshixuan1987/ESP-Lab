"""Standardized products shared by the 5a--5f diagnostic workflows.

The table is intentionally long/tidy: one row represents one metric for one
experiment, initialization, lead/window, variable, and spatial selection.
Large native-grid fields remain in NetCDF diagnostic stores and are referenced
from the accompanying manifest rather than expanded into CSV rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .store import _jsonable


PRODUCT_SCHEMA_VERSION = "1.0.0"
REFERENCE_EXPERIMENT = "Reanalysis"
TEST_EXPERIMENT = "JRA55_FOSIRL"
PAIRED_SIGN_CONVENTION = f"{TEST_EXPERIMENT} - {REFERENCE_EXPERIMENT}"

PRODUCT_COLUMNS = [
    "schema_version", "workflow", "configuration_hash",
    "reference_experiment", "test_experiment", "sign_convention",
    "start_date", "init_year", "init_month", "member_aggregation",
    "component", "variable", "units",
    "native_grid_id", "region", "layer_depth",
    "lead_units", "lead_start", "lead_end", "baseline",
    "metric_name", "value", "uncertainty_lower", "uncertainty_upper",
    "reference_product", "sample_count", "bootstrap_unit",
    "significance_method",
]


def resolve_experiment_roles(labels: Iterable[str]) -> tuple[str, str]:
    """Return ``(reference, test)`` without relying on mapping insertion order."""
    ordered = list(labels)
    if REFERENCE_EXPERIMENT in ordered and TEST_EXPERIMENT in ordered:
        return REFERENCE_EXPERIMENT, TEST_EXPERIMENT
    if len(ordered) != 2:
        raise ValueError(f"Exactly two experiment labels are required; got {ordered}")
    return ordered[0], ordered[1]


def validate_product_table(table: pd.DataFrame, *, require_values: bool = True) -> None:
    """Raise ``ValueError`` when a product table violates the shared contract."""
    missing = [name for name in PRODUCT_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"Product table is missing required columns: {missing}")
    versions = set(table["schema_version"].dropna().astype(str))
    if versions - {PRODUCT_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported product schema versions: {sorted(versions)}")
    paired = table["metric_name"].fillna("").astype(str).str.contains(
        r"paired|delta|difference|attribution", case=False, regex=True
    )
    signs = set(table.loc[paired, "sign_convention"].dropna().astype(str))
    if signs - {PAIRED_SIGN_CONVENTION}:
        raise ValueError(
            "Paired products must use sign convention "
            f"{PAIRED_SIGN_CONVENTION!r}; found {sorted(signs)}"
        )
    if require_values and table["value"].isna().all():
        raise ValueError("Product table contains no finite metric values.")


def standardize_product_table(
    rows: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    workflow: str,
    configuration_hash: str,
    defaults: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return rows in the canonical column order with provenance defaults."""
    table = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    supplied = dict(defaults or {})
    supplied.update({
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "workflow": workflow,
        "configuration_hash": configuration_hash,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "test_experiment": TEST_EXPERIMENT,
        "sign_convention": PAIRED_SIGN_CONVENTION,
        "member_aggregation": "ensemble_mean_then_paired_init_year",
        "bootstrap_unit": "initialization_year",
    })
    for name in PRODUCT_COLUMNS:
        if name not in table:
            table[name] = supplied.get(name, np.nan)
        elif name in supplied:
            table[name] = table[name].fillna(supplied[name])
    table = table[PRODUCT_COLUMNS]
    validate_product_table(table)
    return table


def write_product_bundle(
    output_dir: str | Path,
    table: pd.DataFrame,
    *,
    workflow: str,
    configuration_hash: str,
    field_paths: Iterable[str | Path] = (),
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a tidy CSV and manifest atomically and return their paths."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    canonical = standardize_product_table(
        table, workflow=workflow, configuration_hash=configuration_hash
    )
    if set(canonical["workflow"].dropna().astype(str)) != {workflow}:
        raise ValueError("Product table workflow does not match bundle workflow.")
    if set(canonical["configuration_hash"].dropna().astype(str)) != {configuration_hash}:
        raise ValueError("Product table configuration hash does not match bundle hash.")
    table_path = root / "products.csv"
    table_tmp = root / "products.csv.tmp"
    canonical.to_csv(table_tmp, index=False)
    table_tmp.replace(table_path)

    manifest = {
        "schema_version": PRODUCT_SCHEMA_VERSION,
        "workflow": workflow,
        "configuration_hash": configuration_hash,
        "reference_experiment": REFERENCE_EXPERIMENT,
        "test_experiment": TEST_EXPERIMENT,
        "sign_convention": PAIRED_SIGN_CONVENTION,
        "table": table_path.name,
        "field_paths": [str(Path(path)) for path in field_paths],
        "metadata": _jsonable(metadata or {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
    }
    manifest_path = root / "product_manifest.json"
    manifest_tmp = root / "product_manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    manifest_tmp.replace(manifest_path)
    return table_path, manifest_path


def read_product_bundle(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and validate a bundle from its directory or manifest path."""
    source = Path(path)
    manifest_path = source / "product_manifest.json" if source.is_dir() else source
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete product bundle: {manifest_path}")
    if manifest.get("schema_version") != PRODUCT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported product schema {manifest.get('schema_version')!r} in {manifest_path}"
        )
    table_path = manifest_path.parent / manifest["table"]
    table = pd.read_csv(table_path)
    validate_product_table(table)
    if set(table["workflow"].dropna().astype(str)) != {str(manifest["workflow"])}:
        raise ValueError(f"Table/manifest workflow mismatch in {manifest_path}")
    if set(table["configuration_hash"].dropna().astype(str)) != {
        str(manifest["configuration_hash"])
    }:
        raise ValueError(f"Table/manifest configuration mismatch in {manifest_path}")
    return table, manifest


def combine_product_bundles(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load compatible 5b--5e products for synthesis in 5f."""
    tables: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for path in paths:
        table, manifest = read_product_bundle(path)
        tables.append(table)
        manifests.append(manifest)
    if not tables:
        return pd.DataFrame(columns=PRODUCT_COLUMNS), manifests
    combined = pd.concat(tables, ignore_index=True)
    validate_product_table(combined)
    return combined, manifests


def drift_results_to_product_table(
    results: Mapping[int, Mapping[str, Any]],
    bootstrap: Mapping[int, Mapping[str, Any]],
    *,
    workflow: str,
    configuration_hash: str,
    variable: str,
    units: str,
    component: str,
    lead_units: str,
    baseline: str,
    window_defs: Mapping[str, tuple[int, int]],
    region: str = "global",
    native_grid_id: str = "unknown",
    reference_product: str | None = None,
) -> pd.DataFrame:
    """Reduce monthly/daily store results to canonical scalar and start rows."""
    rows: list[dict[str, Any]] = []
    for init_month, month_results in sorted(results.items()):
        for window, (lead_start, lead_end) in window_defs.items():
            values = month_results.get(window)
            if not isinstance(values, Mapping):
                continue
            boot = bootstrap.get(init_month, {}).get(window, {})
            for key in (
                "bias_ref", "bias_test", "adj_ref", "adj_test",
                "paired_diff", "paired_effect_size",
            ):
                array = values.get(key)
                if array is None:
                    continue
                scalar = float(array.mean(skipna=True)) if hasattr(array, "mean") else float(array)
                row = {
                    "init_month": int(init_month), "component": component,
                    "variable": variable, "units": units,
                    "native_grid_id": native_grid_id, "region": region,
                    "lead_units": lead_units, "lead_start": int(lead_start),
                    "lead_end": int(lead_end), "baseline": baseline,
                    "metric_name": key, "value": scalar,
                    "reference_product": reference_product,
                }
                if key == "paired_diff":
                    lower = boot.get("lower")
                    if lower is None:
                        lower = boot.get("ci_lower")
                    upper = boot.get("upper")
                    if upper is None:
                        upper = boot.get("ci_upper")
                    if lower is not None:
                        row["uncertainty_lower"] = float(lower.mean(skipna=True))
                    if upper is not None:
                        row["uncertainty_upper"] = float(upper.mean(skipna=True))
                rows.append(row)

            fdr = values.get("fdr_significant")
            if fdr is not None:
                rows.append({
                    "init_month": int(init_month), "component": component,
                    "variable": variable, "units": "fraction",
                    "native_grid_id": native_grid_id, "region": region,
                    "lead_units": lead_units, "lead_start": int(lead_start),
                    "lead_end": int(lead_end), "baseline": baseline,
                    "metric_name": "fdr_significant_fraction",
                    "value": float(fdr.mean(skipna=True)),
                    "reference_product": reference_product,
                    "significance_method": "Benjamini-Hochberg FDR",
                })

            by_year = values.get("paired_diff_by_year")
            if by_year is not None and "Y" in by_year.dims:
                reduce_dims = [dim for dim in by_year.dims if dim != "Y"]
                start_values = by_year.mean(reduce_dims, skipna=True) if reduce_dims else by_year
                for year, value in zip(start_values.Y.values, start_values.values):
                    rows.append({
                        "start_date": f"{int(year):04d}-{int(init_month):02d}-01",
                        "init_year": int(year), "init_month": int(init_month),
                        "component": component, "variable": variable, "units": units,
                        "native_grid_id": native_grid_id, "region": region,
                        "lead_units": lead_units, "lead_start": int(lead_start),
                        "lead_end": int(lead_end), "baseline": baseline,
                        "metric_name": "paired_diff_by_start", "value": float(value),
                        "reference_product": reference_product,
                        "sample_count": 1,
                    })

        full_start = min(first for first, _ in window_defs.values())
        full_end = max(last for _, last in window_defs.values())
        for key in (
            "maximum_adjustment_rate", "day_of_maximum_adjustment",
            "sign_reversal_or_overshoot", "cumulative_adjustment",
            "early_late_adjustment_ratio",
        ):
            array = month_results.get(key)
            if array is None:
                continue
            rows.append({
                "init_month": int(init_month), "component": component,
                "variable": variable, "units": units,
                "native_grid_id": native_grid_id, "region": region,
                "lead_units": lead_units, "lead_start": int(full_start),
                "lead_end": int(full_end), "baseline": baseline,
                "metric_name": key, "value": float(array.mean(skipna=True)),
                "reference_product": reference_product,
            })
    return standardize_product_table(
        rows, workflow=workflow, configuration_hash=configuration_hash
    )


__all__ = [
    "PAIRED_SIGN_CONVENTION", "PRODUCT_COLUMNS", "PRODUCT_SCHEMA_VERSION",
    "REFERENCE_EXPERIMENT", "TEST_EXPERIMENT", "combine_product_bundles",
    "read_product_bundle", "standardize_product_table", "validate_product_table",
    "write_product_bundle", "drift_results_to_product_table", "resolve_experiment_roles",
]
