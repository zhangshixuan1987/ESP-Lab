"""
run_unified.py
==============
Top-level CLI orchestrator for the Unified S2D Diagnostic Architecture (5f).

Executes:
  0. Consume standardized 5b--5e products
  A. Field Drift (B^obs, D^obs, J)
  B. Physical Consistency (EF, BR, coupling slopes, R_apparent)
  C. Model Attractor using an independent historical climatology
  D. IC-to-drift spatial alignment and across-start attribution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd
import xarray as xr

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from . import branch_a_drift as _b_a
from . import branch_b_physics as _b_b
from . import branch_c_attractor as _b_c
from . import branch_d_attribution as _b_d
from . import inventory as _inv
from . import references as _ref
from .config import DEFAULT_OUTPUT_ROOT, DAILY_WINDOWS, MONTHLY_WINDOWS
from esp_lab.diagnostics.products import (
    combine_product_bundles,
    standardize_product_table,
    write_product_bundle,
)
from esp_lab.diagnostics.store import config_fingerprint


def run(
    frequency: str = "monthly",
    variable: str = "TREFHT",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_field: xr.DataArray | None = None,
    obs_ref: xr.DataArray | None = None,
    e3sm_clim: xr.DataArray | None = None,
    physics_fields: Mapping[str, xr.DataArray] | None = None,
    source_paths: dict | None = None,
    product_bundles: list[str | Path] | None = None,
    ic_difference: xr.DataArray | None = None,
    paired_drift: xr.DataArray | None = None,
    spatial_weights: xr.DataArray | None = None,
    start_indicator: xr.DataArray | None = None,
    regional_drift_response: xr.DataArray | None = None,
    baseline_lead: int = 1,
    verbose: bool = True,
) -> dict:
    if verbose:
        print("\n" + "=" * 70)
        print("Unified S2D Diagnostic Architecture — Master Pipeline")
        print("=" * 70)
        print(f"  Frequency : {frequency}")
        print(f"  Variable  : {variable}")
        print(f"  Outputs   : {output_root}")
        print()

    # Branch 0: read standardized products; do not recompute their diagnostics.
    product_table, product_manifests = combine_product_bundles(product_bundles or [])

    # Inventory & Readiness
    inv_summary = _inv.run_inventory(
        output_root=output_root, source_paths=source_paths, verbose=verbose
    )
    if model_field is None and not product_bundles:
        raise ValueError(
            "run_unified requires product_bundles from 5b--5e (preferred) or a "
            "validated model_field for backward-compatible direct execution."
        )

    lead_dim = "d" if frequency == "daily" else "L"
    windows = DAILY_WINDOWS if frequency == "daily" else MONTHLY_WINDOWS
    results = {
        "inventory": inv_summary,
        "products": {"table": product_table, "manifests": product_manifests},
    }
    if model_field is not None:
        results.update({
        "field_drift": _b_a.run_branch_a(
            model_field, obs_ref, baseline_lead=baseline_lead, lead_dim=lead_dim
        ),
        "model_attractor": _b_c.run_branch_c(
            model_field, obs_ref, e3sm_clim, windows,
            lead_dim=lead_dim, baseline_lead=baseline_lead,
        ),
        })
    if physics_fields is not None:
        required = ("lhflx_ref", "shflx_ref", "lhflx_test", "shflx_test")
        missing = [name for name in required if name not in physics_fields]
        if missing:
            raise ValueError(f"physics_fields missing required arrays: {missing}")
        results["physical_consistency"] = _b_b.run_branch_b(
            *(physics_fields[name] for name in required),
            window_defs=windows, lead_dim=lead_dim,
        )
    if (ic_difference is None) != (paired_drift is None):
        raise ValueError("ic_difference and paired_drift must be supplied together.")
    if ic_difference is not None:
        spatial_dims = tuple(
            dim for dim in ("lat", "lon", "ncol")
            if dim in ic_difference.dims and dim in paired_drift.dims
        )
        if not spatial_dims:
            raise ValueError("Could not infer common spatial dimensions for Branch D.")
        results["ic_drift_attribution"] = _b_d.run_branch_d(
            ic_difference, paired_drift, spatial_dims=spatial_dims,
            weights=spatial_weights, start_indicator=start_indicator,
            regional_drift_response=regional_drift_response,
        )
    attribution_rows = []
    for group, dataset in results.get("ic_drift_attribution", {}).items():
        for metric_name, array in dataset.data_vars.items():
            if metric_name == "sample_count":
                continue
            attribution_rows.append({
                "component": "matched_native_grid", "variable": variable,
                "units": array.attrs.get("units", "dimensionless"),
                "native_grid_id": "matched_ic_drift_grid", "region": "global",
                "lead_units": "day" if frequency == "daily" else "month",
                "baseline": f"lead {baseline_lead}",
                "metric_name": f"attribution_{group}_{metric_name}",
                "value": float(array.mean(skipna=True)),
                "sample_count": int(dataset["sample_count"].max())
                if "sample_count" in dataset else None,
            })
    if not product_table.empty or attribution_rows:
        source_hashes = [manifest["configuration_hash"] for manifest in product_manifests]
        synthesis_hash = config_fingerprint(
            {"frequency": frequency, "variable": variable, "baseline_lead": baseline_lead},
            variable=variable, context={"source_configuration_hashes": source_hashes},
        )
        source_rows = product_table.copy()
        if not source_rows.empty:
            source_rows["workflow"] = "5f_unified_synthesis"
            source_rows["configuration_hash"] = synthesis_hash
        attribution_table = (
            standardize_product_table(
                attribution_rows, workflow="5f_unified_synthesis",
                configuration_hash=synthesis_hash,
            )
            if attribution_rows else pd.DataFrame(columns=source_rows.columns)
        )
        synthesis_table = pd.concat([source_rows, attribution_table], ignore_index=True)
        write_product_bundle(
            Path(output_root) / "products", synthesis_table,
            workflow="5f_unified_synthesis", configuration_hash=synthesis_hash,
            metadata={
                "source_manifests": [str(path) for path in (product_bundles or [])],
                "causal_language": "association unless spatial, cross-start, and pathway evidence agree",
            },
        )
        results["synthesis_product_table"] = synthesis_table
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Unified S2D Orchestrator.")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="monthly")
    parser.add_argument("--variable", default="TREFHT")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    run(frequency=args.frequency, variable=args.variable, output_root=args.output_root)


if __name__ == "__main__":
    main()
