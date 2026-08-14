"""
run_physical.py
===============
Top-level CLI orchestrator for the S2D Physical-Consistency Diagnostic Branch.

Executes:
  1. Flux partitioning (EF & Bowen ratio)
  2. Land coupling (SM -> LH, SM -> TREFHT)
  3. Precip-SM lag response (PRECT -> ΔSM)
  4. Ocean coupling (SST vs TREFHT)
  5. Apparent surface energy residual
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Mapping

import xarray as xr
from esp_lab.diagnostics.products import resolve_experiment_roles
from esp_lab.diagnostics.products import standardize_product_table, write_product_bundle
from esp_lab.diagnostics.store import config_fingerprint

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from . import energy_budget as _eb
from . import flux_partitioning as _fp
from . import land_coupling as _lc
from . import ocean_coupling as _oc
from . import precip_sm_response as _psr
from . import water_budget as _wb
from .config import DEFAULT_OUTPUT_ROOT


def run(
    frequency: str = "daily",
    season: str = "may",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    fields_by_experiment: Mapping[str, Mapping[str, xr.DataArray]] | None = None,
    window_defs: Mapping[str, tuple[int, int]] | None = None,
    soil_layer_thickness: Mapping[str, list[float]] | None = None,
    write_products: bool = False,
    verbose: bool = True,
) -> dict:
    if verbose:
        print("\n" + "=" * 70)
        print("S2D Physical-Consistency Diagnostics Branch")
        print("=" * 70)
        print(f"  Frequency : {frequency}")
        print(f"  Season    : {season}")
        print(f"  Outputs   : {output_root}")
        print()

    if fields_by_experiment is None:
        raise ValueError(
            "run_physical requires fields_by_experiment={ref: {...}, test: {...}}. "
            "Use physical_io.load_physical_variables_daily/monthly after the "
            "inventory gate, then pass the resulting lazy DataArrays here."
        )
    labels = list(fields_by_experiment)
    if len(labels) != 2:
        raise ValueError("Physical diagnostics require exactly two experiments.")
    ref_label, test_label = resolve_experiment_roles(labels)
    ref = fields_by_experiment[ref_label]
    test = fields_by_experiment[test_label]
    lead_dim = "d" if frequency == "daily" else "L"
    if window_defs is None:
        from .config import DAILY_WINDOWS, MONTHLY_WINDOWS
        window_defs = DAILY_WINDOWS if frequency == "daily" else MONTHLY_WINDOWS

    results: dict = {}

    def _have(*names: str) -> bool:
        return all(name in ref and name in test for name in names)

    if _have("LHFLX", "SHFLX"):
        results["flux_partitioning"] = _fp.run_flux_partitioning(
            ref["LHFLX"], ref["SHFLX"], test["LHFLX"], test["SHFLX"],
            dict(window_defs), day_dim=lead_dim,
        )
    if _have("H2OSOI", "LHFLX", "TREFHT"):
        thickness = soil_layer_thickness or {}
        results["land_coupling"] = _lc.run_land_coupling(
            ref["H2OSOI"], ref["LHFLX"], ref["TREFHT"],
            test["H2OSOI"], test["LHFLX"], test["TREFHT"],
            dict(window_defs), day_dim=lead_dim,
            layer_thickness_ref=thickness.get(ref_label),
            layer_thickness_test=thickness.get(test_label),
        )
    if frequency == "daily" and _have("PRECT", "H2OSOI"):
        results["precip_sm_response"] = _psr.run_precip_sm_response(
            ref["PRECT"], ref["H2OSOI"], test["PRECT"], test["H2OSOI"],
            day_dim=lead_dim,
        )
    if _have("TREFHT", "TS"):
        results["ocean_coupling"] = _oc.run_ocean_coupling(
            ref["TREFHT"], ref["TS"], test["TREFHT"], test["TS"],
            dict(window_defs), day_dim=lead_dim,
        )
    if _have("FSNS", "FLNS", "LHFLX", "SHFLX"):
        results["energy_budget"] = _eb.run_energy_budget(
            ref["FSNS"], ref["FLNS"], ref["LHFLX"], ref["SHFLX"],
            test["FSNS"], test["FLNS"], test["LHFLX"], test["SHFLX"],
            dict(window_defs), day_dim=lead_dim,
        )
    if _have("DSTORAGE_DT", "PRECT", "ET", "RUNOFF"):
        results["land_water_budget"] = _wb.run_water_budget(
            ref["DSTORAGE_DT"], ref["PRECT"], ref["ET"], ref["RUNOFF"],
            test["DSTORAGE_DT"], test["PRECT"], test["ET"], test["RUNOFF"],
            dict(window_defs), lead_dim=lead_dim,
        )

    if not results:
        raise ValueError("No physical diagnostic has all of its required fields.")

    rows = []
    for diagnostic, diagnostic_results in results.items():
        if not isinstance(diagnostic_results, Mapping):
            continue
        # Most branches are keyed by lead window; lag diagnostics are represented
        # as one all-leads group.
        groups = diagnostic_results
        if any(isinstance(value, xr.DataArray) for value in groups.values()):
            groups = {"all_leads": diagnostic_results}
        for window, metrics in groups.items():
            if not isinstance(metrics, Mapping):
                continue
            first, last = window_defs.get(window, (None, None))
            for metric_name, value in metrics.items():
                if not isinstance(value, xr.DataArray):
                    continue
                rows.append({
                    "init_month": {"may": 5, "nov": 11}.get(season),
                    "component": "coupled", "variable": diagnostic,
                    "units": value.attrs.get("units", "diagnostic_native_units"),
                    "native_grid_id": "native_component_grid",
                    "region": "global",
                    "lead_units": "day" if frequency == "daily" else "month",
                    "lead_start": first, "lead_end": last,
                    "baseline": "metric evaluated within each lead window",
                    "metric_name": metric_name,
                    "value": float(value.mean(skipna=True)),
                    "sample_count": value.sizes.get("Y"),
                })
    if rows and write_products:
        fingerprint = config_fingerprint(
            {"frequency": frequency, "season": season, "windows": dict(window_defs)},
            variable="physical_consistency",
        )
        product_table = standardize_product_table(
            rows, workflow="5e_physical_consistency", configuration_hash=fingerprint,
            defaults={
                "reference_product": "5c/5d validated native-grid products",
                "significance_method": "paired initialization-year bootstrap where available",
            },
        )
        write_product_bundle(
            Path(output_root) / frequency / season / "products", product_table,
            workflow="5e_physical_consistency", configuration_hash=fingerprint,
            metadata={
                "use_synthetic_demo": False,
                "bowen_ratio_definition": "SHFLX / LHFLX",
                "energy_diagnostic": "apparent residual",
                "soil_layer_thickness_required": True,
            },
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="S2D Physical-Consistency Orchestrator.")
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="daily")
    parser.add_argument("--season", choices=["may", "nov", "both"], default="may")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    run(
        frequency=args.frequency, season=args.season,
        output_root=args.output_root, write_products=True,
    )


if __name__ == "__main__":
    main()
