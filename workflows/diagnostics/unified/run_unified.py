"""
run_unified.py
==============
Top-level CLI orchestrator for the Unified S2D Diagnostic Architecture (5f).

Executes:
  1. Multi-source Inventory & Readiness Gate
  2. Reference Assembly (O_X and M_X)
  3. Branch A: Field Drift (B^obs, D^obs, J)
  4. Branch B: Physical Consistency (EF, BR, coupling slopes, R_apparent)
  5. Branch C: Model Attractor (d^obs, d^model, δd^obs, δd^model, 4-Category Map, 2-Axis Plot)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import xarray as xr

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from . import branch_a_drift as _b_a
from . import branch_b_physics as _b_b
from . import branch_c_attractor as _b_c
from . import inventory as _inv
from . import references as _ref
from .config import DEFAULT_OUTPUT_ROOT, DAILY_WINDOWS, MONTHLY_WINDOWS


def run(
    frequency: str = "monthly",
    variable: str = "TREFHT",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_field: xr.DataArray | None = None,
    obs_ref: xr.DataArray | None = None,
    e3sm_clim: xr.DataArray | None = None,
    physics_fields: Mapping[str, xr.DataArray] | None = None,
    source_paths: dict | None = None,
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

    # Step 1: Inventory & Readiness
    inv_summary = _inv.run_inventory(
        output_root=output_root, source_paths=source_paths, verbose=verbose
    )
    if model_field is None:
        raise ValueError(
            "run_unified requires a validated model_field. The inventory report "
            "was written, but diagnostic branches were not marked complete."
        )

    lead_dim = "d" if frequency == "daily" else "L"
    windows = DAILY_WINDOWS if frequency == "daily" else MONTHLY_WINDOWS
    results = {
        "inventory": inv_summary,
        "field_drift": _b_a.run_branch_a(
            model_field, obs_ref, baseline_lead=1, lead_dim=lead_dim
        ),
        "model_attractor": _b_c.run_branch_c(
            model_field, obs_ref, e3sm_clim, windows,
            lead_dim=lead_dim, baseline_lead=1,
        ),
    }
    if physics_fields is not None:
        required = ("lhflx_ref", "shflx_ref", "lhflx_test", "shflx_test")
        missing = [name for name in required if name not in physics_fields]
        if missing:
            raise ValueError(f"physics_fields missing required arrays: {missing}")
        results["physical_consistency"] = _b_b.run_branch_b(
            *(physics_fields[name] for name in required),
            window_defs=windows, lead_dim=lead_dim,
        )
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
