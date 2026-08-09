#!/usr/bin/env python3
"""
02_compare_netcdf_structure.py
==============================
Step 2: For every file flagged DIFFERENT in the Step 1 manifest, compare
        the NetCDF variable/dimension schema and classify variables as
        physical, metadata-only, or unknown.

What this script does
---------------------
1. Load the manifests from Step 1.
2. Filter to rows with status == DIFFERENT or INCOMPATIBLE_STRUCTURE.
3. For each such (date, component) pair:
   - Open both restart files (metadata only; no data read into memory).
   - Compute a StructureDiff (common vars, only_ref, only_test, dim mismatches).
   - Classify each variable as 'physical', 'metadata', or 'unknown'.
4. Write per-date schema comparison CSVs to output/file_comparison/.
5. Print a summary table.

Usage
-----
    # Pilot date only (reads manifests from output/manifests/)
    python 02_compare_netcdf_structure.py

    # Full campaign
    python 02_compare_netcdf_structure.py --full-campaign

    # Single date
    python 02_compare_netcdf_structure.py --date 1980-05-01-00000

Outputs
-------
    output/file_comparison/<date>_<component>_schema.csv
    output/file_comparison/<date>_variable_classification.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.ic_core import (
    AuditResult,
    StructureDiff,
    classify_variable,
    compare_nc_schema,
)
from esp_lab.diagnostics.ic_io import (
    load_audit_csv,
    open_restart_file,
)

from .config import build_ic_config, load_config


def run(
    config_path: Path,
    pilot_only: bool | None = None,
    single_date: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the NetCDF structure comparison step.

    Returns
    -------
    DataFrame of variable classification results.
    """
    cfg = load_config(config_path)
    ic_cfg = build_ic_config(cfg, pilot_only=pilot_only)

    out_root = Path(config_path.parent) / ic_cfg.output_root
    manifests_dir = out_root / cfg.get("output", {}).get("subdirs", {}).get(
        "manifests", "manifests"
    )
    fc_dir = out_root / cfg.get("output", {}).get("subdirs", {}).get(
        "file_comparison", "file_comparison"
    )
    fc_dir.mkdir(parents=True, exist_ok=True)

    active_dates = [single_date] if single_date else ic_cfg.active_dates

    if verbose:
        print("=" * 70)
        print("IC Analysis — Step 2: NetCDF Structure Comparison")
        print("=" * 70)

    all_var_rows: list[dict] = []
    all_schema_rows: list[dict] = []

    for date in active_dates:
        manifest_csv = manifests_dir / f"{date}.csv"
        if not manifest_csv.exists():
            if verbose:
                print(f"  [SKIP] {date} — manifest not found: {manifest_csv}")
            continue

        manifest = load_audit_csv(manifest_csv)
        diff_rows = manifest[
            manifest["status"].isin([
                AuditResult.DIFFERENT.value,
                AuditResult.INCOMPATIBLE_STRUCTURE.value,
            ])
        ]

        if verbose:
            print(f"\n  Date: {date}  ({len(diff_rows)} DIFFERENT/INCOMPATIBLE rows)")

        for _, row in diff_rows.iterrows():
            comp = row["component"]
            ref_path = Path(row["ref_path"])
            test_path = Path(row["test_path"])

            if not ref_path.is_file() or not test_path.is_file():
                if verbose:
                    print(f"    [{comp}] File(s) not accessible; skipping schema.")
                continue

            try:
                ds_ref = open_restart_file(ref_path, chunks=None)
                ds_test = open_restart_file(test_path, chunks=None)
                diff: StructureDiff = compare_nc_schema(ds_ref, ds_test)
                ds_ref.close()
                ds_test.close()
            except Exception as exc:
                if verbose:
                    print(f"    [{comp}] Schema comparison failed: {exc}")
                continue

            # Schema-level summary row
            schema_row = {
                "start_date": date,
                "component": comp,
                "member": row.get("member", ""),
                "ref_path": str(ref_path),
                "test_path": str(test_path),
                **diff.to_dict(),
            }
            all_schema_rows.append(schema_row)

            # Variable classification rows
            comp_spec = next(
                (c for c in ic_cfg.components if c.name == comp), None
            )
            priority_kws = comp_spec.priority_keywords if comp_spec else None

            for var in diff.common_vars:
                classification = classify_variable(var, comp, priority_kws)
                all_var_rows.append({
                    "start_date": date,
                    "component": comp,
                    "variable": var,
                    "location": "common",
                    "classification": classification,
                })

            for var in diff.only_in_ref:
                all_var_rows.append({
                    "start_date": date,
                    "component": comp,
                    "variable": var,
                    "location": "only_ref",
                    "classification": classify_variable(var, comp, priority_kws),
                })

            for var in diff.only_in_test:
                all_var_rows.append({
                    "start_date": date,
                    "component": comp,
                    "variable": var,
                    "location": "only_test",
                    "classification": classify_variable(var, comp, priority_kws),
                })

            if verbose:
                print(
                    f"    [{comp}] common={len(diff.common_vars)} "
                    f"only_ref={len(diff.only_in_ref)} "
                    f"only_test={len(diff.only_in_test)} "
                    f"dim_mismatch={len(diff.dim_mismatches)}"
                )

        # Write per-date schema CSV
        date_schema_rows = [row for row in all_schema_rows if row["start_date"] == date]
        if date_schema_rows:
            schema_df = pd.DataFrame(date_schema_rows)
            schema_df.to_csv(fc_dir / f"{date}_schema.csv", index=False)

        # Write per-date variable classification CSV
        date_var_rows = [row for row in all_var_rows if row["start_date"] == date]
        if date_var_rows:
            var_df = pd.DataFrame(date_var_rows)
            var_df.to_csv(fc_dir / f"{date}_variable_classification.csv", index=False)

    # Combined output
    combined_var = pd.DataFrame(all_var_rows)
    if not combined_var.empty:
        combined_var.to_csv(fc_dir / "all_variable_classifications.csv", index=False)
        if verbose:
            print("\n--- Variable Classification Summary ---")
            print(
                combined_var.groupby(["component", "classification"])
                .size()
                .unstack(fill_value=0)
                .to_string()
            )

    return combined_var


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC Analysis Step 2: Compare NetCDF structure of DIFFERENT files."
    )
    parser.add_argument("--config", default=str(_SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--full-campaign", action="store_true")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    run(
        config_path=Path(args.config),
        pilot_only=False if args.full_campaign else None,
        single_date=args.date,
    )


if __name__ == "__main__":
    main()
