#!/usr/bin/env python3
"""
03_compute_ic_statistics.py
============================
Step 3: Compute variable-level IC difference statistics for every file
        flagged DIFFERENT in the Step 1 manifest.

What this script does
---------------------
1. Load manifests from Step 1 and variable classifications from Step 2.
2. For each DIFFERENT (date, component) pair and each *physical* variable:
   - Load both restart fields into memory (or via Dask).
   - Compute ic_variable_stats: RMSE, MAD, mean_diff, pattern_corr,
     frac_differing, integral diff.  Area-weighted where possible.
3. Write per-date statistics CSVs and NetCDF diff files.

Usage
-----
    python 03_compute_ic_statistics.py                    # pilot only
    python 03_compute_ic_statistics.py --full-campaign
    python 03_compute_ic_statistics.py --date 1980-05-01-00000
    python 03_compute_ic_statistics.py --components ocn lnd

Outputs
-------
    output/variable_statistics/<date>_<component>_stats.csv
    output/variable_statistics/<date>_<component>_diff.nc   (ΔX fields)
"""

from __future__ import annotations

import argparse
import sys
import warnings
import re
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.ic_core import (
    AuditResult,
    classify_variable,
    ic_variable_stats,
)
from esp_lab.diagnostics.ic_io import (
    load_audit_csv,
    open_restart_file,
)

from .config import build_ic_config, load_config


# ---------------------------------------------------------------------------
# Area-weight loader for MPAS components
# ---------------------------------------------------------------------------

def _load_area_weights(ds: xr.Dataset, component: str) -> xr.DataArray | None:
    """Attempt to extract native cell-area weights from a restart dataset."""
    # MPAS ocean and sea ice
    if component in ("ocn", "ice"):
        for var in ("areaCell", "area", "cellArea"):
            if var in ds:
                return ds[var]
    # ELM land (area is not always in the restart; skip)
    return None


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    config_path: Path,
    pilot_only: bool | None = None,
    single_date: str | None = None,
    filter_components: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run variable-level IC statistics computation."""
    cfg = load_config(config_path)
    ic_cfg = build_ic_config(cfg, pilot_only=pilot_only)

    out_root = Path(config_path.parent) / ic_cfg.output_root
    subdirs = cfg.get("output", {}).get("subdirs", {})
    manifests_dir = out_root / subdirs.get("manifests", "manifests")
    fc_dir = out_root / subdirs.get("file_comparison", "file_comparison")
    vs_dir = out_root / subdirs.get("variable_statistics", "variable_statistics")
    vs_dir.mkdir(parents=True, exist_ok=True)

    active_dates = [single_date] if single_date else ic_cfg.active_dates
    nc_chunks = cfg.get("netcdf", {}).get("chunks", {}) or None

    if verbose:
        print("=" * 70)
        print("IC Analysis — Step 3: Compute Variable-Level IC Statistics")
        print("=" * 70)

    all_stat_rows: list[dict] = []

    for date in active_dates:
        manifest_csv = manifests_dir / f"{date}.csv"
        if not manifest_csv.exists():
            if verbose:
                print(f"  [SKIP] {date} — manifest not found")
            continue

        manifest = load_audit_csv(manifest_csv)
        diff_rows = manifest[manifest["status"] == AuditResult.DIFFERENT.value]

        # Load pre-computed variable classifications if available
        classif_csv = fc_dir / f"{date}_variable_classification.csv"
        known_physical: dict[str, set[str]] = {}  # comp → set of physical vars
        if classif_csv.exists():
            classif_df = pd.read_csv(classif_csv)
            for comp, grp in classif_df.groupby("component"):
                phys = set(
                    grp.loc[
                        (grp["classification"] == "physical") &
                        (grp["location"] == "common"),
                        "variable",
                    ]
                )
                known_physical[comp] = phys

        for _, row in diff_rows.iterrows():
            comp = row["component"]
            if filter_components and comp not in filter_components:
                continue

            ref_path = Path(row["ref_path"])
            test_path = Path(row["test_path"])
            if not ref_path.is_file() or not test_path.is_file():
                continue

            member_value = row.get("member", "")
            member = "nomember" if pd.isna(member_value) or not str(member_value) else str(member_value)
            member_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", member)

            if verbose:
                print(f"\n  [{date}] {comp} {member}")

            try:
                ds_ref = open_restart_file(ref_path, chunks=nc_chunks)
                ds_test = open_restart_file(test_path, chunks=nc_chunks)
            except Exception as exc:
                if verbose:
                    print(f"    Cannot open files: {exc}")
                continue

            # Determine which variables to process
            comp_spec = next(
                (c for c in ic_cfg.components if c.name == comp), None
            )
            priority_kws = comp_spec.priority_keywords if comp_spec else None

            if comp in known_physical:
                target_vars = known_physical[comp]
            else:
                # Fall back to classifying variables on the fly
                target_vars = {
                    v for v in ds_ref.data_vars
                    if classify_variable(v, comp, priority_kws) == "physical"
                    and v in ds_test.data_vars
                }

            area_weights = _load_area_weights(ds_ref, comp)

            stat_rows: list[dict] = []
            for var in sorted(target_vars):
                try:
                    # Materialize one variable pair at a time.  Chunked restart
                    # reads remain Dask-backed up to this compute boundary.
                    da_ref, da_test = dask.compute(ds_ref[var], ds_test[var])

                    # Use area weights only if dimensions align
                    aw = None
                    if area_weights is not None:
                        try:
                            aw = area_weights.broadcast_like(da_ref)
                        except Exception:
                            pass

                    stats = ic_variable_stats(da_ref, da_test, area_weights=aw)
                    stat_row = {
                        "start_date": date,
                        "component": comp,
                        "member": member,
                        "variable": var,
                        "season": ic_cfg.date_season(date),
                        **stats,
                    }
                    stat_rows.append(stat_row)
                    all_stat_rows.append(stat_row)

                    # Persist each variable immediately so a large component
                    # never accumulates every restart field in memory.
                    diff_da = (da_test - da_ref).rename(var)
                    var_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", var)
                    diff_path = vs_dir / f"{date}_{member_token}_{comp}_{var_token}_diff.nc"
                    diff_ds = diff_da.to_dataset(name=var)
                    diff_ds.attrs.update({
                        "ref_experiment": ic_cfg.experiment_pair.ref_label,
                        "test_experiment": ic_cfg.experiment_pair.test_label,
                        "start_date": date,
                        "component": comp,
                        "member": member,
                    })
                    diff_ds.to_netcdf(
                        diff_path,
                        encoding={var: {"zlib": True, "complevel": 2}},
                    )

                    if verbose:
                        print(
                            f"    {var:40s}  RMSE={stats['rmse']:.4g}  "
                            f"MAD={stats['mad']:.4g}  "
                            f"r={stats['pattern_corr']:.3f}"
                        )
                except Exception as exc:
                    warnings.warn(f"    {var}: {exc}", stacklevel=2)
                    continue

            ds_ref.close()
            ds_test.close()

            # Write per-component statistics CSV
            if stat_rows:
                stat_df = pd.DataFrame(stat_rows)
                stat_df.to_csv(
                    vs_dir / f"{date}_{member_token}_{comp}_stats.csv", index=False
                )

    combined = pd.DataFrame(all_stat_rows)
    if not combined.empty:
        combined.to_csv(vs_dir / "all_variable_stats.csv", index=False)
        if verbose:
            print("\n--- RMSE Summary (mean across dates) ---")
            summary = (
                combined.groupby(["component", "variable"])["rmse"]
                .mean()
                .reset_index()
                .sort_values(["component", "rmse"], ascending=[True, False])
            )
            print(summary.to_string(index=False))

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC Analysis Step 3: Compute variable-level IC statistics."
    )
    parser.add_argument("--config", default=str(_SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--full-campaign", action="store_true")
    parser.add_argument("--date", default=None)
    parser.add_argument(
        "--components", nargs="+", default=None,
        help="Limit to specific components, e.g. --components ocn lnd"
    )
    args = parser.parse_args()
    run(
        config_path=Path(args.config),
        pilot_only=False if args.full_campaign else None,
        single_date=args.date,
        filter_components=args.components,
    )


if __name__ == "__main__":
    main()
