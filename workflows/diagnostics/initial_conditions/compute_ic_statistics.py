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
import hashlib
import json
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


def _grid_integrity(da: xr.DataArray) -> tuple[str, str]:
    """Return a stable native-grid signature and explicit coordinate ordering."""
    payload = {"dims": [(dim, int(da.sizes[dim])) for dim in da.dims], "coords": {}}
    ordering = []
    for name in da.dims:
        if name not in da.coords or da.coords[name].ndim != 1:
            ordering.append(f"{name}:index")
            continue
        values = np.asarray(da.coords[name].values)
        if values.size < 2 or not np.issubdtype(values.dtype, np.number):
            direction = "index"
        else:
            delta = np.diff(values.astype(float))
            direction = "ascending" if np.all(delta > 0) else (
                "descending" if np.all(delta < 0) else "nonmonotonic"
            )
        ordering.append(f"{name}:{direction}")
        payload["coords"][name] = {
            "size": int(values.size), "first": str(values[0]) if values.size else None,
            "last": str(values[-1]) if values.size else None, "ordering": direction,
        }
    for name, coord in da.coords.items():
        if coord.ndim != 1 or name in payload["coords"]:
            continue
        values = np.ascontiguousarray(coord.values)
        payload["coords"][name] = {
            "dims": list(coord.dims), "size": int(values.size),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest()[:16],
        }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16], ",".join(ordering)


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
    if component in ("lnd", "rof"):
        for var in ("area", "AREA", "cellArea", "areacella"):
            if var not in ds:
                continue
            weights = ds[var]
            if component == "lnd":
                for fraction in ("landfrac", "LANDFRAC", "land_fraction"):
                    if fraction in ds:
                        try:
                            weights = weights * ds[fraction]
                        except Exception:
                            pass
                        break
            return weights
    return None


def _load_volume_weights(
    ds: xr.Dataset, da: xr.DataArray, area_weights: xr.DataArray | None
) -> xr.DataArray | None:
    """Return MPAS cell volume when layer thickness is compatible."""
    if area_weights is None or "nVertLevels" not in da.dims:
        return None
    for name in ("layerThickness", "restingThickness"):
        if name not in ds:
            continue
        thickness = ds[name]
        extra_dims = [d for d in thickness.dims if d not in da.dims]
        if extra_dims:
            thickness = thickness.isel({d: 0 for d in extra_dims})
        try:
            return (area_weights * thickness).broadcast_like(da)
        except Exception:
            continue
    return None


def _meaningful_threshold(
    cfg: dict, component: str, variable: str, da_ref: xr.DataArray
) -> tuple[float, str]:
    """Resolve a variable threshold, falling back to a scale-aware rule."""
    diagnostics = cfg.get("diagnostics", {})
    overrides = diagnostics.get("meaningful_thresholds", {})
    component_overrides = overrides.get(component, {})
    if variable in component_overrides:
        return float(component_overrides[variable]), "configured_absolute"

    sigma_fraction = float(diagnostics.get("default_threshold_sigma_fraction", 0.01))
    ref_std = float(da_ref.std(skipna=True).values)
    if np.isfinite(ref_std) and ref_std > 0:
        return sigma_fraction * ref_std, f"{sigma_fraction:g}_reference_sigma"
    return float(diagnostics.get("constant_field_threshold", 1.0e-10)), "numerical_floor"


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
                    ref_var = ds_ref[var]
                    test_var = ds_test[var]
                    if "nCells" in ref_var.dims:
                        native_coords = {
                            name: ds_ref[name]
                            for name in ("latCell", "lonCell", "areaCell")
                            if name in ds_ref and ds_ref[name].dims == ("nCells",)
                        }
                        if native_coords:
                            ref_var = ref_var.assign_coords(native_coords)
                            test_var = test_var.assign_coords(native_coords)
                    da_ref, da_test = dask.compute(ref_var, test_var)

                    # Use area weights only if dimensions align
                    aw = None
                    weighting = "unweighted"
                    if area_weights is not None:
                        try:
                            aw = area_weights.broadcast_like(da_ref)
                            weighting = (
                                "native_effective_land_area" if comp == "lnd"
                                else "native_cell_area"
                            )
                        except Exception:
                            pass
                    if comp == "ocn":
                        volume_weights = _load_volume_weights(ds_ref, da_ref, area_weights)
                        if volume_weights is not None:
                            aw = volume_weights
                            weighting = "native_cell_volume"

                    threshold, threshold_source = _meaningful_threshold(
                        cfg, comp, var, da_ref
                    )
                    stats = ic_variable_stats(
                        da_ref,
                        da_test,
                        area_weights=aw,
                        meaningful_threshold=threshold,
                    )
                    grid_signature, coordinate_ordering = _grid_integrity(da_ref)
                    stat_row = {
                        "start_date": date,
                        "component": comp,
                        "member": member,
                        "variable": var,
                        "season": ic_cfg.date_season(date),
                        "units": da_ref.attrs.get("units", ""),
                        "threshold_source": threshold_source,
                        "weighting": weighting,
                        "percentile_weighting": "unweighted",
                        "grid_signature": grid_signature,
                        "coordinate_ordering": coordinate_ordering,
                        **stats,
                    }
                    stat_rows.append(stat_row)
                    all_stat_rows.append(stat_row)

                    # Persist each variable immediately so a large component
                    # never accumulates every restart field in memory.
                    diff_da = (da_test - da_ref).rename("signed_difference")
                    ref_scale = max(abs(float(da_ref.mean(skipna=True).values)), threshold)
                    relative_da = (diff_da / np.maximum(np.abs(da_ref), ref_scale * 1.0e-6)).rename(
                        "relative_difference"
                    )
                    standard_scale = stats["reference_std"]
                    standardized_da = (
                        diff_da / standard_scale if np.isfinite(standard_scale) and standard_scale > 0
                        else xr.full_like(diff_da, np.nan)
                    ).rename("standardized_difference")
                    exceedance_da = (np.abs(diff_da) > threshold).astype("int8").rename(
                        "threshold_exceedance"
                    )
                    var_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", var)
                    diff_path = vs_dir / f"{date}_{member_token}_{comp}_{var_token}_diff.nc"
                    diff_ds = xr.merge([
                        diff_da,
                        relative_da,
                        standardized_da,
                        exceedance_da,
                    ])
                    diff_ds.attrs.update({
                        "ref_experiment": ic_cfg.experiment_pair.ref_label,
                        "test_experiment": ic_cfg.experiment_pair.test_label,
                        "start_date": date,
                        "component": comp,
                        "member": member,
                        "source_variable": var,
                        "source_units": da_ref.attrs.get("units", ""),
                        "meaningful_threshold": threshold,
                        "threshold_source": threshold_source,
                        "weighting": weighting,
                        "percentile_weighting": "unweighted",
                        "grid_signature": grid_signature,
                        "coordinate_ordering": coordinate_ordering,
                    })
                    diff_ds.to_netcdf(
                        diff_path,
                        encoding={
                            name: {"zlib": True, "complevel": 2}
                            for name in diff_ds.data_vars
                        },
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
