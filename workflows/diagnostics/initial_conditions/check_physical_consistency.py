#!/usr/bin/env python3
"""
05_check_physical_consistency.py
=================================
Step 5: Check cross-component physical consistency at coupled interfaces.

Checks performed
----------------
1. Atmosphere surface temperature vs land surface temperature
2. Ocean SST vs sea-ice concentration (below-freezing with no ice)
3. (Optional) Land soil moisture vs atmospheric humidity / precipitation
4. Coupler timestamps vs component restart timestamps

Both the ref and test experiments are checked independently (not differenced),
because inconsistency within one set of ICs is a separate issue from
BruteForce–FOSIRL differences.

Usage
-----
    python 05_check_physical_consistency.py
    python 05_check_physical_consistency.py --full-campaign
    python 05_check_physical_consistency.py --date 1980-05-01-00000

Outputs
-------
    output/file_comparison/<date>_consistency_ref.csv
    output/file_comparison/<date>_consistency_test.csv
    output/file_comparison/consistency_report.csv   (combined)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import xarray as xr

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.ic_core import (
    AuditResult,
    ConsistencyReport,
    check_atm_surface_vs_land,
    check_ocean_ice_consistency,
)
from esp_lab.diagnostics.ic_io import (
    load_audit_csv,
    open_restart_file,
)

from .config import build_ic_config, load_config


# ---------------------------------------------------------------------------
# Variable name discovery helpers
# ---------------------------------------------------------------------------

def _find_var(ds: xr.Dataset, candidates: list[str]) -> xr.DataArray | None:
    """Return the first DataArray whose name is in candidates, or None."""
    for v in candidates:
        if v in ds:
            return ds[v]
    return None


_ATM_TS_CANDIDATES = ["TS", "TREFHT", "tbot", "T_srf", "ts"]
_LND_TGRND_CANDIDATES = ["T_grnd", "tgrnd", "TSOI_T_grnd", "tskin", "t_grnd"]
_OCN_SST_CANDIDATES = ["temperature", "activeTracers,temperature", "SST", "sst", "T"]
_ICE_AICE_CANDIDATES = ["aice", "iceConcentration", "aicen", "SIC"]


def _check_timestamp_consistency(
    ds_atm: xr.Dataset | None,
    ds_lnd: xr.Dataset | None,
    ds_ocn: xr.Dataset | None,
    ds_ice: xr.Dataset | None,
    ds_cpl: xr.Dataset | None,
    experiment: str,
    date: str,
) -> list[dict]:
    """Compare simulation timestamps across components."""
    rows = []
    components = {
        "atm": ds_atm, "lnd": ds_lnd, "ocn": ds_ocn,
        "ice": ds_ice, "cpl": ds_cpl,
    }
    timestamps: dict[str, str] = {}
    for comp, ds in components.items():
        if ds is None:
            continue
        if "time" in ds.coords and ds.coords["time"].size > 0:
            timestamps[comp] = str(ds.coords["time"].values[0])
        else:
            for attr in ("current_year", "year", "stop_n"):
                if attr in ds.attrs:
                    timestamps[comp] = str(ds.attrs[attr])
                    break

    # Check all pairs
    comp_list = list(timestamps.keys())
    for i in range(len(comp_list)):
        for j in range(i + 1, len(comp_list)):
            c1, c2 = comp_list[i], comp_list[j]
            match = timestamps[c1] == timestamps[c2]
            rows.append({
                "check_name": f"timestamp_{c1}_vs_{c2}",
                "experiment": experiment,
                "start_date": date,
                "n_points": 1,
                "mean_abs_mismatch": 0.0 if match else 1.0,
                "max_abs_mismatch": 0.0 if match else 1.0,
                "frac_inconsistent": 0.0 if match else 1.0,
                "threshold": 0.0,
                "n_warnings": 0 if match else 1,
                "notes": f"{c1}={timestamps[c1]}  {c2}={timestamps[c2]}",
            })
    return rows


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    config_path: Path,
    pilot_only: bool | None = None,
    single_date: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    ic_cfg = build_ic_config(cfg, pilot_only=pilot_only)

    out_root = Path(config_path.parent) / ic_cfg.output_root
    subdirs = cfg.get("output", {}).get("subdirs", {})
    manifests_dir = out_root / subdirs.get("manifests", "manifests")
    fc_dir = out_root / subdirs.get("file_comparison", "file_comparison")
    fc_dir.mkdir(parents=True, exist_ok=True)

    active_dates = [single_date] if single_date else ic_cfg.active_dates
    nc_chunks = cfg.get("netcdf", {}).get("chunks", {}) or None

    if verbose:
        print("=" * 70)
        print("IC Analysis — Step 5: Check Physical Consistency")
        print("=" * 70)

    all_rows: list[dict] = []

    for date in active_dates:
        manifest_csv = manifests_dir / f"{date}.csv"
        if not manifest_csv.exists():
            if verbose:
                print(f"  [SKIP] {date} — manifest not found")
            continue

        manifest = load_audit_csv(manifest_csv)

        # Build a path lookup: {comp: {experiment: path}}
        path_lookup: dict[str, dict[str, str]] = {}
        for _, row in manifest.iterrows():
            comp = row["component"]
            path_lookup.setdefault(comp, {})
            exp_ref = ic_cfg.experiment_pair.ref_label
            exp_test = ic_cfg.experiment_pair.test_label
            # Use the first (non-member) file for non-atm components
            if not row.get("member"):
                path_lookup[comp][exp_ref] = row["ref_path"]
                path_lookup[comp][exp_test] = row["test_path"]

        for exp_label in [
            ic_cfg.experiment_pair.ref_label,
            ic_cfg.experiment_pair.test_label,
        ]:
            exp_key = "ref_path" if exp_label == ic_cfg.experiment_pair.ref_label \
                else "test_path"

            def _open(comp: str) -> xr.Dataset | None:
                row_match = manifest[manifest["component"] == comp].copy()
                if row_match.empty:
                    return None
                # Prefer a component-level file; atmosphere restarts are
                # member-specific, so use EN00 as the deterministic check.
                member = row_match["member"].fillna("")
                preferred = row_match[(member == "") | (member == "EN00")]
                if not preferred.empty:
                    row_match = preferred
                p = Path(row_match.iloc[0][exp_key])
                if not p.is_file():
                    return None
                try:
                    return open_restart_file(p, chunks=nc_chunks)
                except Exception as exc:
                    warnings.warn(f"Cannot open {p}: {exc}", stacklevel=3)
                    return None

            ds_atm = _open("atm")
            ds_lnd = _open("lnd")
            ds_ocn = _open("ocn")
            ds_ice = _open("ice")
            ds_cpl = _open("cpl")

            if verbose:
                print(f"\n  {date}  [{exp_label}]")

            check_rows: list[dict] = []

            # Check 1: Atmosphere TS vs Land T_grnd
            da_atm_ts = _find_var(ds_atm, _ATM_TS_CANDIDATES) if ds_atm else None
            da_lnd_tgrnd = _find_var(ds_lnd, _LND_TGRND_CANDIDATES) if ds_lnd else None
            if da_atm_ts is not None and da_lnd_tgrnd is not None:
                try:
                    rpt = check_atm_surface_vs_land(
                        da_atm_ts.load(), da_lnd_tgrnd.load()
                    )
                    row = {**rpt.to_dict(), "experiment": exp_label, "start_date": date}
                    check_rows.append(row)
                    if verbose:
                        print(
                            f"    atm_vs_land: frac_inconsistent="
                            f"{rpt.frac_inconsistent:.3f}  "
                            f"warnings={rpt.warnings}"
                        )
                except Exception as exc:
                    warnings.warn(f"    atm_vs_land check: {exc}", stacklevel=2)
            else:
                if verbose:
                    print("    atm_vs_land: SKIPPED (vars not found)")

            # Check 2: Ocean SST vs sea-ice concentration
            da_sst = _find_var(ds_ocn, _OCN_SST_CANDIDATES) if ds_ocn else None
            da_aice = _find_var(ds_ice, _ICE_AICE_CANDIDATES) if ds_ice else None
            if da_sst is not None and da_aice is not None:
                try:
                    # Use surface layer for 3-D ocean temperature
                    if da_sst.ndim > 2:
                        da_sst = da_sst.isel(**{
                            d: 0 for d in da_sst.dims
                            if d not in ("nCells", "lat", "lon", "x", "y")
                        })
                    # Collapse ensemble dimension for ice
                    if da_aice.ndim > 2:
                        da_aice = da_aice.mean(
                            [d for d in da_aice.dims if d not in ("nCells", "lat", "lon")],
                            skipna=True,
                        )
                    rpt = check_ocean_ice_consistency(da_sst.load(), da_aice.load())
                    row = {**rpt.to_dict(), "experiment": exp_label, "start_date": date}
                    check_rows.append(row)
                    if verbose:
                        print(
                            f"    ocean_vs_ice: frac_inconsistent="
                            f"{rpt.frac_inconsistent:.4f}  "
                            f"warnings={rpt.warnings}"
                        )
                except Exception as exc:
                    warnings.warn(f"    ocean_vs_ice check: {exc}", stacklevel=2)
            else:
                if verbose:
                    print("    ocean_vs_ice: SKIPPED (vars not found)")

            # Check 3: Timestamp consistency across components
            ts_rows = _check_timestamp_consistency(
                ds_atm, ds_lnd, ds_ocn, ds_ice, ds_cpl, exp_label, date
            )
            check_rows.extend(ts_rows)
            if verbose and ts_rows:
                mismatch_ts = [r for r in ts_rows if r["frac_inconsistent"] > 0]
                if mismatch_ts:
                    print(f"    timestamp mismatches: {[r['check_name'] for r in mismatch_ts]}")
                else:
                    print(f"    all timestamps consistent")

            all_rows.extend(check_rows)

            # Write per-date per-experiment CSV
            if check_rows:
                exp_slug = exp_label.replace(" ", "_").replace("/", "_")
                pd.DataFrame(check_rows).to_csv(
                    fc_dir / f"{date}_consistency_{exp_slug}.csv", index=False
                )

            # Close datasets
            for ds in (ds_atm, ds_lnd, ds_ocn, ds_ice, ds_cpl):
                if ds is not None:
                    ds.close()

    combined = pd.DataFrame(all_rows)
    if not combined.empty:
        combined.to_csv(fc_dir / "consistency_report.csv", index=False)
        if verbose:
            print("\n--- Consistency Report Summary ---")
            print(combined[["check_name", "experiment", "frac_inconsistent", "n_warnings"]].to_string(index=False))

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC Analysis Step 5: Check cross-component physical consistency."
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
