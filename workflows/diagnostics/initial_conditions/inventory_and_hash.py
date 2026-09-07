#!/usr/bin/env python3
"""
01_inventory_and_hash.py
========================
Step 1: Discover matched initialization pairs and build a file-level
        SHA-256 audit manifest.

What this script does
---------------------
1. Load config.yaml and build ICConfig.
2. Discover all matched start-date directory pairs (BruteForce ↔ JRA55-FOSIRL).
3. For each matched date and component:
   - Compute SHA-256 for every atmospheric EN00–EN09 member (control check).
   - Compute one SHA-256 per non-atmospheric restart file.
   - Compare checksums → IDENTICAL | DIFFERENT | MISSING |
     INCOMPATIBLE_STRUCTURE.
4. Write manifests CSV to output/manifests/{date}.csv.
5. Write a combined JSON manifest.

Usage
-----
    # Pilot date only (default)
    python 01_inventory_and_hash.py

    # Full campaign
    python 01_inventory_and_hash.py --full-campaign

    # Single date
    python 01_inventory_and_hash.py --date 1980-05-01-00000

    # Skip hashing (just file existence check)
    python 01_inventory_and_hash.py --no-hash

Outputs
-------
    output/manifests/<date>.csv
    output/manifests/combined_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Resolve the package from the repository root so this script can be run
# directly without installing esp_lab.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.ic_core import (
    DEFAULT_COMPONENTS,
    AuditResult,
    ComponentSpec,
    ExperimentPair,
    ICConfig,
    write_manifest_json,
)
from esp_lab.diagnostics.ic_io import (
    AUDIT_MANIFEST_COLUMNS,
    build_file_manifest,
    load_audit_csv,
    match_start_dates,
    write_audit_csv,
)

from .config import build_ic_config, load_config



# ===========================================================================
# Main
# ===========================================================================


def run(
    config_path: Path,
    pilot_only: bool | None = None,
    single_date: str | None = None,
    compute_hash: bool = True,
    reuse_existing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the inventory and hash step.

    Parameters
    ----------
    config_path:
        Path to config.yaml.
    pilot_only:
        Override the YAML pilot_only setting.
    single_date:
        If provided, process only this date (overrides pilot/full-campaign).
    compute_hash:
        Whether to compute SHA-256 checksums (slower but essential for audit).
    reuse_existing:
        Reuse a complete, path-compatible saved manifest instead of hashing the
        same files again. Set to ``False`` after changing source files.
    verbose:
        Print progress to stdout.

    Returns
    -------
    Combined audit manifest DataFrame.
    """
    cfg = load_config(config_path)
    ic_cfg = build_ic_config(cfg, pilot_only=pilot_only)

    pair = ic_cfg.experiment_pair
    ref_root = Path(pair.ref_root)
    test_root = Path(pair.test_root)
    out_root = Path(config_path.parent) / ic_cfg.output_root
    manifests_dir = out_root / cfg.get("output", {}).get(
        "subdirs", {}
    ).get("manifests", "manifests")

    # Determine active dates
    if single_date:
        active_dates = [single_date]
    else:
        active_dates = ic_cfg.active_dates

    if verbose:
        print("=" * 70)
        print("IC Analysis — Step 1: Inventory and Hash")
        print("=" * 70)
        print(f"  ref  : {ref_root}")
        print(f"  test : {test_root}")
        print(f"  dates: {active_dates}")
        print(f"  hash : {compute_hash}")
        print()

    # Discover matched pairs for active dates
    all_pairs = match_start_dates(ref_root, test_root)
    active_pairs = [p for p in all_pairs if p.date_str in active_dates]

    if not active_pairs:
        warnings.warn(
            f"No matched pairs found for active dates={active_dates}. "
            "Check that ref_root and test_root exist and contain the expected "
            "YYYY-MM-DD-00000/ sub-directories.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=AUDIT_MANIFEST_COLUMNS)

    all_dfs = []
    for pair_obj in active_pairs:
        cached_csv = manifests_dir / f"{pair_obj.date_str}.csv"
        if reuse_existing and cached_csv.is_file():
            cached = load_audit_csv(cached_csv)
            required = set(AUDIT_MANIFEST_COLUMNS)
            ref_prefix = str(pair_obj.ref_dir) + os.sep
            test_prefix = str(pair_obj.test_dir) + os.sep
            paths_match = (
                not cached.empty
                and cached.get("ref_path", pd.Series(dtype=str)).fillna("").str.startswith(ref_prefix).all()
                and cached.get("test_path", pd.Series(dtype=str)).fillna("").str.startswith(test_prefix).all()
            )
            hashes_complete = (
                not compute_hash
                or (
                    cached.get("ref_sha256", pd.Series(dtype=str)).fillna("").ne("").all()
                    and cached.get("test_sha256", pd.Series(dtype=str)).fillna("").ne("").all()
                )
            )
            if required.issubset(cached.columns) and paths_match and hashes_complete:
                if verbose:
                    print(f"  [REUSE] {pair_obj.date_str} → {cached_csv}")
                all_dfs.append(cached[AUDIT_MANIFEST_COLUMNS])
                continue

        if verbose:
            status = "OK" if pair_obj.both_exist else "MISSING"
            print(f"  [{status}] {pair_obj.date_str}")

        df = build_file_manifest(
            pairs=[pair_obj],
            components=ic_cfg.components,
            ref_label=pair.ref_label,
            test_label=pair.test_label,
            compute_hash=compute_hash,
            atm_hash_all_members=ic_cfg.atm_hash_all_members,
            non_atm_hash_per_member=ic_cfg.non_atm_hash_per_member,
        )

        # Write per-date CSV
        csv_path = write_audit_csv(df, manifests_dir, prefix=pair_obj.date_str)
        if verbose:
            print(f"    → {csv_path}")

        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Write combined JSON manifest
    if not combined.empty:
        json_path = manifests_dir / "combined_manifest.json"
        records = combined.to_dict(orient="records")
        write_manifest_json(
            records,
            json_path,
            extra_meta={
                "ref_experiment": pair.ref_label,
                "test_experiment": pair.test_label,
                "n_dates": len(active_pairs),
            },
        )
        if verbose:
            print(f"\n  Combined manifest → {json_path}")

    # Print summary
    if verbose and not combined.empty:
        print("\n--- Audit Summary ---")
        print(combined.groupby(["component", "status"]).size().to_string())
        print()

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IC Analysis Step 1: Inventory and hash restart files."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--full-campaign",
        action="store_true",
        help="Process all start dates (overrides pilot_only in config).",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Process only this specific date (YYYY-MM-DD-00000).",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 computation (file existence check only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild manifests even when a compatible saved manifest exists.",
    )
    args = parser.parse_args()

    pilot_override = False if args.full_campaign else None
    run(
        config_path=Path(args.config),
        pilot_only=pilot_override,
        single_date=args.date,
        compute_hash=not args.no_hash,
        reuse_existing=not args.force,
    )


if __name__ == "__main__":
    main()
