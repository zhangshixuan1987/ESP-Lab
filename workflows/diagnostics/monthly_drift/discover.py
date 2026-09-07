"""
discover.py
===========
Step 1: Discover expected monthly output files and build the raw inventory.

Wraps ``monthly_io.discover_all_files``.  Can be imported by inventory.py
or run as a standalone script.

Usage
-----
    # Pilot mode (default)
    python -m workflows.diagnostics.monthly_drift.discover

    # Full campaign
    python -m workflows.diagnostics.monthly_drift.discover --full-campaign

    # Single variable
    python -m workflows.diagnostics.monthly_drift.discover --variable PRECT

    # Single init month
    python -m workflows.diagnostics.monthly_drift.discover --season may

Outputs
-------
    {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/monthly_spatial/monthly_inventory_raw.csv
    (before gate classification)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.monthly_io import discover_all_files

from .config import DEFAULT_OUTPUT_ROOT, build_monthly_config
from esp_lab.diagnostics.monthly_core import MonthlyConfig


def run(
    pilot_only: bool = True,
    variable: str | None = None,
    season: str | None = None,
    verbose: bool = True,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    config: MonthlyConfig | None = None,
) -> "pd.DataFrame":
    """Run the file discovery step.

    Returns
    -------
    Raw inventory DataFrame (before gate classification).
    """
    import pandas as pd

    pm = {"may": [5], "nov": [11], "november": [11], "both": [5, 11]}.get(
        (season or "both").lower(), [5, 11]
    )

    if config is None:
        config = build_monthly_config(
            pilot_only=pilot_only,
            pilot_months=pm if pilot_only else [5, 11],
            variable_filter=[variable] if variable else None,
            output_root=output_root,
        )
    output_root = config.output_root

    if verbose:
        print("=" * 70)
        print("Monthly S2D — Step 1: File Discovery")
        print("=" * 70)
        print(f"  Experiments : {list(config.experiments)}")
        print(f"  Init years  : {config.active_years}")
        print(f"  Init months : {config.active_months}")
        print(f"  Members     : {len(config.members)} ({config.members[0]}–{config.members[-1]})")
        print(f"  Variables   : {[v.native_name for v in config.variables]}")
        print(f"  Data dir    : {config.data_dir}")
        print()

    raw_df = discover_all_files(config, variable=variable, verbose=verbose)

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "monthly_inventory_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    if verbose:
        print(f"\n  Raw inventory → {raw_csv}  ({len(raw_df)} rows)")

    return raw_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly S2D Step 1: discover expected output files."
    )
    parser.add_argument("--full-campaign", action="store_true",
                        help="Process all init years (overrides pilot_only).")
    parser.add_argument("--variable", default=None, help="Limit to one variable.")
    parser.add_argument("--season", default=None,
                        choices=["may", "nov", "both"],
                        help="Limit to one init season.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    run(
        pilot_only=not args.full_campaign,
        variable=args.variable,
        season=args.season,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
