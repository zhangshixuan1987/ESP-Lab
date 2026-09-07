"""
discover.py
===========
Step 3: Discover expected daily output files and construct raw daily inventory.

Wraps ``daily_io.discover_all_daily_files``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.daily_io import discover_all_daily_files

from .config import DEFAULT_OUTPUT_ROOT, build_daily_config
from esp_lab.diagnostics.daily_core import DailyDriftConfig


def run(
    pilot_only: bool = True,
    variable: str | None = None,
    season: str | None = None,
    verbose: bool = True,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    config: DailyDriftConfig | None = None,
) -> "pd.DataFrame":
    import pandas as pd

    pm = {"may": [5], "nov": [11], "november": [11], "both": [5, 11]}.get(
        (season or "both").lower(), [5, 11]
    )

    if config is None:
        config = build_daily_config(
            pilot_only=pilot_only,
            pilot_months=pm if pilot_only else [5, 11],
            variable_filter=[variable] if variable else None,
            output_root=output_root,
        )
    output_root = config.output_root

    if verbose:
        print("=" * 70)
        print("Daily S2D — Step 3: File Discovery")
        print("=" * 70)
        print(f"  Experiments : {list(config.experiments)}")
        print(f"  Init years  : {config.active_years}")
        print(f"  Init months : {config.active_months}")
        print(f"  Members     : {len(config.members)} ({config.members[0]}–{config.members[-1]})")
        print(f"  Variables   : {[v.native_field for v in config.variables]}")
        print(f"  Data dir    : {config.data_dir}")
        print()

    raw_df = discover_all_daily_files(config, variable=variable, verbose=verbose)

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "daily_inventory_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    if verbose:
        print(f"\n  Raw daily inventory → {raw_csv}  ({len(raw_df)} rows)")

    return raw_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily S2D Step 3: discover expected output files."
    )
    parser.add_argument("--full-campaign", action="store_true")
    parser.add_argument("--variable", default=None)
    parser.add_argument("--season", default=None, choices=["may", "nov", "both"])
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
