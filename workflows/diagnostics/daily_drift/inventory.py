"""
inventory.py
============
Steps 4–5: Build the daily inventory, apply window-specific completeness,
           evaluate paired readiness gate, and write daily report files.

Recommended inventory columns (27 total):
  experiment, case_prefix, init_date, init_year, init_month, member, variable,
  stream, file_count, first_timestamp, last_timestamp, expected_days,
  available_days, missing_days, duplicate_days, variable_found, units,
  calendar, grid_signature, member_valid, week_1_status, weeks_2_3_status,
  weeks_4_6_status, weeks_7_12_status, daily_status, paired_status, notes

Outputs
-------
  {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/daily_spatial/daily_inventory.csv
  {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/daily_spatial/daily_missing_data_report.txt
  {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/daily_spatial/daily_inventory_summary.json
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esp_lab.diagnostics.daily_core import (
    DailyDriftConfig,
    DailyGateStatus,
    DailyLeadCoverageResult,
    DailyPairedReadinessReport,
    build_daily_paired_readiness,
    check_daily_lead_coverage,
    classify_daily_analysis_status,
    classify_daily_window_status,
    write_daily_inventory_csv,
    write_daily_inventory_json,
    write_daily_missing_report,
)
from esp_lab.data_access_e3sm import build_init_tags

from .config import DEFAULT_OUTPUT_ROOT, build_daily_config


def _apply_daily_gate(df: pd.DataFrame, config) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        avail = [int(x) for x in str(row.get("available_days", "")).split(",") if x.strip()]
        dup   = [int(x) for x in str(row.get("duplicate_days", "")).split(",") if x.strip()]

        expected = config.lead_days
        missing  = sorted(set(expected) - set(avail))
        coverage = DailyLeadCoverageResult(
            available_days=sorted(avail),
            expected_days=sorted(expected),
            missing_days=missing,
            duplicate_days=sorted(dup),
            extra_days=[],
        )

        win_statuses = classify_daily_window_status(coverage, config.window_defs)
        overall_status = classify_daily_analysis_status(coverage, config.window_defs)

        row = row.copy()
        row["week_1_status"]     = win_statuses.get("week_1", DailyGateStatus.BLOCKED).value
        row["weeks_2_3_status"]  = win_statuses.get("weeks_2_3", DailyGateStatus.BLOCKED).value
        row["weeks_4_6_status"]  = win_statuses.get("weeks_4_6", DailyGateStatus.BLOCKED).value
        row["weeks_7_12_status"] = win_statuses.get("weeks_7_12", DailyGateStatus.BLOCKED).value
        row["daily_status"]      = overall_status.value
        rows.append(row)

    return pd.DataFrame(rows)


def run(
    pilot_only: bool = True,
    variable: str | None = None,
    season: str | None = None,
    skip_discovery: bool = False,
    verbose: bool = True,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    config: DailyDriftConfig | None = None,
) -> tuple[pd.DataFrame, DailyPairedReadinessReport]:
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

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("Daily S2D — Steps 4–5: Daily Inventory & Paired Readiness Gate")
        print("=" * 70)

    raw_csv = out_dir / "daily_inventory_raw.csv"
    if skip_discovery and raw_csv.exists():
        raw_df = pd.read_csv(raw_csv, dtype=str)
        if verbose:
            print(f"  Loaded existing raw daily inventory: {raw_csv}")
    else:
        from .discover import run as discover_run
        raw_df = discover_run(
            pilot_only=pilot_only,
            variable=variable,
            season=season,
            verbose=verbose,
            output_root=output_root,
            config=config,
        )

    df = _apply_daily_gate(raw_df, config)

    experiments = list(config.experiments.keys())
    if len(experiments) < 2:
        raise ValueError("Need at least 2 experiments for daily pairing.")

    ref_label, test_label = experiments[0], experiments[1]
    init_tags = build_init_tags(config.active_years, config.active_months)

    ref_inv  = df[df["experiment"] == ref_label]
    test_inv = df[df["experiment"] == test_label]

    report = build_daily_paired_readiness(
        ref_inventory=ref_inv,
        test_inventory=test_inv,
        window_defs=config.window_defs,
        init_tags=init_tags,
        members=config.members,
    )

    paired_complete = report.common_paired_inits
    df["paired_status"] = df.apply(
        lambda row: "PAIRED_COMPLETE"
        if (row["init_tag"], row["member"]) in paired_complete
        else "NOT_PAIRED",
        axis=1,
    )

    if verbose:
        print("\n" + report.format_report())
        print("\nDaily analysis status breakdown:")
        print(df.groupby(["experiment", "daily_status"]).size().unstack(fill_value=0))

    inv_csv  = write_daily_inventory_csv(df, out_dir / "daily_inventory.csv")
    rpt_txt  = write_daily_missing_report(df, out_dir / "daily_missing_data_report.txt")
    json_out = write_daily_inventory_json(
        df,
        out_dir / "daily_inventory_summary.json",
        extra_meta={
            "pilot_only":       config.pilot_only,
            "ref_experiment":   ref_label,
            "test_experiment":  test_label,
            "n_common_paired":  len(report.common_paired_inits),
            "gate_pass":        report.gate_pass,
        },
    )

    if verbose:
        print(f"\n  Daily Inventory → {inv_csv}")
        print(f"  Missing Report  → {rpt_txt}")
        print(f"  Summary JSON    → {json_out}")

    if not report.gate_pass:
        msg = "Daily paired readiness gate FAILED: no complete paired combinations found."
        if config.gate_strict:
            print(f"\n[STRICT GATE FAIL] {msg}", file=sys.stderr)
            sys.exit(1)
        else:
            warnings.warn(msg, stacklevel=2)
    else:
        if verbose:
            print(f"\n  Daily Gate PASS: {len(paired_complete)} paired complete combinations.")

    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily S2D Steps 4–5: inventory & paired readiness gate."
    )
    parser.add_argument("--full-campaign", action="store_true")
    parser.add_argument("--variable", default=None)
    parser.add_argument("--season", default=None, choices=["may", "nov", "both"])
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    run(
        pilot_only=not args.full_campaign,
        variable=args.variable,
        season=args.season,
        skip_discovery=args.skip_discovery,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
