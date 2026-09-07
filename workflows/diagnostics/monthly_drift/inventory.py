"""
inventory.py
============
Steps 2–4: Build the monthly inventory, apply gate classification,
           check paired readiness, and write all three output files.

Gate logic (window-aware)
-------------------------
  analysis_status:
      ANALYSIS_READY  — all leads required by the requested windows are present.
      BLOCKED         — at least one window lead is missing or duplicated.

  archive_status:
      ARCHIVE_COMPLETE — all 24 monthly leads are present.
      BLOCKED          — one or more of leads 1–24 are missing.

  Example: missing lead 20
    - months_1_3, months_4_6, months_10_12 → NOT blocked
    - second_year (13–24) → BLOCKED
    - archive → BLOCKED

Paired readiness gate
---------------------
  If gate_strict=True  → SystemExit(1) when gate fails (batch mode).
  If gate_strict=False → warning and continue (notebook mode).

Usage
-----
    python -m workflows.diagnostics.monthly_drift.inventory
    python -m workflows.diagnostics.monthly_drift.inventory --full-campaign
    python -m workflows.diagnostics.monthly_drift.inventory --variable TREFHT
    python -m workflows.diagnostics.monthly_drift.inventory --skip-discovery

Outputs
-------
    {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/monthly_spatial/monthly_inventory.csv
    {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/monthly_spatial/monthly_missing_data_report.txt
    {S2D_DIAG_ROOT}/multimodel/leadtime_drift/atm/monthly_spatial/monthly_inventory_summary.json
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

from esp_lab.diagnostics.monthly_core import (
    GateStatus,
    InventoryStatus,
    LeadCoverageResult,
    MonthlyConfig,
    PairedReadinessReport,
    build_paired_readiness,
    check_lead_coverage,
    classify_analysis_status,
    classify_archive_status,
    write_inventory_csv,
    write_inventory_json,
    write_missing_report,
)
from esp_lab.data_access_e3sm import build_init_tags

from .config import DEFAULT_OUTPUT_ROOT, build_monthly_config


def _apply_gate(df: pd.DataFrame, config) -> pd.DataFrame:
    """Re-classify analysis_status using window-aware gate logic."""
    rows = []
    for _, row in df.iterrows():
        avail  = [int(x) for x in str(row.get("available_leads", "")).split(",") if x.strip()]
        dup    = [int(x) for x in str(row.get("duplicate_leads", "")).split(",") if x.strip()]

        # Reconstruct LeadCoverageResult
        expected = config.leads
        missing  = sorted(set(expected) - set(avail))
        coverage = LeadCoverageResult(
            available_leads=sorted(avail),
            expected_leads=sorted(expected),
            missing_leads=missing,
            duplicate_leads=sorted(dup),
            extra_leads=[],
        )

        analysis_status = classify_analysis_status(coverage, config.window_defs)
        archive_status  = classify_archive_status(coverage, nlead=24)

        row = row.copy()
        row["analysis_status"] = analysis_status.value
        row["archive_status"]  = archive_status.value
        rows.append(row)

    return pd.DataFrame(rows)


def run(
    pilot_only: bool = True,
    variable: str | None = None,
    season: str | None = None,
    skip_discovery: bool = False,
    verbose: bool = True,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    config: MonthlyConfig | None = None,
) -> tuple[pd.DataFrame, PairedReadinessReport]:
    """Run inventory classification and paired readiness gate.

    Returns
    -------
    (inventory_df, paired_report)
    """
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

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("Monthly S2D — Steps 2–4: Inventory & Paired Readiness Gate")
        print("=" * 70)

    # Load raw inventory (run discover.py first, or skip_discovery=False)
    raw_csv = out_dir / "monthly_inventory_raw.csv"
    if skip_discovery and raw_csv.exists():
        raw_df = pd.read_csv(raw_csv, dtype=str)
        if verbose:
            print(f"  Loaded existing raw inventory: {raw_csv}")
    else:
        # Run discovery inline
        from .discover import run as discover_run
        raw_df = discover_run(
            pilot_only=pilot_only,
            variable=variable,
            season=season,
            verbose=verbose,
            output_root=output_root,
            config=config,
        )

    # Step 2: Re-classify with window-aware gate
    df = _apply_gate(raw_df, config)

    # Step 3: Paired readiness
    experiments = list(config.experiments.keys())
    if len(experiments) < 2:
        raise ValueError("Need at least 2 experiments for pairing.")

    ref_label, test_label = experiments[0], experiments[1]
    init_tags = build_init_tags(config.active_years, config.active_months)

    ref_inv  = df[df["experiment"] == ref_label]
    test_inv = df[df["experiment"] == test_label]

    report = build_paired_readiness(
        ref_inventory=ref_inv,
        test_inventory=test_inv,
        window_defs=config.window_defs,
        init_tags=init_tags,
        members=config.members,
    )

    # Add paired_status column
    paired_complete = report.common_paired_inits
    df["paired_status"] = df.apply(
        lambda row: "PAIRED_COMPLETE"
        if (row["init_tag"], row["member"]) in paired_complete
        else "NOT_PAIRED",
        axis=1,
    )

    if verbose:
        print("\n" + report.format_report())
        print("\nAnalysis status breakdown:")
        print(df.groupby(["experiment", "analysis_status"]).size().unstack(fill_value=0))
        print("\nArchive status breakdown:")
        print(df.groupby(["experiment", "archive_status"]).size().unstack(fill_value=0))

    # Step 4: Gate
    gate_ok = report.gate_pass
    inv_csv = write_inventory_csv(df, out_dir / "monthly_inventory.csv")
    rpt_txt = write_missing_report(
        df,
        out_dir / "monthly_missing_data_report.txt",
        config_summary=(
            f"years={config.active_years}, months={config.active_months}, "
            f"members={config.members}, leads=1–{max(config.leads)}"
        ),
    )
    json_out = write_inventory_json(
        df,
        out_dir / "monthly_inventory_summary.json",
        extra_meta={
            "pilot_only":       config.pilot_only,
            "ref_experiment":   ref_label,
            "test_experiment":  test_label,
            "n_common_paired":  len(report.common_paired_inits),
            "gate_pass":        gate_ok,
        },
    )

    if verbose:
        print(f"\n  Inventory    → {inv_csv}")
        print(f"  Missing rpt  → {rpt_txt}")
        print(f"  Summary JSON → {json_out}")

    if not gate_ok:
        msg = (
            "Paired readiness gate FAILED: no complete paired (init_tag, member) "
            "tuples found across both experiments.  Check monthly_missing_data_report.txt."
        )
        if config.gate_strict:
            print(f"\n[GATE FAIL] {msg}", file=sys.stderr)
            sys.exit(1)
        else:
            warnings.warn(msg, stacklevel=2)
    else:
        if verbose:
            print(
                f"\n  Gate PASS: {len(paired_complete)} paired complete "
                f"(init_tag, member) combinations."
            )

    return df, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monthly S2D Steps 2–4: Inventory and paired readiness gate."
    )
    parser.add_argument("--full-campaign", action="store_true")
    parser.add_argument("--variable", default=None)
    parser.add_argument("--season", default=None, choices=["may", "nov", "both"])
    parser.add_argument("--skip-discovery", action="store_true",
                        help="Reuse existing monthly_inventory_raw.csv.")
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
